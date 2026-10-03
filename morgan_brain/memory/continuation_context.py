"""Bounded exact source inspection; selections classify text but never verify it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from morgan_brain.memory.revisions import instant
from morgan_brain.models import Memory, MemoryKind, MemoryStatus

SECTIONS = ("current_facts", "completed_progress", "unresolved_questions", "relevant_constraints")
FIELDS = (
    "id",
    "user_id",
    "project",
    "kind",
    "content",
    "source",
    "author_id",
    "origin_kind",
    "scope",
    "instruction_like",
    "status",
    "created_at",
    "recorded_at",
    "revises_event_ids",
    "revision_root_id",
    "revision_state",
    "eligible_leaf_ids",
    "eligible_leaf_count",
    "revision_truncated",
    "support_event_ids",
    "support_state",
    "valid_from",
    "valid_to",
    "superseded_by",
    "last_confirmed",
    "confidence",
)


def validate_request(
    user_id: str,
    project: str,
    evidence_ids: list[str],
    selections: list[dict[str, Any]],
    effective_at: datetime | None,
) -> datetime:
    """Validate the entire caller request before any surface opens its database."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("inspect_context requires a named owner")
    if not isinstance(project, str) or not project.strip():
        raise ValueError("inspect_context requires a named project")
    if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 16:
        raise ValueError("inspect_context requires 1..16 source IDs")
    if any(not isinstance(i, str) or not i.strip() or len(i) > 256 for i in evidence_ids):
        raise ValueError("source IDs must be nonblank strings of at most 256 characters")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("duplicate source IDs")
    if not isinstance(selections, list) or len(selections) > 16:
        raise ValueError("inspect_context accepts at most 16 selections")
    spans: set[tuple[str, int, int]] = set()
    for item in selections:
        if not isinstance(item, dict) or set(item) != {
            "section",
            "event_id",
            "start",
            "end",
            "quote",
        }:
            raise ValueError("selection must have exactly section/event_id/start/end/quote")
        if not isinstance(item["section"], str) or item["section"] not in SECTIONS:
            raise ValueError("unknown selection section")
        if not isinstance(item["event_id"], str) or item["event_id"] not in evidence_ids:
            raise ValueError("selection source must be requested")
        start, end, quote = item["start"], item["end"], item["quote"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("selection offsets must be increasing Unicode code-point integers")
        if not isinstance(quote, str) or not quote.strip() or not 1 <= len(quote) <= 240:
            raise ValueError("quote must have 1..240 nonblank code points")
        if end - start != len(quote):
            raise ValueError("quote length must equal span length")
        span = (item["event_id"], start, end)
        if span in spans:
            raise ValueError("duplicate selection span")
        spans.add(span)
    cutoff = effective_at if effective_at is not None else datetime.now(UTC)
    if not isinstance(cutoff, datetime) or cutoff.utcoffset() is None:
        raise ValueError("effective_at must be a timezone-aware datetime")
    # Refuse malformed Unicode and unbounded owner/project/request strings before opening DB.
    if (
        len(json.dumps([user_id, project, evidence_ids, selections], ensure_ascii=False).encode())
        > 16384
    ):
        raise ValueError("inspection request exceeds 16384 UTF-8 bytes")
    return cutoff


def ineligible(memory: Memory, cutoff: datetime) -> str | None:
    if memory.status == MemoryStatus.QUARANTINED:
        return "quarantined"
    if memory.kind == MemoryKind.SEMANTIC:
        if (
            memory.valid_from is None
            or instant(memory.valid_from) > instant(cutoff)
            or (memory.valid_to is not None and instant(cutoff) >= instant(memory.valid_to))
        ):
            return "inactive_interval"
        if memory.support_state != "current":
            return "unsupported_revised_or_conflicted_fact_support"
        return None
    if memory.revision_truncated or memory.revision_state == "conflicted":
        return "conflicted_or_truncated_lineage"
    if (
        memory.kind != MemoryKind.EPISODIC
        or memory.status != MemoryStatus.STORED
        or memory.revision_state != "active"
        or (memory.created_at is not None and instant(memory.created_at) > instant(cutoff))
    ):
        return "inactive_or_future"
    return None


def source_projection(memory: Memory) -> dict[str, Any]:
    raw = memory.model_dump(mode="json")
    return {
        **{field: raw[field] for field in FIELDS},
        "effective_at": raw["valid_from"]
        if memory.kind == MemoryKind.SEMANTIC
        else raw["created_at"],
    }


def assemble(
    *,
    user_id: str,
    project: str,
    requested_ids: list[str],
    records: list[Memory],
    missing_ids: list[str],
    selections: list[dict[str, Any]],
    cutoff: datetime,
) -> dict[str, Any]:
    by_id = {record.id: record for record in records}
    sections: dict[str, Any] = {name: {"status": "unknown", "items": []} for name in SECTIONS}
    withheld = []
    for selection in selections:
        memory = by_id.get(selection["event_id"])
        if memory is not None:
            start, end = selection["start"], selection["end"]
            if end > len(memory.content) or memory.content[start:end] != selection["quote"]:
                raise ValueError("quote does not exactly match original Unicode source span")
        reason = "missing" if memory is None else ineligible(memory, cutoff)
        section = sections[selection["section"]]
        if reason:
            withheld.append(
                {
                    "event_id": selection["event_id"],
                    "section": selection["section"],
                    "reason": reason,
                }
            )
            if reason == "conflicted_or_truncated_lineage" or (
                memory is not None and memory.support_state == "conflicted_support"
            ):
                section["status"] = "contested"
            continue
        item = dict(selection, classification="unverified_selection")
        if selection["section"] == "completed_progress":
            item["verification"] = "unverified_report"
        section["items"].append(item)
        if section["status"] != "contested":
            section["status"] = "unverified"
    supplied = set(by_id)
    result = {
        "version": "morgan.continuation_context.v1",
        "user_id": user_id,
        "project": project,
        "effective_at": cutoff.isoformat(),
        "requested_ids": list(requested_ids),
        "missing_ids": list(missing_ids),
        "coverage": "requested_ids_only",
        "action_authority": "none",
        "sources": [source_projection(m) for m in records],
        "sections": sections,
        "withheld": withheld,
        "corrections": [
            {
                "event_id": m.id,
                "revises_event_ids": list(m.revises_event_ids),
                "revision_root_id": m.revision_root_id,
                "revision_state": m.revision_state,
                "eligible_leaf_ids": list(m.eligible_leaf_ids),
                "eligible_leaf_count": m.eligible_leaf_count,
                "revision_truncated": m.revision_truncated,
            }
            for m in records
            if m.revises_event_ids
        ],
        "unresolved_support_ids": sorted(
            {i for m in records for i in m.support_event_ids} - supplied
        ),
        "unresolved_branch_ids": sorted(
            {
                i
                for m in records
                for i in [
                    *m.revises_event_ids,
                    *m.eligible_leaf_ids,
                    *([m.revision_root_id] if m.revision_root_id else []),
                ]
            }
            - supplied
        ),
        "bounds": {
            "source_ids": 16,
            "selections": 16,
            "quote_code_points": 240,
            "serialized_utf8_bytes": 16384,
        },
    }
    if (
        len(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        > 16384
    ):
        raise ValueError("continuation context exceeds 16384 serialized UTF-8 bytes")
    return result
