"""The shape of every command's result.

One payload per command, built once and used twice: serialised directly under
``--json``, which is a contract other programs parse, and handed to ``render`` for
the human form. Keeping the two readings of the same dictionary together is what
stops them drifting apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from morgan_brain.memory.gate import ForgetReport, RecallOutcome
from morgan_brain.memory.migrations import Step
from morgan_brain.memory.snapshot import RestoreResult, SnapshotResult
from morgan_brain.memory.store.spaces import EmbeddingSpace
from morgan_brain.models import Memory, TemporalFact


def memory_to_dict(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "project": m.project,
        "kind": m.kind.value,
        "content": m.content,
        "source": m.source.value,
        "importance": m.importance,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def recall_result(outcome: RecallOutcome, *, project: str, all_projects: bool) -> dict[str, Any]:
    """What recall found, and whether it abstained and why: an empty ``results`` is either
    ``empty`` (nothing in scope) or ``declined`` (the floor judged nothing stood out)."""
    return {
        "project": project,
        "all_projects": all_projects,
        "abstained": outcome.abstained,
        "reason": outcome.reason,
        "results": [memory_to_dict(m) for m in outcome.memories],
    }


def fact_to_dict(f: TemporalFact) -> dict[str, Any]:
    return {
        "id": f.id,
        "project": f.project,
        "subject": f.subject,
        "predicate": f.predicate,
        "object": f.object,
        "confidence": f.confidence,
        "source": f.source.value,
        "valid_from": f.valid_from.isoformat() if f.valid_from else None,
        "last_confirmed": f.last_confirmed.isoformat() if f.last_confirmed else None,
    }


def snapshot_to_dict(result: SnapshotResult) -> dict[str, Any]:
    return {
        "path": str(result.path),
        "bytes": result.bytes,
        "user_version": result.user_version,
        "counts": dict(result.counts),
    }


def restore_to_dict(result: RestoreResult) -> dict[str, Any]:
    return {
        "before": dict(result.before),
        "after": dict(result.after),
        "safety_snapshot": str(result.safety.path),
    }


def _step_to_dict(step: Step) -> dict[str, Any]:
    return {"number": step.number, "name": step.name, "heavy": step.heavy}


def migration_plan_to_dict(
    *, database: str, dry_run: bool, user_version: int, code_version: int, pending: Sequence[Step]
) -> dict[str, Any]:
    """What ``morgan migrate`` would run: under ``--dry-run``, or when nothing is pending."""
    return {
        "database": database,
        "dry_run": dry_run,
        "user_version": user_version,
        "code_version": code_version,
        "pending": [_step_to_dict(s) for s in pending],
    }


def migration_status_to_dict(
    *, user_version: int, code_version: int, pending: Sequence[Step]
) -> dict[str, Any]:
    """Where a database stands against this code's steps, as ``morgan doctor`` reads it --
    doctor runs none of them."""
    return {
        "user_version": user_version,
        "code_version": code_version,
        "pending": [_step_to_dict(s) for s in pending],
    }


def migration_to_dict(
    *,
    database: str,
    snapshot: SnapshotResult,
    from_version: int,
    user_version: int,
    code_version: int,
    applied: Sequence[tuple[Step, dict[str, int]]],
    after: dict[str, int],
    quick_check: str,
) -> dict[str, Any]:
    """What ``morgan migrate`` ran. ``before`` is counted off the snapshot and ``after`` off the
    migrated database, so equal counts say no row was lost on the way."""
    return {
        "database": database,
        "dry_run": False,
        "snapshot": str(snapshot.path),
        "from_version": from_version,
        "user_version": user_version,
        "code_version": code_version,
        "steps": [{**_step_to_dict(s), "counts": dict(counts)} for s, counts in applied],
        "before": dict(snapshot.counts),
        "after": dict(after),
        "quick_check": quick_check,
    }


def embedding_space_to_dict(
    space: EmbeddingSpace,
    *,
    fingerprint: str,
    reason: str | None = None,
    strings_digest: str | None = None,
) -> dict[str, Any]:
    """The active space. From ``morgan migrate``, *fingerprint* is ``recorded`` (this run
    recorded it), ``matches`` (it was recorded before and the model still answers it) or
    ``unverified`` (the embedding server did not answer; *reason* says how). From ``morgan
    doctor``, which never records, it is what the comparison found, and *strings_digest* is
    the sha256 of the five strings the fingerprint is made of."""
    shaped: dict[str, Any] = {
        "id": space.id,
        "model": space.model,
        "dims": space.dims,
        "fingerprint": fingerprint,
    }
    if reason is not None:
        shaped["reason"] = reason
    if strings_digest is not None:
        shaped["strings_digest"] = strings_digest
    return shaped


def merge_forget_reports(reports: list[ForgetReport]) -> ForgetReport:
    """Sum ``--all-projects`` reports into one. ``tables_skipped`` is deduplicated: the same
    optional table is either present or absent for the whole database, not per project."""
    merged = ForgetReport()
    skipped: set[str] = set()
    for r in reports:
        merged.memories += r.memories
        merged.facts += r.facts
        merged.history += r.history
        merged.sessions += r.sessions
        merged.turns += r.turns
        merged.digests += r.digests
        skipped.update(r.tables_skipped)
    merged.tables_skipped = sorted(skipped)
    return merged


def forget_result(report: ForgetReport, *, project: str, all_projects: bool) -> dict[str, Any]:
    """The one output that must not lie: a skipped table prints as "not present", never as
    a silent 0."""
    warnings: list[str] = []
    if report.tables_skipped:
        warnings.append(
            "not present in this database, so nothing was erased from (not an error): "
            + ", ".join(report.tables_skipped)
        )
    return {
        "project": project,
        "all_projects": all_projects,
        "memories": report.memories,
        "facts": report.facts,
        "history": report.history,
        "tables_skipped": list(report.tables_skipped),
        "warnings": warnings,
    }
