"""A bounded personal focus plan from explicit memory evidence; no inferred permissions."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from morgan_brain.local_notes import ACTION
from morgan_brain.memory.continuation_context import ineligible
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.revisions import instant
from morgan_brain.models import Memory, MemoryKind, MemorySource, MemoryStatus
from morgan_brain.tasks import TaskKey, TaskRequest

_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


class Commitment(BaseModel):
    """Explicit stored planning data; title text is never an instruction to execute."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["morgan.commitment.v1"]
    title: str = Field(min_length=1, max_length=240)
    due: date
    minutes: int = Field(ge=1, le=240)
    priority: Literal["high", "normal", "low"]
    status: Literal["open", "done"]

    @field_validator("title")
    @classmethod
    def printable_title(cls, value: str) -> str:
        """Keep one visible line and reject control characters rather than hiding them."""
        if not value.strip() or any(not char.isprintable() for char in value):
            raise ValueError("commitment title must be a nonblank printable line")
        return value


@dataclass(frozen=True)
class BriefRequest:
    """One explicit source scope, day, budget and destination task identity."""

    task_key: TaskKey
    evidence_ids: tuple[str, ...]
    day: date
    budget_minutes: int
    follow_revisions: bool = False

    def __post_init__(self) -> None:
        """Bound exact evidence selection and reject implicit or duplicate source selection."""
        if not isinstance(self.follow_revisions, bool):
            raise TypeError("follow_revisions must be a boolean")
        if not 1 <= len(self.evidence_ids) <= 8 or len(set(self.evidence_ids)) != len(
            self.evidence_ids
        ):
            raise ValueError("select 1 to 8 distinct evidence IDs")
        if any(
            not isinstance(i, str) or not i.strip() or len(i) > 256 or not i.isprintable()
            for i in self.evidence_ids
        ):
            raise ValueError("invalid evidence ID")
        if isinstance(self.budget_minutes, bool) or not isinstance(self.budget_minutes, int):
            raise TypeError("budget must be an integer number of minutes")
        if not 1 <= self.budget_minutes <= 480:
            raise ValueError("daily budget must be 1 to 480 minutes")
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise TypeError("planning day requires a calendar date")


@dataclass(frozen=True)
class BriefDraft:
    """A reviewable exact task proposal; constructing it grants no execution authority."""

    task: TaskRequest
    selected_ids: tuple[str, ...]
    deferred_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    planned_minutes: int
    as_of: datetime
    planner: str = "controlled-priority-v1"
    resolved_ids: tuple[tuple[str, str], ...] = ()


def _literal(value: str) -> str:
    """Escape source text as literal Markdown instead of allowing active links or headings."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|>\-])", r"\\\1", html.escape(value, quote=True))


def _render(
    request: BriefRequest,
    items: list[tuple[str, Commitment]],
    deferred: list[tuple[str, Commitment]],
    excluded: list[tuple[str, Commitment]],
    as_of: datetime,
) -> str:
    """Render grounded titles, source IDs and budget decisions with no generated assertions."""
    planned = sum(item.minutes for _, item in items)
    lines = [
        f"# Focus plan — {request.day.isoformat()}",
        "",
        "Planner: controlled-priority-v1 (deterministic; no language model).",
        f"Evidence snapshot: {as_of.isoformat()}.",
        f"Budget: {request.budget_minutes} min; planned: {planned} min.",
        "",
        "## Focus",
        "",
    ]
    if not items:
        lines.append("No selected open commitment fits this day and budget.")
    for identity, item in items:
        timing = "overdue" if item.due < request.day else "due today"
        lines.append(
            f"- [ ] {_literal(item.title)} — {item.minutes} min, {item.priority}, {timing}; "
            f"source: {_literal(identity)}."
        )
    for title, entries in (
        ("Deferred: insufficient remaining time", deferred),
        ("Excluded: completed or future-dated", excluded),
    ):
        if entries:
            lines.extend(["", f"## {title}", ""])
            for identity, item in entries:
                lines.append(f"- {_literal(item.title)} — source: {_literal(identity)}.")
    lines.extend(
        [
            "",
            "Only the explicitly selected source records were considered. This is a snapshot,",
            "not a complete calendar or an assertion that any listed activity has been performed.",
            "Approving delivery authorizes saving this note only.",
            "Checklist items are not tool grants.",
        ]
    )
    return "\n".join(lines) + "\n"


def _current_leaf(memory: Memory, cutoff: datetime) -> str:
    """Follow only a trusted commitment with one complete, unambiguous revision family."""
    if (
        memory.kind != MemoryKind.EPISODIC
        or memory.source != MemorySource.USER_STATED
        or memory.status != MemoryStatus.STORED
        or memory.instruction_like
    ):
        raise ValueError("select trusted commitment sources")
    if memory.created_at is not None and instant(memory.created_at) > instant(cutoff):
        raise ValueError("selected commitment is future-dated")
    if memory.revision_truncated or memory.eligible_leaf_count != 1:
        raise ValueError("commitment revisions are ambiguous or incomplete")
    if len(memory.eligible_leaf_ids) != 1 or len(memory.content.encode("utf-8")) > 4096:
        raise ValueError("invalid commitment revision source")
    Commitment.model_validate_json(memory.content)
    return memory.eligible_leaf_ids[0]


async def _sources(
    gate: MemoryGate, request: BriefRequest, cutoff: datetime
) -> tuple[list[Memory], tuple[tuple[str, str], ...]]:
    """Resolve only explicitly selected families through scoped gate reads at one cutoff."""
    original = await gate.evidence(
        user_id=request.task_key.user_id,
        project=request.task_key.project,
        evidence_ids=list(request.evidence_ids),
        effective_at=cutoff,
    )
    if original.missing_ids:
        raise ValueError("selected evidence is unavailable in this scope")
    if not request.follow_revisions:
        return original.records, tuple((item.id, item.id) for item in original.records)
    links = tuple((item.id, _current_leaf(item, cutoff)) for item in original.records)
    current = await gate.evidence(
        user_id=request.task_key.user_id,
        project=request.task_key.project,
        evidence_ids=list(dict.fromkeys(target for _, target in links)),
        effective_at=cutoff,
    )
    if current.missing_ids:
        raise ValueError("current commitment revision is unavailable")
    by_id = {item.id: item for item in current.records}
    for source, (_, target) in zip(original.records, links, strict=True):
        leaf = by_id[target]
        if (
            leaf.revision_root_id != source.revision_root_id
            or _current_leaf(leaf, cutoff) != target
        ):
            raise ValueError("commitment revisions changed during planning; retry")
    return current.records, links


def _render_resolutions(links: tuple[tuple[str, str], ...]) -> str:
    """Expose followed revisions without presenting the plan as an exhaustive task search."""
    changed = [(source, target) for source, target in links if source != target]
    if not changed:
        return ""
    lines = ["", "## Updated source versions", ""]
    lines.extend(f"- {_literal(source)} → {_literal(target)}" for source, target in changed)
    return "\n".join(lines) + "\n"


async def prepare_brief(
    gate: MemoryGate, request: BriefRequest, *, as_of: datetime | None = None
) -> BriefDraft:
    """Read scoped current evidence, allocate a finite budget, and return an unapproved draft.

    The controlled planner sorts by explicit priority, then due date and source ID. It greedily
    selects fitting items; it is not a semantic model, optimal scheduler or medical adviser.
    """
    cutoff = as_of if as_of is not None else datetime.now(UTC)
    if cutoff.utcoffset() is None:
        raise ValueError("brief snapshot requires a timezone")
    records, resolved = await _sources(gate, request, cutoff)
    candidates: list[tuple[str, Commitment]] = []
    excluded: list[tuple[str, Commitment]] = []
    for memory in records:
        if (
            memory.kind != MemoryKind.EPISODIC
            or memory.source != MemorySource.USER_STATED
            or memory.instruction_like
            or ineligible(memory, cutoff) is not None
        ):
            raise ValueError("select current, unambiguous, user-stated commitment evidence")
        if len(memory.content.encode("utf-8")) > 4096:
            raise ValueError("commitment evidence exceeds 4096 UTF-8 bytes")
        commitment = Commitment.model_validate_json(memory.content)
        target = (
            candidates
            if commitment.status == "open" and commitment.due <= request.day
            else excluded
        )
        target.append((memory.id, commitment))
    candidates.sort(key=lambda pair: (_PRIORITY_RANK[pair[1].priority], pair[1].due, pair[0]))
    selected: list[tuple[str, Commitment]] = []
    deferred: list[tuple[str, Commitment]] = []
    remaining = request.budget_minutes
    for item in candidates:
        if item[1].minutes <= remaining:
            selected.append(item)
            remaining -= item[1].minutes
        else:
            deferred.append(item)
    payload = _render(request, selected, deferred, excluded, cutoff) + _render_resolutions(resolved)
    return BriefDraft(
        task=TaskRequest(request.task_key, ACTION, payload),
        selected_ids=tuple(identity for identity, _ in selected),
        deferred_ids=tuple(identity for identity, _ in deferred),
        excluded_ids=tuple(identity for identity, _ in excluded),
        planned_minutes=request.budget_minutes - remaining,
        as_of=cutoff,
        resolved_ids=resolved,
    )
