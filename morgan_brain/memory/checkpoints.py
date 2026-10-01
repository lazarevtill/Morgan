"""Optional bounded resumable state, encoded in ordinary evidence-backed facts.

Planned steps confer no authorization. Reference IDs are reported history pointers,
not user/tool support. Ownership and validity remain in the fact envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from morgan_brain.models import PERSONAL_PROJECT, Scope, TemporalFact

CHECKPOINT_PREDICATE = "resumable_state_v1"
MAX_CHECKPOINT_BYTES = 32768
Identity = Annotated[str, Field(min_length=1, max_length=256, pattern=r"\S")]
Text = Annotated[str, Field(min_length=1, max_length=240, pattern=r"\S")]


class CheckpointContext(BaseModel):
    """Reported ownership and provenance labels; these do not authorize access."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: str
    project: str = PERSONAL_PROJECT
    author_id: str = ""
    scope: Scope = Scope.PRIVATE


class CheckpointItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: Text
    evidence_ids: list[Identity] = Field(default_factory=list, max_length=16)


class ReportedProgress(CheckpointItem):
    """Always unverified agent-reported progress; never accepted fact support."""

    verification: Literal["unverified_agent_report"] = "unverified_agent_report"
    reference_ids: list[Identity] = Field(default_factory=list, max_length=8)


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["morgan.checkpoint.v1"] = "morgan.checkpoint.v1"
    kind: Literal["goal", "task", "project"]
    subject_entity_id: Identity | None = None
    applies_to: list[Identity] = Field(default_factory=list, max_length=8)
    title: str = Field(min_length=1, max_length=120, pattern=r"\S")
    objective: str = Field(min_length=1, max_length=500, pattern=r"\S")
    status: Literal["active", "blocked", "paused", "completed"] = "active"
    progress: list[ReportedProgress] = Field(default_factory=list, max_length=4)
    open_questions: list[CheckpointItem] = Field(default_factory=list, max_length=4)
    next_steps: list[CheckpointItem] = Field(default_factory=list, max_length=4)

    def encode(self, support_ids: list[str]) -> str:
        """Revalidate nested mutable lists and item lineage before persistence."""
        checked = Checkpoint.model_validate(self.model_dump())
        if len(set(support_ids)) > 16:
            raise ValueError("checkpoint requires at most 16 support IDs")
        for item in [*checked.progress, *checked.open_questions, *checked.next_steps]:
            if not set(item.evidence_ids) <= set(support_ids):
                raise ValueError("checkpoint item evidence must occur in fact support")
        encoded = checked.model_dump_json()
        if len(encoded.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the 32768-byte encoding bound")
        return encoded


class CheckpointResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["morgan.checkpoint.result.v1"] = "morgan.checkpoint.result.v1"
    fact: TemporalFact
    eligibility: Literal[
        "current", "unsupported", "needs_rebuild", "unsupported_version", "invalid_state"
    ]
    state: Checkpoint | None = None

    @model_validator(mode="after")
    def safe_state(self) -> CheckpointResult:
        if self.eligibility not in ("current", "unsupported") and self.state is not None:
            raise ValueError("ineligible checkpoint cannot expose resumable state")
        return self


class AmbiguousCheckpoint(ValueError):
    """Legacy overlapping intervals cannot select a resumable state by rank."""

    def __init__(self, fact_ids: list[str]) -> None:
        self.fact_ids = tuple(sorted(fact_ids))
        super().__init__("multiple effective checkpoint facts; inspect scoped historical evidence")


class StaleCheckpoint(ValueError):
    """The expected checkpoint head no longer matches; explicitly prepare again."""


def checkpoint_subject(identity: str) -> str:
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 128:
        raise ValueError("checkpoint identity requires 1 to 128 nonblank characters")
    return "checkpoint:" + identity
