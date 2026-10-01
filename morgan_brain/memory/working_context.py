"""A bounded, attributed quotation view; categories and links are unverified selections."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from morgan_brain.memory.checkpoints import (
    CHECKPOINT_PREDICATE,
    CheckpointContext,
    checkpoint_subject,
)
from morgan_brain.models import Memory, MemoryKind, MemorySource, MemoryStatus, TemporalFact

WORKING_CONTEXT_PREDICATE = "working_context_v1"
MAX_CONTEXT_BYTES = 16384


class SourceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(min_length=1, max_length=256)
    quote: str = Field(min_length=1, max_length=240, pattern=r"\S")


class SourceSpan(SourceQuote):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)


class DraftDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    choice: SourceQuote
    reason: SourceQuote | None = None


class SelectedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    choice: SourceSpan
    reason: SourceSpan | None = None


class WorkingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["morgan.working_context.v1"] = "morgan.working_context.v1"
    classification: Literal["unverified_model_selection"] = "unverified_model_selection"
    validation: Literal["exact_spans_only"] = "exact_spans_only"
    progress_verification: Literal["unverified_report"] = "unverified_report"
    title: str = Field(min_length=1, max_length=120, pattern=r"\S")
    decisions: list[SelectedDecision] = Field(default_factory=list, max_length=4)
    open_questions: list[SourceSpan] = Field(default_factory=list, max_length=4)
    intentions: list[SourceSpan] = Field(default_factory=list, max_length=4)
    progress: list[SourceSpan] = Field(default_factory=list, max_length=4)

    def spans(self) -> list[SourceSpan]:
        return [
            *[span for item in self.decisions for span in (item.choice, item.reason) if span],
            *self.open_questions,
            *self.intentions,
            *self.progress,
        ]

    def event_ids(self) -> list[str]:
        return list(dict.fromkeys(span.event_id for span in self.spans()))


class WorkingContextDraft(BaseModel):
    """Model emits quotations only; offsets are derived from unique exact matches."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str = Field(min_length=1, max_length=120, pattern=r"\S")
    decisions: list[DraftDecision] = Field(default_factory=list, max_length=4)
    open_questions: list[SourceQuote] = Field(default_factory=list, max_length=4)
    intentions: list[SourceQuote] = Field(default_factory=list, max_length=4)
    progress: list[SourceQuote] = Field(default_factory=list, max_length=4)

    def normalize(self, records: list[Memory]) -> WorkingContext:
        data = self.model_dump()
        sources = {record.id: record.content for record in records}
        spans = [
            *[
                span
                for item in data["decisions"]
                for span in (item["choice"], item["reason"])
                if span
            ],
            *data["open_questions"],
            *data["intentions"],
            *data["progress"],
        ]
        for span in spans:
            text = sources.get(span["event_id"], "")
            offset = text.find(span["quote"])
            if offset < 0 or text.find(span["quote"], offset + 1) >= 0:
                raise ValueError("working context quote requires a unique exact source match")
            span.update(start=offset, end=offset + len(span["quote"]))
        return validate_working_context(WorkingContext.model_validate(data), records)


class AttributedSpan(SourceSpan):
    source: MemorySource
    author_id: str


class WorkingContextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["morgan.working_context.result.v1"] = "morgan.working_context.result.v1"
    fact_id: str
    eligibility: Literal["current", "needs_rebuild", "invalid_state", "unsupported_version"]
    state: WorkingContext | None = None
    sources: list[AttributedSpan] = Field(default_factory=list)


class WorkingContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    view: WorkingContextResult


class WorkingContextList(BaseModel):
    """Bounded structural discovery; ineligible views contain no stale quotations."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["morgan.working_context.list.v1"] = "morgan.working_context.list.v1"
    items: list[WorkingContextEntry] = Field(default_factory=list, max_length=32)
    truncated: bool = False


class WorkingContextPreview(BaseModel):
    """Explicit apply input, not a signed receipt or authorization capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["morgan.working_context.preview.v1"] = "morgan.working_context.preview.v1"
    context_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    context: CheckpointContext
    expected_fact_id: str | None
    generation: int = Field(ge=0, strict=True)
    state: WorkingContext
    evidence_basis: list[Memory] = Field(max_length=16)


def working_subject(identity: str) -> str:
    return "working_" + checkpoint_subject(identity)


def current_source(record: Memory) -> bool:
    return (
        record.kind is MemoryKind.EPISODIC
        and record.status is MemoryStatus.STORED
        and record.revision_state == "active"
        and not record.revision_truncated
        and record.eligible_leaf_count == 1
    )


def validate_working_context(state: WorkingContext, records: list[Memory]) -> WorkingContext:
    """Check exact Unicode-codepoint spans and current raw identities, not entailment."""
    checked = WorkingContext.model_validate(state.model_dump())
    identities = checked.event_ids()
    if not identities or len(identities) > 16:
        raise ValueError("working context requires 1 to 16 exact source events")
    sources = {record.id: record for record in records}
    if len(sources) != len(records) or not set(identities) <= set(sources):
        raise ValueError("working context source unavailable")
    for span in checked.spans():
        record = sources[span.event_id]
        if (
            not current_source(record)
            or span.end <= span.start
            or record.content[span.start : span.end] != span.quote
            or span.end > len(record.content)
        ):
            raise ValueError("working context source or exact span is not current")
    if len(checked.model_dump_json().encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise ValueError("working context encoding exceeds 16384 bytes")
    return checked


def working_fact(preview: WorkingContextPreview) -> TemporalFact:
    state = validate_working_context(preview.state, preview.evidence_basis)
    return TemporalFact(
        user_id=preview.context.user_id,
        project=preview.context.project,
        subject=working_subject(preview.context_id),
        predicate=WORKING_CONTEXT_PREDICATE,
        object=state.model_dump_json(),
        source=MemorySource.AGENT_INFERRED,
        author_id=preview.context.author_id,
        scope=preview.context.scope,
        support_event_ids=[
            record.id
            for record in preview.evidence_basis
            if record.id in state.event_ids()
            and record.source in (MemorySource.USER_STATED, MemorySource.TOOL_OBSERVED)
        ],
    )


def is_organizer_fact(subject: str, predicate: str) -> bool:
    """Reserved predicates are structural only in their canonical subject namespace."""
    if predicate == WORKING_CONTEXT_PREDICATE:
        prefix, encode = "working_checkpoint:", working_subject
    elif predicate == CHECKPOINT_PREDICATE:
        prefix, encode = "checkpoint:", checkpoint_subject
    else:
        return False
    if not subject.startswith(prefix):
        return False
    try:
        return encode(subject[len(prefix) :]) == subject
    except ValueError:
        return False
