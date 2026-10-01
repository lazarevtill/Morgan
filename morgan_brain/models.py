"""Domain models. Everything that persists is ``user_id``-keyed and ``project``-keyed.

Two ideas the design hinges on:

* **Actor attribution** — every memory records who asserted it (``MemorySource``), so the
  assistant never mistakes its own inference for a user-stated fact.
* **Bi-temporal facts** — semantic facts carry validity intervals; updating a fact closes the
  old interval and opens a new one (evolution, not overwrite), so recall is never confidently
  stale and history stays queryable.

Timestamps are passed in, never generated implicitly, so the system stays deterministic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

#: The project a write lands in when the caller names none: outside a repository on the CLI,
#: or an MCP call with no ``project`` argument. There is no silent default -- both surfaces
#: report when they used it.
PERSONAL_PROJECT = "personal"


class Identified(BaseModel):
    """Anything with a stable id and creation time."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime | None = None


class UserScoped(Identified):
    """Anything owned by a specific user. The single tenancy key in the whole system."""

    user_id: str


class Entity(BaseModel):
    """A named entity extracted from input or referenced by a fact."""

    name: str
    type: str = "unknown"


class MemorySource(str, Enum):
    UNKNOWN = "unknown"
    USER_STATED = "user_stated"
    AGENT_INFERRED = "agent_inferred"
    TOOL_OBSERVED = "tool_observed"


class OriginKind(str, Enum):
    """Which path wrote a memory. ``unknown`` is a row from before its writer said."""

    REMEMBER = "remember"
    ASK = "ask"
    ASK_CONFLICT_GUARD = "ask_conflict_guard"
    IMPORT = "import"
    EXTRACTED = "extracted"
    UNKNOWN = "unknown"


class Scope(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"


class MemoryStatus(str, Enum):
    STORED = "stored"
    QUARANTINED = "quarantined"


class MemoryKind(str, Enum):
    EPISODIC = "episodic"  # what happened, when
    SEMANTIC = "semantic"  # what's true


class Memory(UserScoped):
    created_at: datetime | None = Field(
        default=None, validation_alias=AliasChoices("created_at", "effective_at")
    )
    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    kind: MemoryKind = MemoryKind.EPISODIC
    content: str
    source: MemorySource = MemorySource.UNKNOWN
    entities: list[Entity] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    # Provenance: where the row came from. ``cwd`` is where the writer ran, never the project.
    origin_kind: OriginKind = OriginKind.UNKNOWN
    client: str = ""
    session_id: str = ""
    cwd: str = ""
    author_id: str = ""
    scope: Scope = Scope.PRIVATE
    instruction_like: bool = False
    status: MemoryStatus = MemoryStatus.STORED
    # Ingestion time belongs to Morgan; created_at remains the asserted event time.
    recorded_at: datetime | None = None
    # Populated for semantic recall/evidence projections; raw events have no fact interval.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    last_confirmed: datetime | None = None
    support_event_ids: list[str] = Field(default_factory=list, max_length=32)
    revises_event_ids: list[str] = Field(default_factory=list, max_length=8)
    revision_root_id: str | None = None
    revision_state: str | None = None
    eligible_leaf_ids: list[str] = Field(default_factory=list, max_length=32)
    eligible_leaf_count: int = 0
    revision_truncated: bool = False
    support_state: str | None = None

    @model_validator(mode="before")
    @classmethod
    def consistent_time_alias(cls, value: object) -> object:
        if isinstance(value, dict) and "created_at" in value and "effective_at" in value:
            left = value["created_at"]
            right = value["effective_at"]
            if isinstance(left, str):
                left = datetime.fromisoformat(left)
            if isinstance(right, str):
                right = datetime.fromisoformat(right)
            if left != right:
                raise ValueError("revision_effective_time: conflicting event time aliases")
        return value

    @field_validator("revises_event_ids", mode="before")
    @classmethod
    def canonical_parents(cls, value: object) -> list[str]:
        if not isinstance(value, list) or len(value) > 8:
            raise ValueError("revision_parent_limit: expected up to 8 parent IDs")
        if any(
            not isinstance(identity, str) or not identity.strip() or len(identity) > 256
            for identity in value
        ):
            raise ValueError("revision_parent_limit: malformed parent ID")
        if len(set(value)) != len(value):
            raise ValueError("revision_parent_limit: parent IDs must be distinct")
        return sorted(value)

    @model_validator(mode="after")
    def correction_shape(self) -> Memory:
        if self.revises_event_ids:
            if self.source is MemorySource.UNKNOWN or not self.author_id.strip():
                raise ValueError(
                    "revision_actor_boundary: known source and reported author required"
                )
            if self.created_at is None or self.created_at.utcoffset() is None:
                raise ValueError("revision_effective_time: aware correction time required")
        return self

    @property
    def effective_at(self) -> datetime | None:
        return self.created_at


class TemporalFact(UserScoped):
    """A semantic fact with a validity interval. Supersession, not deletion."""

    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    subject: str  # usually an entity name or "user"
    predicate: str  # e.g. "lives_in", "works_at", "prefers"
    object: str  # the value
    source: MemorySource = MemorySource.UNKNOWN
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None  # None = currently valid
    superseded_by: str | None = None  # id of the fact that replaced this one
    last_confirmed: datetime | None = None
    author_id: str = ""
    scope: Scope = Scope.PRIVATE
    recorded_at: datetime | None = None
    support_event_ids: list[str] = Field(default_factory=list, max_length=32)
    support_state: str | None = None


class MemoryQuery(BaseModel):
    """A recall request. Defaults to currently-valid facts via multi-signal retrieval."""

    user_id: str
    #: min_length matches ``Memory.project``. Without it an empty project reached recall and
    #: silently matched nothing in every signal — a wrong answer rather than a refusal,
    #: in the seam whose whole job is refusing.
    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    all_projects: bool = False
    text: str
    top_k: int = 8
    kinds: list[MemoryKind] | None = None
    include_superseded: bool = False
    effective_at: datetime | None = None

    @field_validator("effective_at")
    @classmethod
    def aware_cutoff(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("effective_at must include a timezone")
        return value


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(UserScoped):
    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    role: Role
    content: str
    session_id: str | None = None
