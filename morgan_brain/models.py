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

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

#: The project a write lands in when the caller names none: outside a repository on the CLI,
#: or an MCP call with no ``project`` argument. There is no silent default -- both surfaces
#: report when they used it.
PERSONAL_PROJECT = "personal"


def tool_call_text(tool_name: str, arguments: object) -> str:
    """The text a ``tool_call`` turn stores: one JSON document, ``{"input": …, "tool": …}``, keys
    sorted, compact, ``ensure_ascii=False`` so Cyrillic stays Cyrillic for the gate's rules and
    for FTS5. One document, so the scanner can decode the whole text back and scan its string
    values before the serialisation hides a word boundary behind ``\\n``.
    """
    return json.dumps(
        {"input": arguments, "tool": tool_name},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


#: The two harnesses whose transcripts capture reads.
Harness = Literal["claude-code", "codex"]
#: What a turn is: the owner's words, the model's, a tool the model called, or what it got back.
TurnRole = Literal["user", "assistant", "tool_call", "tool_result"]
#: Which capture form first stored a session's turns; each sets its own ``*_first_at`` once.
CaptureTrigger = Literal["hook", "sweep", "all"]
#: How a captured session's project was resolved from its ``cwd``.
ProjectSource = Literal["git", "path", "owner"]


def session_id_of(harness: str, native_id: str) -> str:
    """The ``sessions.id`` every writer and reader derives rather than looks up."""
    return f"{harness}:{native_id}"


def utc_iso(dt: datetime) -> str:
    """The one shape every new writer stores: UTC, millisecond precision, ``Z`` --
    ``2026-09-22T10:15:00.000Z``. Both harnesses write this shape, so lexical order is time
    order across every writer. A naive datetime is refused: it has no zone to convert from.
    """
    if dt.tzinfo is None:
        raise ValueError("utc_iso needs an aware datetime")
    moment = dt.astimezone(UTC)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


def parse_iso(text: str) -> datetime:
    """An ISO 8601 string -- with ``Z`` or an offset, a date alone, or a naive time taken as
    UTC -- as an aware UTC datetime. Every comparison of timestamps in Python goes through
    it, so a harness's shape and Morgan's own compare as moments, not as strings."""
    normalised = text.strip()
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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
    USER_STATED = "user_stated"
    AGENT_INFERRED = "agent_inferred"
    TOOL_OBSERVED = "tool_observed"


class OriginKind(str, Enum):
    """Which path wrote a memory. ``unknown`` is a row from before its writer said."""

    REMEMBER = "remember"
    ASK = "ask"
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
    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    kind: MemoryKind = MemoryKind.EPISODIC
    content: str
    source: MemorySource = MemorySource.USER_STATED
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


class TemporalFact(UserScoped):
    """A semantic fact with a validity interval. Supersession, not deletion."""

    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    subject: str  # usually an entity name or "user"
    predicate: str  # e.g. "lives_in", "works_at", "prefers"
    object: str  # the value
    source: MemorySource = MemorySource.USER_STATED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None  # None = currently valid
    superseded_by: str | None = None  # id of the fact that replaced this one
    last_confirmed: datetime | None = None
    author_id: str = ""
    scope: Scope = Scope.PRIVATE


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


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(UserScoped):
    project: str = Field(default=PERSONAL_PROJECT, min_length=1)
    role: Role
    content: str
    session_id: str | None = None


class Session(BaseModel):
    """One captured harness session: the row of ``sessions``. ``id`` is
    ``session_id_of(harness, native_id)``; the counts and the three ``*_first_at`` stamps are
    the store's to advance."""

    id: str
    user_id: str
    project: str = Field(min_length=1)
    harness: Harness
    native_id: str
    source_path: str
    cwd: str
    cwd_changed: bool = False
    project_source: ProjectSource
    entrypoint: str = ""
    interactive: bool = True
    harness_version: str = ""
    harness_mode: str = ""
    forked_from: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    erased_since: str | None = None
    hook_first_at: str | None = None
    sweep_first_at: str | None = None
    all_first_at: str | None = None
    turn_count: int = 0
    authored_turn_count: int = 0
    gate_redactions: int = 0
    gate_flags: int = 0
    gate_provider_hits: int = 0
    paused_turns: int = 0
    reader_version: int
    gate_version: int
    imported_at: str = ""
    updated_at: str = ""


class Turn(BaseModel):
    """One turn of a captured session: the row of ``turns``. ``text`` is what is stored -- for
    a ``tool_call`` the one JSON document ``tool_call_text`` makes -- after the gate's scan and
    the cap; ``redactions`` and ``flags`` are the gate's JSON arrays."""

    id: int | None = None
    session_id: str
    user_id: str
    project: str = Field(min_length=1)
    ordinal: int = 0
    native_key: str
    native_uuid: str | None = None
    parent_uuid: str | None = None
    role: TurnRole
    authored: bool = True
    injected_kind: str = ""
    tool_name: str | None = None
    tool_use_id: str | None = None
    morgan_call: str | None = None
    is_error: bool = False
    ts: str | None = None
    text: str
    truncated_chars: int = 0
    redactions: str = "[]"
    flags: str = "[]"
    norm_hash: str | None = None
    is_correction: bool = False


class OpenCall(BaseModel):
    """A call a read met whose later items can still arrive: its result (or output, and for a
    Codex MCP call its item), with whether it is Morgan's own."""

    id: str
    morgan: bool = False
    pending: list[str] = Field(default_factory=list)


class CaptureCursor(BaseModel):
    """Where capture stopped reading one transcript: the row of ``capture_cursors``.
    ``identity`` is the sha256 of the file's first line, so a rewritten file resets;
    ``lease_owner``/``lease_until`` are the lease's own columns; ``open_calls`` holds the calls
    a read met whose later items have not been read yet, so a read that ends between a call
    and its result still skips the result on its next pass."""

    harness: str
    native_id: str
    source_path: str
    byte_offset: int = 0
    size: int = 0
    mtime_ns: int = 0
    identity: str = ""
    status: Literal["new", "ok", "reset"] = "new"
    last_read_at: str = ""
    lease_owner: str | None = None
    lease_until: str | None = None
    open_calls: list[OpenCall] = Field(default_factory=list)


class Exclusion(BaseModel):
    """A session capture must not read again, by its harness's native id: the row of
    ``capture_exclusions``. ``reason`` is ``forget``, ``since``, ``retention`` or ``exclude``."""

    harness: str
    native_id: str
    reason: str
    excluded_at: str


class Pause(BaseModel):
    """One pause interval of a project: the row of ``capture_pauses``. An interval is current
    while ``paused_until`` is ``None`` or in the future; a turn inside any interval is never
    stored, however the interval later ended."""

    id: int | None = None
    user_id: str
    project: str
    paused_from: str
    paused_until: str | None = None
