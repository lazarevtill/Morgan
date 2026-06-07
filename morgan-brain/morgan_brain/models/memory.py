"""Memory domain models.

Two ideas the design hinges on:

* **Actor attribution** — every memory records who asserted it (``MemorySource``), so the
  assistant never mistakes its own inference for a user-stated fact.
* **Bi-temporal facts** — semantic facts carry validity intervals; updating a fact closes the
  old interval and opens a new one (evolution, not overwrite), so recall is never confidently
  stale and history stays queryable.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from morgan_brain.models.base import Entity, UserScoped


class MemorySource(str, Enum):
    USER_STATED = "user_stated"
    AGENT_INFERRED = "agent_inferred"
    TOOL_OBSERVED = "tool_observed"


class MemoryKind(str, Enum):
    EPISODIC = "episodic"   # what happened, when
    SEMANTIC = "semantic"   # what's true
    PROCEDURAL = "procedural"  # how to do something (skills)


class Memory(UserScoped):
    kind: MemoryKind = MemoryKind.EPISODIC
    content: str
    source: MemorySource = MemorySource.USER_STATED
    entities: list[Entity] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None


class TemporalFact(UserScoped):
    """A semantic fact with a validity interval. Supersession, not deletion."""

    subject: str           # usually an entity name or "user"
    predicate: str         # e.g. "lives_in", "works_at", "prefers"
    object: str            # the value
    source: MemorySource = MemorySource.USER_STATED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None         # None = currently valid
    superseded_by: str | None = None         # id of the fact that replaced this one
    last_confirmed: datetime | None = None


class MemoryQuery(BaseModel):
    """A recall request. Defaults to currently-valid facts via multi-signal retrieval."""

    user_id: str
    text: str
    top_k: int = 8
    kinds: list[MemoryKind] | None = None
    include_superseded: bool = False
