"""The operations a model may propose over the fact base.

This is the contract the model is constrained to, not an internal type: it is handed to
the structured-output ladder as a JSON schema and validated on the way back. UPDATE and
DELETE close an interval rather than rewriting a row, which is why the batch says what to
do and never what a fact should end up looking like.
"""

from __future__ import annotations

import builtins
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FactOpKind(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class FactOp(BaseModel):
    """A single fact operation proposed by the LLM."""

    op: FactOpKind
    subject: str
    predicate: str
    object: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = ""
    support_event_ids: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("support_event_ids", mode="before")
    @classmethod
    def declared_supports(cls, value: builtins.object) -> list[str]:
        if not isinstance(value, list) or len(value) > 32:
            raise ValueError("support_event_ids must be a list of at most 32 IDs")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 256 for item in value):
            raise ValueError("support_event_ids contains an invalid ID")
        if len(set(value)) != len(value):
            raise ValueError("support_event_ids must be distinct")
        return sorted(value)


class FactOpBatch(BaseModel):
    """Batch of fact operations — the schema passed to ``generate_structured``."""

    ops: list[FactOp]
