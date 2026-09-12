"""The operations a model may propose over the fact base.

This is the contract the model is constrained to, not an internal type: it is handed to
the structured-output ladder as a JSON schema and validated on the way back. UPDATE and
DELETE close an interval rather than rewriting a row, which is why the batch says what to
do and never what a fact should end up looking like.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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


class FactOpBatch(BaseModel):
    """Batch of fact operations — the schema passed to ``generate_structured``."""

    ops: list[FactOp]
