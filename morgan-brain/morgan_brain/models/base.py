"""Base model mixins. Timestamps are passed in, never generated implicitly, so the
system stays deterministic and testable."""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


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
    type: str = "unknown"  # person | place | org | date | concept | ...
