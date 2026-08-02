"""Memory contract — storage and retrieval only. Memory never decides *what* to learn
(that's Learning) and never decides *how* to apply it (that's Personalization).

All access is user-scoped and must go through the MemoryGate (see security/).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from morgan_brain.models.memory import DEFAULT_PROJECT, Memory, MemoryQuery, TemporalFact


@runtime_checkable
class MemoryStore(Protocol):
    async def store(self, memory: Memory) -> str:
        """Persist an episodic/semantic/procedural memory; returns its id."""
        ...

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        """Multi-signal retrieval (vector + BM25 + entity), currently-valid by default."""
        ...

    async def upsert_fact(self, fact: TemporalFact) -> str:
        """Assert a fact; supersedes any conflicting currently-valid fact (no overwrite)."""
        ...

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = DEFAULT_PROJECT,
        all_projects: bool = False,
    ) -> list[TemporalFact]:
        """Return currently-valid facts (valid_to is None), scoped to *project*.

        ``all_projects=True`` is the explicit cross-project escape hatch (used by
        multi-project consolidation, never as a default).
        """
        ...

    async def close_fact(
        self, fact_id: str, *, user_id: str, project: str, now: datetime | None = None
    ) -> None:
        """Close a fact's validity interval (soft delete). Scoped to *user_id* + *project* so a
        caller who only knows a fact id cannot close a fact belonging to someone else."""
        ...

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        """Overwrite a fact's confidence in place. Scoped to *user_id* + *project*."""
        ...
