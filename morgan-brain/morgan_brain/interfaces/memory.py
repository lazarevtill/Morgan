"""Memory contract — storage and retrieval only. Memory never decides *what* to learn
(that's Learning) and never decides *how* to apply it (that's Personalization).

All access is user-scoped and must go through the MemoryGate (see security/).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from morgan_brain.models.memory import Memory, MemoryQuery, TemporalFact


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

    async def current_facts(self, *, user_id: str, subject: str | None = None) -> list[TemporalFact]:
        """Return currently-valid facts (valid_to is None)."""
        ...
