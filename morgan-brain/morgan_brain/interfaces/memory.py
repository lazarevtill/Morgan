"""Memory contract — storage and retrieval only. Memory never decides *what* to learn
(that's Learning) and never decides *how* to apply it (that's Personalization).

All access is user-scoped and must go through the MemoryGate (see security/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from morgan_brain.models.memory import DEFAULT_PROJECT, Memory, MemoryQuery, TemporalFact


@dataclass
class ForgetReport:
    """What a single ``forget()`` call erased.

    ``champions_flagged`` stays empty: flagging a promoted champion preprompt that may embed
    text mined from this project's conversations requires the ``PromptRegistry``, which
    ``MemoryModule`` does not hold and no caller currently wires in. An empty list here is
    honest scope, not a broken feature -- until that wiring exists, a champion can only be
    reviewed and rolled back by hand.
    """

    memories: int = 0
    facts: int = 0
    signals: int = 0
    history: int = 0
    champions_flagged: list[str] = field(default_factory=list)


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

    async def distinct_projects(self, user_id: str) -> list[str]:
        """Return the distinct project names *user_id* has stored memories under.

        Used to fan a per-user operation (e.g. nightly consolidation) out across every
        project the user actually has, instead of assuming a single project."""
        ...

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        """Erase everything *user_id* stored under *project*, across every index, in one
        transaction. Idempotent -- forgetting an already-empty project returns an
        all-zero report rather than raising."""
        ...
