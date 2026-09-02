"""MemoryGate — the single choke point for all memory reads and writes.

Every store/recall/forget passes through here. It enforces user- and project-scope (the
basis of multi-tenant readiness) and is the one place to add redaction, consent, and audit
later. No caller holds the ``MemoryModule`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from morgan_brain.models import DEFAULT_PROJECT, Memory, MemoryQuery, TemporalFact

if TYPE_CHECKING:
    from morgan_brain.memory.module import MemoryModule


@dataclass
class ForgetReport:
    """What a single ``forget()`` call erased.

    ``history`` is ``0`` for two different reasons that used to be indistinguishable: the
    table exists and genuinely had nothing under this project, or it was never created on
    this connection. ``tables_skipped`` names every table that was absent (and therefore not
    touched) so a caller can print "not tracked here" instead of a false "0 erased".
    """

    memories: int = 0
    facts: int = 0
    history: int = 0
    #: Rows erased from the semantic upper index -- entity nodes, their co-occurrence edges,
    #: and the schemas of this scope. All derived from the memories being erased.
    index_entries: int = 0
    tables_skipped: list[str] = field(default_factory=list)


class MemoryGate:
    def __init__(self, store: MemoryModule) -> None:
        self._store = store

    async def store(self, memory: Memory) -> str:
        self._require_scope(memory.user_id)
        return await self._store.store(memory)

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        self._require_scope(query.user_id)
        return await self._store.recall(query)

    async def upsert_fact(self, fact: TemporalFact) -> str:
        self._require_scope(fact.user_id)
        return await self._store.upsert_fact(fact)

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = DEFAULT_PROJECT,
        all_projects: bool = False,
    ) -> list[TemporalFact]:
        self._require_scope(user_id, None if all_projects else project)
        return await self._store.current_facts(
            user_id=user_id, subject=subject, project=project, all_projects=all_projects
        )

    async def close_fact(
        self, fact_id: str, *, user_id: str, project: str, now: datetime | None = None
    ) -> None:
        self._require_scope(user_id, project)
        await self._store.close_fact(fact_id, user_id=user_id, project=project, now=now)

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        self._require_scope(user_id, project)
        await self._store.set_confidence(fact_id, user_id=user_id, project=project, value=value)

    async def distinct_projects(self, user_id: str) -> list[str]:
        self._require_scope(user_id)
        return await self._store.distinct_projects(user_id)

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        self._require_scope(user_id, project)
        return await self._store.forget(user_id=user_id, project=project)

    @staticmethod
    def _require_scope(user_id: str, project: str | None = None) -> None:
        if not user_id:
            raise PermissionError("memory access requires a user_id")
        if project is not None and not project:
            raise PermissionError("memory access requires a project")
