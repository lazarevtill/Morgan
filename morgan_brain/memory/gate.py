"""MemoryGate — the single choke point for all memory reads and writes.

Every store/recall/forget passes through here. It enforces user- and project-scope (the
basis of multi-tenant readiness) and is the one place to add redaction, consent, and audit
later. No caller holds the ``MemoryModule`` directly.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from morgan_brain.memory.migrations import DatabaseNeedsMigration
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
    tables_skipped: list[str] = field(default_factory=list)


class MemoryGate:
    """*read_only_reason*, when given, refuses every write -- ``store``, ``upsert_fact``,
    ``close_fact``, ``set_confidence``, ``forget`` -- with ``DatabaseNeedsMigration`` carrying
    it: the database behind the gate waits for a heavy migration step (``memory.migrations``).
    Reads are unaffected, because they work on the schema the database already has.
    """

    def __init__(self, store: MemoryModule, read_only_reason: str | None = None) -> None:
        self._store = store
        self._read_only_reason = read_only_reason

    @property
    def read_only_reason(self) -> str | None:
        """Why writes are refused, or ``None`` when they are not."""
        return self._read_only_reason

    async def store(self, memory: Memory) -> str:
        self.require_writable()
        self._require_scope(memory.user_id)
        return await self._store.store(memory)

    async def get(self, memory_id: str, *, user_id: str) -> Memory | None:
        """One memory by id, or ``None`` if this owner has no such memory.

        A read like any other, so it comes through the gate: a caller that reached into the
        episodic store directly would be a caller whose scope nobody checked.
        """
        self._require_scope(user_id)
        return await self._store.get(memory_id, user_id=user_id)

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        self._require_scope(query.user_id)
        return await self._store.recall(query)

    async def upsert_fact(self, fact: TemporalFact) -> str:
        self.require_writable()
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
        self.require_writable()
        self._require_scope(user_id, project)
        await self._store.close_fact(fact_id, user_id=user_id, project=project, now=now)

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        self.require_writable()
        self._require_scope(user_id, project)
        await self._store.set_confidence(fact_id, user_id=user_id, project=project, value=value)

    async def distinct_projects(self, user_id: str) -> list[str]:
        self._require_scope(user_id)
        return await self._store.distinct_projects(user_id)

    def write_transaction(self) -> AbstractContextManager[None]:
        """Make every gate call inside the block one atomic write, under the write lock.

        For a caller whose writes depend on what it reads through the gate: the reads see
        what other processes committed before the block, and nothing else can write until
        the block ends. Nothing inside may await real I/O. ``forget`` is the exception: it
        vacuums once its erasure commits, so it refuses to run inside the block.
        """
        return self._store.write_transaction()

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        self.require_writable()
        self._require_scope(user_id, project)
        return await self._store.forget(user_id=user_id, project=project)

    def require_writable(self) -> None:
        """Raise ``DatabaseNeedsMigration`` if writes are refused; every write method does.

        Public for a command whose write comes after costly work -- a model call, an embedding,
        a history row written outside the gate: it refuses first, and nothing else happens.
        """
        if self._read_only_reason is not None:
            raise DatabaseNeedsMigration(self._read_only_reason)

    @staticmethod
    def _require_scope(user_id: str, project: str | None = None) -> None:
        if not user_id:
            raise PermissionError("memory access requires a user_id")
        if project is not None and not project:
            raise PermissionError("memory access requires a project")
