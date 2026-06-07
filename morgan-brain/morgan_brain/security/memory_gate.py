"""MemoryGate — the single choke point for all memory reads and writes.

Every store/recall passes through here. It enforces user-scope isolation (the basis of
multi-tenant readiness) and is the one place to add redaction, consent, and audit later.
Wrapping the MemoryStore means no module can bypass the gate.
"""
from __future__ import annotations

from morgan_brain.interfaces.memory import MemoryStore
from morgan_brain.models.memory import Memory, MemoryQuery, TemporalFact


class MemoryGate:
    def __init__(self, store: MemoryStore) -> None:
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

    async def current_facts(self, *, user_id: str, subject: str | None = None) -> list[TemporalFact]:
        self._require_scope(user_id)
        return await self._store.current_facts(user_id=user_id, subject=subject)

    @staticmethod
    def _require_scope(user_id: str) -> None:
        if not user_id:
            raise PermissionError("memory access requires a user_id")
