"""MemoryModule — the interfaces.MemoryStore implementation.

Recall is multi-signal: vector (semantic) + FTS5 (keyword) + entity overlap, combined with
reciprocal rank fusion (the single rerank layer). Facts are delegated to the bi-temporal store.
All access is user-scoped; callers reach it only through the MemoryGate.

Every signal is durable: the vector index, the keyword index, the entity index, and the
episodic records themselves each live in SQLite, so recall survives a process restart. Episodic
rehydration reads the full record from ``EpisodicStore`` -- never a subset carried in a vector
payload -- so a memory recovered after a restart is exactly the one that was stored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from morgan_brain.models.memory import (
    Memory,
    MemoryKind,
    MemoryQuery,
    TemporalFact,
)
from morgan_brain.modules.memory.indexing.embedder import Embedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import VectorIndex, VectorRecord


class MemoryModule:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: VectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
        fts: FtsIndex,
        entities: EntityIndex,
        episodics: EpisodicStore,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._fts = fts
        self._entities = entities
        self._episodics = episodics

    async def store(self, memory: Memory) -> str:
        if memory.created_at is None:
            memory.created_at = self._clock()
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        self._episodics.put(memory)
        await self._vectors.upsert(
            VectorRecord(
                id=memory.id,
                user_id=memory.user_id,
                vector=vector,
                payload={"content": memory.content, "user_id": memory.user_id},
            )
        )
        self._fts.add(memory.id, memory.content, user_id=memory.user_id)
        self._entities.add(memory.id, [e.name for e in memory.entities], user_id=memory.user_id)
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        q_vector = await self._embedder.embed(query.text)
        vec_hits = await self._vectors.search(
            user_id=query.user_id, vector=q_vector, top_k=query.top_k * 2
        )
        vector_ranking = [h.id for h in vec_hits]
        fts_ranking = self._fts.search(query.text, user_id=query.user_id, top_k=query.top_k * 2)
        entity_ranking = self._entities.search(
            {t for t in query.text.split()}, user_id=query.user_id, top_k=query.top_k * 2
        )

        fused_ids = reciprocal_rank_fusion([vector_ranking, fts_ranking, entity_ranking])
        episodic = [m for m in (self._episodics.get(mid) for mid in fused_ids) if m is not None]

        # Currently-valid facts are authoritative; surface them alongside episodic recall.
        # Phase 1 includes all current facts (volume is small until Phase 2 extraction);
        # relevance-ranking of facts is a Phase 2 concern.
        facts = await self._temporal.current_facts(user_id=query.user_id)
        fact_memories = [
            Memory(
                user_id=query.user_id,
                kind=MemoryKind.SEMANTIC,
                content=f"{f.subject} {f.predicate} {f.object}".replace("_", " "),
                source=f.source,
            )
            for f in facts
        ]
        return (fact_memories + episodic)[: query.top_k]

    async def upsert_fact(self, fact: TemporalFact) -> str:
        return await self._temporal.upsert_fact(fact, now=self._clock())

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        return await self._temporal.current_facts(user_id=user_id, subject=subject)
