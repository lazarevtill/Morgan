"""MemoryModule — the interfaces.MemoryStore implementation.

Recall is multi-signal: vector (semantic) + BM25 (keyword) + entity overlap, combined with
reciprocal rank fusion (the single rerank layer). Facts are delegated to the bi-temporal store.
All access is user-scoped; callers reach it only through the MemoryGate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from morgan_brain.models.memory import (
    Memory,
    MemoryKind,
    MemoryQuery,
    MemorySource,
    TemporalFact,
)
from morgan_brain.modules.memory.indexing.embedder import Embedder
from morgan_brain.modules.memory.retrieval.bm25 import Bm25Index
from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import VectorHit, VectorIndex, VectorRecord


class MemoryModule:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: VectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._bm25 = Bm25Index()
        self._by_id: dict[str, Memory] = {}
        self._entities: dict[str, set[str]] = {}  # memory_id -> lowercased entity names

    async def store(self, memory: Memory) -> str:
        if memory.created_at is None:
            memory.created_at = self._clock()
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        await self._vectors.upsert(
            VectorRecord(
                id=memory.id,
                user_id=memory.user_id,
                vector=vector,
                # Payload carries enough to fully reconstruct the Memory in another
                # process (kind + source for faithful attribution), so cross-process
                # episodic recall doesn't depend on this process's in-memory record set.
                payload={
                    "content": memory.content,
                    "user_id": memory.user_id,
                    "kind": memory.kind.value,
                    "source": memory.source.value,
                },
            )
        )
        self._bm25.add(memory.id, memory.content)
        self._by_id[memory.id] = memory
        self._entities[memory.id] = {e.name.lower() for e in memory.entities}
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        q_vector = await self._embedder.embed(query.text)
        vec_hits: list[VectorHit] = await self._vectors.search(
            user_id=query.user_id, vector=q_vector, top_k=query.top_k * 2
        )
        vector_ranking = [h.id for h in vec_hits]

        bm25_ranking = [
            mid
            for mid, _ in self._bm25.search(query.text, top_k=query.top_k * 2)
            if self._owned(mid, query.user_id)
        ]

        q_terms = {t.lower() for t in query.text.split()}
        entity_ranking = [
            mid
            for mid in self._by_id
            if self._owned(mid, query.user_id) and (self._entities.get(mid, set()) & q_terms)
        ]

        fused_ids = reciprocal_rank_fusion([vector_ranking, bm25_ranking, entity_ranking])
        # Reconstruct hits stored by another process (Redis 2-process topology): they live in
        # the shared vector index but not this process's `_by_id`. Vector search is already
        # user-scoped, so foreign hits are safe to surface. Local records win (richer: entities).
        hit_payloads = {h.id: h.payload for h in vec_hits}
        episodic: list[Memory] = []
        for mid in fused_ids:
            if mid in self._by_id:
                episodic.append(self._by_id[mid])
            elif mid in hit_payloads:
                episodic.append(_memory_from_payload(hit_payloads[mid]))

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

    def _owned(self, memory_id: str, user_id: str) -> bool:
        mem = self._by_id.get(memory_id)
        return mem is not None and mem.user_id == user_id

    async def upsert_fact(self, fact: TemporalFact) -> str:
        return await self._temporal.upsert_fact(fact, now=self._clock())

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        return await self._temporal.current_facts(user_id=user_id, subject=subject)


def _memory_from_payload(payload: dict[str, Any]) -> Memory:
    """Reconstruct an episodic ``Memory`` from a vector-index payload.

    Used for cross-process recall: a hit stored by another process is recovered from its
    payload rather than dropped. Missing ``kind``/``source`` (older payloads) default to
    episodic / agent_inferred so reconstruction is always safe.
    """
    kind = payload.get("kind")
    source = payload.get("source")
    return Memory(
        user_id=payload.get("user_id", ""),
        kind=MemoryKind(kind) if kind else MemoryKind.EPISODIC,
        content=payload.get("content", ""),
        source=MemorySource(source) if source else MemorySource.AGENT_INFERRED,
    )
