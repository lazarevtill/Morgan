"""VectorIndex: user-scoped vector storage + cosine search. InMemoryVectorIndex for tests;
QdrantVectorIndex for runtime. Both satisfy the same Protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorRecord:
    id: str
    user_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(self, record: VectorRecord) -> None: ...
    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, record: VectorRecord) -> None:
        self._records[record.id] = record

    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]:
        scored = [
            VectorHit(id=r.id, score=_cosine(vector, r.vector), payload=r.payload)
            for r in self._records.values()
            if r.user_id == user_id
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


class QdrantVectorIndex:
    """Runtime adapter. Uses a single collection with a `user_id` payload filter for scoping.
    Smoke-tested against a live Qdrant, not in unit tests."""

    def __init__(self, url: str, collection: str = "morgan_memories", dim: int = 1024) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models as qm

        self._client = AsyncQdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._qm = qm

    async def ensure_collection(self) -> None:
        qm = self._qm
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    async def upsert(self, record: VectorRecord) -> None:
        qm = self._qm
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload={**record.payload, "user_id": record.user_id},
                )
            ],
        )

    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]:
        qm = self._qm
        res = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
            ),
        )
        return [
            VectorHit(id=str(p.id), score=p.score, payload=dict(p.payload or {}))
            for p in res.points
        ]
