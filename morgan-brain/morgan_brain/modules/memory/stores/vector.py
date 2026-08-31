"""VectorIndex: user-scoped vector storage + cosine search. InMemoryVectorIndex for tests;
QdrantVectorIndex for runtime. Both satisfy the same Protocol."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from morgan_brain.models.memory import DEFAULT_PROJECT

# Stable namespace for UUID5 derivation of Qdrant point ids.
# Using a fixed, project-specific UUID so ids are deterministic across restarts.
QDRANT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID_URL namespace


@dataclass
class VectorRecord:
    id: str
    user_id: str
    vector: list[float]
    project: str = DEFAULT_PROJECT
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(self, record: VectorRecord) -> None: ...
    async def search(
        self,
        *,
        user_id: str,
        vector: list[float],
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
    ) -> list[VectorHit]: ...
    async def delete(self, ids: list[str]) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, record: VectorRecord) -> None:
        self._records[record.id] = record

    async def search(
        self,
        *,
        user_id: str,
        vector: list[float],
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
    ) -> list[VectorHit]:
        scored = [
            VectorHit(id=r.id, score=_cosine(vector, r.vector), payload=r.payload)
            for r in self._records.values()
            if r.user_id == user_id and (project is None or r.project == project)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    async def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._records.pop(mid, None)


class QdrantVectorIndex:
    """Runtime adapter. Uses a single collection with a `user_id` payload filter for scoping.

    Point ids in Qdrant must be integers or UUIDs. String memory ids are mapped to
    deterministic UUID5 values (via ``QDRANT_ID_NAMESPACE``) on upsert; the original string
    id is stored in ``payload["mem_id"]`` so search can return the original value.

    Smoke-tested against a live Qdrant via ``@pytest.mark.live`` tests; unit tests inject
    a fake client via the ``_client`` constructor parameter.
    """

    def __init__(
        self,
        url: str,
        collection: str = "morgan_memories",
        dim: int = 1024,
        *,
        _client: Any = None,
    ) -> None:
        from qdrant_client.http import models as qm

        if _client is not None:
            self._client = _client
        else:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._qm = qm

    @staticmethod
    def _point_id(mem_id: str) -> str:
        """Derive a stable UUID5 string for *mem_id* (Qdrant-acceptable point id)."""
        return str(uuid.uuid5(QDRANT_ID_NAMESPACE, mem_id))

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
        point_id = self._point_id(record.id)
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=record.vector,
                    payload={
                        **record.payload,
                        "user_id": record.user_id,
                        "project": record.project,
                        # Store the original string id so search can return it.
                        "mem_id": record.id,
                    },
                )
            ],
        )

    async def search(
        self,
        *,
        user_id: str,
        vector: list[float],
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
    ) -> list[VectorHit]:
        qm = self._qm
        must = [qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
        if project is not None:
            must.append(qm.FieldCondition(key="project", match=qm.MatchValue(value=project)))
        res = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=qm.Filter(must=must),
        )
        return [
            VectorHit(
                # Return the original string id if stored; fall back to str(point.id).
                id=dict(p.payload or {}).get("mem_id") or str(p.id),
                score=p.score,
                payload=dict(p.payload or {}),
            )
            for p in res.points
        ]

    async def delete(self, ids: list[str]) -> None:
        qm = self._qm
        point_ids = [self._point_id(mid) for mid in ids]
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=point_ids),
        )
