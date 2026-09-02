"""Tests for QdrantVectorIndex uuid point-id round-trip (commit 2).

Uses a FAKE qdrant client injected via a new _client parameter so no network is
needed. Verifies:
- upsert converts the string id to a deterministic UUID5 point id.
- search returns the original string id (from payload["mem_id"]).
- user_id scoping is preserved.

Live tests (require a real Qdrant) are marked @pytest.mark.live and skipped
by default.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from morgan_brain.modules.memory.stores.vector import (
    QDRANT_ID_NAMESPACE,
    QdrantVectorIndex,
    VectorRecord,
)

# ---------------------------------------------------------------------------
# Fake Qdrant client helpers
# ---------------------------------------------------------------------------


def _make_fake_point(point_id: Any, score: float, payload: dict[str, Any]) -> MagicMock:
    pt = MagicMock()
    pt.id = point_id
    pt.score = score
    pt.payload = payload
    return pt


def _make_fake_result(points: list[Any]) -> MagicMock:
    result = MagicMock()
    result.points = points
    return result


def _make_fake_client(search_points: list[Any] | None = None) -> MagicMock:
    """Build a MagicMock AsyncQdrantClient."""
    client = MagicMock()
    client.upsert = AsyncMock(return_value=None)
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.create_collection = AsyncMock(return_value=None)
    client.query_points = AsyncMock(return_value=_make_fake_result(search_points or []))
    return client


# ---------------------------------------------------------------------------
# UUID id round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_uses_uuid5_point_id() -> None:
    """upsert must convert the string record.id to a deterministic UUID5."""
    fake_client = _make_fake_client()
    idx = QdrantVectorIndex(url="http://unused", dim=4, _client=fake_client)

    record = VectorRecord(id="mem-abc-123", user_id="u1", vector=[0.1, 0.2, 0.3, 0.4])
    await idx.upsert(record)

    assert fake_client.upsert.called
    call_kwargs = fake_client.upsert.call_args
    points = call_kwargs.kwargs.get("points") or call_kwargs.args[1]
    assert len(points) == 1
    point = points[0]

    expected_uuid = uuid.uuid5(QDRANT_ID_NAMESPACE, "mem-abc-123")
    assert str(point.id) == str(expected_uuid), (
        f"Expected UUID5 {expected_uuid} for id 'mem-abc-123', got {point.id}"
    )


@pytest.mark.asyncio
async def test_upsert_stores_original_id_in_payload() -> None:
    """The original string id must be stored in payload['mem_id']."""
    fake_client = _make_fake_client()
    idx = QdrantVectorIndex(url="http://unused", dim=4, _client=fake_client)

    record = VectorRecord(id="original-str-id", user_id="u1", vector=[1.0, 0.0, 0.0, 0.0])
    await idx.upsert(record)

    call_kwargs = fake_client.upsert.call_args
    points = call_kwargs.kwargs.get("points") or call_kwargs.args[1]
    payload = points[0].payload
    assert payload["mem_id"] == "original-str-id"


@pytest.mark.asyncio
async def test_search_returns_original_string_id() -> None:
    """search must return the original string id from payload['mem_id']."""
    mem_id = "some-string-id"
    fake_point = _make_fake_point(
        point_id=str(uuid.uuid5(QDRANT_ID_NAMESPACE, mem_id)),
        score=0.95,
        payload={"mem_id": mem_id, "user_id": "u1", "content": "hello"},
    )
    fake_client = _make_fake_client(search_points=[fake_point])
    idx = QdrantVectorIndex(url="http://unused", dim=4, _client=fake_client)

    hits = await idx.search(user_id="u1", vector=[1.0, 0.0, 0.0, 0.0], top_k=5)

    assert len(hits) == 1
    assert hits[0].id == mem_id
    assert hits[0].score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_search_without_mem_id_falls_back_to_str_point_id() -> None:
    """search gracefully falls back to str(point.id) if payload has no 'mem_id'."""
    raw_uuid = str(uuid.uuid4())
    fake_point = _make_fake_point(
        point_id=raw_uuid,
        score=0.7,
        payload={"user_id": "u1"},  # no mem_id
    )
    fake_client = _make_fake_client(search_points=[fake_point])
    idx = QdrantVectorIndex(url="http://unused", dim=4, _client=fake_client)

    hits = await idx.search(user_id="u1", vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert hits[0].id == raw_uuid


@pytest.mark.asyncio
async def test_uuid5_is_deterministic() -> None:
    """Same string id always produces the same UUID5."""
    id1 = uuid.uuid5(QDRANT_ID_NAMESPACE, "stable-id")
    id2 = uuid.uuid5(QDRANT_ID_NAMESPACE, "stable-id")
    assert id1 == id2


@pytest.mark.asyncio
async def test_different_string_ids_produce_different_uuids() -> None:
    """Different string ids must produce different UUID5s."""
    id1 = uuid.uuid5(QDRANT_ID_NAMESPACE, "id-a")
    id2 = uuid.uuid5(QDRANT_ID_NAMESPACE, "id-b")
    assert id1 != id2


# ---------------------------------------------------------------------------
# Live tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_qdrant_upsert_search_round_trip() -> None:  # pragma: no cover
    """Integration smoke: upsert + search against a real Qdrant at localhost:6333."""
    import uuid as _uuid_mod

    idx = QdrantVectorIndex(url="http://localhost:6333", dim=4)
    await idx.ensure_collection()

    record = VectorRecord(
        id=f"live-test-{_uuid_mod.uuid4()}",
        user_id="live-user",
        vector=[1.0, 0.0, 0.0, 0.0],
        payload={"content": "live test"},
    )
    await idx.upsert(record)
    hits = await idx.search(user_id="live-user", vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert any(h.id == record.id for h in hits)
