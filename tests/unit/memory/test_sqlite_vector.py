import pytest

from morgan_brain.memory.db import open_db
from morgan_brain.memory.vectors import SqliteVectorIndex, VectorRecord


def _idx(tmp_path, dim=4):
    return SqliteVectorIndex(open_db(str(tmp_path / "m.db")), dim=dim)


async def test_upsert_then_search_returns_the_record(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(
        VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0], payload={"content": "x"})
    )
    hits = await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]
    assert hits[0].payload["content"] == "x"


async def test_search_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="b", user_id="u2", vector=[1, 0, 0, 0]))
    hits = await idx.search(user_id="u1", vector=[1, 0, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    idx = SqliteVectorIndex(open_db(path), dim=4)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[0, 1, 0, 0]))
    reopened = SqliteVectorIndex(open_db(path), dim=4)
    hits = await reopened.search(user_id="u", vector=[0, 1, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_delete_removes_the_vector(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0]))
    await idx.delete(["a"])
    assert await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=5) == []


async def test_upsert_replaces_rather_than_duplicates(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[0, 1, 0, 0]))
    hits = await idx.search(user_id="u", vector=[0, 1, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_scoping_happens_inside_the_knn_not_after(tmp_path):
    """Regression: post-filtering a global KNN silently drops the caller's own neighbours.

    Two users share the store. u1 owns the exact match AND the second-nearest vector, but
    u2's identical vector would crowd the top-k of an unscoped query. With top_k=2, a
    correct implementation returns BOTH of u1's vectors.
    """
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="u1-exact", user_id="u1", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="u2-exact", user_id="u2", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="u1-near", user_id="u1", vector=[0.9, 0.1, 0, 0]))
    await idx.upsert(VectorRecord(id="u2-near", user_id="u2", vector=[0.9, 0.1, 0, 0]))

    hits = await idx.search(user_id="u1", vector=[1, 0, 0, 0], top_k=2)
    assert [h.id for h in hits] == ["u1-exact", "u1-near"]


async def test_score_is_cosine_similarity_not_raw_distance(tmp_path):
    """Pins the SCALE, not just the ordering.

    sqlite-vec reports cosine *distance* (1 - similarity, range 0..2). The score must be
    converted back to true cosine similarity on -1..1, matching InMemoryVectorIndex and
    QdrantVectorIndex, so an exact match scores ~1.0 and an orthogonal vector scores ~0.0.
    Negating distance (-distance) would instead yield -0.0 and -1.0 here — this test fails
    against that.
    """
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="exact", user_id="u", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="orthogonal", user_id="u", vector=[0, 1, 0, 0]))

    hits = await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=2)
    scores = {h.id: h.score for h in hits}
    assert scores["exact"] == pytest.approx(1.0)
    assert scores["orthogonal"] == pytest.approx(0.0)
