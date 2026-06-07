from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex, VectorRecord


async def test_upsert_and_search_returns_nearest_first():
    idx = InMemoryVectorIndex()
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1.0, 0.0], payload={"content": "A"}))
    await idx.upsert(VectorRecord(id="b", user_id="u1", vector=[0.0, 1.0], payload={"content": "B"}))
    hits = await idx.search(user_id="u1", vector=[0.9, 0.1], top_k=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score >= hits[1].score


async def test_search_is_user_scoped():
    idx = InMemoryVectorIndex()
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1.0, 0.0], payload={}))
    await idx.upsert(VectorRecord(id="b", user_id="u2", vector=[1.0, 0.0], payload={}))
    hits = await idx.search(user_id="u1", vector=[1.0, 0.0], top_k=5)
    assert [h.id for h in hits] == ["a"]
