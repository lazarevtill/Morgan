from datetime import UTC, datetime

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.module import MemoryModule
from morgan_brain.models import Entity, Memory, MemoryKind, MemoryQuery, TemporalFact


def _module() -> MemoryModule:
    return build_memory_module(
        open_db(":memory:"),
        embedder=FakeEmbedder(dim=16),
        dim=16,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


async def test_store_then_recall_finds_memory():
    m = _module()
    await m.store(Memory(user_id="u1", kind=MemoryKind.EPISODIC, content="I love hiking in Berlin"))
    hits = await m.recall(MemoryQuery(user_id="u1", text="hiking Berlin", top_k=5))
    assert any("hiking" in h.content for h in hits)


async def test_recall_is_user_scoped():
    m = _module()
    await m.store(Memory(user_id="u1", kind=MemoryKind.EPISODIC, content="secret note"))
    hits = await m.recall(MemoryQuery(user_id="u2", text="secret", top_k=5))
    assert hits == []


async def test_current_facts_are_surfaced_by_recall():
    m = _module()
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    hits = await m.recall(MemoryQuery(user_id="u1", text="where do I live", top_k=5))
    assert any(h.kind is MemoryKind.SEMANTIC and "Berlin" in h.content for h in hits)


async def test_upsert_supersedes_instead_of_overwriting():
    m = _module()
    first = await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Lisbon")
    )
    current = await m.current_facts(user_id="u1")
    assert [f.object for f in current] == ["Lisbon"]
    assert first not in [f.id for f in current]


async def test_entity_overlap_boosts_recall():
    m = _module()
    await m.store(
        Memory(user_id="u1", content="met Alice yesterday", entities=[Entity(name="Alice")])
    )
    await m.store(Memory(user_id="u1", content="random unrelated text"))
    hits = await m.recall(MemoryQuery(user_id="u1", text="Alice", top_k=2))
    assert hits and hits[0].content == "met Alice yesterday"


async def test_recall_facts_are_user_scoped():
    m = _module()
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    hits = await m.recall(MemoryQuery(user_id="u2", text="anything", top_k=5))
    assert hits == []
