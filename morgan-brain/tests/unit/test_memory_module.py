from datetime import datetime

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.store import MemoryModule


def _module() -> MemoryModule:
    conn = open_db(":memory:")
    return MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime(2026, 1, 1),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
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


async def test_entity_overlap_boosts_recall():
    m = _module()
    await m.store(
        Memory(user_id="u1", content="met Alice yesterday", entities=[Entity(name="Alice")])
    )
    await m.store(Memory(user_id="u1", content="random unrelated text"))
    hits = await m.recall(MemoryQuery(user_id="u1", text="Alice", top_k=2))
    assert hits and hits[0].content == "met Alice yesterday"


async def test_facts_delegate_to_temporal_store():
    m = _module()
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    facts = await m.current_facts(user_id="u1")
    assert len(facts) == 1 and facts[0].object == "Berlin"


async def test_recall_surfaces_current_facts():
    m = _module()
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    hits = await m.recall(MemoryQuery(user_id="u1", text="anything", top_k=5))
    assert any("Berlin" in h.content for h in hits)


async def test_recall_facts_are_user_scoped():
    m = _module()
    await m.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    hits = await m.recall(MemoryQuery(user_id="u2", text="anything", top_k=5))
    assert hits == []
