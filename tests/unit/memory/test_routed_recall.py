"""Recall with the semantic upper index wired in.

The narrowing has to be pushed *into* each signal's query, not applied to its output.
Filtering the top-k after the fact is a different mechanism with a different result: a
relevant memory sitting below the cut is never seen by a post-filter, and reaching it is
the entire point of routing. `test_routing_reaches_past_the_unrouted_cut` is the test
that tells the two apart, so it is the one that must not be weakened.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.fts import FtsIndex
from morgan_brain.memory.vectors import SqliteVectorIndex, VectorRecord
from morgan_brain.models import Memory, MemoryQuery

U = "u1"
P = "acme"
_PROBE = [0.0, *([0.0] * 6), 1.0]


@pytest.fixture
def wiring():
    conn = open_db(":memory:")
    module = build_memory_module(
        conn,
        embedder=FakeEmbedder(dim=8),
        dim=8,
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    semantic = module._semantic
    semantic.ensure_schemas(user_id=U, project=P)
    yield module, semantic, conn
    conn.close()


async def _store(module, content: str, entity_names: list[str]) -> str:
    from morgan_brain.models import Entity

    return await module.store(
        Memory(
            user_id=U,
            project=P,
            content=content,
            entities=[Entity(name=n) for n in entity_names],
        )
    )


async def test_recall_without_a_routable_term_is_unchanged(wiring):
    """The index is wired in and has nothing to say, so recall behaves exactly as it did
    before it existed. This is the invariant every other behaviour sits on."""
    module, _semantic, _conn = wiring
    await _store(module, "the deploy was blocked", [])

    hits = await module.recall(MemoryQuery(user_id=U, project=P, text="deploy", top_k=8))
    assert [m.content for m in hits] == ["the deploy was blocked"]


async def test_routing_excludes_a_memory_outside_the_pool(wiring):
    module, semantic, _conn = wiring
    await _store(module, "Harbor blocked the deploy", ["harbor"])
    await _store(module, "dentist appointment was moved", ["dentist"])
    semantic.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    semantic.assign(user_id=U, project=P, entity="dentist", schema_name="health")

    hits = await module.recall(MemoryQuery(user_id=U, project=P, text="Harbor", top_k=8))
    contents = [m.content for m in hits]
    assert "Harbor blocked the deploy" in contents
    assert "dentist appointment was moved" not in contents


async def test_sqlite_vector_searches_inside_the_pool_not_after_it(wiring):
    """The test that tells a pre-filter from a post-filter.

    Five vectors; m0 and m1 are the two globally nearest to the probe. Ask for k=2
    restricted to {m3, m4}. A post-filter would take the global top-2 (m0, m1), find
    neither in the pool, and return nothing. A real pre-filter returns m3 and m4. This is
    the whole value of routing: a memory that ranks below the cut in the full store is
    still reachable inside a small pool.
    """
    _module, _semantic, conn = wiring
    vectors = SqliteVectorIndex(conn, dim=8)
    for i in range(5):
        await vectors.upsert(
            VectorRecord(
                id=f"m{i}", user_id=U, project=P, vector=[float(i), *([0.0] * 6), 1.0], payload={}
            )
        )

    unrestricted = await vectors.search(user_id=U, vector=_PROBE, top_k=2, project=P)
    assert [h.id for h in unrestricted] == ["m0", "m1"]

    restricted = await vectors.search(
        user_id=U, vector=_PROBE, top_k=2, project=P, restrict_ids=["m3", "m4"]
    )
    assert sorted(h.id for h in restricted) == ["m3", "m4"]


async def test_fts_searches_inside_the_pool_not_after_it(wiring):
    """Same property for the keyword signal: the LIMIT applies after the pool, not before."""
    _module, _semantic, conn = wiring
    fts = FtsIndex(conn)
    for i in range(5):
        fts.add(f"k{i}", "deploy note about the pipeline", user_id=U, project=P)

    assert len(fts.search("deploy", user_id=U, top_k=2, project=P)) == 2
    assert fts.search("deploy", user_id=U, top_k=2, project=P, restrict_ids=["k4"]) == ["k4"]


async def test_an_empty_pool_never_silences_recall(wiring):
    """The entity is known to the index but files nothing, so routing must stand down
    rather than narrow the search to nothing."""
    module, semantic, _conn = wiring
    await _store(module, "the deploy was blocked", [])
    semantic.assign(user_id=U, project=P, entity="harbor", schema_name="work")

    hits = await module.recall(MemoryQuery(user_id=U, project=P, text="Harbor deploy", top_k=8))
    assert [m.content for m in hits] == ["the deploy was blocked"]
