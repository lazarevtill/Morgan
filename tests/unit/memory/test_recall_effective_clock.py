"""Present recall uses the injected clock, rather than the latest timeline head."""

from datetime import UTC, datetime

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import MemoryQuery, TemporalFact


async def test_recall_respects_effective_boundary_and_scope():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    future = datetime(2027, 1, 1, tzinfo=UTC)
    conn = open_db(":memory:")
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: now)
    )
    try:
        await gate.upsert_fact(
            TemporalFact(user_id="owner", subject="user", predicate="drink", object="tea")
        )
        await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=future,
            )
        )
        for owner, project in (("other", "personal"), ("owner", "work")):
            await gate.upsert_fact(
                TemporalFact(
                    user_id=owner,
                    project=project,
                    subject="user",
                    predicate="drink",
                    object="private coffee",
                    valid_from=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
        assert [fact.object for fact in await gate.current_facts(user_id="owner")] == ["tea"]
        primitive = SqliteTemporalStore(conn=conn, initialize=False)
        assert [fact.object for fact in await primitive.current_facts(user_id="owner")] == ["water"]
        query = MemoryQuery(user_id="owner", text="user drink")
        now = datetime(2026, 9, 1, tzinfo=UTC)
        assert [memory.content for memory in (await gate.recall(query)).memories] == [
            "user drink tea"
        ]
        now = future
        assert [memory.content for memory in (await gate.recall(query)).memories] == [
            "user drink water"
        ]
    finally:
        conn.close()
