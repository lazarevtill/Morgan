"""Durable assertion identity is separate from index preparation and ingestion time."""

import asyncio
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.episodic import EventIdentityConflict
from morgan_brain.models import Memory, MemorySource


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return await super().embed(text)


@pytest.fixture
def stack():
    conn = open_db(":memory:")
    embedder = CountingEmbedder()
    now = datetime(2026, 9, 1, tzinfo=UTC)
    module = build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: now)
    yield MemoryGate(module), conn, embedder, now
    conn.close()


def event(**changes):
    return Memory.model_validate(
        {"id": "stable", "user_id": "owner", "content": "Synthetic tea preference", **changes}
    )


async def test_replay_does_no_embedding_or_index_writes_and_does_not_mutate_caller(stack):
    gate, conn, embedder, now = stack
    original = event(created_at=datetime(2020, 1, 1, tzinfo=UTC))
    before_caller = original.model_dump()
    assert await gate.store(original) == "stable"
    assert original.model_dump() == before_caller
    stored = await gate.get("stable", user_id="owner")
    assert stored.recorded_at == now
    assert stored.created_at == original.created_at
    before_db = conn.serialize()
    changes = conn.total_changes
    assert await gate.store(event()) == "stable"
    assert embedder.calls == 1
    assert conn.total_changes == changes
    assert conn.serialize() == before_db
    assert await gate.get("stable", user_id="owner") == stored


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_id", "intruder"),
        ("project", "work"),
        ("content", "Changed assertion"),
        ("source", "user_stated"),
        ("author_id", "other author"),
        ("created_at", datetime(2025, 1, 1, tzinfo=UTC)),
    ],
)
async def test_changed_identity_is_atomic_and_requires_no_embedding(stack, field, value):
    gate, conn, embedder, _ = stack
    await gate.store(event())
    before_db = conn.serialize()
    with pytest.raises(EventIdentityConflict) as exc:
        await gate.store(event(**{field: value}))
    assert field in exc.value.fields
    assert "Synthetic tea preference" not in str(exc.value)
    assert embedder.calls == 1
    assert conn.serialize() == before_db


async def test_identical_text_with_distinct_ids_preserves_two_assertions(stack):
    gate, conn, embedder, _ = stack
    await gate.store(event(id="first"))
    await gate.store(event(id="second"))
    assert embedder.calls == 2
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM vec_meta").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM fts_memories").fetchone()[0] == 2


async def test_reported_author_is_independent_of_assertion_source(stack):
    gate, _, _, _ = stack
    await gate.store(event(id="agent", author_id="agent:client"))
    await gate.store(event(id="user", author_id="owner", source=MemorySource.USER_STATED))
    agent = await gate.get("agent", user_id="owner")
    user = await gate.get("user", user_id="owner")
    assert agent.source is MemorySource.UNKNOWN and agent.author_id == "agent:client"
    assert user.source is MemorySource.USER_STATED and user.author_id == "owner"


async def test_recorded_time_is_sampled_after_embedding_preparation(stack, monkeypatch):
    gate, _, embedder, initial = stack
    inserted = datetime(2026, 9, 2, tzinfo=UTC)
    clock = [initial]
    monkeypatch.setattr(gate._store, "_clock", lambda: clock[0])
    original_embed = embedder.embed

    async def delayed_embed(text):
        clock[0] = inserted
        return await original_embed(text)

    monkeypatch.setattr(embedder, "embed", delayed_embed)
    await gate.store(event())
    stored = await gate.get("stable", user_id="owner")
    assert stored.created_at == initial
    assert stored.recorded_at == inserted


async def test_version_seven_migration_preserves_legacy_provenance_and_unknown_ingestion(stack):
    gate, conn, _, now = stack
    await gate.store(event(source=MemorySource.USER_STATED))
    conn.execute("ALTER TABLE memories DROP COLUMN recorded_at")
    conn.execute("PRAGMA user_version=7")
    conn.commit()
    reopened = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: now)
    )
    legacy = await reopened.get("stable", user_id="owner")
    assert legacy.source is MemorySource.USER_STATED
    assert legacy.recorded_at is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 8
    await reopened.store(event(id="new"))
    assert (await reopened.get("new", user_id="owner")).recorded_at == now


@pytest.mark.parametrize("same_owner", [False, True])
async def test_racing_preparations_recheck_identity_before_any_index_write(tmp_path, same_owner):
    ready = asyncio.Event()
    calls = 0

    class BarrierEmbedder(FakeEmbedder):
        async def embed(self, text):
            nonlocal calls
            calls += 1
            if calls == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=5)
            return await super().embed(text)

    conns = [open_db(str(tmp_path / "race.db")) for _ in range(2)]
    embedder = BarrierEmbedder(dim=4)
    gates = [MemoryGate(build_memory_module(conn, embedder=embedder, dim=4)) for conn in conns]
    try:
        results = await asyncio.gather(
            gates[0].store(event(user_id="owner")),
            gates[1].store(event(user_id="owner" if same_owner else "intruder")),
            return_exceptions=True,
        )
        assert results.count("stable") == (2 if same_owner else 1)
        assert sum(isinstance(result, EventIdentityConflict) for result in results) == (
            0 if same_owner else 1
        )
        assert conns[0].execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conns[0].execute("SELECT COUNT(*) FROM vec_meta").fetchone()[0] == 1
        assert conns[0].execute("SELECT COUNT(*) FROM fts_memories").fetchone()[0] == 1
        assert conns[0].execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        for conn in conns:
            conn.close()
