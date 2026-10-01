"""A committed erasure invalidates store preparation across SQLite connections."""

import asyncio
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory import module as module_impl
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store import erasure
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Entity, Memory, TemporalFact


class PausedEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, text):
        self.entered.set()
        await self.release.wait()
        return await super().embed(text)


def gate(conn, embedder):
    return MemoryGate(
        build_memory_module(
            conn, embedder=embedder, dim=4, clock=lambda: datetime(2026, 10, 1, tzinfo=UTC)
        )
    )


@pytest.mark.parametrize(
    ("erased_owner", "erased_project"),
    [("owner", "personal"), ("owner", "other-context"), ("other-owner", "personal")],
)
async def test_committed_forget_cancels_prepared_store_globally_but_allows_fresh_store(
    tmp_path, erased_owner, erased_project
):
    path = str(tmp_path / "synthetic.db")
    writer_conn, eraser_conn = open_db(path), open_db(path)
    embedder = PausedEmbedder()
    writer = gate(writer_conn, embedder)
    eraser = gate(eraser_conn, FakeEmbedder(dim=4))
    await eraser.store(
        Memory(
            id="erased", user_id=erased_owner, project=erased_project, content="synthetic forgotten"
        )
    )
    pending = asyncio.create_task(
        writer.store(
            Memory(
                id="pending",
                user_id="owner",
                content="synthetic pending",
                entities=[Entity(name="synthetic")],
            )
        )
    )
    try:
        await asyncio.wait_for(embedder.entered.wait(), timeout=5)
        await eraser.forget(user_id=erased_owner, project=erased_project)
        after_forget = eraser_conn.serialize()
        assert erasure.read_generation(eraser_conn) == 1
        embedder.release.set()
        with pytest.raises(erasure.StoreInterruptedByForget, match="retry explicitly"):
            await asyncio.wait_for(pending, timeout=5)
        # The rejected write leaves no memory, index or repository metadata behind.
        assert writer_conn.serialize() == after_forget
        for sql in (
            "SELECT COUNT(*) FROM memories",
            "SELECT COUNT(*) FROM vec_meta",
            "SELECT COUNT(*) FROM vec_items",
            "SELECT COUNT(*) FROM fts_memories",
            "SELECT COUNT(*) FROM memory_entities",
        ):
            assert writer_conn.execute(sql).fetchone()[0] == 0
        assert writer_conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0
        await writer.store(Memory(id="fresh", user_id="owner", content="synthetic fresh"))
        assert await writer.get("fresh", user_id="owner") is not None
        assert erasure.read_generation(writer_conn) == 1
    finally:
        embedder.release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        writer_conn.close()
        eraser_conn.close()


async def test_failed_forget_rolls_back_generation_and_partial_deletes(tmp_path, monkeypatch):
    path = str(tmp_path / "rollback.db")
    writer_conn, eraser_conn = open_db(path), open_db(path)
    embedder = PausedEmbedder()
    writer, eraser = gate(writer_conn, embedder), gate(eraser_conn, FakeEmbedder(dim=4))
    await eraser.store(Memory(id="keep", user_id="owner", content="synthetic retained"))
    before = eraser_conn.serialize()

    def fail_after_memory_erased(conn, request):
        assert erasure.read_generation(conn) == 1
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        raise RuntimeError("synthetic deletion failure")

    monkeypatch.setitem(module_impl._DELETERS, "facts", fail_after_memory_erased)
    pending = asyncio.create_task(
        writer.store(
            Memory(id="pending", user_id="owner", content="synthetic independent prepared event")
        )
    )
    try:
        await asyncio.wait_for(embedder.entered.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="synthetic deletion failure"):
            await eraser.forget(user_id="owner", project="personal")
        assert eraser_conn.serialize() == before
        assert erasure.read_generation(eraser_conn) == 0
        embedder.release.set()
        assert await asyncio.wait_for(pending, timeout=5) == "pending"
        assert await writer.get("keep", user_id="owner") is not None
    finally:
        embedder.release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        writer_conn.close()
        eraser_conn.close()


async def test_empty_scope_forget_still_invalidates_prepared_writes(tmp_path):
    conn = open_db(str(tmp_path / "empty.db"))
    try:
        memory = gate(conn, FakeEmbedder(dim=4))
        report = await memory.forget(user_id="absent", project="absent")
        assert (report.memories, report.facts, report.history) == (0, 0, 0)
        assert erasure.read_generation(conn) == 1
    finally:
        conn.close()


async def test_light_generation_migration_preserves_legacy_assertions(tmp_path):
    path = str(tmp_path / "legacy.db")
    conn = open_db(path)
    memory = gate(conn, FakeEmbedder(dim=4))
    await memory.store(Memory(id="legacy", user_id="owner", content="synthetic legacy"))
    await memory.upsert_fact(
        TemporalFact(id="fact", user_id="owner", subject="user", predicate="drink", object="tea")
    )
    event = dict(conn.execute("SELECT * FROM memories WHERE id='legacy'").fetchone())
    fact = dict(conn.execute("SELECT * FROM facts WHERE id='fact'").fetchone())
    conn.execute("DROP TABLE erasure_state")
    conn.execute("PRAGMA user_version=9")
    conn.commit()
    conn.close()
    conn = open_db(path)
    try:
        reopened = gate(conn, FakeEmbedder(dim=4))
        assert dict(conn.execute("SELECT * FROM memories WHERE id='legacy'").fetchone()) == event
        assert dict(conn.execute("SELECT * FROM facts WHERE id='fact'").fetchone()) == fact
        assert erasure.read_generation(conn) == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10
        await reopened.forget(user_id="owner", project="personal")
        assert erasure.read_generation(conn) == 1
        assert [row["name"] for row in conn.execute("PRAGMA table_info(erasure_state)")] == [
            "singleton",
            "generation",
        ]
    finally:
        conn.close()
