"""An archive import cannot resume old work across a committed forget."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from morgan_brain.app.chatgpt_import import (
    ARCHIVE_PROJECT,
    import_chatgpt,
    is_held_out,
)
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
from morgan_brain.models import Memory


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return await super().embed(text)


def _export(path, turns=1):
    conversation = next(f"c{i}" for i in range(50) if not is_held_out(f"c{i}"))
    path.write_text(
        json.dumps(
            [
                {
                    "id": conversation,
                    "mapping": {
                        str(i): {
                            "message": {
                                "id": f"synthetic-{i}",
                                "author": {"role": "user"},
                                "create_time": 1700000000 + i,
                                "content": {"parts": [f"synthetic archive message {i}"]},
                            }
                        }
                        for i in range(turns)
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _gate(conn, embedder):
    return MemoryGate(build_memory_module(conn, embedder=embedder, dim=4))


async def _finish(pending, release):
    release.set()
    if not pending.done():
        pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)


async def test_forget_during_archive_read_cancels_import_before_embedding(tmp_path, monkeypatch):
    path = _export(tmp_path / "synthetic.json")
    conn = open_db(str(tmp_path / "synthetic.db"))
    eraser_conn = open_db(str(tmp_path / "synthetic.db"))
    embedder = CountingEmbedder()
    gate, eraser = _gate(conn, embedder), _gate(eraser_conn, FakeEmbedder(dim=4))
    await eraser.store(
        Memory(
            id="forgotten",
            user_id="owner",
            project=ARCHIVE_PROJECT,
            content="synthetic previous archive",
        )
    )
    entered, release = threading.Event(), threading.Event()
    original_read = Path.read_text

    def paused_read(self, *args, **kwargs):
        if self == path:
            entered.set()
            assert release.wait(timeout=5), "synthetic archive read was not released"
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", paused_read)
    pending = asyncio.create_task(import_chatgpt(path, gate=gate, user_id="owner"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await eraser.forget(user_id="owner", project=ARCHIVE_PROJECT)
        after_forget = eraser_conn.serialize()
        release.set()
        with pytest.raises(StoreInterruptedByForget, match="retry explicitly"):
            await asyncio.wait_for(pending, timeout=5)
        assert embedder.calls == 0
        assert conn.serialize() == after_forget
    finally:
        await _finish(pending, release)
        conn.close()
        eraser_conn.close()


async def test_forget_during_import_canary_cancels_later_archive_writes(tmp_path, monkeypatch):
    path = _export(tmp_path / "synthetic.json", turns=2)
    conn = open_db(str(tmp_path / "synthetic.db"))
    eraser_conn = open_db(str(tmp_path / "synthetic.db"))
    embedder = CountingEmbedder()
    gate, eraser = _gate(conn, embedder), _gate(eraser_conn, FakeEmbedder(dim=4))
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused_canary():
        entered.set()
        await release.wait()

    monkeypatch.setattr(gate, "check_embedding_space", paused_canary)
    pending = asyncio.create_task(import_chatgpt(path, gate=gate, user_id="owner", canary_every=1))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        await eraser.forget(user_id="owner", project=ARCHIVE_PROJECT)
        after_forget = eraser_conn.serialize()
        release.set()
        with pytest.raises(StoreInterruptedByForget, match="retry explicitly"):
            await asyncio.wait_for(pending, timeout=5)
        assert embedder.calls == 1
        assert conn.serialize() == after_forget
    finally:
        await _finish(pending, release)
        conn.close()
        eraser_conn.close()


async def test_forget_during_replay_read_refuses_cached_assertion(tmp_path, monkeypatch):
    path = _export(tmp_path / "synthetic.json")
    conn = open_db(str(tmp_path / "synthetic.db"))
    eraser_conn = open_db(str(tmp_path / "synthetic.db"))
    embedder = CountingEmbedder()
    gate, eraser = _gate(conn, embedder), _gate(eraser_conn, FakeEmbedder(dim=4))
    await import_chatgpt(path, gate=gate, user_id="owner")
    entered, release = asyncio.Event(), asyncio.Event()
    original_get = gate.get

    async def paused_get(memory_id, *, user_id):
        existing = await original_get(memory_id, user_id=user_id)
        entered.set()
        await release.wait()
        return existing

    monkeypatch.setattr(gate, "get", paused_get)
    pending = asyncio.create_task(import_chatgpt(path, gate=gate, user_id="owner"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        await eraser.forget(user_id="owner", project=ARCHIVE_PROJECT)
        after_forget = eraser_conn.serialize()
        release.set()
        with pytest.raises(StoreInterruptedByForget, match="retry explicitly"):
            await asyncio.wait_for(pending, timeout=5)
        assert embedder.calls == 1
        assert conn.serialize() == after_forget
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    finally:
        await _finish(pending, release)
        conn.close()
        eraser_conn.close()


@pytest.mark.parametrize("existing", [False, True])
async def test_explicit_stale_generation_refuses_new_store_and_replay(tmp_path, existing):
    conn = open_db(str(tmp_path / "synthetic.db"))
    embedder = CountingEmbedder()
    gate = _gate(conn, embedder)
    memory = Memory(id="synthetic", user_id="owner", content="synthetic source")
    try:
        if existing:
            await gate.store(memory)
        generation = gate.capture_erasure_generation()
        # A global guard also cancels work across an erasure of another context.
        await gate.forget(user_id="owner", project="unrelated")
        before, calls = conn.serialize(), embedder.calls
        with pytest.raises(StoreInterruptedByForget, match="retry explicitly"):
            await gate.store(memory, expected_generation=generation)
        assert conn.serialize() == before
        assert embedder.calls == calls
        # A deliberately new call can use the current generation without a retry loop.
        assert (
            await gate.store(memory, expected_generation=gate.capture_erasure_generation())
            == memory.id
        )
        assert embedder.calls == calls + (not existing)
    finally:
        conn.close()


@pytest.mark.parametrize("generation", [True, False, -1, 0.0, "0"])
async def test_generation_guard_requires_a_nonnegative_integer(tmp_path, generation):
    conn = open_db(str(tmp_path / "synthetic.db"))
    embedder = CountingEmbedder()
    gate = _gate(conn, embedder)
    try:
        before = conn.serialize()
        with pytest.raises(ValueError, match="captured erasure generation"):
            await gate.store(
                Memory(user_id="owner", content="synthetic"), expected_generation=generation
            )
        assert embedder.calls == 0
        assert conn.serialize() == before
    finally:
        conn.close()


async def test_default_store_preserves_old_adapter_signature():
    class OldAdapter:
        async def store(self, memory):
            return memory.id

    gate = MemoryGate(OldAdapter())
    memory = Memory(user_id="owner", content="synthetic")
    assert await gate.store(memory) == memory.id
    assert await gate.store(memory, expected_generation=None) == memory.id
