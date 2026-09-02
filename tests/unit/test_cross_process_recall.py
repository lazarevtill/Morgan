"""Cross-process episodic recall (Redis 2-process topology).

A memory stored by one process must be recallable by a second, independent process that opens
its own connection to the same on-disk database file -- this is what the Redis 2-process
topology (brain-api + learning-worker) actually needs. A fresh ``MemoryModule`` built over the
same path is the faithful simulation: unlike the old in-process ``_by_id`` dict this test used
to clear, the durable stores (vectors, FTS, entities, episodics) hold no per-process state, so a
second instance genuinely reads what the first one wrote.
"""

from __future__ import annotations

import pytest

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, MemorySource
from tests.unit.memory.conftest import build_memory_module as _module


@pytest.mark.asyncio
async def test_recall_reconstructs_foreign_episodic_from_vector_payload(tmp_path) -> None:
    path = str(tmp_path / "m.db")
    await _module(path).store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="I love Rust",
            source=MemorySource.USER_STATED,
        )
    )
    # Simulate the other process: a fresh MemoryModule instance over the same file.
    hits = await _module(path).recall(MemoryQuery(user_id="u1", text="Rust"))
    rust = next((h for h in hits if "Rust" in h.content), None)
    assert rust is not None, "foreign episodic was dropped instead of reconstructed"
    # Reconstructed faithfully (kind + source recovered from the durable episodic record).
    assert rust.kind is MemoryKind.EPISODIC
    assert rust.source is MemorySource.USER_STATED


@pytest.mark.asyncio
async def test_reconstructed_recall_stays_user_scoped(tmp_path) -> None:
    path = str(tmp_path / "m.db")
    await _module(path).store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="secret for u1",
            source=MemorySource.USER_STATED,
        )
    )
    other = await _module(path).recall(MemoryQuery(user_id="u2", text="secret"))
    assert not any("secret" in h.content for h in other)
