"""Cross-process episodic recall (Redis 2-process topology).

An episodic stored by another process lives only in the shared vector index, not in this
process's in-memory record set (``_by_id``). recall() must reconstruct it from the vector
payload rather than drop it. Clearing ``_by_id`` faithfully simulates the cross-process case
without needing a real Qdrant + two processes.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, MemorySource
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex

T0 = datetime(2026, 1, 1)


def _mm() -> MemoryModule:
    return MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: T0,
    )


@pytest.mark.asyncio
async def test_recall_reconstructs_foreign_episodic_from_vector_payload() -> None:
    mm = _mm()
    await mm.store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="I love Rust",
            source=MemorySource.USER_STATED,
        )
    )
    # Simulate the other process: drop this process's record set; the shared index keeps it.
    mm._by_id.clear()  # noqa: SLF001

    hits = await mm.recall(MemoryQuery(user_id="u1", text="Rust"))
    rust = next((h for h in hits if "Rust" in h.content), None)
    assert rust is not None, "foreign episodic was dropped instead of reconstructed"
    # Reconstructed faithfully (kind + source recovered from the payload).
    assert rust.kind is MemoryKind.EPISODIC
    assert rust.source is MemorySource.USER_STATED


@pytest.mark.asyncio
async def test_reconstructed_recall_stays_user_scoped() -> None:
    mm = _mm()
    await mm.store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="secret for u1",
            source=MemorySource.USER_STATED,
        )
    )
    mm._by_id.clear()  # noqa: SLF001

    other = await mm.recall(MemoryQuery(user_id="u2", text="secret"))
    assert not any("secret" in h.content for h in other)
