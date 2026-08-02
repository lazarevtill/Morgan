"""Fixtures + scorer for the memory-quality suite (design spec §13)."""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.models.memory import Memory, MemoryKind
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex


@pytest.fixture
def memory() -> MemoryModule:
    conn = open_db(":memory:")
    return MemoryModule(
        embedder=FakeEmbedder(dim=32),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime(2026, 1, 1),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )


async def seed(memory: MemoryModule, user_id: str, contents: list[str]) -> None:
    for c in contents:
        await memory.store(Memory(user_id=user_id, kind=MemoryKind.EPISODIC, content=c))


def recall_at_k(results: list[Memory], expected_substring: str, k: int) -> float:
    """1.0 if any of the top-k recalled memories contains the expected substring, else 0.0."""
    return 1.0 if any(expected_substring.lower() in m.content.lower() for m in results[:k]) else 0.0
