"""Shared MemoryGate fixture for the security test suite."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.security.memory_gate import MemoryGate


@pytest.fixture
def gate() -> MemoryGate:
    """A MemoryGate over a fresh in-memory MemoryModule."""
    conn = open_db(":memory:")
    module = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime.now(UTC),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
    return MemoryGate(module)
