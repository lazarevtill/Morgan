"""Shared MemoryModule test factory for the durable memory-store test suite.

``FakeEmbedder`` is sha256-based (see ``modules/memory/indexing/embedder.py``), so identical
text embeds identically across processes -- unlike the builtin ``hash()``, which
``PYTHONHASHSEED`` randomises per process. That determinism matters here: these tests build a
*fresh* MemoryModule instance per call to simulate a restart, and cross-instance vector recall
would silently break if the embedder's output changed between instances.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore


def build_memory_module(path: str, *, dim: int = 4) -> MemoryModule:
    """Build a MemoryModule over the durable stack rooted at *path* (or ``:memory:``)."""
    conn = open_db(path)
    return MemoryModule(
        embedder=FakeEmbedder(dim=dim),
        vectors=SqliteVectorIndex(conn, dim=dim),
        temporal=SqliteTemporalStore(path),
        clock=lambda: datetime.now(UTC),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
