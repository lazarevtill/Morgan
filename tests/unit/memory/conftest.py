"""Shared MemoryModule factory for the memory tests.

``FakeEmbedder`` is sha256-based, so identical text embeds identically across processes --
these tests build a *fresh* MemoryModule per call to simulate a restart.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from morgan_brain.composition import build_memory_module as _build
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory


def build_memory_module(
    path: str, *, dim: int = 4, floor_margin: float | None = None
) -> MemoryModule:
    """Build a MemoryModule over the durable stack rooted at *path* (or ``:memory:``)."""
    return _build(
        open_db(path),
        embedder=FakeEmbedder(dim=dim),
        dim=dim,
        clock=lambda: datetime.now(UTC),
        floor_margin=floor_margin,
    )


#: ``vec_items`` and ``fts_memories`` as every Morgan up to ``user_version`` 5 created them:
#: without the ``status``, ``scope`` and ``author_id`` columns migration step 6 adds.
_VERSION_FIVE_INDEXES = (
    """
    CREATE VIRTUAL TABLE vec_items USING vec0(
        embedding float[{dim}] distance_metric=cosine,
        user_id TEXT,
        project TEXT
    )
    """,
    """
    CREATE VIRTUAL TABLE fts_memories USING fts5(
        memory_id UNINDEXED,
        user_id   UNINDEXED,
        project   UNINDEXED,
        content,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
)


def a_version_five_database(
    path: str,
    *,
    dim: int,
    memories: Sequence[Memory],
    vector: Callable[[str], list[float]],
) -> sqlite3.Connection:
    """A database at ``user_version`` 5, opened by this code: what ``morgan migrate`` finds
    when step 6 is the one step left.

    ``vec_items`` and ``fts_memories`` are created first, at the version-5 DDL, so the stores
    this code opens next leave them as they are. Each memory is written the way a version-5
    Morgan wrote it: its row, its ``vec_meta`` row, its vector (``vector(content)``, packed as
    ``vectors.py`` packs one) and its keyword row -- the last two without the new columns.
    """
    conn = open_db(path)
    for statement in _VERSION_FIVE_INDEXES:
        conn.execute(statement.format(dim=dim))
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    module = _build(conn, embedder=FakeEmbedder(dim=dim), dim=dim, clock=lambda: datetime.now(UTC))
    for memory in memories:
        module._episodics.put(memory)
        rowid = conn.execute(
            "INSERT INTO vec_meta (id, user_id, project, payload) VALUES (?, ?, ?, ?)",
            (memory.id, memory.user_id, memory.project, json.dumps({"content": memory.content})),
        ).lastrowid
        embedding = vector(memory.content)
        conn.execute(
            "INSERT INTO vec_items (rowid, embedding, user_id, project) VALUES (?, ?, ?, ?)",
            (rowid, struct.pack(f"{len(embedding)}f", *embedding), memory.user_id, memory.project),
        )
        conn.execute(
            "INSERT INTO fts_memories (memory_id, user_id, project, content) VALUES (?, ?, ?, ?)",
            (memory.id, memory.user_id, memory.project, memory.content),
        )
    conn.commit()
    return conn
