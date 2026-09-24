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
from pathlib import Path

from morgan_brain.composition import build_memory_module as _build
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.projects import ProjectStore
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


def a_history_store(
    conn: sqlite3.Connection | None = None, *, clock: Callable[[], datetime]
) -> SessionHistoryStore:
    """A session history store over a connection that carries ``projects`` as well.

    A turn registers its project, and ``projects`` is ``ProjectStore``'s table, not the
    history store's. The composition root opens both on the one connection; a test that builds
    a history store on a connection of its own needs the same pair, and this is where that
    pairing lives rather than in the history store's own ``__init__``.
    """
    conn = conn if conn is not None else open_db(":memory:")
    ProjectStore(conn)
    return SessionHistoryStore(conn, clock=clock)


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


#: ``projects``, ``memories`` and ``facts`` exactly as phase 0's head (``3193f97``) created them:
#: without step 8's columns. A version-7 fixture is built from these and never by this branch's
#: stores, which create the latest DDL and would hide the step.
_VERSION_SEVEN_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS projects (
        name                TEXT PRIMARY KEY,
        classification      TEXT NOT NULL DEFAULT 'unclassified',
        remote              TEXT,
        root                TEXT,
        capture_enabled     INTEGER NOT NULL DEFAULT 1,
        retention_days      INTEGER,
        paused_until        TEXT,
        consolidate_enabled INTEGER NOT NULL DEFAULT 1,
        created_at          TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id               TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        -- pre-phase-0 default; kept for fresh/migrated parity, no writer relies on it
        project          TEXT NOT NULL DEFAULT 'default',
        kind             TEXT NOT NULL,
        source           TEXT NOT NULL,
        content          TEXT NOT NULL,
        importance       REAL NOT NULL,
        entities         TEXT NOT NULL,
        created_at       TEXT,
        origin_kind      TEXT NOT NULL DEFAULT 'unknown',
        client           TEXT NOT NULL DEFAULT '',
        session_id       TEXT NOT NULL DEFAULT '',
        cwd              TEXT NOT NULL DEFAULT '',
        author_id        TEXT NOT NULL DEFAULT '',
        scope            TEXT NOT NULL DEFAULT 'private',
        instruction_like INTEGER NOT NULL DEFAULT 0,
        status           TEXT NOT NULL DEFAULT 'stored'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id)",
    """
    CREATE TABLE IF NOT EXISTS facts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        -- pre-phase-0 default; kept for fresh/migrated parity, no writer relies on it
        project TEXT NOT NULL DEFAULT 'default',
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        superseded_by TEXT,
        last_confirmed TEXT,
        author_id TEXT NOT NULL DEFAULT '',
        scope TEXT NOT NULL DEFAULT 'private'
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_one_current "
    "ON facts (user_id, project, subject, predicate) WHERE valid_to IS NULL",
)


def a_version_seven_database(path: str) -> None:
    """A database at ``user_version`` 7 as phase 0's ``morgan migrate`` left the live file:
    ``projects``, ``memories`` and ``facts`` at the frozen DDL above, one row each, and none of
    the tables or columns this branch adds. Opened by nothing of this branch's stores."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(path)
    for statement in _VERSION_SEVEN_TABLES:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO projects (name, classification, created_at) "
        "VALUES ('p', 'unclassified', '2026-09-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO memories (id, user_id, project, kind, source, content, importance, "
        "entities, created_at, author_id) VALUES ('m-1', 'u', 'p', 'episodic', 'user_stated', "
        "'the harbor mirror', 0.5, '[]', '2026-09-01T00:00:00+00:00', 'u')"
    )
    conn.execute(
        "INSERT INTO facts (id, user_id, project, subject, predicate, object, source, confidence, "
        "valid_from, author_id) VALUES ('f-1', 'u', 'p', 'harbor', 'runs_on', 'k8s', "
        "'user_stated', 1.0, '2026-09-01T00:00:00+00:00', 'u')"
    )
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()


def a_version_two_database(path: str) -> None:
    """A database at ``user_version`` 2, what every Morgan before phase 0 wrote: ``memories``
    and ``facts`` at the pre-phase-0 DDL ``test_provenance_columns`` freezes, one row each. It
    opens read-only at 3 and reaches 8 only under ``morgan migrate``, because steps 4-6 are
    heavy."""
    # Imported here, not at module level: that test module imports this conftest.
    from tests.unit.memory.test_provenance_columns import _PRE_PHASE_ZERO_DDL

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(path)
    for statement in _PRE_PHASE_ZERO_DDL.values():
        conn.execute(statement)
    conn.execute(
        "INSERT INTO memories VALUES ('m-1', 'u', 'p', 'episodic', 'user_stated', "
        '\'the harbor mirror\', 0.5, \'[{"name": "harbor", "type": "unknown"}]\', '
        "'2026-09-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO facts VALUES ('f-1', 'u', 'p', 'harbor', 'runs_on', 'k8s', 'user_stated', "
        "1.0, '2026-09-01T00:00:00+00:00', NULL, NULL, '2026-09-01T00:00:00+00:00')"
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
