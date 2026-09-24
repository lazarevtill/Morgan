"""Durable episodic records -- the rehydration source that in-process dicts used to be.

Every stored ``Memory`` (kind, source, entities, importance -- the full record, not the
subset that used to ride along in a vector-index payload) is persisted here so recall can
rebuild it after a restart or from a second process sharing the same database file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.tables import ANSWERED_BY, Erasure, project_tables
from morgan_brain.models import PERSONAL_PROJECT, Entity, Memory, MemoryKind, MemorySource

#: Nothing here is keyed by a session; the session grain passes the table over.
HOLDS_NOTHING_PER_SESSION: tuple[str, ...] = ("memories",)

#: The columns migration step 4 added. ``Memory`` validates each from its stored text.
_PROVENANCE = (
    "origin_kind",
    "client",
    "session_id",
    "cwd",
    "author_id",
    "scope",
    "instruction_like",
    "status",
)

#: The columns migration step 8 added. ``Memory`` validates each from its stored text.
_GATE = ("redactions", "flags")


def _entities_json(entities: list[Entity]) -> str:
    return json.dumps([{"name": e.name, "type": e.type} for e in entities])


class EpisodicStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
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
                status           TEXT NOT NULL DEFAULT 'stored',
                redactions       TEXT NOT NULL DEFAULT '[]',
                flags            TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
            """
        )
        conn.commit()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
        if "project" not in cols:
            conn.execute(
                "ALTER TABLE memories ADD COLUMN project TEXT NOT NULL "
                f"DEFAULT '{PERSONAL_PROJECT}'"
            )
            conn.commit()

    def put(self, memory: Memory) -> None:
        with write_transaction(self._conn):
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, user_id, project, kind, source, content, importance, entities, created_at,
                     origin_kind, client, session_id, cwd, author_id, scope, instruction_like,
                     status, redactions, flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.project,
                    memory.kind.value,
                    memory.source.value,
                    memory.content,
                    memory.importance,
                    _entities_json(memory.entities),
                    memory.created_at.isoformat() if memory.created_at else None,
                    memory.origin_kind.value,
                    memory.client,
                    memory.session_id,
                    memory.cwd,
                    memory.author_id,
                    memory.scope.value,
                    int(memory.instruction_like),
                    memory.status.value,
                    memory.redactions,
                    memory.flags,
                ),
            )

    def set_entities(self, memory_id: str, entities: list[Entity]) -> None:
        """Rewrite one memory's stored entity list, and nothing else in its row.

        Migration step 1 writes through this rather than ``put``: it runs on databases from
        before step 4, whose ``memories`` lacks the provenance columns ``put`` names. This
        statement names only columns every version of the table has.
        """
        with write_transaction(self._conn):
            self._conn.execute(
                "UPDATE memories SET entities = ? WHERE id = ?",
                (_entities_json(entities), memory_id),
            )

    def get(self, memory_id: str) -> Memory | None:
        """The memory stored under *memory_id*, or ``None``.

        Reads whatever columns the row has. A database still waiting for migration step 4 has
        no provenance columns, and it opens read-only with its reads answering, so each of
        those fields is taken from the row when present and left to ``Memory``'s default when
        not, and the gate's two columns, absent until step 8 runs.
        """
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        # Membership in a Row tests its values, so the column names are taken out first.
        present = set(row.keys())
        provenance: dict[str, Any] = {c: row[c] for c in (*_PROVENANCE, *_GATE) if c in present}
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            project=row["project"],
            kind=MemoryKind(row["kind"]),
            source=MemorySource(row["source"]),
            content=row["content"],
            importance=row["importance"],
            entities=[Entity(**e) for e in json.loads(row["entities"])],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            **provenance,
        )

    def ids(self) -> list[str]:
        """Every stored memory's id, across users and projects, in a stable order."""
        return [str(r["id"]) for r in self._conn.execute("SELECT id FROM memories ORDER BY id")]

    def delete(self, ids: list[str]) -> None:
        with write_transaction(self._conn):
            _delete_memories(self._conn, json.dumps(ids))

    def distinct_projects(self, user_id: str) -> list[str]:
        """Return every project *user_id* has data under, across every project-keyed table.

        `memories` alone is not the answer: `facts` and `session_history` are independently
        project-keyed, and `Orchestrator._persist_turn` writes history synchronously while the
        episodic memory is written by the worker off the bus. If the worker is down -- or the
        bounded in-proc queue drops the event -- a project accumulates transcripts with zero
        memory rows. Enumerating from `memories` made `forget --all-projects` skip such a
        project silently while reporting a clean sweep. `store/tables.py::project_tables` is
        the one registry of which tables to check, shared with `MemoryModule.forget`.

        A table name cannot be a bound parameter, so each is interpolated into the SQL text
        rather than bound -- safe here because every name comes from `project_tables`, never
        from a caller: `PROJECT_TABLES` is a module-level constant and its `embedding_spaces`
        additions are table names Morgan itself registered, not query input. A table in
        `ANSWERED_BY` is skipped: the FTS tables' rows are the `turns` rows'.
        """
        projects: set[str] = set()
        for table in project_tables(self._conn):
            if table in ANSWERED_BY:
                continue
            if not self._table_exists(table):
                continue
            # Table name from the registry above, never from a caller -- see the docstring.
            sql = f"SELECT DISTINCT project FROM {table} WHERE user_id = ?"  # noqa: S608 # nosec B608
            projects.update(r["project"] for r in self._conn.execute(sql, (user_id,)))
        return sorted(projects)

    def _table_exists(self, name: str) -> bool:
        """A table may legitimately be absent -- the CLI opens the database without building
        every store's schema, and `forget()` reports those as skipped rather than failing."""
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
        ).fetchone()
        return row is not None


def _delete_memories(
    conn: sqlite3.Connection,
    memory_ids: str,
    user_id: str | None = None,
    project: str | None = None,
) -> int:
    """The one delete of ``memories``: the rows of the ids in the JSON array *memory_ids*, and
    every row of *user_id* under *project*. Called with ids alone, *user_id* and *project* are
    ``None``, and ``user_id = NULL`` is true of no row, so exactly the ids' rows go."""
    return conn.execute(
        "DELETE FROM memories WHERE id IN (SELECT value FROM json_each(?)) "
        "OR (user_id = ? AND project = ?)",
        (memory_ids, user_id, project),
    ).rowcount


def delete_memories(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """`forget()`'s deleter for ``memories``: the erased memories' rows."""
    return _delete_memories(conn, erasure.memory_ids, erasure.user_id, erasure.project)
