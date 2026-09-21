"""Durable episodic records -- the rehydration source that in-process dicts used to be.

Every stored ``Memory`` (kind, source, entities, importance -- the full record, not the
subset that used to ride along in a vector-index payload) is persisted here so recall can
rebuild it after a restart or from a second process sharing the same database file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.tables import project_tables
from morgan_brain.models import DEFAULT_PROJECT, Entity, Memory, MemoryKind, MemorySource


class EpisodicStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                project    TEXT NOT NULL DEFAULT 'default',
                kind       TEXT NOT NULL,
                source     TEXT NOT NULL,
                content    TEXT NOT NULL,
                importance REAL NOT NULL,
                entities   TEXT NOT NULL,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
            """
        )
        conn.commit()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
        if "project" not in cols:
            conn.execute(
                f"ALTER TABLE memories ADD COLUMN project TEXT NOT NULL DEFAULT '{DEFAULT_PROJECT}'"
            )
            conn.commit()

    def put(self, memory: Memory) -> None:
        with write_transaction(self._conn):
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, user_id, project, kind, source, content, importance, entities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.project,
                    memory.kind.value,
                    memory.source.value,
                    memory.content,
                    memory.importance,
                    json.dumps([{"name": e.name, "type": e.type} for e in memory.entities]),
                    memory.created_at.isoformat() if memory.created_at else None,
                ),
            )

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
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
        )

    def ids(self) -> list[str]:
        """Every stored memory's id, across users and projects, in a stable order."""
        return [str(r["id"]) for r in self._conn.execute("SELECT id FROM memories ORDER BY id")]

    def delete(self, ids: list[str]) -> None:
        with write_transaction(self._conn):
            for mid in ids:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))

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
        additions are table names Morgan itself registered, not query input.
        """
        projects: set[str] = set()
        for table in project_tables(self._conn):
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
