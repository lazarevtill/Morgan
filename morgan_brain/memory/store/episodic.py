"""Durable episodic records -- the rehydration source that in-process dicts used to be.

Every stored ``Memory`` (kind, source, entities, importance -- the full record, not the
subset that used to ride along in a vector-index payload) is persisted here so recall can
rebuild it after a restart or from a second process sharing the same database file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import ClassVar

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
        self._conn.commit()

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

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        self._conn.commit()

    #: Every project-keyed table `forget()` erases from. `memories` alone is not the answer:
    #: `facts`, `interaction_signals` and `session_history` are independently project-keyed,
    #: and `Orchestrator._persist_turn` writes history and the base signal synchronously while
    #: the episodic memory is written by the worker off the bus. If the worker is down — or the
    #: bounded in-proc queue drops the event — a project accumulates transcripts and signals
    #: with zero memory rows. Enumerating from `memories` made `forget --all-projects` skip
    #: such a project silently while reporting a clean sweep.
    #:
    #: A table name cannot be a bound parameter, so each is a literal statement rather than a
    #: name interpolated into SQL.
    _PROJECT_TABLE_SQL: ClassVar[dict[str, str]] = {
        "memories": "SELECT DISTINCT project FROM memories WHERE user_id = ?",
        "facts": "SELECT DISTINCT project FROM facts WHERE user_id = ?",
        "interaction_signals": (
            "SELECT DISTINCT project FROM interaction_signals WHERE user_id = ?"
        ),
        "session_history": "SELECT DISTINCT project FROM session_history WHERE user_id = ?",
    }

    def distinct_projects(self, user_id: str) -> list[str]:
        """Return every project *user_id* has data under, across all project-keyed tables."""
        projects: set[str] = set()
        for table, sql in self._PROJECT_TABLE_SQL.items():
            if not self._table_exists(table):
                continue
            projects.update(r["project"] for r in self._conn.execute(sql, (user_id,)))
        return sorted(projects)

    def _table_exists(self, name: str) -> bool:
        """A table may legitimately be absent -- the CLI opens the database without building
        every store's schema, and `forget()` reports those as skipped rather than failing."""
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
        ).fetchone()
        return row is not None
