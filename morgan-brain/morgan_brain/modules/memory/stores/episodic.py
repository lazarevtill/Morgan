"""Durable episodic records -- the rehydration source that in-process dicts used to be.

Every stored ``Memory`` (kind, source, entities, importance -- the full record, not the
subset that used to ride along in a vector-index payload) is persisted here so recall can
rebuild it after a restart or from a second process sharing the same database file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import DEFAULT_PROJECT, Memory, MemoryKind, MemorySource


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
