"""Persistent entity-overlap index — the third recall signal.

Ordering is defined here rather than left to dict iteration: most matched entities first,
then memory id, so fusion input is stable across processes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable


class EntityIndex:
    """Persistent entity-overlap index over memory entity names, backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                user_id   TEXT NOT NULL,
                name      TEXT NOT NULL,
                PRIMARY KEY (memory_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_lookup
                ON memory_entities (user_id, name);
            """
        )
        conn.commit()

    def add(self, memory_id: str, names: Iterable[str], *, user_id: str) -> None:
        self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO memory_entities (memory_id, user_id, name) VALUES (?, ?, ?)",
            [(memory_id, user_id, n.lower()) for n in names],
        )
        self._conn.commit()

    def search(self, terms: Iterable[str], *, user_id: str, top_k: int) -> list[str]:
        wanted = [t.lower() for t in terms]
        if not wanted:
            return []
        placeholders = ",".join("?" * len(wanted))
        rows = self._conn.execute(
            f"""
            SELECT memory_id, COUNT(*) AS hits
            FROM memory_entities
            WHERE user_id = ? AND name IN ({placeholders})
            GROUP BY memory_id
            ORDER BY hits DESC, memory_id ASC
            LIMIT ?
            """,
            (user_id, *wanted, top_k),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
        self._conn.commit()
