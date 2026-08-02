"""Persistent entity-overlap index — the third recall signal.

Ordering is defined here rather than left to dict iteration: most matched entities first,
then memory id, so fusion input is stable across processes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from morgan_brain.models.memory import DEFAULT_PROJECT


class EntityIndex:
    """Persistent entity-overlap index over memory entity names, backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        # The index is created separately, after the project-column migration below runs -- for
        # a pre-existing database `memory_entities` exists without `project` at this point, and
        # a CREATE INDEX referencing that column here would fail before the ALTER TABLE gets a
        # chance to add it.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                user_id   TEXT NOT NULL,
                project   TEXT NOT NULL DEFAULT 'default',
                name      TEXT NOT NULL,
                PRIMARY KEY (memory_id, name)
            );
            """
        )
        conn.commit()
        self._migrate_project_column()
        conn.executescript(
            "CREATE INDEX IF NOT EXISTS idx_entities_lookup ON memory_entities (user_id, project, name);"
        )
        conn.commit()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(memory_entities)")}
        if "project" not in cols:
            self._conn.execute(
                f"ALTER TABLE memory_entities ADD COLUMN project TEXT NOT NULL "
                f"DEFAULT '{DEFAULT_PROJECT}'"
            )
            # The old index doesn't cover `project`; drop it so the index script below (run
            # after this migration) recreates it with the new column.
            self._conn.execute("DROP INDEX IF EXISTS idx_entities_lookup")
            self._conn.commit()

    def add(
        self,
        memory_id: str,
        names: Iterable[str],
        *,
        user_id: str,
        project: str = DEFAULT_PROJECT,
    ) -> None:
        self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO memory_entities (memory_id, user_id, project, name) "
            "VALUES (?, ?, ?, ?)",
            [(memory_id, user_id, project, n.lower()) for n in names],
        )
        self._conn.commit()

    def search(
        self,
        terms: Iterable[str],
        *,
        user_id: str,
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
    ) -> list[str]:
        wanted = [t.lower() for t in terms]
        if not wanted:
            return []
        placeholders = ",".join("?" * len(wanted))
        sql = f"""
            SELECT memory_id, COUNT(*) AS hits
            FROM memory_entities
            WHERE user_id = ? AND name IN ({placeholders})
            """
        params: list[object] = [user_id, *wanted]
        if project is not None:
            sql += " AND project = ?"
            params.append(project)
        sql += " GROUP BY memory_id ORDER BY hits DESC, memory_id ASC LIMIT ?"
        params.append(top_k)
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
        self._conn.commit()
