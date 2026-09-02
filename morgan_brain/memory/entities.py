"""Persistent entity-overlap index — the third recall signal.

Ordering is defined here rather than left to dict iteration: most matched entities first,
then memory id, so fusion input is stable across processes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

from morgan_brain.models import DEFAULT_PROJECT


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
            "CREATE INDEX IF NOT EXISTS idx_entities_lookup "
            "ON memory_entities (user_id, project, name);"
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
        restrict_ids: list[str] | None = None,
    ) -> list[str]:
        """Rank memories by how many of *terms* they mention.

        *restrict_ids* is the semantic index's candidate pool, applied inside the query
        before ``LIMIT`` for the same reason as in ``FtsIndex.search``. ``None`` means
        search everything.
        """
        wanted = [t.lower() for t in terms]
        if not wanted:
            return []
        # One literal statement, no SQL assembled from data: the term list arrives as a bound
        # JSON array through json_each, and "project IS NULL means every project" is expressed
        # as a bound flag rather than by appending a clause.
        rows = self._conn.execute(
            """
            SELECT memory_id, COUNT(*) AS hits
            FROM memory_entities
            WHERE user_id = ?
              AND name IN (SELECT value FROM json_each(?))
              AND (? OR project = ?)
              AND (? OR memory_id IN (SELECT value FROM json_each(?)))
            GROUP BY memory_id
            ORDER BY hits DESC, memory_id ASC
            LIMIT ?
            """,
            (
                user_id,
                json.dumps(wanted),
                project is None,
                project,
                restrict_ids is None,
                json.dumps(restrict_ids or []),
                top_k,
            ),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
        self._conn.commit()
