"""FTS5 keyword index — the persistent replacement for the in-process BM25 index.

Two traps this module exists to handle:

* Raw user text is **not** a valid FTS5 ``MATCH`` expression. Hyphens, quotes and bare
  ``AND``/``OR`` produce syntax errors that surface as silent recall failures, so every
  token is extracted and quoted.
* The previous tokenizer was ``[a-z0-9]+``, which dropped Cyrillic entirely. ``unicode61``
  indexes it, so keyword recall works for the intended corpus.
"""

from __future__ import annotations

import json
import re
import sqlite3

from morgan_brain.models.memory import DEFAULT_PROJECT

_TOKEN = re.compile(r"\w+", re.UNICODE)


def to_match_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression (OR over quoted tokens)."""
    tokens = _TOKEN.findall(text)
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


class FtsIndex:
    """Persistent keyword index over memory content, backed by SQLite FTS5."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                memory_id UNINDEXED,
                user_id   UNINDEXED,
                project   UNINDEXED,
                content,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        conn.commit()
        self._migrate_project_column()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed.

        FTS5 virtual tables cannot be ``ALTER``ed, so ``fts_memories`` is self-contained
        (it carries its own ``content`` column, not an external-content reference) -- its
        existing rows are read out, the table is dropped and recreated with the ``project``
        column, and the rows are reinserted with ``DEFAULT_PROJECT`` backfilled.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(fts_memories)")}
        if "project" not in cols:
            rows = self._conn.execute(
                "SELECT memory_id, user_id, content FROM fts_memories"
            ).fetchall()
            self._conn.execute("DROP TABLE fts_memories")
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE fts_memories USING fts5(
                    memory_id UNINDEXED,
                    user_id   UNINDEXED,
                    project   UNINDEXED,
                    content,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
            for r in rows:
                self._conn.execute(
                    "INSERT INTO fts_memories (memory_id, user_id, project, content) "
                    "VALUES (?, ?, ?, ?)",
                    (r["memory_id"], r["user_id"], DEFAULT_PROJECT, r["content"]),
                )
            self._conn.commit()

    def add(
        self, memory_id: str, content: str, *, user_id: str, project: str = DEFAULT_PROJECT
    ) -> None:
        self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            "INSERT INTO fts_memories (memory_id, user_id, project, content) VALUES (?, ?, ?, ?)",
            (memory_id, user_id, project, content),
        )
        self._conn.commit()

    def search(
        self,
        text: str,
        *,
        user_id: str,
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
        restrict_ids: list[str] | None = None,
    ) -> list[str]:
        """Rank memories by keyword match.

        *restrict_ids* narrows the searched set to the semantic index's candidate pool.
        It is applied inside the query, before ``LIMIT``, so a memory that would rank
        below the cut in the whole store can still be returned when the pool is small --
        which is the reason routing exists. Filtering the result afterwards would give a
        different (worse) answer. ``None`` means no pool: search everything.
        """
        match = to_match_query(text)
        if not match:
            return []
        sql = "SELECT memory_id FROM fts_memories WHERE fts_memories MATCH ? AND user_id = ?"
        params: list[object] = [match, user_id]
        if project is not None:
            sql += " AND project = ?"
            params.append(project)
        if restrict_ids is not None:
            sql += " AND memory_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(restrict_ids))
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (mid,))
        self._conn.commit()
