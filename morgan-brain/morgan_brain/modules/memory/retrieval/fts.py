"""FTS5 keyword index — the persistent replacement for the in-process BM25 index.

Two traps this module exists to handle:

* Raw user text is **not** a valid FTS5 ``MATCH`` expression. Hyphens, quotes and bare
  ``AND``/``OR`` produce syntax errors that surface as silent recall failures, so every
  token is extracted and quoted.
* The previous tokenizer was ``[a-z0-9]+``, which dropped Cyrillic entirely. ``unicode61``
  indexes it, so keyword recall works for the owner's real corpus.
"""

from __future__ import annotations

import re
import sqlite3

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
                content,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        conn.commit()

    def add(self, memory_id: str, content: str, *, user_id: str) -> None:
        self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            "INSERT INTO fts_memories (memory_id, user_id, content) VALUES (?, ?, ?)",
            (memory_id, user_id, content),
        )
        self._conn.commit()

    def search(self, text: str, *, user_id: str, top_k: int) -> list[str]:
        match = to_match_query(text)
        if not match:
            return []
        rows = self._conn.execute(
            """
            SELECT memory_id FROM fts_memories
            WHERE fts_memories MATCH ? AND user_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, user_id, top_k),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (mid,))
        self._conn.commit()
