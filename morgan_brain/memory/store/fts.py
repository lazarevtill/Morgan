"""FTS5 keyword index — the persistent replacement for the in-process BM25 index.

Two traps this module exists to handle:

* Raw user text is **not** a valid FTS5 ``MATCH`` expression. Hyphens, quotes and bare
  ``AND``/``OR`` produce syntax errors that surface as silent recall failures, so every
  token is extracted and quoted.
* The previous tokenizer was ``[a-z0-9]+``, which dropped every non-Latin script. Keyword
  recall silently returned nothing for such text -- a wrong answer, not an empty one.
  ``unicode61`` indexes it, so recall works for a multilingual corpus.
"""

from __future__ import annotations

import json
import re
import sqlite3

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.tables import Erasure
from morgan_brain.models import PERSONAL_PROJECT, MemoryStatus, Scope

_TOKEN = re.compile(r"\w+", re.UNICODE)

#: ``fts_memories`` as this code creates it. ``status``, ``scope`` and ``author_id`` follow
#: ``content`` rather than precede it, so ``content`` keeps the column index it had. Migration
#: step 6 recreates an older table from its own frozen copy of this statement
#: (``migrations._FTS_MEMORIES_AT_STEP_SIX``); a test holds the two equal. SQLite does not
#: record ``IF NOT EXISTS`` as part of the DDL.
_CREATE_FTS_MEMORIES = """CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
    memory_id UNINDEXED,
    user_id   UNINDEXED,
    project   UNINDEXED,
    content,
    status    UNINDEXED,
    scope     UNINDEXED,
    author_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
)"""


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
        conn.execute(_CREATE_FTS_MEMORIES)
        conn.commit()
        self._migrate_project_column()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed.

        FTS5 virtual tables cannot be ``ALTER``ed, so ``fts_memories`` is self-contained
        (it carries its own ``content`` column, not an external-content reference) -- its
        existing rows are read out, the table is dropped and recreated with the ``project``
        column, and the rows are reinserted with ``PERSONAL_PROJECT`` backfilled.

        The table is recreated as it stood when ``project`` was added, without the columns
        migration step 6 adds: a database this old was written before phase 0, opens
        read-only until ``morgan migrate`` runs, and gets them from step 6 then. A table that
        has ``project`` -- every one this code or step 6 created -- is left alone.
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
                    (r["memory_id"], r["user_id"], PERSONAL_PROJECT, r["content"]),
                )
            self._conn.commit()

    def add(
        self,
        memory_id: str,
        content: str,
        *,
        user_id: str,
        project: str = PERSONAL_PROJECT,
        status: MemoryStatus = MemoryStatus.STORED,
        scope: Scope = Scope.PRIVATE,
        author_id: str = "",
    ) -> None:
        """Index *content* under *memory_id*, replacing what was indexed for it before. The
        memory's status, scope and author are stored beside it, unindexed; keyword search does
        not filter on them in phase 0."""
        with write_transaction(self._conn):
            self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
            self._conn.execute(
                "INSERT INTO fts_memories "
                "(memory_id, user_id, project, content, status, scope, author_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, user_id, project, content, status.value, scope.value, author_id),
            )

    def search(
        self,
        text: str,
        *,
        user_id: str,
        top_k: int,
        project: str | None = PERSONAL_PROJECT,
    ) -> list[str]:
        """Rank memories by keyword match."""
        match = to_match_query(text)
        if not match:
            return []
        sql = "SELECT memory_id FROM fts_memories WHERE fts_memories MATCH ? AND user_id = ?"
        params: list[object] = [match, user_id]
        if project is not None:
            sql += " AND project = ?"
            params.append(project)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        with write_transaction(self._conn):
            _delete_keywords(self._conn, json.dumps(ids))


def _delete_keywords(
    conn: sqlite3.Connection,
    memory_ids: str,
    user_id: str | None = None,
    project: str | None = None,
) -> int:
    """The one delete of ``fts_memories``: the keyword rows of the memory ids in the JSON array
    *memory_ids*, and every row of *user_id* under *project*, whether or not its memory still
    exists. Called with ids alone, *user_id* and *project* are ``None``, and
    ``user_id = NULL`` is true of no row, so exactly the ids' rows go."""
    return conn.execute(
        "DELETE FROM fts_memories WHERE memory_id IN (SELECT value FROM json_each(?)) "
        "OR (user_id = ? AND project = ?)",
        (memory_ids, user_id, project),
    ).rowcount


def delete_keywords(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """`forget()`'s deleter for ``fts_memories``: the erased keyword rows, and then their words.

    FTS5 answers a DELETE with a tombstone and keeps the row's terms in its segment b-tree,
    ``fts_memories_data``, until a merge -- which a small or quiet database may never reach, so
    the words would outlive the rows and the ``VACUUM`` after them. ``optimize`` merges every
    segment into one now, inside ``forget()``'s transaction, and the deleted terms are left out
    of it. ``optimize`` rather than the ``secure-delete`` option, which needs SQLite 3.42.
    """
    erased = _delete_keywords(conn, erasure.memory_ids, erasure.user_id, erasure.project)
    conn.execute("INSERT INTO fts_memories(fts_memories) VALUES ('optimize')")
    return erased
