"""Valid-time fact store (SQLite). A fact is currently valid when valid_to IS NULL. Asserting a
new value for the same (user, subject, predicate) closes the old interval (sets valid_to = now,
superseded_by = new id) instead of deleting it — so history stays queryable and recall is never
confidently stale."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from itertools import pairwise
from typing import Any

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.models import PERSONAL_PROJECT, MemorySource, TemporalFact

#: The columns migration step 4 added. ``TemporalFact`` validates each from its stored text.
_PROVENANCE = ("author_id", "scope")

# The index is created separately, after the project-column migration below runs -- for a
# pre-existing database the `facts` table exists without `project` at this point, and a
# CREATE INDEX referencing that column here would fail before the ALTER TABLE gets a chance to
# add it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT 'default',
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    superseded_by TEXT,
    last_confirmed TEXT,
    author_id TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'private'
);
"""

#: A key has at most one currently-valid fact. The index enforces it, so a write that would
#: leave two -- the race upsert_fact now holds the lock against -- fails instead of succeeding
#: silently. It replaces the plain index of the same shape, which only served lookups.
_ONE_CURRENT_FACT_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_one_current "
    "ON facts (user_id, project, subject, predicate) WHERE valid_to IS NULL"
)

#: Every currently-valid fact whose key has another one, oldest first within each key.
_DUPLICATED_CURRENT_FACTS = """
SELECT f.id, f.user_id, f.project, f.subject, f.predicate, f.valid_from
FROM facts f
JOIN (
    SELECT user_id, project, subject, predicate FROM facts
    WHERE valid_to IS NULL
    GROUP BY user_id, project, subject, predicate
    HAVING COUNT(*) > 1
) d ON f.user_id = d.user_id AND f.project = d.project
   AND f.subject = d.subject AND f.predicate = d.predicate
WHERE f.valid_to IS NULL
ORDER BY f.user_id, f.project, f.subject, f.predicate, f.valid_from, f.rowid
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class SqliteTemporalStore:
    def __init__(self, path: str = ":memory:", *, conn: sqlite3.Connection | None = None) -> None:
        """Build the store over *conn* (a shared connection, e.g. from ``open_db``) when given,
        so facts live in the same database file as every other store; otherwise opens its own
        connection at *path* (``:memory:`` default) for isolated/test use.
        """
        # check_same_thread=False so it can be used from the async server's threadpool.
        self._conn = conn if conn is not None else sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_project_column()
        self._enforce_one_current_fact_per_key()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(facts)")}
        if "project" not in cols:
            self._conn.execute(
                f"ALTER TABLE facts ADD COLUMN project TEXT NOT NULL DEFAULT '{PERSONAL_PROJECT}'"
            )
            # The old index doesn't cover `project`; drop it so the index created after this
            # migration covers the new column.
            self._conn.execute("DROP INDEX IF EXISTS idx_facts_current")
            self._conn.commit()

    def _enforce_one_current_fact_per_key(self) -> None:
        """Repair keys left with more than one current fact, then index so none can be again.

        Before upsert_fact took the write lock, two processes asserting the same key at once
        could both insert, leaving the key with two currently-valid facts. Each such key keeps
        its newest; every older one is closed at the moment the next one became valid and
        marked as superseded by it -- what a serial run would have written. Nothing is deleted.
        Idempotent, and done under the write lock so two processes opening at once repair once.
        """
        indexed = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_facts_one_current'"
        ).fetchone()
        if indexed is not None:
            return
        with write_transaction(self._conn):
            rows = self._conn.execute(_DUPLICATED_CURRENT_FACTS).fetchall()
            for older, newer in pairwise(rows):
                same_key = all(
                    older[c] == newer[c] for c in ("user_id", "project", "subject", "predicate")
                )
                if same_key:
                    self._conn.execute(
                        "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                        (newer["valid_from"], newer["id"], older["id"]),
                    )
            # The plain index of the same shape only served lookups; the unique one does both.
            self._conn.execute("DROP INDEX IF EXISTS idx_facts_current")
            self._conn.execute(_ONE_CURRENT_FACT_INDEX)

    def _row_to_fact(self, row: sqlite3.Row) -> TemporalFact:
        """Read whatever columns the row has. A database still waiting for migration step 4
        has no ``author_id`` or ``scope``, and recall reads current facts on it while it is
        read-only; each is taken from the row when present and left to the default when not.
        """
        # Membership in a Row tests its values, so the column names are taken out first.
        present = set(row.keys())
        provenance: dict[str, Any] = {c: row[c] for c in _PROVENANCE if c in present}
        return TemporalFact(
            id=row["id"],
            user_id=row["user_id"],
            project=row["project"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            source=MemorySource(row["source"]),
            confidence=row["confidence"],
            valid_from=_dt(row["valid_from"]),
            valid_to=_dt(row["valid_to"]),
            superseded_by=row["superseded_by"],
            last_confirmed=_dt(row["last_confirmed"]),
            **provenance,
        )

    async def upsert_fact(self, fact: TemporalFact, *, now: datetime) -> str:
        # The current facts are looked up inside the write transaction, which holds the lock
        # from its first statement. Two processes can share this database file -- two
        # `morgan consolidate` runs, say -- and a lookup made before the lock leaves a window
        # in which both find the same current fact, both insert, and both close it: the key is
        # left with two currently-valid facts.
        with write_transaction(self._conn):
            cur = self._conn.execute(
                "SELECT id FROM facts WHERE user_id=? AND project=? AND subject=? AND predicate=? "
                "AND valid_to IS NULL",
                (fact.user_id, fact.project, fact.subject, fact.predicate),
            )
            existing = [r["id"] for r in cur.fetchall()]
            fact = fact.model_copy(deep=True)
            if fact.valid_from is None:
                fact.valid_from = now
            fact.last_confirmed = now
            # Closed before the new fact is inserted: a key may hold one current fact at a time,
            # and the unique index checks that at each statement, not at commit.
            for old_id in existing:
                self._conn.execute(
                    "UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                    (_iso(now), fact.id, old_id),
                )
            self._conn.execute(
                "INSERT INTO facts (id, user_id, project, subject, predicate, object, source, "
                "confidence, valid_from, valid_to, superseded_by, last_confirmed, author_id, "
                "scope) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fact.id,
                    fact.user_id,
                    fact.project,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.source.value,
                    fact.confidence,
                    _iso(fact.valid_from),
                    _iso(fact.valid_to),
                    fact.superseded_by,
                    _iso(fact.last_confirmed),
                    fact.author_id,
                    fact.scope.value,
                ),
            )
        return fact.id

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = PERSONAL_PROJECT,
    ) -> list[TemporalFact]:
        sql = "SELECT * FROM facts WHERE user_id=? AND valid_to IS NULL"
        params: list[object] = [user_id]
        if project is not None:
            sql += " AND project=?"
            params.append(project)
        if subject is not None:
            sql += " AND subject=?"
            params.append(subject)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def history(self, *, user_id: str, subject: str, predicate: str) -> list[TemporalFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND predicate=? ORDER BY valid_from",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def close_fact(self, fact_id: str, *, user_id: str, project: str, now: datetime) -> None:
        """Close a fact's validity interval by setting ``valid_to = now``.

        This is the "soft delete" operation — the fact is retained in history
        with its interval closed, but will no longer appear in ``current_facts``.
        Scoped to *user_id* + *project*: a fact belonging to another user or another
        project is left untouched even if its id is known, so this is a no-op (not an
        error) both when *fact_id* doesn't exist and when it exists but is out of scope.
        """
        with write_transaction(self._conn):
            self._conn.execute(
                "UPDATE facts SET valid_to=? WHERE id=? AND user_id=? AND project=? "
                "AND valid_to IS NULL",
                (_iso(now), fact_id, user_id, project),
            )

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        """Overwrite the ``confidence`` for *fact_id* in-place, scoped to *user_id* + *project*.

        Used by the decay worker to persist decayed confidence scores.
        """
        with write_transaction(self._conn):
            self._conn.execute(
                "UPDATE facts SET confidence=? WHERE id=? AND user_id=? AND project=?",
                (value, fact_id, user_id, project),
            )
