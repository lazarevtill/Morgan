"""Valid-time fact store (SQLite). A fact is currently valid when valid_to IS NULL. Asserting a
new value for the same (user, subject, predicate) closes the old interval (sets valid_to = now,
superseded_by = new id) instead of deleting it — so history stays queryable and recall is never
confidently stale."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from morgan_brain.models.memory import DEFAULT_PROJECT, MemorySource, TemporalFact

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
    last_confirmed TEXT
);
"""

_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts (user_id, project, subject, predicate) WHERE valid_to IS NULL;
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class SqliteTemporalStore:
    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False so it can be used from the async server's threadpool.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_project_column()
        self._conn.executescript(_INDEX_SCHEMA)
        self._conn.commit()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(facts)")}
        if "project" not in cols:
            self._conn.execute(
                f"ALTER TABLE facts ADD COLUMN project TEXT NOT NULL DEFAULT '{DEFAULT_PROJECT}'"
            )
            # The old index doesn't cover `project`; drop it so the index script below (run
            # after this migration) recreates it with the new column.
            self._conn.execute("DROP INDEX IF EXISTS idx_facts_current")
            self._conn.commit()

    def _row_to_fact(self, row: sqlite3.Row) -> TemporalFact:
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
        )

    async def upsert_fact(self, fact: TemporalFact, *, now: datetime) -> str:
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
        self._conn.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
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
            ),
        )
        for old_id in existing:
            self._conn.execute(
                "UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                (_iso(now), fact.id, old_id),
            )
        self._conn.commit()
        return fact.id

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = DEFAULT_PROJECT,
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
        self._conn.execute(
            "UPDATE facts SET valid_to=? WHERE id=? AND user_id=? AND project=? "
            "AND valid_to IS NULL",
            (_iso(now), fact_id, user_id, project),
        )
        self._conn.commit()

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        """Overwrite the ``confidence`` for *fact_id* in-place, scoped to *user_id* + *project*.

        Used by the decay worker to persist decayed confidence scores.
        """
        self._conn.execute(
            "UPDATE facts SET confidence=? WHERE id=? AND user_id=? AND project=?",
            (value, fact_id, user_id, project),
        )
        self._conn.commit()
