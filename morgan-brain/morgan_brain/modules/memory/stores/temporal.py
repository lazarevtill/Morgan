"""Bi-temporal fact store (SQLite). A fact is currently valid when valid_to IS NULL. Asserting a
new value for the same (user, subject, predicate) closes the old interval (sets valid_to = now,
superseded_by = new id) instead of deleting it — so history stays queryable and recall is never
confidently stale."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from morgan_brain.models.memory import MemorySource, TemporalFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts (user_id, subject, predicate) WHERE valid_to IS NULL;
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

    def _row_to_fact(self, row: sqlite3.Row) -> TemporalFact:
        return TemporalFact(
            id=row["id"], user_id=row["user_id"], subject=row["subject"],
            predicate=row["predicate"], object=row["object"],
            source=MemorySource(row["source"]), confidence=row["confidence"],
            valid_from=_dt(row["valid_from"]), valid_to=_dt(row["valid_to"]),
            superseded_by=row["superseded_by"], last_confirmed=_dt(row["last_confirmed"]),
        )

    async def upsert_fact(self, fact: TemporalFact, *, now: datetime) -> str:
        cur = self._conn.execute(
            "SELECT id FROM facts WHERE user_id=? AND subject=? AND predicate=? AND valid_to IS NULL",
            (fact.user_id, fact.subject, fact.predicate),
        )
        existing = [r["id"] for r in cur.fetchall()]
        if fact.valid_from is None:
            fact.valid_from = now
        fact.last_confirmed = now
        self._conn.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (fact.id, fact.user_id, fact.subject, fact.predicate, fact.object,
             fact.source.value, fact.confidence, _iso(fact.valid_from), _iso(fact.valid_to),
             fact.superseded_by, _iso(fact.last_confirmed)),
        )
        for old_id in existing:
            self._conn.execute(
                "UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                (_iso(now), fact.id, old_id),
            )
        self._conn.commit()
        return fact.id

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        sql = "SELECT * FROM facts WHERE user_id=? AND valid_to IS NULL"
        params: list[object] = [user_id]
        if subject is not None:
            sql += " AND subject=?"
            params.append(subject)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def history(
        self, *, user_id: str, subject: str, predicate: str
    ) -> list[TemporalFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND predicate=? ORDER BY valid_from",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]
