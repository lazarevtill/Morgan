"""Valid-time fact store (SQLite). A fact is currently valid when valid_to IS NULL. Asserting a
new value for the same (user, subject, predicate) closes the old interval (sets valid_to = now,
superseded_by = new id) instead of deleting it — so history stays queryable and recall is never
confidently stale."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from morgan_brain.memory.errors import SourceProtectionError
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.tables import Erasure
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
    -- pre-phase-0 default; kept for fresh/migrated parity, no writer relies on it
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
    scope TEXT NOT NULL DEFAULT 'private',
    recorded_at TEXT,
    support_event_ids TEXT NOT NULL DEFAULT '[]'
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


def _instant(value: datetime) -> datetime:
    """Compare offsets by instant, interpreting legacy naive timestamps as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class SqliteTemporalStore:
    def __init__(
        self,
        path: str = ":memory:",
        *,
        conn: sqlite3.Connection | None = None,
        initialize: bool = True,
    ) -> None:
        """Build the store over *conn* (a shared connection, e.g. from ``open_db``) when given,
        so facts live in the same database file as every other store; otherwise opens its own
        connection at *path* (``:memory:`` default) for isolated/test use.
        """
        # check_same_thread=False so it can be used from the async server's threadpool.
        self._conn = conn if conn is not None else sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if not initialize:
            return
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
        if "recorded_at" in present:
            provenance["recorded_at"] = _dt(row["recorded_at"])
        if "support_event_ids" in present:
            provenance["support_event_ids"] = json.loads(row["support_event_ids"])
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
                "SELECT id, valid_from, valid_to, last_confirmed, source FROM facts "
                "WHERE user_id=? AND project=? "
                "AND subject=? AND predicate=? "
                "AND valid_to IS NULL",
                (fact.user_id, fact.project, fact.subject, fact.predicate),
            )
            existing_rows = cur.fetchall()
            if not existing_rows and (
                fact.valid_from is None or _instant(fact.valid_from) >= _instant(now)
            ):
                # Cancelling a scheduled head leaves its predecessor with a finite
                # future end. A new present/future assertion closes that effective row too,
                # without treating already-ended history as a current head.
                bounded_rows = self._conn.execute(
                    "SELECT id, valid_from, valid_to, last_confirmed, source FROM facts "
                    "WHERE user_id=? AND project=? AND subject=? AND predicate=? "
                    "AND valid_to IS NOT NULL",
                    (fact.user_id, fact.project, fact.subject, fact.predicate),
                ).fetchall()
                instant = _instant(now)
                existing_rows = [
                    row
                    for row in bounded_rows
                    if (end := _dt(row["valid_to"])) is not None
                    and instant < _instant(end)
                    and (
                        (start := _dt(row["valid_from"])) is None
                        or _instant(start) <= instant
                        # An implicit writer may have captured its clock before this
                        # predecessor committed. The scheduled-start guard below
                        # rejects real future schedules; ordinary starts are clamped.
                        # Empty cancelled intervals must never become predecessors.
                        or (fact.valid_from is None and _instant(start) < _instant(end))
                    )
                ]
            if fact.source in (MemorySource.UNKNOWN, MemorySource.AGENT_INFERRED):
                # Even an empty incoming interval would close these predecessors.
                if any(row["source"] == MemorySource.USER_STATED.value for row in existing_rows):
                    raise SourceProtectionError(
                        "Unattributed or inferred fact cannot replace a user statement"
                    )
                # Protection is independent of the closure candidates: a backdated
                # assertion skips the finite-predecessor fallback above, but must
                # still not overlap a user's historical or future interval.
                user_rows = self._conn.execute(
                    "SELECT valid_from, valid_to FROM facts WHERE user_id=? AND project=? "
                    "AND subject=? AND predicate=? AND source=?",
                    (
                        fact.user_id,
                        fact.project,
                        fact.subject,
                        fact.predicate,
                        MemorySource.USER_STATED.value,
                    ),
                ).fetchall()
                incoming_start = _instant(fact.valid_from or now)
                incoming_end = _instant(fact.valid_to) if fact.valid_to is not None else None
                for row in user_rows:
                    start_value, end_value = _dt(row["valid_from"]), _dt(row["valid_to"])
                    user_start = _instant(start_value) if start_value is not None else None
                    user_end = _instant(end_value) if end_value is not None else None
                    # Structural user heads are also protected because legacy
                    # historical closure semantics would mutate them regardless
                    # of overlap. Empty cancelled intervals overlap nothing.
                    overlap = (
                        (incoming_end is None or incoming_start < incoming_end)
                        and (user_end is None or user_start is None or user_start < user_end)
                        and (user_end is None or incoming_start < user_end)
                        and (
                            incoming_end is None or user_start is None or user_start < incoming_end
                        )
                    )
                    if user_end is None or overlap:
                        raise SourceProtectionError(
                            "Unattributed or inferred fact cannot replace a user statement"
                        )
            fact = fact.model_copy(deep=True)
            implicit_start = fact.valid_from is None
            if fact.valid_from is None:
                fact.valid_from = now
            if any(
                (start := _dt(row["valid_from"])) is not None
                and _instant(start) > _instant(now)
                and _instant(start) > _instant(_dt(row["last_confirmed"]) or now)
                and _instant(start) > _instant(fact.valid_from)
                for row in existing_rows
            ):
                raise ValueError("Fact cannot precede the existing scheduled timeline head")
            if implicit_start:
                # Other writers can capture their clocks before acquiring the lock.
                # Serialize ordinary assertions monotonically, rather than mistaking
                # a slightly later committed start for an intentional future schedule.
                for row in existing_rows:
                    start = _dt(row["valid_from"])
                    if start is not None and _instant(start) > _instant(fact.valid_from):
                        fact.valid_from = start
            fact.last_confirmed = fact.valid_from if implicit_start else now
            # A scheduled assertion must not close today's predecessor before its
            # effective start. Historical writes retain their existing closure semantics.
            supersedes_at = fact.valid_from if _instant(fact.valid_from) > _instant(now) else now
            # Closed before the new fact is inserted: a key may hold one current fact at a time,
            # and the unique index checks that at each statement, not at commit.
            for row in existing_rows:
                old_end = _dt(row["valid_to"])
                closes_at = (
                    old_end
                    if old_end is not None and _instant(old_end) < _instant(supersedes_at)
                    else supersedes_at
                )
                self._conn.execute(
                    "UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                    (_iso(closes_at), fact.id, row["id"]),
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
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")}
            if "recorded_at" in columns:
                self._conn.execute(
                    "UPDATE facts SET recorded_at=?, support_event_ids=? WHERE id=?",
                    (_iso(now), json.dumps(fact.support_event_ids), fact.id),
                )
        return fact.id

    async def get_fact(self, fact_id: str, *, user_id: str, project: str) -> TemporalFact | None:
        """Resolve a durable fact identity only within its named owner and applicability."""
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id=? AND user_id=? AND project=?",
            (fact_id, user_id, project),
        ).fetchone()
        return self._row_to_fact(row) if row is not None else None

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = PERSONAL_PROJECT,
        at: datetime | None = None,
    ) -> list[TemporalFact]:
        """Return effective facts at *at*, or structural unclosed heads when omitted.

        Validity is half-open: ``[valid_from, valid_to)``. Present-time retrieval
        supplies its injected clock; version checks retain the default head semantics.
        """
        sql = "SELECT * FROM facts WHERE user_id=?"
        if at is None:
            sql += " AND valid_to IS NULL"
        params: list[object] = [user_id]
        if project is not None:
            sql += " AND project=?"
            params.append(project)
        if subject is not None:
            sql += " AND subject=?"
            params.append(subject)
        if at is not None:
            # SQLite narrows candidates before Python objects are built. Its date
            # parser has lower precision and accepts fewer ISO forms than Python,
            # so retain a one-second boundary margin and every unparsed bound.
            # The exact Python interval check below remains authoritative.
            sql += (
                " AND (julianday(valid_from) IS NULL OR "
                "julianday(valid_from) <= julianday(?) + 1.0 / 86400.0)"
                " AND (julianday(valid_to) IS NULL OR "
                "julianday(valid_to) >= julianday(?) - 1.0 / 86400.0)"
            )
            clock = _instant(at).isoformat()
            params.extend((clock, clock))
        rows = self._conn.execute(sql, params).fetchall()
        facts = [self._row_to_fact(r) for r in rows]
        if at is None:
            return facts
        instant = _instant(at)
        return [
            fact
            for fact in facts
            if (fact.valid_from is None or _instant(fact.valid_from) <= instant)
            and (fact.valid_to is None or instant < _instant(fact.valid_to))
        ]

    async def history(self, *, user_id: str, subject: str, predicate: str) -> list[TemporalFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND predicate=? ORDER BY valid_from",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def close_fact(self, fact_id: str, *, user_id: str, project: str, now: datetime) -> None:
        """Shorten a fact's interval at *now*, retaining its history and successor.

        A predecessor may already have a scheduled future end, yet still be effective
        now. Ended history is never extended. Cancelling a future fact closes it at its
        start, giving an empty interval rather than a negative one; this does not reopen
        its predecessor. Missing or out-of-scope identities are no-ops.
        """
        with write_transaction(self._conn):
            row = self._conn.execute(
                "SELECT valid_from, valid_to FROM facts WHERE id=? AND user_id=? AND project=?",
                (fact_id, user_id, project),
            ).fetchone()
            if row is None:
                return
            start = _dt(row["valid_from"])
            end = _dt(row["valid_to"])
            closes_at = start if start is not None and _instant(start) > _instant(now) else now
            if end is not None and _instant(end) <= _instant(closes_at):
                return
            self._conn.execute(
                "UPDATE facts SET valid_to=? WHERE id=? AND user_id=? AND project=?",
                (_iso(closes_at), fact_id, user_id, project),
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


def delete_facts(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """`forget()`'s deleter for ``facts``: every fact of the erased project, current and
    closed alike."""
    return conn.execute(
        "DELETE FROM facts WHERE user_id = ? AND project = ?",
        (erasure.user_id, erasure.project),
    ).rowcount
