"""``call_log``: one row per ``recall``, ``remember``, ``facts`` and ``ask`` call, read back by
time window and counted by command and degrade reason. Nothing here registers a project: a
call is not the owner's data.

Every statement runs with ``execute``, never ``executescript``, so a schema function can run
inside a caller's own write transaction without an implicit commit ending it partway through.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS call_log (
        id                INTEGER PRIMARY KEY,
        ts                TEXT NOT NULL,
        surface           TEXT NOT NULL,
        client            TEXT NOT NULL,
        native_session_id TEXT NOT NULL DEFAULT '',
        command           TEXT NOT NULL,
        user_id           TEXT NOT NULL,
        project           TEXT NOT NULL,
        all_projects      INTEGER NOT NULL DEFAULT 0,
        outcome           TEXT NOT NULL,
        embed_outcome     TEXT,
        degraded          TEXT,
        degrade_reason    TEXT,
        embed_latency_ms  REAL,
        total_ms          REAL NOT NULL,
        query_language    TEXT,
        reason            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_call_log_ts ON call_log (ts)",
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create ``call_log`` and its index, joining *conn*'s current transaction."""
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


class CallLogStore:
    """Creates ``call_log``; every query over it is a module-level function, like
    ``ProjectStore``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()


@dataclass(frozen=True)
class CallRecord:
    """One call as ``call_log`` holds it. ``client`` is the MCP client's name or the CLI's
    harness marker; ``native_session_id`` the harness session that made the call, or ``''``."""

    ts: str
    surface: str
    client: str
    native_session_id: str
    command: str
    user_id: str
    project: str
    all_projects: bool
    outcome: str
    embed_outcome: str | None
    degraded: str | None
    degrade_reason: str | None
    embed_latency_ms: float | None
    total_ms: float
    query_language: str | None
    reason: str | None
    id: int | None = None


_COLUMNS: tuple[str, ...] = (
    "ts",
    "surface",
    "client",
    "native_session_id",
    "command",
    "user_id",
    "project",
    "all_projects",
    "outcome",
    "embed_outcome",
    "degraded",
    "degrade_reason",
    "embed_latency_ms",
    "total_ms",
    "query_language",
    "reason",
)


def _row_to_record(row: sqlite3.Row) -> CallRecord:
    values = dict(row)
    values["all_projects"] = bool(values["all_projects"])
    return CallRecord(**values)


def insert_call(conn: sqlite3.Connection, record: CallRecord) -> int:
    """Insert *record* and return its rowid. Joins the caller's transaction."""
    values = {name: getattr(record, name) for name in _COLUMNS}
    values["all_projects"] = int(record.all_projects)
    cursor = conn.execute(
        "INSERT INTO call_log (ts, surface, client, native_session_id, command, user_id, "
        "project, all_projects, outcome, embed_outcome, degraded, degrade_reason, "
        "embed_latency_ms, total_ms, query_language, reason) VALUES (:ts, :surface, :client, "
        ":native_session_id, :command, :user_id, :project, :all_projects, :outcome, "
        ":embed_outcome, :degraded, :degrade_reason, :embed_latency_ms, :total_ms, "
        ":query_language, :reason)",
        values,
    )
    if cursor.lastrowid is None:
        raise RuntimeError("call_log insert returned no rowid")
    return int(cursor.lastrowid)


def calls_between(
    conn: sqlite3.Connection, *, user_id: str, since: str, until: str
) -> list[CallRecord]:
    """*user_id*'s calls with ``since <= ts < until``, oldest first."""
    rows = conn.execute(
        "SELECT * FROM call_log WHERE user_id = ? AND ts >= ? AND ts < ? ORDER BY ts, id",
        (user_id, since, until),
    )
    return [_row_to_record(r) for r in rows]


@dataclass(frozen=True)
class CallCounts:
    """Calls since a moment, by command: recalls, the degraded ones among them by reason,
    remembers, the ones among them that failed on the embedding, facts and asks."""

    recalls: int
    degraded: dict[str, int]
    remembers: int
    remember_failed: int
    facts: int
    asks: int


def call_counts_since(conn: sqlite3.Connection, *, user_id: str, since: str) -> CallCounts:
    by_command: dict[str, int] = {}
    for row in conn.execute(
        "SELECT command, COUNT(*) AS n FROM call_log WHERE user_id = ? AND ts >= ? "
        "GROUP BY command",
        (user_id, since),
    ):
        by_command[str(row["command"])] = int(row["n"])
    degraded = {
        str(row["degrade_reason"]): int(row["n"])
        for row in conn.execute(
            "SELECT degrade_reason, COUNT(*) AS n FROM call_log WHERE user_id = ? AND ts >= ? "
            "AND command = 'recall' AND degraded IS NOT NULL GROUP BY degrade_reason "
            "ORDER BY degrade_reason",
            (user_id, since),
        )
    }
    remember_failed = int(
        conn.execute(
            "SELECT COUNT(*) FROM call_log WHERE user_id = ? AND ts >= ? "
            "AND command = 'remember' AND outcome = 'failed'",
            (user_id, since),
        ).fetchone()[0]
    )
    return CallCounts(
        recalls=by_command.get("recall", 0),
        degraded=degraded,
        remembers=by_command.get("remember", 0),
        remember_failed=remember_failed,
        facts=by_command.get("facts", 0),
        asks=by_command.get("ask", 0),
    )
