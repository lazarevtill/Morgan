"""The archive's session tables: ``sessions``, ``turns``, ``turns_fts``, ``corrections_fts`` and
``turn_links``, their writers and the reads built on them.

No vectors: the archive is keyword-indexed only, and capture never embeds. ``turns_fts`` and
``corrections_fts`` are regular FTS5 tables whose rowid is ``turns.id``. Each is created with
FTS5's ``secure-delete`` option, so a deleted row's terms are overwritten rather than left in
the segment tree -- set by the ``CREATE`` that makes the table and by nothing else, because
setting it is a write, and an open of an existing database must take no write lock. Every
statement runs with ``execute``, never ``executescript``: a schema function can then run inside
a caller's own write transaction without an implicit commit ending it partway through.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from morgan_brain.memory.store.fts import to_match_query
from morgan_brain.models import CaptureTrigger, Session, Turn

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id                  TEXT PRIMARY KEY,
        user_id             TEXT NOT NULL,
        project             TEXT NOT NULL,
        harness             TEXT NOT NULL,
        native_id           TEXT NOT NULL,
        source_path         TEXT NOT NULL,
        cwd                 TEXT NOT NULL,
        cwd_changed         INTEGER NOT NULL DEFAULT 0,
        project_source      TEXT NOT NULL,
        entrypoint          TEXT NOT NULL DEFAULT '',
        interactive         INTEGER NOT NULL DEFAULT 1,
        harness_version     TEXT NOT NULL DEFAULT '',
        harness_mode        TEXT NOT NULL DEFAULT '',
        forked_from         TEXT,
        started_at          TEXT,
        ended_at            TEXT,
        erased_since        TEXT,
        hook_first_at       TEXT,
        sweep_first_at      TEXT,
        all_first_at        TEXT,
        turn_count          INTEGER NOT NULL DEFAULT 0,
        authored_turn_count INTEGER NOT NULL DEFAULT 0,
        gate_redactions     INTEGER NOT NULL DEFAULT 0,
        gate_flags          INTEGER NOT NULL DEFAULT 0,
        gate_provider_hits  INTEGER NOT NULL DEFAULT 0,
        paused_turns        INTEGER NOT NULL DEFAULT 0,
        reader_version      INTEGER NOT NULL,
        gate_version        INTEGER NOT NULL,
        imported_at         TEXT NOT NULL,
        updated_at          TEXT NOT NULL,
        UNIQUE (harness, native_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        id              INTEGER PRIMARY KEY,
        session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        user_id         TEXT NOT NULL,
        project         TEXT NOT NULL,
        ordinal         INTEGER NOT NULL,
        native_key      TEXT NOT NULL,
        native_uuid     TEXT,
        parent_uuid     TEXT,
        role            TEXT NOT NULL
                            CHECK (role IN ('user', 'assistant', 'tool_call', 'tool_result')),
        authored        INTEGER NOT NULL DEFAULT 1,
        injected_kind   TEXT NOT NULL DEFAULT '',
        tool_name       TEXT,
        tool_use_id     TEXT,
        morgan_call     TEXT,
        is_error        INTEGER NOT NULL DEFAULT 0,
        ts              TEXT,
        text            TEXT NOT NULL,
        truncated_chars INTEGER NOT NULL DEFAULT 0,
        redactions      TEXT NOT NULL DEFAULT '[]',
        flags           TEXT NOT NULL DEFAULT '[]',
        norm_hash       TEXT,
        is_correction   INTEGER NOT NULL DEFAULT 0,
        UNIQUE (session_id, native_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turns_project_ts ON turns (project, user_id, ts)",
    """
    CREATE INDEX IF NOT EXISTS idx_turns_native_uuid ON turns (native_uuid)
        WHERE native_uuid IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_turns_correction ON turns (is_correction, ts)
        WHERE is_correction = 1
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
        text,
        user_id    UNINDEXED,
        project    UNINDEXED,
        session_id UNINDEXED,
        role       UNINDEXED,
        authored   UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS corrections_fts USING fts5(
        norm,
        project UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turn_links (
        turn_id         INTEGER NOT NULL,
        earlier_turn_id INTEGER NOT NULL,
        user_id         TEXT NOT NULL,
        project         TEXT NOT NULL,
        earlier_project TEXT NOT NULL,
        similarity      REAL NOT NULL,
        lexicon_version INTEGER NOT NULL,
        computed_at     TEXT NOT NULL,
        PRIMARY KEY (turn_id, earlier_turn_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turn_links_earlier ON turn_links (earlier_turn_id)",
)

#: The FTS5 tables whose rowid is ``turns.id``, each created with ``secure-delete`` on.
FTS_TABLES: tuple[str, ...] = ("turns_fts", "corrections_fts")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the archive's tables and indexes, joining *conn*'s current transaction rather than
    committing one of its own. An FTS table's ``secure-delete`` option is set only when the
    ``CREATE`` here made the table: on an existing database this function writes nothing.
    """
    created = {table: not _table_exists(conn, table) for table in FTS_TABLES}
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)
    for table in FTS_TABLES:
        if created[table]:
            # `table` is one of FTS_TABLES, never caller input.
            conn.execute(f"INSERT INTO {table}({table}, rank) VALUES ('secure-delete', 1)")  # noqa: S608 # nosec B608


class SessionStore:
    """Creates the archive's tables. Every query over them is one of the module-level
    functions below, which take a plain connection rather than this store, like
    ``ProjectStore``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()


# --- rows --------------------------------------------------------------------------------------

_SESSION_COLUMNS: tuple[str, ...] = tuple(Session.model_fields)
_TURN_COLUMNS: tuple[str, ...] = tuple(name for name in Turn.model_fields if name != "id")
_TURN_BOOLS = ("authored", "is_error", "is_correction")


def _row_to_session(row: sqlite3.Row) -> Session:
    values = dict(row)
    values["cwd_changed"] = bool(values["cwd_changed"])
    values["interactive"] = bool(values["interactive"])
    return Session(**values)


def _row_to_turn(row: sqlite3.Row) -> Turn:
    values = dict(row)
    for name in _TURN_BOOLS:
        values[name] = bool(values[name])
    return Turn(**values)


# --- sessions ----------------------------------------------------------------------------------


def upsert_session(conn: sqlite3.Connection, session: Session, *, now: str) -> None:
    """Insert *session*, or update the fields a later capture may change.

    Mutable: ``ended_at``, ``cwd_changed``, ``interactive``, ``reader_version``,
    ``gate_version``, ``forked_from`` and ``updated_at`` take the new values; ``entrypoint``,
    ``harness_version`` and ``harness_mode`` take a new value only when it is not ``''``, so a
    capture that does not know them never blanks what an earlier one knew; ``started_at`` is set
    once, by the first capture that knows it. The project, the file and the counts are not
    touched here: ``add_session_counts`` and ``mark_trigger_first`` own the counts and the
    stamps, and ``imported_at`` is set only on insert.
    """
    values = session.model_dump()
    values["cwd_changed"] = int(session.cwd_changed)
    values["interactive"] = int(session.interactive)
    values["imported_at"] = session.imported_at or now
    values["updated_at"] = now
    columns = ", ".join(_SESSION_COLUMNS)
    marks = ", ".join(f":{name}" for name in _SESSION_COLUMNS)
    # The column names are the model's fields, never caller input.
    conn.execute(
        f"INSERT INTO sessions ({columns}) VALUES ({marks}) "  # noqa: S608 # nosec B608
        "ON CONFLICT (harness, native_id) DO UPDATE SET "
        "ended_at = excluded.ended_at, cwd_changed = excluded.cwd_changed, "
        "interactive = excluded.interactive, reader_version = excluded.reader_version, "
        "gate_version = excluded.gate_version, forked_from = excluded.forked_from, "
        "updated_at = excluded.updated_at, "
        "started_at = COALESCE(sessions.started_at, excluded.started_at), "
        "entrypoint = CASE WHEN excluded.entrypoint = '' THEN sessions.entrypoint "
        "ELSE excluded.entrypoint END, "
        "harness_version = CASE WHEN excluded.harness_version = '' "
        "THEN sessions.harness_version ELSE excluded.harness_version END, "
        "harness_mode = CASE WHEN excluded.harness_mode = '' THEN sessions.harness_mode "
        "ELSE excluded.harness_mode END",
        values,
    )


_TRIGGER_COLUMNS: dict[str, str] = {
    "hook": "hook_first_at",
    "sweep": "sweep_first_at",
    "all": "all_first_at",
}


def mark_trigger_first(
    conn: sqlite3.Connection, session_id: str, *, trigger: CaptureTrigger, now: str
) -> None:
    """Record that *trigger* first captured turns of *session_id* at *now*, once: a stamp
    already set is never overwritten."""
    column = _TRIGGER_COLUMNS[trigger]
    # `column` is one of the three literals above, never caller input.
    conn.execute(
        f"UPDATE sessions SET {column} = ? WHERE id = ? AND {column} IS NULL",  # noqa: S608 # nosec B608
        (now, session_id),
    )


@dataclass(frozen=True)
class SessionDelta:
    """What one batch adds to a session's counts; ``ended_at`` replaces the stored value when
    it is not ``None``."""

    turns: int
    authored: int
    redactions: int
    flags: int
    provider_hits: int
    paused_turns: int
    ended_at: str | None


def add_session_counts(
    conn: sqlite3.Connection, session_id: str, delta: SessionDelta, *, now: str
) -> None:
    conn.execute(
        "UPDATE sessions SET turn_count = turn_count + ?, "
        "authored_turn_count = authored_turn_count + ?, "
        "gate_redactions = gate_redactions + ?, gate_flags = gate_flags + ?, "
        "gate_provider_hits = gate_provider_hits + ?, paused_turns = paused_turns + ?, "
        "ended_at = COALESCE(?, ended_at), updated_at = ? WHERE id = ?",
        (
            delta.turns,
            delta.authored,
            delta.redactions,
            delta.flags,
            delta.provider_hits,
            delta.paused_turns,
            delta.ended_at,
            now,
            session_id,
        ),
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return None if row is None else _row_to_session(row)


def find_session(conn: sqlite3.Connection, *, harness: str, native_id: str) -> Session | None:
    row = conn.execute(
        "SELECT * FROM sessions WHERE harness = ? AND native_id = ?", (harness, native_id)
    ).fetchone()
    return None if row is None else _row_to_session(row)


def list_sessions(
    conn: sqlite3.Connection, *, user_id: str, project: str | None, since: str | None
) -> list[Session]:
    """*user_id*'s sessions, newest ``started_at`` first (a session without one last); every
    project when *project* is ``None``; with *since*, those started at or after it."""
    sql = "SELECT * FROM sessions WHERE user_id = ?"
    params: list[object] = [user_id]
    if project is not None:
        sql += " AND project = ?"
        params.append(project)
    if since is not None:
        sql += " AND started_at >= ?"
        params.append(since)
    sql += " ORDER BY started_at IS NULL, started_at DESC, id"
    return [_row_to_session(r) for r in conn.execute(sql, params)]


# --- turns -------------------------------------------------------------------------------------


def insert_turns(conn: sqlite3.Connection, turns: Sequence[Turn]) -> list[int]:
    """Insert *turns*, all of one session, and return the rowids inserted, in order.

    Each inserted turn gets the next ``ordinal`` after the session's current maximum.
    ``INSERT OR IGNORE`` on ``(session_id, native_key)`` makes a re-read of the same lines a
    no-op; a turn whose ``native_uuid`` is already stored under another session of the same
    harness -- a resumed Claude Code session copying its parent's records -- is skipped. Every
    inserted row is written to ``turns_fts`` at the same rowid, so the keyword index and the
    table are erased by one list of ids.
    """
    if not turns:
        return []
    session_id = turns[0].session_id
    if any(turn.session_id != session_id for turn in turns):
        raise ValueError("insert_turns takes the turns of one session")
    next_ordinal = int(
        conn.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    )
    columns = ", ".join(_TURN_COLUMNS)
    marks = ", ".join(f":{name}" for name in _TURN_COLUMNS)
    inserted: list[int] = []
    for turn in turns:
        if turn.native_uuid is not None and _stored_under_another_session(conn, turn):
            continue
        values = turn.model_dump(exclude={"id"})
        values["ordinal"] = next_ordinal
        for name in _TURN_BOOLS:
            values[name] = int(values[name])
        # The column names are the model's fields, never caller input.
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO turns ({columns}) VALUES ({marks})",  # noqa: S608 # nosec B608
            values,
        )
        if cursor.rowcount != 1 or cursor.lastrowid is None:
            continue
        rowid = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO turns_fts (rowid, text, user_id, project, session_id, role, authored) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                rowid,
                turn.text,
                turn.user_id,
                turn.project,
                turn.session_id,
                turn.role,
                int(turn.authored),
            ),
        )
        inserted.append(rowid)
        next_ordinal += 1
    return inserted


def _stored_under_another_session(conn: sqlite3.Connection, turn: Turn) -> bool:
    """Whether a turn with this ``native_uuid`` is already stored under another session of the
    same harness, through ``idx_turns_native_uuid``."""
    row = conn.execute(
        "SELECT 1 FROM turns t JOIN sessions s ON s.id = t.session_id "
        "WHERE t.native_uuid = ? AND t.session_id != ? "
        "AND s.harness = (SELECT harness FROM sessions WHERE id = ?)",
        (turn.native_uuid, turn.session_id, turn.session_id),
    ).fetchone()
    return row is not None


def turns_of(conn: sqlite3.Connection, session_id: str, *, since: str | None = None) -> list[Turn]:
    """The turns of *session_id* by ``ordinal``; with *since*, those whose ``ts >= since``."""
    sql = "SELECT * FROM turns WHERE session_id = ?"
    params: list[object] = [session_id]
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    sql += " ORDER BY ordinal"
    return [_row_to_turn(r) for r in conn.execute(sql, params)]


@dataclass(frozen=True)
class SearchHit:
    session_id: str
    harness: str
    native_id: str
    ordinal: int
    role: str
    ts: str | None
    snippet: str


def search_turns(
    conn: sqlite3.Connection, *, user_id: str, project: str | None, query: str, limit: int
) -> list[SearchHit]:
    """FTS5 over ``turns_fts``: *query* as ``to_match_query`` makes it, ranked by FTS5's
    ``rank``, at most *limit* hits, each with a twelve-token snippet. Every project when
    *project* is ``None``; a query with no token finds nothing."""
    match = to_match_query(query)
    if not match:
        return []
    sql = (
        "SELECT t.session_id, s.harness, s.native_id, t.ordinal, t.role, t.ts, "
        "snippet(turns_fts, 0, '', '', '…', 12) AS snippet "
        "FROM turns_fts JOIN turns t ON t.id = turns_fts.rowid "
        "JOIN sessions s ON s.id = t.session_id "
        "WHERE turns_fts MATCH ? AND t.user_id = ?"
    )
    params: list[object] = [match, user_id]
    if project is not None:
        sql += " AND t.project = ?"
        params.append(project)
    sql += " ORDER BY turns_fts.rank LIMIT ?"
    params.append(limit)
    return [
        SearchHit(
            session_id=r["session_id"],
            harness=r["harness"],
            native_id=r["native_id"],
            ordinal=r["ordinal"],
            role=r["role"],
            ts=r["ts"],
            snippet=r["snippet"],
        )
        for r in conn.execute(sql, params)
    ]
