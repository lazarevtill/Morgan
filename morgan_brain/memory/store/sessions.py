"""The archive's session tables: ``sessions``, ``turns``, ``turns_fts``, ``corrections_fts`` and
``turn_links``, and the tables capture keeps beside them -- ``capture_cursors`` with its lease,
``capture_exclusions``, ``capture_state`` and ``capture_pauses`` -- their writers and the reads
built on them.

No vectors: the archive is keyword-indexed only, and capture never embeds. ``turns_fts`` and
``corrections_fts`` are regular FTS5 tables whose rowid is ``turns.id``. Each is created with
FTS5's ``secure-delete`` option, so a deleted row's terms are overwritten rather than left in
the segment tree -- set by the ``CREATE`` that makes the table and by nothing else, because
setting it is a write, and an open of an existing database must take no write lock. Every
statement runs with ``execute``, never ``executescript``: a schema function can then run inside
a caller's own write transaction without an implicit commit ending it partway through.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.fts import to_match_query
from morgan_brain.memory.store.tables import Erasure
from morgan_brain.models import (
    CaptureCursor,
    CaptureTrigger,
    Exclusion,
    Pause,
    Session,
    Turn,
    parse_iso,
)

#: Of this module's registered tables, only ``capture_pauses`` holds nothing per session: a
#: pause interval outlives every session, so only a project-grain forget removes it, and the
#: session grain passes it over.
HOLDS_NOTHING_PER_SESSION: tuple[str, ...] = ("capture_pauses",)

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
    """
    CREATE TABLE IF NOT EXISTS capture_cursors (
        harness      TEXT NOT NULL,
        native_id    TEXT NOT NULL,
        source_path  TEXT NOT NULL,
        byte_offset  INTEGER NOT NULL,
        size         INTEGER NOT NULL,
        mtime_ns     INTEGER NOT NULL,
        identity     TEXT NOT NULL,
        status       TEXT NOT NULL,
        last_read_at TEXT NOT NULL,
        lease_owner  TEXT,
        lease_until  TEXT,
        open_calls   TEXT NOT NULL DEFAULT '[]',
        PRIMARY KEY (harness, native_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS capture_exclusions (
        harness     TEXT NOT NULL,
        native_id   TEXT NOT NULL,
        reason      TEXT NOT NULL,
        excluded_at TEXT NOT NULL,
        PRIMARY KEY (harness, native_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS capture_pauses (
        id           INTEGER PRIMARY KEY,
        user_id      TEXT NOT NULL,
        project      TEXT NOT NULL,
        paused_from  TEXT NOT NULL,
        paused_until TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_capture_pauses_project
        ON capture_pauses (user_id, project, paused_from)
    """,
    """
    CREATE TABLE IF NOT EXISTS capture_state (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)

#: The FTS5 tables whose rowid is ``turns.id``, each created with ``secure-delete`` on.
FTS_TABLES: tuple[str, ...] = ("turns_fts", "corrections_fts")

_OBJECT_NAME_RE = re.compile(r"CREATE (?:VIRTUAL TABLE|TABLE|INDEX) IF NOT EXISTS (\w+)")


def _schema_object_names() -> frozenset[str]:
    """Every table and index name ``_SCHEMA_STATEMENTS`` creates, read from the statements
    themselves so a name appended there is checked here too, with no second list kept by hand
    beside it and liable to fall out of step."""
    names: set[str] = set()
    for statement in _SCHEMA_STATEMENTS:
        match = _OBJECT_NAME_RE.search(statement)
        if match is None:
            raise ValueError(f"no table or index name found in schema statement: {statement!r}")
        names.add(match.group(1))
    return frozenset(names)


#: Every table and index this module owns. Used only to decide, with no lock, whether an open
#: has anything left to build.
_SCHEMA_OBJECT_NAMES: frozenset[str] = _schema_object_names()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _schema_object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'index') AND name = ?", (name,)
    ).fetchone()
    return row is not None


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the archive's tables and indexes when any of them is missing, and set each newly
    made FTS table's ``secure-delete`` option.

    The ``CREATE`` statements and the option inserts run under one ``write_transaction``, so a
    lock timeout or an interruption between them commits nothing rather than leaving a table
    without the option for good. Called by ``SessionStore`` at open and by migration step 8, so
    both leave the same DDL: called from inside a caller's own write transaction, the work
    joins it as a savepoint instead of starting a new one, and step 8's DDL then rolls back
    with the rest of its wave when a later step fails. An open that finds every table and
    index already in place takes no write lock and writes nothing.
    """
    if all(_schema_object_exists(conn, name) for name in _SCHEMA_OBJECT_NAMES):
        return
    with write_transaction(conn):
        # Re-read under the lock: another connection may have finished this since the
        # lock-free check above.
        created = {table: not _table_exists(conn, table) for table in FTS_TABLES}
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        for table in FTS_TABLES:
            if created[table]:
                # `table` is one of FTS_TABLES, never caller input.
                conn.execute(f"INSERT INTO {table}({table}, rank) VALUES ('secure-delete', 1)")  # noqa: S608 # nosec B608


class SessionStore:
    """Creates the archive's tables when any is missing. Every query over them is one of the
    module-level functions below, which take a plain connection rather than this store, like
    ``ProjectStore``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)


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

    ``interactive``, ``reader_version``, ``gate_version`` and ``updated_at`` take the new value
    outright. ``entrypoint``, ``harness_version`` and ``harness_mode`` take a new value only
    when it is not ``''``, and ``forked_from`` only when it is not ``NULL``, so a capture that
    does not know one of them never blanks what an earlier one knew. ``cwd_changed`` is true
    once it has ever been true: a later capture that does not know the cwd changed cannot undo
    an earlier one that saw it. ``ended_at`` keeps the later of the two known timestamps, so a
    capture built from an older slice of the transcript cannot roll the session's end back.
    ``started_at`` is set once, by the first capture that knows it. The project, the file and
    the counts are not touched here: ``add_session_counts`` and ``mark_trigger_first`` own the
    counts and the stamps, and ``imported_at`` is set only on insert.
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
        "interactive = excluded.interactive, reader_version = excluded.reader_version, "
        "gate_version = excluded.gate_version, updated_at = excluded.updated_at, "
        "ended_at = COALESCE("
        "MAX(sessions.ended_at, excluded.ended_at), sessions.ended_at, excluded.ended_at"
        "), "
        "cwd_changed = MAX(sessions.cwd_changed, excluded.cwd_changed), "
        "forked_from = COALESCE(excluded.forked_from, sessions.forked_from), "
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
    """What one batch adds to a session's counts; ``ended_at`` keeps the later of the two known
    timestamps, as ``upsert_session`` does for the same column, rather than whatever value a
    batch happens to supply."""

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
        "UPDATE sessions SET turn_count = turn_count + :turns, "
        "authored_turn_count = authored_turn_count + :authored, "
        "gate_redactions = gate_redactions + :redactions, "
        "gate_flags = gate_flags + :flags, "
        "gate_provider_hits = gate_provider_hits + :provider_hits, "
        "paused_turns = paused_turns + :paused_turns, "
        "ended_at = COALESCE(MAX(ended_at, :ended_at), ended_at, :ended_at), "
        "updated_at = :now WHERE id = :session_id",
        {
            "turns": delta.turns,
            "authored": delta.authored,
            "redactions": delta.redactions,
            "flags": delta.flags,
            "provider_hits": delta.provider_hits,
            "paused_turns": delta.paused_turns,
            "ended_at": delta.ended_at,
            "now": now,
            "session_id": session_id,
        },
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


# --- capture cursors and the lease ---------------------------------------------------------------

_CURSOR_COLUMNS: tuple[str, ...] = tuple(CaptureCursor.model_fields)


def _row_to_cursor(row: sqlite3.Row) -> CaptureCursor:
    values = dict(row)
    values["open_calls"] = json.loads(values["open_calls"])
    return CaptureCursor(**values)


def cursor_get(conn: sqlite3.Connection, *, harness: str, native_id: str) -> CaptureCursor | None:
    row = conn.execute(
        "SELECT * FROM capture_cursors WHERE harness = ? AND native_id = ?", (harness, native_id)
    ).fetchone()
    return None if row is None else _row_to_cursor(row)


def cursors(conn: sqlite3.Connection) -> dict[tuple[str, str], CaptureCursor]:
    """Every cursor, keyed by ``(harness, native_id)``."""
    return {
        (r["harness"], r["native_id"]): _row_to_cursor(r)
        for r in conn.execute("SELECT * FROM capture_cursors ORDER BY harness, native_id")
    }


def cursor_put(conn: sqlite3.Connection, cursor: CaptureCursor) -> None:
    """Insert or replace the cursor's own columns, ``open_calls`` included. The two lease
    columns are the lease's (``lease_claim``/``lease_release``): set from the model on insert,
    left alone on update, so a batch that writes its cursor never drops or forges a lease."""
    values = cursor.model_dump()
    values["open_calls"] = json.dumps(values["open_calls"])
    columns = ", ".join(_CURSOR_COLUMNS)
    marks = ", ".join(f":{name}" for name in _CURSOR_COLUMNS)
    # The column names are the model's fields, never caller input.
    conn.execute(
        f"INSERT INTO capture_cursors ({columns}) VALUES ({marks}) "  # noqa: S608 # nosec B608
        "ON CONFLICT (harness, native_id) DO UPDATE SET source_path = excluded.source_path, "
        "byte_offset = excluded.byte_offset, size = excluded.size, mtime_ns = excluded.mtime_ns, "
        "identity = excluded.identity, status = excluded.status, "
        "last_read_at = excluded.last_read_at, open_calls = excluded.open_calls",
        values,
    )


def lease_claim(
    conn: sqlite3.Connection,
    *,
    harness: str,
    native_id: str,
    source_path: str,
    owner: str,
    now: str,
    until: str,
) -> bool:
    """Take the lease on one transcript for *owner* until *until*: the cursor row is inserted
    when missing (``status "new"``), and the lease is taken when nobody holds it, *owner*
    already holds it, or the holder's lease lapsed before *now*. ``False`` when another owner
    holds it into the future. *now* and *until* are ``utc_iso`` strings, so the comparison is
    lexical. Joins the caller's write transaction rather than opening one of its own -- the
    caller owns the one short write transaction per file."""
    conn.execute(
        "INSERT OR IGNORE INTO capture_cursors (harness, native_id, source_path, byte_offset, "
        "size, mtime_ns, identity, status, last_read_at) "
        "VALUES (?, ?, ?, 0, 0, 0, '', 'new', ?)",
        (harness, native_id, source_path, now),
    )
    taken = conn.execute(
        "UPDATE capture_cursors SET lease_owner = ?, lease_until = ? "
        "WHERE harness = ? AND native_id = ? AND (lease_owner IS NULL OR lease_owner = ? "
        "OR lease_until IS NULL OR lease_until < ?)",
        (owner, until, harness, native_id, owner, now),
    )
    return taken.rowcount == 1


def lease_release(conn: sqlite3.Connection, *, harness: str, native_id: str, owner: str) -> None:
    """Drop the lease *owner* holds; another owner's lease is left alone."""
    conn.execute(
        "UPDATE capture_cursors SET lease_owner = NULL, lease_until = NULL "
        "WHERE harness = ? AND native_id = ? AND lease_owner = ?",
        (harness, native_id, owner),
    )


# --- exclusions ------------------------------------------------------------------------------


def exclusion_add(
    conn: sqlite3.Connection, *, harness: str, native_id: str, reason: str, now: str
) -> None:
    conn.execute(
        "INSERT INTO capture_exclusions (harness, native_id, reason, excluded_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (harness, native_id) DO UPDATE SET "
        "reason = excluded.reason, excluded_at = excluded.excluded_at",
        (harness, native_id, reason, now),
    )


def exclusion_remove(conn: sqlite3.Connection, *, harness: str, native_id: str) -> bool:
    removed = conn.execute(
        "DELETE FROM capture_exclusions WHERE harness = ? AND native_id = ?", (harness, native_id)
    )
    return removed.rowcount == 1


def exclusions(conn: sqlite3.Connection) -> list[Exclusion]:
    return [
        Exclusion(**dict(r))
        for r in conn.execute("SELECT * FROM capture_exclusions ORDER BY harness, native_id")
    ]


def is_excluded(conn: sqlite3.Connection, *, harness: str, native_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM capture_exclusions WHERE harness = ? AND native_id = ?",
        (harness, native_id),
    ).fetchone()
    return row is not None


# --- state -------------------------------------------------------------------------------------

STATE_ENABLED = "enabled"
STATE_ENABLED_AT = "enabled_at"
STATE_REPORT_PATH = "report_path"
STATE_LAST_SWEEP = "last_sweep"


def state_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM capture_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def state_set(conn: sqlite3.Connection, key: str, value: str, *, now: str) -> None:
    conn.execute(
        "INSERT INTO capture_state (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value, now),
    )


def capture_enabled(conn: sqlite3.Connection) -> bool:
    """Off until a capture is explicitly turned on."""
    return state_get(conn, STATE_ENABLED) == "1"


# --- pauses --------------------------------------------------------------------------------------


def _row_to_pause(row: sqlite3.Row) -> Pause:
    return Pause(**dict(row))


def pause_open(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    project: str,
    paused_from: str,
    paused_until: str | None,
) -> Pause:
    """Record a pause interval; ``None`` for an open-ended one. The row goes only with the
    project."""
    cursor = conn.execute(
        "INSERT INTO capture_pauses (user_id, project, paused_from, paused_until) "
        "VALUES (?, ?, ?, ?)",
        (user_id, project, paused_from, paused_until),
    )
    return Pause(
        id=cursor.lastrowid,
        user_id=user_id,
        project=project,
        paused_from=paused_from,
        paused_until=paused_until,
    )


def pause_close(conn: sqlite3.Connection, *, user_id: str, project: str, now: str) -> int:
    """End every *current* interval of the project at *now* -- one whose end is NULL or after
    *now* -- and return how many, so ending a pause early cuts a longer running one short too.
    A past interval is history and is left as it is."""
    closed = conn.execute(
        "UPDATE capture_pauses SET paused_until = ? WHERE user_id = ? AND project = ? "
        "AND (paused_until IS NULL OR paused_until > ?)",
        (now, user_id, project, now),
    )
    return closed.rowcount


def pauses_of(conn: sqlite3.Connection, *, user_id: str, project: str) -> list[Pause]:
    return [
        _row_to_pause(r)
        for r in conn.execute(
            "SELECT * FROM capture_pauses WHERE user_id = ? AND project = ? "
            "ORDER BY paused_from, id",
            (user_id, project),
        )
    ]


def current_pause(
    conn: sqlite3.Connection, *, user_id: str, project: str, now: str
) -> Pause | None:
    """The interval whose end is NULL or after *now*, the latest ``paused_from`` when several;
    ``None`` when the project is not paused."""
    row = conn.execute(
        "SELECT * FROM capture_pauses WHERE user_id = ? AND project = ? "
        "AND (paused_until IS NULL OR paused_until > ?) ORDER BY paused_from DESC, id DESC "
        "LIMIT 1",
        (user_id, project, now),
    ).fetchone()
    return None if row is None else _row_to_pause(row)


def in_pause(ts: str, pauses: Sequence[Pause]) -> bool:
    """Whether the moment *ts* lies inside any interval: ``paused_from <= ts`` and the end NULL
    or ``ts < paused_until``. Pure, and compares parsed moments, so a harness's own timestamp
    shape compares with ``utc_iso``'s by time."""
    moment = parse_iso(ts)
    for pause in pauses:
        if parse_iso(pause.paused_from) > moment:
            continue
        if pause.paused_until is None or moment < parse_iso(pause.paused_until):
            return True
    return False


# --- the deleters, at both grains -----------------------------------------------------------


def _session_scope(erasure: Erasure) -> tuple[str, tuple[object, ...]]:
    """The ``WHERE`` that picks the erasure's sessions: the listed ids at the session grain, the
    owner's whole project at the project grain."""
    if erasure.grain == "sessions":
        return "id IN (SELECT value FROM json_each(?))", (erasure.session_ids,)
    return "user_id = ? AND project = ?", (erasure.user_id, erasure.project)


def delete_sessions(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """The erased sessions' rows, their cursors, and one exclusion each at ``erasure.erased_at``
    with ``erasure.exclusion_reason``: the transcript is still on disk, and a cursor alone
    would let the next sweep bring the session back. The pairs are read before the rows go."""
    if not erasure.erased_at:
        raise ValueError("Erasure.erased_at must be set before delete_sessions runs")
    where, params = _session_scope(erasure)
    # `where` is one of the two literals above, never caller input.
    pairs = conn.execute(
        f"SELECT harness, native_id FROM sessions WHERE {where}",  # noqa: S608 # nosec B608
        params,
    ).fetchall()
    for row in pairs:
        conn.execute(
            "DELETE FROM capture_cursors WHERE harness = ? AND native_id = ?",
            (row["harness"], row["native_id"]),
        )
        exclusion_add(
            conn,
            harness=row["harness"],
            native_id=row["native_id"],
            reason=erasure.exclusion_reason,
            now=erasure.erased_at,
        )
    # `where` is one of `_session_scope`'s two literals, never caller input.
    return conn.execute(
        f"DELETE FROM sessions WHERE {where}",  # noqa: S608 # nosec B608
        params,
    ).rowcount


def delete_turns(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """The erased turns by id at both grains, and the owner's whole project at the project
    grain. The caller building *erasure* decides which turns ``turn_ids`` names; this deleter
    erases exactly those, plus the project's whole set at the project grain."""
    erased = conn.execute(
        "DELETE FROM turns WHERE id IN (SELECT value FROM json_each(?))", (erasure.turn_ids,)
    ).rowcount
    if erasure.grain == "project":
        erased += conn.execute(
            "DELETE FROM turns WHERE user_id = ? AND project = ?",
            (erasure.user_id, erasure.project),
        ).rowcount
    return erased


def _delete_fts_rows(conn: sqlite3.Connection, table: str, erasure: Erasure) -> int:
    """The FTS rows at the erased turns' rowids, never by a scan of an UNINDEXED column, then
    ``optimize``, so the deleted terms leave the segment b-tree inside this transaction (the
    ``secure-delete`` option has already overwritten them in place)."""
    # `table` is one of FTS_TABLES, never caller input.
    erased = conn.execute(
        f"DELETE FROM {table} WHERE rowid IN (SELECT value FROM json_each(?))",  # noqa: S608 # nosec B608
        (erasure.turn_ids,),
    ).rowcount
    # `table` is one of FTS_TABLES, never caller input.
    conn.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")  # noqa: S608 # nosec B608
    return erased


def delete_turns_fts(conn: sqlite3.Connection, erasure: Erasure) -> int:
    return _delete_fts_rows(conn, "turns_fts", erasure)


def delete_corrections_fts(conn: sqlite3.Connection, erasure: Erasure) -> int:
    return _delete_fts_rows(conn, "corrections_fts", erasure)


def delete_turn_links(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """Links from either end: a link whose later or earlier turn is erased goes at both
    grains; at the project grain so does every link of the owner whose either project is the
    erased one."""
    erased = conn.execute(
        "DELETE FROM turn_links WHERE turn_id IN (SELECT value FROM json_each(?)) "
        "OR earlier_turn_id IN (SELECT value FROM json_each(?))",
        (erasure.turn_ids, erasure.turn_ids),
    ).rowcount
    if erasure.grain == "project":
        erased += conn.execute(
            "DELETE FROM turn_links WHERE user_id = ? AND (project = ? OR earlier_project = ?)",
            (erasure.user_id, erasure.project, erasure.project),
        ).rowcount
    return erased


def delete_capture_pauses(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """Project grain only: a pause interval outlives every session."""
    if erasure.grain != "project":
        return 0
    return conn.execute(
        "DELETE FROM capture_pauses WHERE user_id = ? AND project = ?",
        (erasure.user_id, erasure.project),
    ).rowcount
