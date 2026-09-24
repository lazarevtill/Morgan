"""The archive's session tables: the DDL, FTS rows at the turns' rowids, the ``secure-delete``
option set by the ``CREATE`` and by nothing else, and their writers and readers. No vectors
anywhere: the archive is keyword-indexed only."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from morgan_brain.memory.store import sessions
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.memory.store.sessions import FTS_TABLES, SessionDelta, SessionStore
from morgan_brain.models import Session, Turn, parse_iso, session_id_of, utc_iso

NOW = "2026-09-22T10:15:00.000Z"
LATER = "2026-09-22T11:00:00.000Z"
EARLIER = "2026-09-21T09:00:00.000Z"
TABLES = ("sessions", "turns", "turns_fts", "corrections_fts", "turn_links")
INDEXES = (
    "idx_turns_project_ts",
    "idx_turns_native_uuid",
    "idx_turns_correction",
    "idx_turn_links_earlier",
)


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = open_db(str(tmp_path / "m.db"))
    SessionStore(connection)
    return connection


def _session(native_id: str = "s-1", *, project: str = "p", harness: str = "claude-code", **more):
    values = {
        "id": session_id_of(harness, native_id),
        "user_id": "u",
        "project": project,
        "harness": harness,
        "native_id": native_id,
        "source_path": f"/transcripts/{native_id}.jsonl",
        "cwd": "/src/p",
        "project_source": "git",
        "started_at": NOW,
        "reader_version": 1,
        "gate_version": 1,
    }
    values.update(more)
    return Session(**values)


def _turn(session: Session, key: str, text: str, *, role: str = "user", ts: str = NOW, **more):
    values = {
        "session_id": session.id,
        "user_id": session.user_id,
        "project": session.project,
        "native_key": key,
        "role": role,
        "text": text,
        "ts": ts,
    }
    values.update(more)
    return Turn(**values)


def _put(conn: sqlite3.Connection, session: Session, *turns: Turn) -> list[int]:
    with write_transaction(conn):
        sessions.upsert_session(conn, session, now=NOW)
        return sessions.insert_turns(conn, list(turns))


def _ddl(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        r["name"]: r["sql"]
        for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")
    }


def _option(conn: sqlite3.Connection, table: str) -> int | None:
    # `table` is one of FTS_TABLES, never data.
    row = conn.execute(f"SELECT v FROM {table}_config WHERE k = 'secure-delete'").fetchone()  # noqa: S608
    return None if row is None else int(row[0])


# --- the schema -----------------------------------------------------------------------------


def test_the_archive_tables_and_indexes_exist_and_each_fts_table_has_secure_delete_on(conn):
    ddl = _ddl(conn)
    assert set(TABLES) <= set(ddl)
    assert set(INDEXES) <= set(ddl)
    assert "REFERENCES sessions(id) ON DELETE CASCADE" in ddl["turns"]
    assert "UNIQUE (harness, native_id)" in ddl["sessions"]
    assert "tokenize = 'unicode61 remove_diacritics 2'" in ddl["turns_fts"]
    assert all(_option(conn, table) == 1 for table in FTS_TABLES)


def test_a_second_open_changes_nothing_and_writes_nothing(tmp_path):
    """Opening an existing database must take no write lock: a read-only command opens the
    stores too, and so does the digest hook under a short busy timeout. The lock is held by a
    second connection throughout; the option is already set, so nothing is written."""
    path = str(tmp_path / "m.db")
    first = open_db(path)
    SessionStore(first)
    before = _ddl(first)
    holder = open_db(path)
    holder.execute("BEGIN IMMEDIATE")
    try:
        again = open_db(path, busy_timeout_ms=50)
        SessionStore(again)  # would raise "database is locked" if it wrote
        assert _ddl(again) == before
        assert all(_option(again, table) == 1 for table in FTS_TABLES)
    finally:
        holder.rollback()


def test_a_first_open_that_meets_lock_contention_leaves_no_table_half_made(tmp_path):
    """The whole schema is built under one lock. A writer that already holds the database when
    the first open runs makes it fail outright -- not partway through, with one table caught
    without its ``secure-delete`` option for good."""
    path = str(tmp_path / "m.db")
    reader = open_db(path)
    holder = open_db(path)
    holder.execute("BEGIN IMMEDIATE")
    try:
        blocked = open_db(path, busy_timeout_ms=50)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            SessionStore(blocked)
        assert _ddl(reader) == {}
    finally:
        holder.rollback()
    SessionStore(reader)
    assert set(TABLES) <= set(_ddl(reader))
    assert all(_option(reader, table) == 1 for table in FTS_TABLES)


class _RaisesOnStatement:
    """Wraps a real connection; the first ``execute`` whose SQL contains *trigger* raises
    instead of running, and every other call is delegated untouched. Simulates a process that
    dies, or a lock it loses, partway through building the schema."""

    def __init__(self, conn: sqlite3.Connection, trigger: str) -> None:
        self._conn = conn
        self._trigger = trigger

    def execute(self, sql, *args, **kwargs):
        if self._trigger in sql:
            raise RuntimeError("simulated interruption")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_an_interruption_between_two_creates_commits_nothing(tmp_path):
    """The whole schema is built under one write transaction: a failure between two ``CREATE``
    statements rolls every earlier one in the same attempt back too, rather than leaving
    ``turns_fts`` committed without its ``secure-delete`` option because ``corrections_fts``
    never got made."""
    path = str(tmp_path / "m.db")
    real = open_db(path)
    interrupted = _RaisesOnStatement(real, "CREATE VIRTUAL TABLE IF NOT EXISTS corrections_fts")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sessions.create_schema(interrupted)
    assert _ddl(real) == {}

    sessions.create_schema(real)
    assert set(TABLES) <= set(_ddl(real))
    assert all(_option(real, table) == 1 for table in FTS_TABLES)


def test_a_partially_created_schema_is_completed_atomically_on_the_next_open(tmp_path):
    """A database left with only one table -- an earlier open that stopped between two
    ``CREATE``s -- is finished in one write: every remaining table appears together, and each
    FTS table it makes gets ``secure-delete`` set."""
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    conn.execute(sessions._SCHEMA_STATEMENTS[0])  # `sessions` alone; nothing else exists yet
    conn.commit()
    SessionStore(conn)
    assert set(TABLES) <= set(_ddl(conn))
    assert all(_option(conn, table) == 1 for table in FTS_TABLES)


def test_session_id_of_is_the_harness_and_the_native_id():
    assert session_id_of("codex", "abc-123") == "codex:abc-123"


# --- the timestamp helpers ------------------------------------------------------------------


def test_utc_iso_writes_one_shape_and_parse_iso_reads_every_harness_shape():
    moment = datetime(2026, 9, 22, 12, 30, 45, 123456, tzinfo=UTC)
    assert utc_iso(moment) == "2026-09-22T12:30:45.123Z"
    assert parse_iso("2026-09-22T12:30:45.123Z") == moment.replace(microsecond=123000)
    assert parse_iso("2026-09-22T14:30:45+02:00") == moment.replace(microsecond=0)
    assert parse_iso("2026-09-22") == datetime(2026, 9, 22, tzinfo=UTC)
    assert parse_iso("2026-09-22T12:30:45") == moment.replace(microsecond=0)
    with pytest.raises(ValueError, match="aware"):
        utc_iso(datetime(2026, 9, 22, 12, 30, 45))  # noqa: DTZ001 -- the naive case under test
    assert utc_iso(moment) < utc_iso(moment.replace(second=46))  # lexical order is time order


# --- sessions -------------------------------------------------------------------------------


def test_a_session_is_inserted_then_updated_in_its_mutable_fields_only(conn):
    session = _session(entrypoint="cli", harness_version="2.1.280")
    _put(conn, session)
    stored = sessions.get_session(conn, session.id)
    assert stored is not None
    assert (stored.imported_at, stored.updated_at, stored.entrypoint) == (NOW, NOW, "cli")

    resumed = _session(
        entrypoint="",
        harness_version="",
        ended_at=LATER,
        cwd_changed=True,
        project="q",
        source_path="/elsewhere.jsonl",
        started_at=EARLIER,
        reader_version=2,
    )
    with write_transaction(conn):
        sessions.upsert_session(conn, resumed, now=LATER)
    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    assert (updated.ended_at, updated.cwd_changed, updated.reader_version) == (LATER, True, 2)
    assert (updated.imported_at, updated.updated_at) == (NOW, LATER)
    # A capture that does not know a value never blanks what an earlier one knew.
    assert (updated.entrypoint, updated.harness_version) == ("cli", "2.1.280")
    # Fixed at insert: the project, the file, the first start time.
    assert (updated.project, updated.source_path, updated.started_at) == (
        "p",
        session.source_path,
        NOW,
    )
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_upsert_never_blanks_forked_from_cwd_changed_or_ended_at(conn):
    """A later capture that does not know a value -- the model's own defaults, ``None`` and
    ``False`` -- must never undo what an earlier capture already established."""
    session = _session(forked_from="claude-code:parent", cwd_changed=True, ended_at=NOW)
    _put(conn, session)
    later_unknown = _session(forked_from=None, cwd_changed=False, ended_at=None)
    with write_transaction(conn):
        sessions.upsert_session(conn, later_unknown, now=LATER)
    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    assert (updated.forked_from, updated.cwd_changed, updated.ended_at) == (
        "claude-code:parent",
        True,
        NOW,
    )


def test_upsert_still_advances_forked_from_cwd_changed_and_ended_at_when_known(conn):
    session = _session()
    _put(conn, session)
    later_known = _session(forked_from="claude-code:parent", cwd_changed=True, ended_at=LATER)
    with write_transaction(conn):
        sessions.upsert_session(conn, later_known, now=LATER)
    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    assert (updated.forked_from, updated.cwd_changed, updated.ended_at) == (
        "claude-code:parent",
        True,
        LATER,
    )


def test_upsert_keeps_the_later_ended_at_when_a_capture_arrives_out_of_order(conn):
    """A sweep over an older slice of the transcript must not roll a session's end back below
    what a later-running capture already recorded."""
    session = _session(ended_at=LATER)
    _put(conn, session)
    out_of_order = _session(ended_at=NOW)
    with write_transaction(conn):
        sessions.upsert_session(conn, out_of_order, now=LATER)
    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    assert updated.ended_at == LATER


def test_upsert_keeps_harness_mode_when_a_later_capture_does_not_know_it(conn):
    session = _session(harness_mode="print")
    _put(conn, session)
    later_unknown = _session(harness_mode="")
    with write_transaction(conn):
        sessions.upsert_session(conn, later_unknown, now=LATER)
    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    assert updated.harness_mode == "print"


def test_mark_trigger_first_sets_each_stamp_once(conn):
    session = _session()
    _put(conn, session)
    with write_transaction(conn):
        sessions.mark_trigger_first(conn, session.id, trigger="hook", now=NOW)
        sessions.mark_trigger_first(conn, session.id, trigger="hook", now=LATER)
    stored = sessions.get_session(conn, session.id)
    assert stored is not None
    assert (stored.hook_first_at, stored.sweep_first_at, stored.all_first_at) == (NOW, None, None)


def test_add_session_counts_accumulates(conn):
    session = _session()
    _put(conn, session)
    with write_transaction(conn):
        sessions.add_session_counts(
            conn, session.id, SessionDelta(3, 2, 1, 1, 1, 0, ended_at=None), now=NOW
        )
        sessions.add_session_counts(
            conn, session.id, SessionDelta(2, 1, 0, 2, 0, 1, ended_at=LATER), now=LATER
        )
    stored = sessions.get_session(conn, session.id)
    assert stored is not None
    assert (stored.turn_count, stored.authored_turn_count) == (5, 3)
    assert (stored.gate_redactions, stored.gate_flags, stored.gate_provider_hits) == (1, 3, 1)
    assert (stored.paused_turns, stored.ended_at, stored.updated_at) == (1, LATER, LATER)


def test_add_session_counts_keeps_the_later_ended_at_when_a_batch_arrives_out_of_order(conn):
    """A sweep that lands after the hook already recorded a later end must not roll it back."""
    session = _session()
    _put(conn, session)
    with write_transaction(conn):
        sessions.add_session_counts(
            conn, session.id, SessionDelta(1, 1, 0, 0, 0, 0, ended_at=LATER), now=NOW
        )
        sessions.add_session_counts(
            conn, session.id, SessionDelta(1, 1, 0, 0, 0, 0, ended_at=NOW), now=LATER
        )
    stored = sessions.get_session(conn, session.id)
    assert stored is not None
    assert stored.ended_at == LATER


# --- turns ----------------------------------------------------------------------------------


def test_turns_get_ordinals_after_the_maximum_and_a_repeated_key_is_ignored(conn):
    session = _session()
    first = _put(
        conn, session, _turn(session, "a:0", "the harbor"), _turn(session, "b:0", "mirror")
    )
    assert first == [1, 2]
    second = _put(
        conn, session, _turn(session, "b:0", "mirror again"), _turn(session, "c:0", "tide")
    )
    assert second == [3]
    stored = sessions.turns_of(conn, session.id)
    assert [(t.id, t.ordinal, t.text) for t in stored] == [
        (1, 0, "the harbor"),
        (2, 1, "mirror"),
        (3, 2, "tide"),
    ]


def test_the_fts_row_follows_at_the_same_rowid(conn):
    session = _session()
    ids = _put(
        conn, session, _turn(session, "a:0", "the harbor mirror"), _turn(session, "b:0", "tide")
    )
    rows = conn.execute("SELECT rowid, text, session_id, role, authored FROM turns_fts").fetchall()
    assert [(r[0], r[1], r[2], r[3], r[4]) for r in rows] == [
        (ids[0], "the harbor mirror", session.id, "user", 1),
        (ids[1], "tide", session.id, "user", 1),
    ]
    assert conn.execute("SELECT COUNT(*) FROM corrections_fts").fetchone()[0] == 0


def test_a_turn_stored_under_another_session_of_the_same_harness_is_skipped(conn):
    parent = _session("parent")
    _put(conn, parent, _turn(parent, "u1:0", "copied", native_uuid="u1"))
    resumed = _session("child", forked_from=parent.id)
    assert _put(conn, resumed, _turn(resumed, "u1:0", "copied", native_uuid="u1")) == []
    other_harness = _session("rollout", harness="codex")
    assert _put(conn, other_harness, _turn(other_harness, "0:0", "copied", native_uuid="u1")) == [2]


def test_insert_turns_refuses_turns_of_two_sessions_in_one_call(conn):
    session = _session()
    other = _session("other")
    _put(conn, session)
    _put(conn, other)
    with pytest.raises(ValueError, match="one session"), write_transaction(conn):
        sessions.insert_turns(conn, [_turn(session, "a:0", "here"), _turn(other, "b:0", "there")])


def test_a_turn_of_an_unknown_session_is_refused_by_the_foreign_key(conn):
    ghost = _session("ghost")
    with pytest.raises(sqlite3.IntegrityError), write_transaction(conn):
        sessions.insert_turns(conn, [_turn(ghost, "a:0", "no session row")])


def test_turns_of_one_session_only_and_since(conn):
    session = _session()
    _put(
        conn,
        session,
        _turn(session, "a:0", "early", ts=EARLIER),
        _turn(session, "b:0", "now", ts=NOW),
        _turn(session, "c:0", "late", ts=LATER),
    )
    other = _session("other")
    _put(conn, other, _turn(other, "a:0", "elsewhere"))
    assert [t.text for t in sessions.turns_of(conn, session.id)] == ["early", "now", "late"]
    assert [t.text for t in sessions.turns_of(conn, session.id, since=NOW)] == ["now", "late"]
    assert [t.text for t in sessions.turns_of(conn, other.id)] == ["elsewhere"]


# --- readers --------------------------------------------------------------------------------


def test_find_session_by_harness_and_native_id(conn):
    _put(conn, _session("s-1"))
    found = sessions.find_session(conn, harness="claude-code", native_id="s-1")
    assert found is not None and found.id == "claude-code:s-1"
    assert sessions.find_session(conn, harness="codex", native_id="s-1") is None
    assert sessions.get_session(conn, "codex:nope") is None


def test_list_sessions_is_newest_first_and_optionally_across_projects(conn):
    _put(conn, _session("a", started_at=NOW))
    _put(conn, _session("b", project="q", started_at=LATER))
    _put(conn, _session("c", started_at=EARLIER))
    _put(conn, _session("d", started_at=None))
    in_p = sessions.list_sessions(conn, user_id="u", project="p", since=None)
    assert [s.native_id for s in in_p] == ["a", "c", "d"]
    everywhere = sessions.list_sessions(conn, user_id="u", project=None, since=None)
    assert [s.native_id for s in everywhere] == ["b", "a", "c", "d"]
    recent = sessions.list_sessions(conn, user_id="u", project=None, since=NOW)
    assert [s.native_id for s in recent] == ["b", "a"]
    assert sessions.list_sessions(conn, user_id="v", project=None, since=None) == []


def test_search_turns_matches_the_keyword_index_and_snips(conn):
    session = _session()
    _put(
        conn,
        session,
        _turn(session, "a:0", "the harbor mirror failed on the second try"),
        _turn(session, "b:0", "tide tables", role="assistant"),
    )
    other = _session("other", project="q")
    _put(conn, other, _turn(other, "a:0", "another mirror elsewhere"))

    hits = sessions.search_turns(conn, user_id="u", project="p", query="mirror", limit=10)
    assert [(h.session_id, h.harness, h.native_id, h.ordinal, h.role, h.ts) for h in hits] == [
        (session.id, "claude-code", "s-1", 0, "user", NOW)
    ]
    assert "mirror" in hits[0].snippet
    everywhere = sessions.search_turns(conn, user_id="u", project=None, query="mirror", limit=10)
    assert {h.session_id for h in everywhere} == {session.id, other.id}
    assert len(sessions.search_turns(conn, user_id="u", project=None, query="mirror", limit=1)) == 1
    assert sessions.search_turns(conn, user_id="u", project="p", query="", limit=10) == []
    assert sessions.search_turns(conn, user_id="v", project=None, query="mirror", limit=10) == []
