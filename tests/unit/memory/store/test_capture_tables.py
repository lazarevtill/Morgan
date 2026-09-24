"""The tables capture keeps beside the sessions: the cursor with its lease, the exclusions that
keep a forgotten session out, the on/off state, and the pause intervals that outlive
``pause --off``. Every timestamp here is ``utc_iso``'s shape; ``in_pause`` compares moments, so a
harness's own shape is inside or outside an interval by time, not by string."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.memory.store import sessions
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.memory.store.sessions import SessionStore
from morgan_brain.models import CaptureCursor, Exclusion, OpenCall, Pause, utc_iso

T0 = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _at(minutes: int) -> str:
    return utc_iso(T0 + timedelta(minutes=minutes))


NOW = _at(0)
TABLES = ("capture_cursors", "capture_exclusions", "capture_state", "capture_pauses")


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = open_db(str(tmp_path / "m.db"))
    SessionStore(connection)
    return connection


def _cursor(native_id: str = "s-1", **more) -> CaptureCursor:
    values = {
        "harness": "claude-code",
        "native_id": native_id,
        "source_path": f"/transcripts/{native_id}.jsonl",
        "byte_offset": 120,
        "size": 400,
        "mtime_ns": 1_700_000_000_000_000_000,
        "identity": "ab" * 32,
        "status": "ok",
        "last_read_at": NOW,
        "open_calls": [OpenCall(id="toolu_m", morgan=True, pending=["result"])],
    }
    values.update(more)
    return CaptureCursor(**values)


# --- the schema -----------------------------------------------------------------------------


def test_the_four_tables_and_the_pause_index_exist_and_a_second_open_changes_nothing(tmp_path):
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    SessionStore(conn)
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert set(TABLES) <= names and "idx_capture_pauses_project" in names
    before = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    SessionStore(open_db(path))
    after = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    assert [tuple(r) for r in after] == [tuple(r) for r in before]


# --- cursors and the lease -------------------------------------------------------------------


def test_a_cursor_round_trips_and_an_upsert_replaces_its_columns_but_not_the_lease(conn):
    with write_transaction(conn):
        sessions.cursor_put(conn, _cursor())
    stored = sessions.cursor_get(conn, harness="claude-code", native_id="s-1")
    assert stored == _cursor()
    assert sessions.cursor_get(conn, harness="codex", native_id="s-1") is None

    with write_transaction(conn):
        assert sessions.lease_claim(
            conn,
            harness="claude-code",
            native_id="s-1",
            source_path="/transcripts/s-1.jsonl",
            owner="host:1",
            now=NOW,
            until=_at(1),
        )
    with write_transaction(conn):
        sessions.cursor_put(conn, _cursor(byte_offset=400, status="ok", last_read_at=_at(2)))
    advanced = sessions.cursor_get(conn, harness="claude-code", native_id="s-1")
    assert advanced is not None
    assert (advanced.byte_offset, advanced.last_read_at) == (400, _at(2))
    assert (advanced.lease_owner, advanced.lease_until) == ("host:1", _at(1))  # the lease's own
    assert advanced.open_calls == [OpenCall(id="toolu_m", morgan=True, pending=["result"])]
    with write_transaction(conn):
        sessions.cursor_put(conn, _cursor(open_calls=[]))
    cleared = sessions.cursor_get(conn, harness="claude-code", native_id="s-1")
    assert cleared is not None and cleared.open_calls == []
    assert sessions.cursors(conn) == {("claude-code", "s-1"): cleared}


def test_a_lease_is_claimed_once_and_reclaimed_by_its_owner_or_after_it_lapses(conn):
    claim = {"harness": "codex", "native_id": "r-1", "source_path": "/rollouts/r-1.jsonl"}
    with write_transaction(conn):
        assert sessions.lease_claim(conn, **claim, owner="a", now=NOW, until=_at(1))
    new_row = sessions.cursor_get(conn, harness="codex", native_id="r-1")
    assert new_row is not None
    assert (new_row.status, new_row.byte_offset, new_row.lease_owner, new_row.open_calls) == (
        "new",
        0,
        "a",
        [],
    )
    with write_transaction(conn):
        assert not sessions.lease_claim(conn, **claim, owner="b", now=_at(0), until=_at(1))
    with write_transaction(conn):
        assert sessions.lease_claim(conn, **claim, owner="a", now=_at(0), until=_at(2))  # its owner
    with write_transaction(conn):
        assert sessions.lease_claim(conn, **claim, owner="b", now=_at(3), until=_at(4))  # lapsed
    with write_transaction(conn):
        sessions.lease_release(conn, harness="codex", native_id="r-1", owner="a")  # not the holder
    held = sessions.cursor_get(conn, harness="codex", native_id="r-1")
    assert held is not None and held.lease_owner == "b"
    with write_transaction(conn):
        sessions.lease_release(conn, harness="codex", native_id="r-1", owner="b")
    released = sessions.cursor_get(conn, harness="codex", native_id="r-1")
    assert released is not None and (released.lease_owner, released.lease_until) == (None, None)
    assert not conn.in_transaction  # every claim and release above ran in its own transaction


def test_a_claim_nested_inside_a_larger_write_joins_it_as_a_savepoint(conn):
    """A per-file claim, wrapped in its own write transaction the way a batch of files will
    wrap each one, still lands correctly when that batch itself runs inside a still-larger
    write: the inner transaction joins the outer one as a savepoint instead of failing to
    start a second transaction."""
    with write_transaction(conn):
        with write_transaction(conn):
            assert sessions.lease_claim(
                conn,
                harness="codex",
                native_id="r-2",
                source_path="/r-2.jsonl",
                owner="a",
                now=NOW,
                until=_at(1),
            )
        assert conn.in_transaction
    assert sessions.cursor_get(conn, harness="codex", native_id="r-2") is not None


# --- exclusions -------------------------------------------------------------------------------


def test_exclusions_are_added_listed_and_removed(conn):
    with write_transaction(conn):
        sessions.exclusion_add(conn, harness="codex", native_id="r-1", reason="forget", now=NOW)
        sessions.exclusion_add(conn, harness="codex", native_id="r-1", reason="since", now=_at(1))
        sessions.exclusion_add(
            conn, harness="claude-code", native_id="s-9", reason="exclude", now=NOW
        )
    assert sessions.exclusions(conn) == [
        Exclusion(harness="claude-code", native_id="s-9", reason="exclude", excluded_at=NOW),
        Exclusion(harness="codex", native_id="r-1", reason="since", excluded_at=_at(1)),
    ]
    assert sessions.is_excluded(conn, harness="codex", native_id="r-1")
    assert not sessions.is_excluded(conn, harness="codex", native_id="r-2")
    with write_transaction(conn):
        assert sessions.exclusion_remove(conn, harness="codex", native_id="r-1")
        assert not sessions.exclusion_remove(conn, harness="codex", native_id="r-1")
    assert not sessions.is_excluded(conn, harness="codex", native_id="r-1")


# --- state ------------------------------------------------------------------------------------


def test_the_state_is_a_key_value_table_and_enabled_means_1(conn):
    assert sessions.state_get(conn, sessions.STATE_ENABLED) is None
    assert not sessions.capture_enabled(conn)
    with write_transaction(conn):
        sessions.state_set(conn, sessions.STATE_ENABLED, "1", now=NOW)
        sessions.state_set(conn, sessions.STATE_REPORT_PATH, "/state/gate-report.json", now=NOW)
        sessions.state_set(conn, sessions.STATE_REPORT_PATH, "/state/newer.json", now=_at(1))
    assert sessions.capture_enabled(conn)
    assert sessions.state_get(conn, sessions.STATE_REPORT_PATH) == "/state/newer.json"
    assert (sessions.STATE_ENABLED_AT, sessions.STATE_LAST_SWEEP) == ("enabled_at", "last_sweep")


# --- pauses -----------------------------------------------------------------------------------


def test_an_interval_is_current_while_its_end_is_null_or_in_the_future(conn):
    with write_transaction(conn):
        past = sessions.pause_open(
            conn, user_id="u", project="p", paused_from=_at(-120), paused_until=_at(-60)
        )
        bounded = sessions.pause_open(
            conn, user_id="u", project="p", paused_from=_at(-10), paused_until=_at(120)
        )
    assert past.id is not None and bounded.id is not None and bounded.id > past.id
    assert [p.paused_from for p in sessions.pauses_of(conn, user_id="u", project="p")] == [
        _at(-120),
        _at(-10),
    ]
    assert sessions.current_pause(conn, user_id="u", project="p", now=NOW) == bounded
    assert sessions.current_pause(conn, user_id="u", project="p", now=_at(121)) is None
    assert sessions.current_pause(conn, user_id="u", project="q", now=NOW) is None
    with write_transaction(conn):
        open_ended = sessions.pause_open(
            conn, user_id="u", project="p", paused_from=_at(-5), paused_until=None
        )
    assert sessions.current_pause(conn, user_id="u", project="p", now=NOW) == open_ended  # latest


def test_pause_close_ends_every_current_interval_now_and_leaves_the_past_ones(conn):
    with write_transaction(conn):
        sessions.pause_open(
            conn, user_id="u", project="p", paused_from=_at(-120), paused_until=_at(-60)
        )
        sessions.pause_open(
            conn, user_id="u", project="p", paused_from=_at(-10), paused_until=_at(120)
        )
        sessions.pause_open(conn, user_id="u", project="p", paused_from=_at(-5), paused_until=None)
        sessions.pause_open(conn, user_id="u", project="q", paused_from=_at(-5), paused_until=None)
        assert sessions.pause_close(conn, user_id="u", project="p", now=NOW) == 2
    ends = [p.paused_until for p in sessions.pauses_of(conn, user_id="u", project="p")]
    assert ends == [_at(-60), NOW, NOW]  # the longer, bounded interval was cut short too
    assert sessions.current_pause(conn, user_id="u", project="p", now=NOW) is None
    assert sessions.current_pause(conn, user_id="u", project="q", now=NOW) is not None
    with write_transaction(conn):
        assert sessions.pause_close(conn, user_id="u", project="p", now=_at(1)) == 0


def test_in_pause_is_pure_and_compares_moments():
    pauses = [
        Pause(user_id="u", project="p", paused_from=_at(0), paused_until=_at(10)),
        Pause(user_id="u", project="p", paused_from=_at(60), paused_until=None),
    ]
    assert sessions.in_pause(_at(0), pauses)  # the start is inside
    assert sessions.in_pause(_at(5), pauses)
    assert not sessions.in_pause(_at(10), pauses)  # the end is outside
    assert not sessions.in_pause(_at(30), pauses)
    assert sessions.in_pause(_at(600), pauses)  # open-ended
    assert sessions.in_pause("2026-09-22T10:05:00Z", pauses)  # a harness's own shape
    assert sessions.in_pause("2026-09-22T12:05:00+02:00", pauses)
    assert not sessions.in_pause(_at(5), [])
