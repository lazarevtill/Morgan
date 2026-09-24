"""``call_log``: one row per call, read back by time window and counted by command and
degrade reason."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from morgan_brain.memory.store import calls
from morgan_brain.memory.store.calls import CallCounts, CallLogStore, CallRecord
from morgan_brain.memory.store.db import open_db, write_transaction


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = open_db(str(tmp_path / "m.db"))
    CallLogStore(connection)
    return connection


def _call(ts: str, command: str = "recall", **more) -> CallRecord:
    values = {
        "ts": ts,
        "surface": "cli",
        "client": "cli:claude-code/cli",
        "native_session_id": "s-1",
        "command": command,
        "user_id": "u",
        "project": "p",
        "all_projects": False,
        "outcome": "ok",
        "embed_outcome": "ok",
        "degraded": None,
        "degrade_reason": None,
        "embed_latency_ms": 12.5,
        "total_ms": 40.0,
        "query_language": "en",
        "reason": None,
    }
    values.update(more)
    return CallRecord(**values)


def test_the_schema_and_a_second_open_change_nothing(tmp_path):
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    CallLogStore(conn)
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"call_log", "idx_call_log_ts"} <= names
    before = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    CallLogStore(open_db(path))
    after = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    assert [tuple(r) for r in after] == [tuple(r) for r in before]


def test_a_call_round_trips_with_its_id(conn):
    record = _call("2026-09-22T10:00:00.000Z")
    with write_transaction(conn):
        rowid = calls.insert_call(conn, record)
    assert rowid == 1
    [stored] = calls.calls_between(
        conn, user_id="u", since="2026-09-22T00:00:00.000Z", until="2026-09-23T00:00:00.000Z"
    )
    assert stored == replace(record, id=1)


def test_calls_between_is_half_open_and_owner_scoped(conn):
    with write_transaction(conn):
        for ts in (
            "2026-09-21T23:59:59.999Z",
            "2026-09-22T00:00:00.000Z",
            "2026-09-22T12:00:00.000Z",
        ):
            calls.insert_call(conn, _call(ts))
        calls.insert_call(conn, _call("2026-09-22T12:00:00.000Z", user_id="v"))
    found = calls.calls_between(
        conn, user_id="u", since="2026-09-22T00:00:00.000Z", until="2026-09-22T12:00:00.000Z"
    )
    assert [c.ts for c in found] == ["2026-09-22T00:00:00.000Z"]


def test_call_counts_since_counts_by_command_and_degrade_reason(conn):
    with write_transaction(conn):
        calls.insert_call(conn, _call("2026-09-22T10:00:00.000Z"))
        calls.insert_call(
            conn,
            _call(
                "2026-09-22T10:01:00.000Z", degraded="keyword_only", degrade_reason="unreachable"
            ),
        )
        calls.insert_call(
            conn,
            _call(
                "2026-09-22T10:02:00.000Z", degraded="keyword_only", degrade_reason="over_budget"
            ),
        )
        calls.insert_call(
            conn,
            _call(
                "2026-09-22T10:03:00.000Z", degraded="keyword_only", degrade_reason="over_budget"
            ),
        )
        calls.insert_call(conn, _call("2026-09-22T10:04:00.000Z", command="remember"))
        calls.insert_call(
            conn, _call("2026-09-22T10:05:00.000Z", command="remember", outcome="failed")
        )
        calls.insert_call(conn, _call("2026-09-22T10:06:00.000Z", command="facts"))
        calls.insert_call(conn, _call("2026-09-22T10:07:00.000Z", command="ask"))
        calls.insert_call(conn, _call("2026-09-20T10:00:00.000Z", command="ask"))  # before
        calls.insert_call(conn, _call("2026-09-22T10:08:00.000Z", user_id="v"))  # another owner
    counts = calls.call_counts_since(conn, user_id="u", since="2026-09-22T00:00:00.000Z")
    assert counts == CallCounts(
        recalls=4,
        degraded={"over_budget": 2, "unreachable": 1},
        remembers=2,
        remember_failed=1,
        facts=1,
        asks=1,
    )
