"""``digests`` and its three companions: the exact text of every render with its line refs and
one ``digest_refs`` row per quoted id, the line and link ratings, and the reads by window, by
project and by quoted id built over them."""

from __future__ import annotations

import json
import sqlite3

import pytest

from morgan_brain.memory.store import digests
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.memory.store.digests import (
    DigestLineRef,
    DigestRating,
    DigestRef,
    DigestRow,
    DigestStore,
    LinkRating,
)

TABLES = ("digests", "digest_refs", "digest_ratings", "link_ratings")
T = "2026-09-22T10:00:00.000Z"


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = open_db(str(tmp_path / "m.db"))
    DigestStore(connection)
    return connection


def _row(digest_id: str, ts: str = T, **more) -> DigestRow:
    values = {
        "id": digest_id,
        "ts": ts,
        "user_id": "u",
        "project": "p",
        "harness": "claude-code",
        "native_session_id": "s-1",
        "source": "startup",
        "entrypoint": "cli",
        "entrypoint_source": "env",
        "first_for_session": True,
        "text": "project: p\n| fact: user likes tea\n| memory: 2026-09-21 the harbor mirror\n",
        "lines": (
            DigestLineRef(None, "project", None),
            DigestLineRef(1, "fact", "f-1"),
            DigestLineRef(2, "memory", "m-1"),
        ),
        "refs": (DigestRef("fact", "f-1"), DigestRef("memory", "m-1"), DigestRef("turn", "7")),
        "chars": 70,
        "ms": 12,
    }
    values.update(more)
    return DigestRow(**values)


def _insert(conn: sqlite3.Connection, *rows: DigestRow) -> None:
    with write_transaction(conn):
        for row in rows:
            digests.insert_digest(conn, row)


def test_the_schema_and_a_second_open_change_nothing(tmp_path):
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    DigestStore(conn)
    ddl = {r["name"]: r["sql"] for r in conn.execute("SELECT name, sql FROM sqlite_master")}
    assert set(TABLES) <= set(ddl)
    assert {"idx_digests_session", "idx_digest_refs_ref"} <= set(ddl)
    assert "entrypoint_source TEXT NOT NULL DEFAULT ''" in ddl["digests"]
    assert ddl["digests"].index("entrypoint ") < ddl["digests"].index("entrypoint_source")
    before = sorted(ddl.items())
    DigestStore(open_db(path))
    after = sorted(
        (r["name"], r["sql"]) for r in conn.execute("SELECT name, sql FROM sqlite_master")
    )
    assert after == before


def test_a_digest_round_trips_with_its_lines_and_one_ref_row_per_quoted_id(conn):
    row = _row("d-1")
    _insert(conn, row)
    assert digests.get_digest(conn, user_id="u", digest_id="d-1") == row
    assert digests.get_digest(conn, user_id="v", digest_id="d-1") is None
    refs = conn.execute("SELECT kind, ref_id FROM digest_refs ORDER BY kind, ref_id").fetchall()
    assert [tuple(r) for r in refs] == [("fact", "f-1"), ("memory", "m-1"), ("turn", "7")]
    stored_lines = conn.execute("SELECT lines FROM digests WHERE id = 'd-1'").fetchone()[0]
    assert json.loads(stored_lines) == [
        {"no": None, "kind": "project", "ref": None},
        {"no": 1, "kind": "fact", "ref": "f-1"},
        {"no": 2, "kind": "memory", "ref": "m-1"},
    ]
    assert digests.has_first_for_session(conn, harness="claude-code", native_session_id="s-1")
    assert not digests.has_first_for_session(conn, harness="codex", native_session_id="s-1")


def test_last_digest_is_the_newest_of_the_project(conn):
    _insert(
        conn,
        _row("d-1", ts="2026-09-22T09:00:00.000Z"),
        _row("d-2", ts=T),
        _row("d-q", project="q"),
    )
    last = digests.last_digest(conn, user_id="u", project="p")
    assert last is not None and last.id == "d-2"
    assert digests.last_digest(conn, user_id="u", project="r") is None


def test_newest_unrated_first_skips_rated_excluded_and_later_renders(conn):
    def excluded(entrypoint: str) -> bool:
        return entrypoint in ("", "exec") or entrypoint.startswith("sdk")

    _insert(
        conn,
        _row("old-unrated", ts="2026-09-22T08:00:00.000Z"),
        _row("rated", ts="2026-09-22T09:00:00.000Z"),
        _row("exec", ts="2026-09-22T10:00:00.000Z", entrypoint="exec"),
        _row("unknown", ts="2026-09-22T10:30:00.000Z", entrypoint="", entrypoint_source="none"),
        _row("later-render", ts="2026-09-22T11:00:00.000Z", first_for_session=False),
        _row("other-owner", ts="2026-09-22T12:00:00.000Z", user_id="v"),
    )
    with write_transaction(conn):
        digests.rate_line(conn, digest_id="rated", line_no=1, rating="right", now=T)
        digests.rate_line(conn, digest_id="rated", line_no=2, rating="useless", now=T)
    found = digests.newest_unrated_first(
        conn, user_id="u", project=None, excluded_entrypoints=excluded
    )
    assert found is not None and found.id == "old-unrated"
    assert (
        digests.newest_unrated_first(conn, user_id="u", project="q", excluded_entrypoints=excluded)
        is None
    )
    with write_transaction(conn):
        assert (
            digests.backfill_entrypoint(
                conn, harness="claude-code", native_session_id="s-1", entrypoint="cli"
            )
            == 1
        )
    found = digests.newest_unrated_first(
        conn, user_id="u", project=None, excluded_entrypoints=excluded
    )
    assert found is not None and found.id == "unknown"
    assert found.entrypoint_source == "backfill"
    assert digests.get_digest(conn, user_id="u", digest_id="exec") == _row(
        "exec", ts="2026-09-22T10:00:00.000Z", entrypoint="exec"
    )  # a known entrypoint is never back-filled


def test_ratings_are_upserted_and_read_back(conn):
    _insert(conn, _row("d-1"))
    with write_transaction(conn):
        digests.rate_line(conn, digest_id="d-1", line_no=1, rating="wrong", now=T)
        digests.rate_line(
            conn, digest_id="d-1", line_no=1, rating="right", now="2026-09-22T11:00:00.000Z"
        )
        digests.rate_link(
            conn, turn_id=9, earlier_turn_id=4, user_id="u", project="p", rating="right", now=T
        )
        digests.rate_link(
            conn, turn_id=9, earlier_turn_id=4, user_id="u", project="p", rating="wrong", now=T
        )
    assert digests.ratings_of(conn, "d-1") == [
        DigestRating("d-1", 1, "right", "2026-09-22T11:00:00.000Z")
    ]
    assert digests.link_ratings_between(
        conn, user_id="u", since="2026-09-22T00:00:00.000Z", until="2026-09-23T00:00:00.000Z"
    ) == [LinkRating(9, 4, "u", "p", "wrong", T)]
    with pytest.raises(sqlite3.IntegrityError), write_transaction(conn):
        digests.rate_line(conn, digest_id="d-1", line_no=1, rating="meh", now=T)


def test_digests_between_and_digests_quoting(conn):
    _insert(
        conn,
        _row("d-1", ts="2026-09-22T10:00:00.000Z"),
        _row("d-2", ts="2026-09-23T10:00:00.000Z", refs=(DigestRef("turn", "7"),)),
        _row("d-3", ts="2026-09-24T10:00:00.000Z", refs=(DigestRef("memory", "m-9"),)),
    )
    window = digests.digests_between(
        conn, user_id="u", since="2026-09-22T00:00:00.000Z", until="2026-09-24T00:00:00.000Z"
    )
    assert [d.id for d in window] == ["d-1", "d-2"]
    assert digests.digests_quoting(conn, kind="turn", ref_ids=json.dumps([7, 8])) == ["d-1", "d-2"]
    assert digests.digests_quoting(conn, kind="memory", ref_ids=json.dumps(["m-1"])) == ["d-1"]
    assert digests.digests_quoting(conn, kind="fact", ref_ids=json.dumps([])) == []


def test_every_writer_names_the_entrypoint_source(conn):
    """The column's ``DEFAULT ''`` is never relied on: a row always says where its entrypoint
    came from."""
    _insert(conn, _row("d-1", entrypoint="", entrypoint_source="none"))
    stored = conn.execute("SELECT entrypoint, entrypoint_source FROM digests").fetchone()
    assert tuple(stored) == ("", "none")
    with pytest.raises(TypeError):
        DigestRow(**{k: v for k, v in _row("d-2").__dict__.items() if k != "entrypoint_source"})
