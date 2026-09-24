import sqlite3

import pytest

from morgan_brain.memory.store.db import (
    MIN_SQLITE,
    SQLiteTooOld,
    open_db,
    open_readonly,
    write_transaction,
)


def test_open_db_enables_wal_and_vec(tmp_path):
    conn = open_db(str(tmp_path / "m.db"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    # sqlite-vec is loaded when vec_version() resolves
    assert conn.execute("SELECT vec_version()").fetchone()[0]


def test_open_db_is_reopenable(tmp_path):
    path = str(tmp_path / "m.db")
    open_db(path).execute("CREATE TABLE t (a TEXT)")
    conn2 = open_db(path)
    assert conn2.execute("SELECT count(*) FROM t").fetchone()[0] == 0


def test_memory_path_is_supported_for_tests():
    conn = open_db(":memory:")
    assert conn.execute("SELECT vec_version()").fetchone()[0]


def _two_connections(tmp_path):
    """A Morgan connection, and a bare second one that gives up at once instead of waiting."""
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.commit()
    return conn, sqlite3.connect(path, timeout=0)


def test_write_transaction_commits_the_block(tmp_path):
    conn, other = _two_connections(tmp_path)
    with write_transaction(conn):
        conn.execute("INSERT INTO t VALUES ('x')")
    assert other.execute("SELECT a FROM t").fetchall() == [("x",)]
    assert not conn.in_transaction


def test_write_transaction_rolls_back_when_the_block_raises(tmp_path):
    conn, other = _two_connections(tmp_path)
    with pytest.raises(RuntimeError), write_transaction(conn):
        conn.execute("INSERT INTO t VALUES ('x')")
        raise RuntimeError("boom")
    assert other.execute("SELECT a FROM t").fetchall() == []
    assert not conn.in_transaction


def test_write_transaction_holds_the_write_lock_before_the_first_statement(tmp_path):
    """The lock is what makes a read inside the block safe to act on."""
    conn, other = _two_connections(tmp_path)
    with write_transaction(conn), pytest.raises(sqlite3.OperationalError, match="locked"):
        other.execute("BEGIN IMMEDIATE")
    other.execute("BEGIN IMMEDIATE")
    other.rollback()


def test_a_nested_block_is_undone_with_the_outer_one(tmp_path):
    conn, other = _two_connections(tmp_path)
    with pytest.raises(RuntimeError), write_transaction(conn):
        with write_transaction(conn):
            conn.execute("INSERT INTO t VALUES ('inner')")
        raise RuntimeError("outer fails after the inner block finished")
    assert other.execute("SELECT a FROM t").fetchall() == []


def test_a_nested_block_that_raises_undoes_only_its_own_statements(tmp_path):
    conn, other = _two_connections(tmp_path)
    with write_transaction(conn):
        conn.execute("INSERT INTO t VALUES ('outer')")
        with pytest.raises(RuntimeError), write_transaction(conn):
            conn.execute("INSERT INTO t VALUES ('inner')")
            raise RuntimeError("inner fails")
    assert other.execute("SELECT a FROM t").fetchall() == [("outer",)]


def _a_database_with_one_row(path):
    conn = open_db(str(path))
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("INSERT INTO t VALUES ('kept')")
    conn.commit()
    conn.close()


def test_open_readonly_cannot_write(tmp_path):
    """What ``doctor`` opens with: a write through it is SQLite's own refusal, not a promise."""
    _a_database_with_one_row(tmp_path / "m.db")
    conn = open_readonly(str(tmp_path / "m.db"))
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            conn.execute("INSERT INTO t VALUES ('written')")
        assert conn.execute("SELECT vec_version()").fetchone()[0]
    finally:
        conn.close()


def test_open_readonly_creates_no_missing_file(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        open_readonly(str(tmp_path / "absent.db"))

    assert list(tmp_path.iterdir()) == []


def test_open_readonly_opens_the_path_it_is_given_whatever_its_characters(tmp_path):
    """In a bare ``file:`` URI a ``#`` starts the fragment: SQLite would open the path cut off
    there, read-write, creating it. The path is percent-encoded, so it is the file that opens."""
    folder = tmp_path / "a #1 %20"
    folder.mkdir()
    _a_database_with_one_row(folder / "m.db")

    conn = open_readonly(str(folder / "m.db"))
    try:
        assert conn.execute("SELECT a FROM t").fetchone()[0] == "kept"
    finally:
        conn.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a #1 %20"]


def test_open_refuses_an_sqlite_below_the_floor_naming_both_versions(tmp_path, monkeypatch):
    """The check runs before anything else: no file is created, and the read-only open refuses
    the same way. ``sqlite_version_info`` is read at call time, so patching the module attribute
    is enough."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.40.1")
    path = tmp_path / "m.db"
    with pytest.raises(SQLiteTooOld) as raised:
        open_db(str(path))
    assert str(raised.value) == (
        "SQLite 3.40.1 is too old: Morgan needs 3.42.0 for FTS5's secure-delete"
    )
    assert not path.exists()
    with pytest.raises(SQLiteTooOld, match=r"3\.40\.1"):
        open_readonly(str(path))
    assert MIN_SQLITE == (3, 42, 0)


def test_every_connection_opens_with_secure_delete_on(tmp_path):
    for conn in (open_db(str(tmp_path / "m.db")), open_db(":memory:")):
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1


def test_the_floor_version_is_accepted_and_the_version_just_below_it_is_refused(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 42, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.42.0")
    conn = open_db(str(tmp_path / "at_floor.db"))
    assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1

    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 41, 9))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.41.9")
    with pytest.raises(SQLiteTooOld, match=r"3\.41\.9"):
        open_db(str(tmp_path / "below_floor.db"))
