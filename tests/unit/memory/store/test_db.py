import sqlite3

import pytest

from morgan_brain.memory.store.db import open_db, write_transaction


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
