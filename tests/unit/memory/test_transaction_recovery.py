"""A rejected COMMIT must not leak pending work or a lock into the next request."""

import sqlite3

import pytest

from morgan_brain.memory.store.transactions import read_transaction, write_transaction


def test_deferred_constraint_failure_rolls_back_and_releases_writer(tmp_path):
    path = tmp_path / "commit.db"
    conn = sqlite3.connect(path)
    other = sqlite3.connect(path, timeout=0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            "CREATE TABLE parent(id PRIMARY KEY);"
            "CREATE TABLE child(id REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED);"
        )
        with (
            pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"),
            write_transaction(conn),
        ):
            conn.execute("INSERT INTO child VALUES (1)")
            assert conn.execute("SELECT COUNT(*) FROM child").fetchone()[0] == 1
        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM child").fetchone()[0] == 0
        with write_transaction(other):
            other.execute("INSERT INTO parent VALUES (1)")
        with write_transaction(conn):
            conn.execute("INSERT INTO child VALUES (1)")
        assert other.execute("SELECT COUNT(*) FROM child").fetchone()[0] == 1
    finally:
        other.close()
        conn.close()


@pytest.mark.parametrize("transaction", [read_transaction, write_transaction])
def test_rejected_commit_closes_owned_transaction(transaction):
    conn = sqlite3.connect(":memory:")

    def deny_commit(action, argument, _second, _database, _trigger):
        if action == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        conn.set_authorizer(deny_commit)
        with (
            pytest.raises(sqlite3.DatabaseError, match="not authorized"),
            transaction(conn),
        ):
            conn.execute("SELECT 1")
        assert not conn.in_transaction
        conn.set_authorizer(None)
        with transaction(conn):
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        assert not conn.in_transaction
    finally:
        conn.close()
