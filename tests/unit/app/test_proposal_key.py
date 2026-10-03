import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from morgan_brain.memory.store.proposal_key import create_schema, read_key
from tests.unit.app.test_working_context import event, stack


def test_missing_key_read_is_fail_closed_and_does_not_initialize():
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="needs upgrading"):
            read_key(conn)
        assert conn.total_changes == 0
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
        with pytest.raises(ValueError, match="write transaction"):
            create_schema(conn)
    finally:
        conn.close()


def test_concurrent_initializers_keep_one_durable_key(tmp_path):
    path = tmp_path / "key.db"

    def initialize(_):
        conn = sqlite3.connect(path, timeout=10)
        try:
            conn.execute("BEGIN IMMEDIATE")
            create_schema(conn)
            conn.commit()
            return read_key(conn)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(initialize, range(4)))
    assert len(set(keys)) == 1
    assert len(keys[0]) == 32
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        assert read_key(conn) == keys[0]
        assert conn.total_changes == 0
    finally:
        conn.close()


async def test_version_eleven_database_upgrades_key_without_touching_sources(tmp_path):
    path = tmp_path / "old.db"
    conn, gate, _, _ = stack(path)
    await event(gate)
    conn.execute("DROP TABLE proposal_integrity")
    conn.execute("PRAGMA user_version=11")
    conn.commit()
    before = conn.execute("SELECT * FROM memories").fetchall()
    conn.close()
    conn, gate, _, _ = stack(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 12
        assert len(read_key(conn)) == 32
        assert conn.execute("SELECT * FROM memories").fetchall() == before
    finally:
        conn.close()
