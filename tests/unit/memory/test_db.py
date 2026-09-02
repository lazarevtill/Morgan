from morgan_brain.memory.db import open_db


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
