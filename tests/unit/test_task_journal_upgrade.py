"""The additive receipt column preserves v1 tasks and rolls back with its version marker."""

import sqlite3

import pytest

from morgan_brain.tasks import TaskJournal, TaskKey


def legacy(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        "PRAGMA application_id=0x4D54534B; PRAGMA user_version=1;"
        "CREATE TABLE tasks (user_id TEXT, project TEXT, task_id TEXT, action TEXT, payload TEXT, "
        "status TEXT, attempt_id TEXT, cancel_requested INTEGER, result TEXT, created_at TEXT, "
        "started_at TEXT, finished_at TEXT, PRIMARY KEY(user_id,project,task_id));"
        "INSERT INTO tasks VALUES ('owner','project','one','uppercase-v1','old input',"
        "'outcome_unknown','attempt',1,NULL,'created','started','finished');"
    )
    conn.close()


def test_v1_upgrade_preserves_every_existing_field(tmp_path):
    path = tmp_path / "tasks.db"
    legacy(path)
    conn = sqlite3.connect(path)
    before = conn.execute("SELECT * FROM tasks").fetchone()
    conn.close()
    journal = TaskJournal(path)
    try:
        task = journal.get(TaskKey("owner", "project", "one"))
        assert task.status == "outcome_unknown" and task.attempt_id == "attempt"
        assert task.cancel_requested and task.payload == "old input" and task.receipt_json is None
    finally:
        journal.close()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT * FROM tasks").fetchone() == (*before, None)
    finally:
        conn.close()


def test_failed_upgrade_commit_keeps_v1_schema_and_marker(tmp_path, monkeypatch):
    path = tmp_path / "tasks.db"
    legacy(path)
    connect = sqlite3.connect

    def deny_commit(action, argument, _second, _db, _trigger):
        if action == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def rejecting_connection(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_authorizer(deny_commit)
        return conn

    monkeypatch.setattr(sqlite3, "connect", rejecting_connection)
    with pytest.raises(sqlite3.DatabaseError):
        TaskJournal(path)
    conn = connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert "receipt_json" not in {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert conn.execute("SELECT status FROM tasks").fetchone()[0] == "outcome_unknown"
    finally:
        conn.close()
