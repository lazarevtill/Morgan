"""Restoring a snapshot puts back exactly what was in it.

The safety copy is taken first and never skipped: a restore is the one command whose mistake
cannot be undone by running it again.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.surfaces.cli.__main__ import main


def test_restore_returns_the_snapshot_rows(tmp_path, monkeypatch, capsys):
    _db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)

    assert main(["restore", str(snap), "--yes", "--json"]) == 0

    out = _json(capsys)
    assert out["after"]["memories"] == 1 != out["before"]["memories"]
    assert out["safety_snapshot"].endswith("-before-restore.db")


def test_a_stray_wal_is_removed(tmp_path, monkeypatch, capsys):
    db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)
    Path(db + "-wal").write_bytes(b"stale")

    main(["restore", str(snap), "--yes", "--json"])

    assert not Path(db + "-wal").exists()


def test_a_snapshot_from_a_newer_morgan_is_refused(tmp_path, monkeypatch, capsys):
    _db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)
    with sqlite3.connect(snap) as conn:
        conn.execute("PRAGMA user_version = 99")

    assert main(["restore", str(snap), "--yes", "--json"]) == 1
    assert "user_version 99" in _json(capsys)["error"]


def test_restore_without_yes_previews_and_does_not_touch_the_database(
    tmp_path, monkeypatch, capsys
):
    """Step 3 of the brief: without ``--yes`` the CLI prints what it would replace and
    returns 2 -- nothing is read from the snapshot and nothing changes."""
    db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)

    assert main(["restore", str(snap), "--json"]) == 2

    out = _json(capsys)
    # settings.temporal_db_url stores a posix-style path (see config.py); compare as paths.
    assert Path(out["database"]) == Path(db)
    with sqlite3.connect(db) as conn:
        # Both memories the helper stored are still there -- nothing was replaced.
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 2


def test_the_snapshot_survives_a_restore_byte_identical_and_still_listed(
    tmp_path, monkeypatch, capsys
):
    """Review finding (CRITICAL): ``restore`` must never delete the snapshot it restores
    from (SPEC-phase0 SS3.2). It only ever reads *source* -- ``morgan snapshot --list`` must
    still show it afterwards, unchanged."""
    _db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)
    original_bytes = snap.read_bytes()

    assert main(["restore", str(snap), "--yes", "--json"]) == 0
    _json(capsys)  # drain restore's own output

    assert snap.exists()
    assert snap.read_bytes() == original_bytes

    assert main(["snapshot", "--list", "--json"]) == 0
    listed = [Path(s["path"]).name for s in _json(capsys)["snapshots"]]
    assert snap.name in listed


def test_a_live_wal_from_the_pre_restore_database_does_not_leak_into_the_restore(
    tmp_path, monkeypatch, capsys
):
    """Review finding (IMPORTANT, plan-mandated): a *genuinely valid* WAL left behind by the
    pre-restore database -- not merely stray/garbage bytes, which SQLite already ignores on
    its own -- must not be consultable once ``db_path`` holds the restored content. This is
    only true if ``db_path``'s own ``-wal``/``-shm`` are dropped *before* the swap, not after:
    proves the fixed ordering closes the crash window the review flagged.

    A closed SQLite connection auto-checkpoints and deletes its own WAL when it is the last
    one open, so a genuine (non-empty, valid-header) WAL is captured here via a bystander
    connection that is only closed *after* the bytes are read back out, then written back to
    disk with no connection open at all -- as if the writer had crashed before a clean
    checkpoint, rather than exited normally.
    """
    db, snap = _database_then_snapshot_then_more_rows(tmp_path, monkeypatch, capsys)

    bystander = sqlite3.connect(db)
    bystander.execute("SELECT 1")
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(
        "INSERT INTO memories "
        "(id, user_id, project, kind, source, content, importance, entities, created_at) "
        "VALUES ('never-checkpointed', 'owner', 'p', 'episodic', 'user_stated', "
        "'never checkpointed', 0.5, '[]', NULL)"
    )
    writer.commit()
    writer.close()
    wal_path = Path(db + "-wal")
    assert wal_path.exists() and wal_path.stat().st_size > 0
    real_wal_bytes = wal_path.read_bytes()
    bystander.close()  # the last connection closing checkpoints-and-deletes its own copy

    # Put the genuinely valid WAL back with no connection open at all -- as a crash, not a
    # clean shutdown, would leave it.
    wal_path.write_bytes(real_wal_bytes)
    assert wal_path.exists()

    assert main(["restore", str(snap), "--yes", "--json"]) == 0

    assert not wal_path.exists()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT content FROM memories").fetchall()
    assert rows == [("the first memory",)]


def _database_then_snapshot_then_more_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[str, Path]:
    """Build a database with one memory, snapshot it, then store a second memory on top.

    Returns ``(db_path, snapshot_path)``. ``MORGAN_DATA_DIR``/``MORGAN_SNAPSHOT_DIR`` point at
    *tmp_path*, and ``MORGAN_EMBEDDING_BACKEND=hash`` so nothing calls a live model.
    """
    data_dir = tmp_path / "data"
    snapshot_dir = tmp_path / "snapshots"
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_SNAPSHOT_DIR", str(snapshot_dir))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")

    assert main(["remember", "the first memory", "--json"]) == 0
    capsys.readouterr()  # discard -- only the snapshot call's output is needed below

    assert main(["snapshot", "--reason", "seed", "--json"]) == 0
    snap = Path(_json(capsys)["path"])

    assert main(["remember", "the second memory", "--json"]) == 0
    capsys.readouterr()  # discard -- leave capsys clean for the test body's own read

    return str(data_dir / "morgan.db"), snap


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)
