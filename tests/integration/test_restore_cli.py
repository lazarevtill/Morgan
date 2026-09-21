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

from morgan_brain.config import get_settings
from morgan_brain.surfaces.cli.__main__ import main


@pytest.fixture(autouse=True)
def _fresh_settings() -> Any:
    """``get_settings`` is ``lru_cache``d; these tests change the environment under it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
