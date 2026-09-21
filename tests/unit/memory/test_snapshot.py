"""A snapshot is a copy that can be restored, or it is not a snapshot.

VACUUM INTO writes a consistent copy while other processes hold the file open under WAL, which
a filesystem copy of a live database does not. Every snapshot is verified before it is
announced: a corrupt one announced as good is worse than none.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.memory import snapshot
from morgan_brain.models import Memory
from tests.unit.memory.conftest import build_memory_module


async def _db_with_one_memory(path: Path) -> str:
    module = build_memory_module(str(path))
    await module.store(Memory(user_id="u", project="p", content="Harbor upgrade plan"))
    module._conn.commit()
    return str(path)


async def test_a_snapshot_holds_the_same_rows(tmp_path):
    db = await _db_with_one_memory(tmp_path / "morgan.db")

    result = snapshot.take(db, into=tmp_path / "snapshots", reason="test", clock=_clock)

    assert result.path.is_file()
    assert result.counts["memories"] == 1
    assert result.user_version == _version_of(db)
    assert _memory_rows(result.path) == _memory_rows(db)


async def test_the_name_carries_the_time_and_the_reason(tmp_path):
    db = await _db_with_one_memory(tmp_path / "morgan.db")
    result = snapshot.take(db, into=tmp_path / "s", reason="migrate", clock=_clock)
    assert result.path.name == "morgan-20260921T101500Z-migrate.db"


async def test_no_room_refuses_by_name(tmp_path, monkeypatch):
    db = await _db_with_one_memory(tmp_path / "morgan.db")
    monkeypatch.setattr(snapshot, "_free_bytes", lambda _p: 1)

    with pytest.raises(snapshot.NotEnoughSpace) as exc:
        snapshot.take(db, into=tmp_path / "s", reason="test", clock=_clock)

    assert "MORGAN_SNAPSHOT_DIR" in str(exc.value)
    assert not list((tmp_path / "s").glob("*.db"))


async def test_a_copy_that_fails_its_check_is_deleted(tmp_path, monkeypatch):
    db = await _db_with_one_memory(tmp_path / "morgan.db")
    monkeypatch.setattr(snapshot, "_quick_check", lambda _p: "malformed")

    with pytest.raises(snapshot.SnapshotCorrupt):
        snapshot.take(db, into=tmp_path / "s", reason="test", clock=_clock)

    assert not list((tmp_path / "s").glob("*.db"))


async def test_a_same_second_collision_gets_a_free_name_and_never_overwrites(tmp_path):
    """``restore``'s safety snapshot always uses the fixed reason ``before-restore``, so a
    second one in the same UTC second is not hypothetical -- it is the first caller that
    hits it. The first snapshot must survive untouched under its own name."""
    db = await _db_with_one_memory(tmp_path / "morgan.db")
    first = snapshot.take(db, into=tmp_path / "s", reason="before-restore", clock=_clock)

    module = build_memory_module(db)
    await module.store(Memory(user_id="u", project="p", content="a second memory"))
    module._conn.commit()

    second = snapshot.take(db, into=tmp_path / "s", reason="before-restore", clock=_clock)

    assert first.path.name == "morgan-20260921T101500Z-before-restore.db"
    assert second.path.name == "morgan-20260921T101500Z-before-restore-2.db"
    assert first.path.exists() and second.path.exists()
    # The first file was never overwritten with the later state.
    assert len(_memory_rows(first.path)) == 1
    assert len(_memory_rows(second.path)) == 2


def _clock() -> datetime:
    return datetime(2026, 9, 21, 10, 15, tzinfo=UTC)


def _read_only(path: str | Path) -> sqlite3.Connection:
    """A read-only connection to *path* -- describing a copy must never mutate it."""
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def _version_of(path: str | Path) -> int:
    conn = _read_only(path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def _memory_rows(path: str | Path) -> list[tuple]:
    conn = _read_only(path)
    try:
        return conn.execute(
            "SELECT id, user_id, project, content FROM memories ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
