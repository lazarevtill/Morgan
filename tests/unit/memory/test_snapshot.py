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


async def _closed_db_with_one_memory(path: Path) -> str:
    """Like ``_db_with_one_memory``, but with its connection explicitly closed -- so no
    lingering handle keeps a memory-mapped ``-shm`` (or an inconsistent ``-wal``) around
    while a test injects its own sidecar files next to *path*."""
    module = build_memory_module(str(path))
    await module.store(Memory(user_id="u", project="p", content="Harbor upgrade plan"))
    module._conn.commit()
    module._conn.close()
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


async def test_the_old_databases_wal_is_unlinked_before_the_swap_not_after(tmp_path, monkeypatch):
    """Review finding (IMPORTANT): removing ``db_path``'s own ``-wal``/``-shm`` *after*
    ``os.replace`` left a window where a crash could pair the just-restored file with the
    *old* database's WAL on a later open. Proven by recording call order directly -- a plain
    "does the file still exist at replace time" check is not reliable here, because
    ``take()``'s own connection close already folds and clears a stale WAL as an incidental
    side effect in a single-connection test, regardless of ``restore()``'s own ordering --
    so this pins down the one thing that actually changed: ``restore()`` must call
    ``Path(f"{db_path}-wal").unlink`` before it calls ``os.replace``, not after.
    """
    source_db = await _db_with_one_memory(tmp_path / "source.db")
    source_snapshot = snapshot.take(source_db, into=tmp_path / "snaps", reason="seed", clock=_clock)

    live_db = await _closed_db_with_one_memory(tmp_path / "morgan.db")
    wal_path = Path(f"{live_db}-wal")

    events: list[str] = []
    real_unlink = Path.unlink
    real_replace = snapshot.os.replace

    def _tracking_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == wal_path:
            events.append("unlink-wal")
        real_unlink(self, *args, **kwargs)

    def _tracking_replace(src: object, dst: object) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(Path, "unlink", _tracking_unlink)
    monkeypatch.setattr(snapshot.os, "replace", _tracking_replace)

    snapshot.restore(live_db, source=source_snapshot.path, into=tmp_path / "safety", clock=_clock)

    assert "unlink-wal" in events and "replace" in events
    assert events.index("unlink-wal") < events.index("replace")


async def test_a_failure_mid_restore_leaves_no_restoring_temp_file(tmp_path, monkeypatch):
    """Review finding (MINOR, folded into the same fix): ``restore`` copies *source* to a
    scratch file (``f"{db_path}.restoring"``) next to ``db_path`` before swapping it in.
    Any failure between that copy and the swap -- simulated here at ``os.replace`` itself --
    must not leave that scratch file behind."""
    source_db = await _db_with_one_memory(tmp_path / "source.db")
    source_snapshot = snapshot.take(source_db, into=tmp_path / "snaps", reason="seed", clock=_clock)

    # Closed, not `_db_with_one_memory`'s lingering-open connection: an open connection can
    # itself make db_path's own `-wal` unlink raise a (real, separately-behaved) Windows
    # PermissionError, which would test that interaction instead of the one this test is for.
    live_db = await _closed_db_with_one_memory(tmp_path / "morgan.db")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(snapshot.os, "replace", _explode)

    with pytest.raises(OSError, match="simulated failure"):
        snapshot.restore(
            live_db, source=source_snapshot.path, into=tmp_path / "safety", clock=_clock
        )

    assert not _exists(Path(f"{live_db}.restoring"))


def _clock() -> datetime:
    return datetime(2026, 9, 21, 10, 15, tzinfo=UTC)


def _exists(path: Path) -> bool:
    """A plain function, not an inline ``Path.exists()``: ruff's ASYNC240 flags a blocking
    pathlib call written directly in an ``async def`` test body."""
    return path.exists()


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
