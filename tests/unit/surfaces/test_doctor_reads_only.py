"""doctor reads the database and changes nothing in it.

It is the command the owner runs because something is wrong, so it must not be one more thing
that changes the file. Building the stores to count rows created every missing table, ran the
light migration steps -- on a database another install still writes to -- and fixed the vector
table at whatever width was set that minute, before any embedding space was registered.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import migrations, snapshot
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor
from tests.fakes import model_server

#: ``memories`` and ``facts`` as every older Morgan created them, and ``vec_items`` and
#: ``fts_memories`` without the columns migration step 6 adds.
_VERSION_TWO_DDL = (
    """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
        kind TEXT NOT NULL, source TEXT NOT NULL, content TEXT NOT NULL,
        importance REAL NOT NULL, entities TEXT NOT NULL, created_at TEXT
    )
    """,
    """
    CREATE TABLE facts (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
        subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
        source TEXT NOT NULL, confidence REAL NOT NULL, valid_from TEXT, valid_to TEXT,
        superseded_by TEXT, last_confirmed TEXT
    )
    """,
    """
    CREATE VIRTUAL TABLE vec_items USING vec0(
        embedding float[4] distance_metric=cosine, user_id TEXT, project TEXT
    )
    """,
    """
    CREATE VIRTUAL TABLE fts_memories USING fts5(
        memory_id UNINDEXED, user_id UNINDEXED, project UNINDEXED, content,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
)


@pytest.fixture
def chat() -> Iterator[str]:
    """A chat server that answers at once, so the probe these tests do not look at costs
    nothing and no default endpoint -- possibly a real server here -- is contacted."""
    with model_server() as url:
        yield url


def _settings(data_dir: Path, chat: str, **fields: Any) -> Settings:
    return Settings(
        **{
            "data_dir": str(data_dir),
            "embedding_backend": "hash",
            "llm_endpoint": chat,
            "doctor_probe_timeout_seconds": 5.0,
            **fields,
        }
    )


async def _report(settings: Settings, *, project: str = "p") -> dict[str, Any]:
    return await build_doctor_report(settings, project=project, all_projects=False)


def _a_version_two_database(path: Path) -> None:
    conn = open_db(str(path))
    for statement in _VERSION_TWO_DDL:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO memories VALUES ('m1', 'owner', 'default', 'episodic', 'user_stated', "
        "'the first memory', 0.5, '[]', '2026-09-01T00:00:00+00:00')"
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()


def _schema(path: Path) -> tuple[int, list[tuple[str, str, str]]]:
    conn = sqlite3.connect(path)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        rows = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
        return version, [(str(t), str(n), str(s)) for t, n, s in rows]
    finally:
        conn.close()


async def test_doctor_on_a_version_two_database_runs_no_step_and_creates_no_table(tmp_path, chat):
    db = tmp_path / "morgan.db"
    _a_version_two_database(db)
    before = _schema(db)

    report = await _report(_settings(tmp_path, chat))

    assert _schema(db) == before
    migration = report["migration"]
    assert (migration["user_version"], migration["code_version"]) == (2, len(migrations._STEPS))
    assert migration["pending"] == [
        {"number": s.number, "name": s.name, "heavy": s.heavy}
        for s in migrations._STEPS
        if s.number > 2
    ]
    # Read, not built: the tables step 3 would create are named absent, never counted as 0.
    assert report["projects"] is None and "projects" in report["projects_reason"]
    assert report["embedding_space"] is None
    assert "embedding_spaces" in report["embedding_space_reason"]
    assert report["rows_all_projects"]["memories"] == 1
    assert report["rows_all_projects"]["history"] is None


def _journal(path: Path) -> tuple[int, int]:
    """Header bytes 18 and 19, the file format's write and read versions: 1 for a rollback
    journal, 2 for WAL. Read from the bytes, so looking changes nothing."""
    header = path.read_bytes()[:20]
    return header[18], header[19]


def _sidecars(db: Path) -> list[str]:
    return sorted(
        p.name for p in db.parent.iterdir() if p.name in (f"{db.name}-wal", f"{db.name}-shm")
    )


def _wal_bytes(db: Path) -> int:
    return Path(f"{db}-wal").stat().st_size


def _block_the_wal(db: Path) -> None:
    """A directory where the write-ahead log of *db* belongs, so no reader can open it.

    The ``-wal``, not the ``-shm``: a directory in the way of the shared-memory file stops a
    read-only open on Windows only -- on Linux the connection falls back to a wal-index of its
    own and reads the database anyway. A ``-wal`` that cannot be opened stops it on both.
    """
    Path(f"{db}-wal").mkdir()


async def _store(settings: Settings, contents: list[str]) -> Path:
    """Store *contents* in project ``p`` and close; the file is left in WAL mode, with no
    ``-wal`` or ``-shm`` beside it once the last connection has closed."""
    ctx = build_memory_context(settings)
    try:
        for content in contents:
            await ctx.gate.store(
                Memory(
                    user_id=settings.owner_user_id,
                    project="p",
                    content=content,
                    author_id=settings.owner_user_id,
                )
            )
    finally:
        ctx.conn.close()
    return Path(sqlite_path(settings.temporal_db_url))


async def test_a_rollback_journal_database_is_left_byte_for_byte_as_it_was(tmp_path, chat):
    """A snapshot is written in rollback-journal mode and ``morgan restore`` puts one in place
    without opening it to write: doctor, run next, must not be what switches it to WAL."""
    settings = _settings(tmp_path, chat)
    db = await _store(settings, ["the first memory", "the second memory"])
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    before = db.read_bytes()
    assert _journal(db) == (1, 1) and _sidecars(db) == []

    report = await _report(settings)

    assert report["database_error"] is None
    assert report["rows"]["memories"] == 2
    assert db.read_bytes() == before
    assert _journal(db) == (1, 1)
    assert _sidecars(db) == []


async def test_a_wal_database_another_process_is_writing_is_counted_with_its_wal(tmp_path, chat):
    """Rows a live writer has committed to the ``-wal`` and not yet checkpointed into the file
    are rows: a read-only open reads them, as every SQLite reader of a WAL database does."""
    settings = _settings(tmp_path, chat)
    db = await _store(settings, ["in the file"])
    writer = build_memory_context(settings)
    try:
        writer.conn.execute("PRAGMA wal_autocheckpoint=0")
        for content in ("in the wal", "also in the wal"):
            await writer.gate.store(
                Memory(
                    user_id=settings.owner_user_id,
                    project="p",
                    content=content,
                    author_id=settings.owner_user_id,
                )
            )
        assert _wal_bytes(db) > 0

        report = await _report(settings)
    finally:
        writer.conn.close()

    assert report["database_error"] is None
    assert report["rows"] == {"scope": "project 'p'", "memories": 3, "fts": 3, "vectors": 3}


async def test_a_wal_database_that_cannot_be_read_read_only_says_so_on_the_database_line(
    tmp_path, chat
):
    """A reader of a WAL database needs its sidecars; when one of them cannot be opened, the
    read-only open cannot proceed. doctor says so where the database is named, on every line
    that reads it, and never falls back to an open that writes."""
    settings = _settings(tmp_path, chat)
    db = await _store(settings, ["a memory"])
    _block_the_wal(db)
    before = db.read_bytes()

    report = await _report(settings)

    assert report["database_error"].startswith("failed to open database read-only: ")
    assert report["migration_reason"] == report["database_error"]
    assert report["rows"] is None
    assert "probe_errors" not in report and "count_errors" not in report
    assert db.read_bytes() == before


def _a_snapshot_in(folder: Path, settings: Settings, staging: Path) -> snapshot.SnapshotResult:
    """A snapshot of the settings' database, taken in *staging* and moved into *folder*: made
    without reading anything through *folder*'s name, so only doctor's reading is tested."""
    taken = snapshot.take(
        sqlite_path(settings.temporal_db_url),
        into=staging,
        reason="doctor-test",
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )
    folder.mkdir()
    shutil.move(taken.path, folder / taken.path.name)
    staging.rmdir()
    return taken


async def test_a_hash_in_the_snapshot_directory_creates_nothing_and_counts_its_snapshots(
    tmp_path, chat
):
    """doctor describes each snapshot through a ``file:`` URI; a ``#`` in the directory name
    must not cut that path short and have SQLite create a file where it was cut."""
    folder = tmp_path / "snap#dir"
    settings = _settings(tmp_path / "data", chat, snapshot_dir=str(folder))
    await _store(settings, ["a memory"])
    taken = _a_snapshot_in(folder, settings, tmp_path / "staging")
    listing = sorted(p.name for p in tmp_path.iterdir())

    report = await _report(settings)

    assert sorted(p.name for p in tmp_path.iterdir()) == listing
    assert report["snapshots"] == {
        "dir": str(folder),
        "count": 1,
        "newest": taken.path.name,
        "bytes": taken.bytes,
    }
    assert "probe_errors" not in report


async def test_a_missing_database_is_reported_and_not_created(tmp_path, chat):
    data_dir = tmp_path / "never-opened"
    settings = _settings(data_dir, chat)
    path = sqlite_path(settings.temporal_db_url)

    report = await _report(settings)

    assert report["database_error"] == f"no database yet at {path}"
    assert not data_dir.exists()
    assert report["rows"] is None and report["rows_all_projects"] is None
    assert report["migration"] is None
    # What the library can do does not depend on a database existing.
    assert report["sqlite_vec"] and report["fts5"] is True
    assert f"database: {report['database']} (no database yet)" in _render_doctor(report)


async def test_the_snapshots_projects_and_code_roots_are_read_as_they_are(tmp_path, chat):
    settings = _settings(
        tmp_path / "data",
        chat,
        code_roots=[str(tmp_path / "code"), str(tmp_path / "gone")],
    )
    (tmp_path / "code").mkdir()
    build_memory_context(settings).conn.close()
    conn = open_db(sqlite_path(settings.temporal_db_url))
    conn.execute(
        "INSERT INTO projects (name, classification, capture_enabled, consolidate_enabled, "
        "created_at) VALUES ('Morgan', 'personal', 1, 0, '2026-09-21T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    taken = snapshot.take(
        sqlite_path(settings.temporal_db_url),
        into=Path(settings.snapshot_dir),
        reason="doctor-test",
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    report = await _report(settings)

    assert report["snapshots"] == {
        "dir": settings.snapshot_dir,
        "count": 1,
        "newest": taken.path.name,
        "bytes": taken.bytes,
    }
    assert report["projects"] == [
        {
            "name": "Morgan",
            "classification": "personal",
            "capture_enabled": True,
            "paused_until": None,
            "retention_days": None,
            "consolidate_enabled": False,
        }
    ]
    assert report["code_roots"] == [
        {"path": str(tmp_path / "code"), "is_directory": True},
        {"path": str(tmp_path / "gone"), "is_directory": False},
    ]
    assert report["migration"]["pending"] == []
    lines = _render_doctor(report).splitlines()
    assert "project 'Morgan': personal, capture on, consolidate off" in lines
    assert f"code_root: {tmp_path / 'gone'} (not a directory)" in lines


def test_the_env_files_render_one_per_line():
    rendered = _render_doctor(
        {
            "env_files": [
                {"path": "/home/you/.config/morgan/.env", "present": True},
                {"path": "/home/you/code/.env", "present": False},
            ]
        }
    )

    assert rendered.splitlines() == [
        "env_file: /home/you/.config/morgan/.env (present)",
        "env_file: /home/you/code/.env (absent)",
    ]
