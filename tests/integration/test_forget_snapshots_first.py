"""Forget takes a snapshot before it erases, on both surfaces.

Forget is project-grain erasure with no undo. The snapshot is the undo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings, settings_for
from morgan_brain.memory import migrations
from morgan_brain.memory.migrations import DatabaseNeedsMigration
from morgan_brain.memory.store import sessions as sessions_store
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.models import Memory, Session, Turn, session_id_of
from morgan_brain.surfaces.cli.__main__ import main
from morgan_brain.surfaces.cli.commands import cmd_forget

#: A fixed ISO timestamp for the session and turns this file seeds.
_NOW = "2026-09-22T10:00:00.000Z"


def test_cli_forget_writes_a_snapshot_before_erasing(tmp_path, monkeypatch, capsys):
    _a_database_with_one_memory(tmp_path, monkeypatch)

    assert main(["forget", "--project", "p", "--json"]) == 0

    payload = _json(capsys)
    snapshot = Path(payload["snapshot"])
    assert snapshot.is_file()
    assert _memories_in(snapshot) == 1 and payload["memories"] == 1


def test_cli_forget_reports_the_sessions_and_turns_it_erased(tmp_path, monkeypatch, capsys):
    """A privacy operation that erased a captured session must say so: printing only
    ``memories``/``facts``/``history`` while sessions, turns and digests quietly went too
    would understate what ``forget`` just did."""
    _a_database_with_one_memory_and_one_session(tmp_path, monkeypatch)

    assert main(["forget", "--project", "p", "--json"]) == 0

    payload = _json(capsys)
    assert payload["memories"] == 1
    assert payload["sessions"] == 1
    assert payload["turns"] == 2
    assert payload["digests"] == 0


def test_cli_forget_prints_only_the_nonzero_archive_counts(tmp_path, monkeypatch, capsys):
    """The human render names ``sessions`` and ``turns``, which this forget erased, and
    leaves out ``digests``, which it did not."""
    _a_database_with_one_memory_and_one_session(tmp_path, monkeypatch)

    assert main(["forget", "--project", "p"]) == 0

    out = capsys.readouterr().out
    assert "sessions=1" in out
    assert "turns=2" in out
    assert "digests=" not in out


async def test_the_mcp_tool_does_the_same(tmp_path, monkeypatch):
    settings = _a_database_with_one_memory(tmp_path, monkeypatch)

    result = await cmd_forget(argparse.Namespace(all_projects=False), settings, "p")

    assert _is_file(Path(result["snapshot"]))


async def test_a_read_only_database_refuses_before_any_snapshot_is_written(tmp_path, monkeypatch):
    """A read-only context refuses *before* the snapshot is taken, so a forget that cannot
    run leaves no undo file behind either."""
    settings = _a_read_only_database(tmp_path, monkeypatch)

    with pytest.raises(DatabaseNeedsMigration):
        await cmd_forget(argparse.Namespace(all_projects=False), settings, "p")

    assert _snapshot_files(Path(settings.snapshot_dir)) == []


def _a_database_with_one_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORGAN_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    settings = settings_for("cli")

    async def seed() -> None:
        ctx = build_memory_context(settings)
        try:
            await ctx.gate.store(
                Memory(
                    user_id=settings.owner_user_id,
                    project="p",
                    content="a harbor mirror note",
                )
            )
        finally:
            ctx.conn.close()

    _run_isolated(seed())
    return settings


def _a_database_with_one_memory_and_one_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    """The one-memory database above, plus one session of two turns in the same project,
    written through the session store's own write path, the way
    ``tests/unit/memory/test_forget_reaches_every_project_keyed_table.py`` seeds the archive
    tables for its own forget tests."""
    settings = _a_database_with_one_memory(tmp_path, monkeypatch)

    async def seed() -> None:
        ctx = build_memory_context(settings)
        try:
            session = Session(
                id=session_id_of("claude-code", "s-p"),
                user_id=settings.owner_user_id,
                project="p",
                harness="claude-code",
                native_id="s-p",
                source_path="/transcripts/s-p.jsonl",
                cwd="/src/p",
                project_source="git",
                started_at=_NOW,
                reader_version=1,
                gate_version=1,
            )
            with write_transaction(ctx.conn):
                sessions_store.upsert_session(ctx.conn, session, now=_NOW)
                sessions_store.insert_turns(
                    ctx.conn,
                    [
                        Turn(
                            session_id=session.id,
                            user_id=settings.owner_user_id,
                            project="p",
                            native_key="a:0",
                            role="user",
                            text="a harbor mirror turn",
                            ts=_NOW,
                        ),
                        Turn(
                            session_id=session.id,
                            user_id=settings.owner_user_id,
                            project="p",
                            native_key="b:0",
                            role="user",
                            text="another harbor mirror turn",
                            ts=_NOW,
                        ),
                    ],
                )
        finally:
            ctx.conn.close()

    _run_isolated(seed())
    return settings


def _a_read_only_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """One memory in project ``p`` at ``user_version`` 2, under code whose step 3 is heavy --
    the same fixture shape as ``test_read_only_refuses_before_work.py``."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORGAN_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")

    version_two = migrations._STEPS[:2]
    monkeypatch.setattr(migrations, "_STEPS", version_two)
    settings = settings_for("cli")

    async def seed() -> None:
        ctx = build_memory_context(settings)
        try:
            await ctx.gate.store(
                Memory(
                    user_id=settings.owner_user_id,
                    project="p",
                    content="a harbor mirror note",
                )
            )
        finally:
            ctx.conn.close()

    _run_isolated(seed())

    heavy = migrations.Step(3, "a heavy step", True, lambda c, s: None)
    monkeypatch.setattr(migrations, "_STEPS", (*version_two, heavy))
    return settings_for("cli")


def _run_isolated(coro: Any) -> Any:
    """Run *coro* to completion whether or not an event loop is already running here.

    A sync test has none; an ``async def`` test under ``asyncio_mode = auto`` already has
    one, and ``asyncio.run`` refuses to nest inside it.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the calling thread below
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _is_file(path: Path) -> bool:
    """A plain function, not an inline ``Path.is_file()``: ruff's ASYNC240 flags a blocking
    pathlib call written directly in an ``async def`` test body."""
    return path.is_file()


def _snapshot_files(directory: Path) -> list[Path]:
    """Same reason as ``_is_file`` -- a plain function keeps the blocking glob out of the
    ``async def`` test body."""
    return list(directory.glob("*.db"))


def _memories_in(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    finally:
        conn.close()


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)
