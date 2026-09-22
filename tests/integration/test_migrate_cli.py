"""`morgan migrate` runs what an open may not, behind a snapshot; a dry run changes nothing.

The fixture is what every older Morgan wrote -- a database at ``user_version`` 2 --
opened by code whose third step is heavy. Until ``migrate`` runs that step the database is
read-only on both surfaces, and every write says why.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from morgan_brain.memory import migrations
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.surfaces.cli.__main__ import main
from morgan_brain.surfaces.mcp_server import build_server
from tests.fakes import _unit_vector, flaky_model_server, model_server
from tests.unit.memory.conftest import a_version_five_database

_BLOCKED = "writes are blocked until `morgan migrate` runs: 1 step pending (3 a heavy step)"


def _touch_every_memory(conn: sqlite3.Connection, stores: migrations.Stores) -> dict[str, int]:
    """Heavy, because it rewrites rows -- each one to itself, so nothing is lost."""
    return {"memories": conn.execute("UPDATE memories SET content = content").rowcount}


def test_a_dry_run_lists_the_pending_step_and_changes_nothing(tmp_path, monkeypatch, capsys):
    db, snapshots = _a_version_two_database(tmp_path, monkeypatch, capsys, _touch_every_memory)
    before = Path(db).read_bytes()

    assert main(["migrate", "--dry-run", "--json"]) == 0

    out = _json(capsys)
    assert out["pending"] == [{"number": 3, "name": "a heavy step", "heavy": True}]
    assert (out["user_version"], out["code_version"]) == (2, 3)
    assert Path(db).read_bytes() == before
    assert list(snapshots.glob("*.db")) == []


def test_migrate_snapshots_first_and_reaches_the_codes_version(tmp_path, monkeypatch, capsys):
    db, snapshots = _a_version_two_database(tmp_path, monkeypatch, capsys, _touch_every_memory)

    assert main(["migrate", "--json"]) == 0

    out = _json(capsys)
    snapshot = Path(out["snapshot"])
    assert snapshot.parent == snapshots and snapshot.name.endswith("-migrate.db")
    assert _version(str(snapshot)) == 2
    assert _version(db) == len(migrations._STEPS) == 3
    assert out["steps"] == [
        {"number": 3, "name": "a heavy step", "heavy": True, "counts": {"memories": 1}}
    ]
    assert out["before"] == out["after"] and out["after"]["memories"] == 1
    assert out["quick_check"] == "ok"

    assert main(["remember", "written after the migration", "--project", "p", "--json"]) == 0


def test_a_failing_step_leaves_the_version_the_rows_and_the_snapshot(tmp_path, monkeypatch, capsys):
    def erase_then_fail(conn: sqlite3.Connection, stores: migrations.Stores) -> None:
        conn.execute("DELETE FROM memories")
        raise RuntimeError("the heavy step failed")

    db, snapshots = _a_version_two_database(tmp_path, monkeypatch, capsys, erase_then_fail)

    assert main(["migrate", "--json"]) == 1

    error = _json(capsys)["error"]
    assert "the heavy step failed" in error
    assert _version(db) == 2 and _memories(db) == 1
    [snapshot] = snapshots.glob("*-migrate.db")
    assert snapshot.name in error
    assert _memories(str(snapshot)) == 1


def test_a_write_before_migrate_is_refused_by_name_and_a_read_answers(
    tmp_path, monkeypatch, capsys
):
    _a_version_two_database(tmp_path, monkeypatch, capsys, _touch_every_memory)

    assert main(["remember", "not yet", "--project", "p", "--json"]) == 1
    assert _json(capsys)["error"] == _BLOCKED

    assert main(["recall", "first memory", "--project", "p", "--json"]) == 0
    assert _json(capsys)["results"]


def test_an_mcp_client_is_told_the_same_as_an_error_it_can_read(tmp_path, monkeypatch, capsys):
    """Through the SDK's own request handling, not ``call_tool``'s shortcut: what a client
    receives is the result the server turns the exception into."""
    _a_version_two_database(tmp_path, monkeypatch, capsys, _touch_every_memory)

    async def remember_then_recall() -> tuple[CallToolResult, CallToolResult]:
        async with create_connected_server_and_client_session(build_server().mcp) as session:
            refused = await session.call_tool("remember", {"text": "not yet", "project": "p"})
            answered = await session.call_tool("recall", {"query": "first memory", "project": "p"})
        return refused, answered

    # A plain test run on its own loop: the fixture's `main()` calls `asyncio.run` itself.
    refused, answered = asyncio.run(remember_then_recall())

    assert refused.isError is True
    assert _BLOCKED in refused.content[0].text
    assert answered.isError is False


def test_migrate_registers_the_space_and_says_it_is_unverified_when_the_embedder_is_down(
    tmp_path, monkeypatch, capsys
):
    """The wave does not wait on the embedding host: port 1 refuses, and the migration stands."""
    _short_budgets(monkeypatch)
    db = _a_version_five_database(tmp_path, monkeypatch, endpoint="http://127.0.0.1:1/v1")

    assert main(["migrate"]) == 0

    assert "space 1 unverified; the first call will verify it" in capsys.readouterr().out
    assert _version(db) == len(migrations._STEPS)
    space = _active_space(db)
    assert space is not None
    assert (space.id, space.model, space.dims, space.table_name, space.fingerprint) == (
        1,
        _MODEL,
        4,
        "vec_items",
        None,
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, "refused the request: HTTP 401"),
        # A llama-server started without --embeddings: refused at once, never waited on.
        (501, "the server does not serve embeddings"),
        (503, "answered too slowly or dropped: "),
    ],
)
def test_migrate_says_the_space_is_unverified_when_the_embedder_refuses_or_stays_slow(
    tmp_path, monkeypatch, capsys, status, reason
):
    """A host that refuses the request, or answers only errors until the budget is spent, is no
    reason to fail a migration that has committed: the space is left for the first call to
    verify, and the line says why."""
    _short_budgets(monkeypatch)
    with flaky_model_server(fail_times=99, status=status, embedding_dim=4) as url:
        db = _a_version_five_database(tmp_path, monkeypatch, endpoint=url)
        assert main(["migrate"]) == 0

    out = capsys.readouterr().out
    assert "space 1 unverified; the first call will verify it" in out
    assert reason in out
    assert _version(db) == len(migrations._STEPS)


def test_migrate_records_the_fingerprint_when_the_embedder_answers(tmp_path, monkeypatch, capsys):
    """The stored vectors are the ones this server gives their texts, so the sample matches."""
    with model_server(embedding_dim=4) as url:
        db = _a_version_five_database(tmp_path, monkeypatch, endpoint=url)
        assert main(["migrate", "--json"]) == 0

    out = _json(capsys)
    assert out["embedding_space"] == {
        "id": 1,
        "model": _MODEL,
        "dims": 4,
        "fingerprint": "recorded",
    }
    space = _active_space(db)
    assert space is not None and space.fingerprint is not None


def test_a_space_refused_after_the_wave_says_the_migration_committed(tmp_path, monkeypatch, capsys):
    """The width check runs once the wave has committed; its refusal must not read as a
    failed migration, which a rerun would then report as nothing to do."""
    db = _a_version_five_database(tmp_path, monkeypatch, endpoint="http://127.0.0.1:1/v1")
    monkeypatch.setenv("MORGAN_EMBEDDING_DIM", "8")

    assert main(["migrate", "--json"]) == 1

    error = _json(capsys)["error"]
    assert "migrated to user_version" in error and "-migrate.db" in error
    assert "created 4 wide but MORGAN_EMBEDDING_DIM is 8" in error
    assert _version(db) == len(migrations._STEPS)
    assert _active_space(db) is None


_MODEL = "an-embedding-model"


def _a_version_five_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, endpoint: str
) -> str:
    """One memory in a database at ``user_version`` 5, whose vector is the one ``model_server``
    gives its text, under settings that send embeddings to *endpoint*. Returns its path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "provider")
    monkeypatch.setenv("MORGAN_EMBEDDING_ENDPOINT", endpoint)
    monkeypatch.setenv("MORGAN_EMBEDDING_MODEL", _MODEL)
    monkeypatch.setenv("MORGAN_EMBEDDING_DIM", "4")
    db = str(data_dir / "morgan.db")
    a_version_five_database(
        db,
        dim=4,
        memories=[Memory(id="m0", user_id="owner", project="p", content="the first memory")],
        vector=lambda text: _unit_vector(text, 4),
    ).close()
    return db


def _short_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budgets for a suite, not a cold host, for the tests whose embedder never answers: port 1
    and a failing fake give up in a second or two. A test whose embedder answers keeps the
    defaults, so a loaded machine cannot spend its budget before the answer comes."""
    monkeypatch.setenv("MORGAN_EMBEDDING_UNREACHABLE_BUDGET_SECONDS", "1.0")
    monkeypatch.setenv("MORGAN_EMBEDDING_RETRY_BUDGET_SECONDS", "1.5")
    monkeypatch.setenv("MORGAN_EMBEDDING_RETRY_BACKOFF_SECONDS", "0.05")


def _active_space(path: str) -> spaces.EmbeddingSpace | None:
    conn = open_db(path)
    try:
        return spaces.active(conn)
    finally:
        conn.close()


def _a_version_two_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    heavy: Callable[[sqlite3.Connection, migrations.Stores], dict[str, int] | None],
) -> tuple[str, Path]:
    """One memory in a database at ``user_version`` 2, under code whose step 3 runs *heavy*.

    Returns ``(db_path, snapshot_dir)``. The code's own steps are sliced to the first two, so
    the steps appended later leave this fixture where it is.
    """
    data_dir = tmp_path / "data"
    snapshots = tmp_path / "snapshots"
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_SNAPSHOT_DIR", str(snapshots))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")

    version_two = migrations._STEPS[:2]
    monkeypatch.setattr(migrations, "_STEPS", version_two)
    assert main(["remember", "the first memory", "--project", "p", "--json"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        migrations, "_STEPS", (*version_two, migrations.Step(3, "a heavy step", True, heavy))
    )

    db = str(data_dir / "morgan.db")
    assert _version(db) == 2
    return db, snapshots


def _version(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def _memories(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    finally:
        conn.close()


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)
