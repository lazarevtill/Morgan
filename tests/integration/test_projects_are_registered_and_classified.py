"""Every project gets a row, and a CLI write from inside its repository classifies it.

Two halves, both proven from a real terminal: below the gate, any project-keyed write registers
its project (``unclassified``, no remote, no root), so no project written to after migration
step 7 goes without a row; above it, a CLI command that *writes* and whose project was named by
the enclosing git repository records what that repository is -- ``classify(remote, globs)``, the
remote and the root -- recomputed on every such write, because both derive from the repository
and the settings.

A read records nothing. ``morgan-mcp`` records nothing either, whatever directory the server
was started in: the project comes from the tool argument, and the client's repository may be on
another machine entirely.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import morgan_brain.composition as composition
from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store.db import open_db
from morgan_brain.surfaces.cli.__main__ import main
from morgan_brain.surfaces.mcp_server import build_server
from tests.fakes import FakeChatClient

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")

_WORK_REMOTE = "https://gitlab.work.example/team/harbor.git"
_PERSONAL_REMOTE = "https://example.com/someone/harbor.git"
_GLOBS = "gitlab.work.example"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _repository(path: Path, *, remote: str | None = None) -> Path:
    path.mkdir(parents=True)
    _git("init", "-q", cwd=path)
    if remote is not None:
        _git("remote", "add", "origin", remote, cwd=path)
    return path


def _env(data_dir: Path, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "MORGAN_DATA_DIR": str(data_dir),
        "MORGAN_EMBEDDING_BACKEND": "hash",
        "MORGAN_WORK_REMOTE_GLOBS": _GLOBS,
        **extra,
    }


def _run(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.surfaces.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
        check=False,
    )


def _row(data_dir: Path, project: str) -> projects_store.Project | None:
    conn = open_db(str(data_dir / "morgan.db"))
    try:
        return projects_store.get(conn, project)
    finally:
        conn.close()


def test_a_remember_inside_a_work_repository_records_what_the_repository_is(tmp_path):
    """The label is recomputed on every write, because it derives from the remote and the
    settings: point the same repository at a personal remote and the next ``remember`` says
    ``personal``."""
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)
    env = _env(data_dir)

    assert _run(["remember", "the mirror blocked the deploy"], env, repo).returncode == 0

    recorded = _row(data_dir, "harbor")
    assert recorded is not None
    assert (recorded.classification, recorded.remote, recorded.root) == (
        "work",
        _WORK_REMOTE,
        str(repo.resolve()),
    )

    _git("remote", "set-url", "origin", _PERSONAL_REMOTE, cwd=repo)
    assert _run(["remember", "and then it did not"], env, repo).returncode == 0

    rerecorded = _row(data_dir, "harbor")
    assert rerecorded is not None
    assert (rerecorded.classification, rerecorded.remote) == ("personal", _PERSONAL_REMOTE)
    assert rerecorded.created_at == recorded.created_at


def test_a_project_named_by_the_flag_is_registered_but_not_classified(tmp_path):
    """``--project`` is a name, not a repository: the caller may be anywhere, so nothing about
    the directory they happen to stand in is recorded against it."""
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)

    assert _run(["remember", "x", "--project", "elsewhere"], _env(data_dir), repo).returncode == 0

    named = _row(data_dir, "elsewhere")
    assert named is not None
    assert (named.classification, named.remote, named.root) == ("unclassified", None, None)
    assert _row(data_dir, "harbor") is None


@pytest.mark.parametrize("command", [["recall", "anything"], ["facts"]])
def test_a_read_inside_a_repository_records_nothing(tmp_path, command):
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)
    env = _env(data_dir)
    # A database with a project of its own, so the read runs against one that exists.
    assert _run(["remember", "x", "--project", "elsewhere"], env, tmp_path).returncode == 0

    out = _run([*command, "--json"], env, repo)

    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["project"] == "harbor"
    assert _row(data_dir, "harbor") is None


@pytest.mark.parametrize("command", [["ask", "what blocked the deploy?"], ["consolidate"]])
def test_the_other_write_commands_record_too(tmp_path, monkeypatch, command):
    """``ask`` and ``consolidate`` are writes gated by ``require_writable`` like ``remember``,
    and each records the repository it ran in. Run in-process so neither needs a model server:
    the chat client is a double, and consolidation proposes nothing."""
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MORGAN_WORK_REMOTE_GLOBS", _GLOBS)
    monkeypatch.setattr(composition, "build_chat_client", lambda settings: FakeChatClient())

    async def proposes_nothing(self, user_id: str, *, project: str):
        return []

    monkeypatch.setattr(MemoryConsolidator, "consolidate", proposes_nothing)
    monkeypatch.chdir(repo)
    # `consolidate` records the row the project already has; `ask` registers one itself.
    assert main(["remember", "the mirror blocked the deploy"]) == 0
    conn = open_db(str(data_dir / "morgan.db"))
    conn.execute("UPDATE projects SET classification = 'unclassified', remote = NULL")
    conn.commit()
    conn.close()

    assert main(command) == 0

    recorded = _row(data_dir, "harbor")
    assert recorded is not None
    assert (recorded.classification, recorded.remote, recorded.root) == (
        "work",
        _WORK_REMOTE,
        str(repo.resolve()),
    )


def test_a_database_waiting_for_migrate_refuses_the_write_and_records_nothing(
    tmp_path, monkeypatch, capsys
):
    """The recording takes the same refusal as the write it rides with: the owner is told to
    run ``morgan migrate``, and the database is left exactly as it was."""
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MORGAN_WORK_REMOTE_GLOBS", _GLOBS)
    version_two = migrations._STEPS[:2]
    monkeypatch.setattr(migrations, "_STEPS", version_two)
    build_memory_context(Settings()).conn.close()
    heavy = migrations.Step(3, "a heavy step", True, lambda c, s: None)
    monkeypatch.setattr(migrations, "_STEPS", (*version_two, heavy))
    monkeypatch.chdir(repo)

    code = main(["remember", "the mirror blocked the deploy", "--json"])

    assert code == 1
    assert "morgan migrate" in json.loads(capsys.readouterr().out)["error"]
    conn = open_db(sqlite_path(Settings().temporal_db_url))
    try:
        assert projects_store.list_all(conn) == []
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    finally:
        conn.close()


async def test_a_write_through_the_mcp_server_registers_but_never_classifies(tmp_path, monkeypatch):
    """An MCP server may run on another machine than the client, so the directory it was
    started in says nothing about the project the tool names: the row is registered and stays
    ``unclassified`` until a CLI write from inside the repository."""
    data_dir = tmp_path / "data"
    repo = _repository(tmp_path / "harbor", remote=_WORK_REMOTE)
    monkeypatch.setenv("MORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MORGAN_WORK_REMOTE_GLOBS", _GLOBS)
    monkeypatch.chdir(repo)

    await build_server().call_tool("remember", {"text": "stored by a client", "project": "harbor"})

    registered = _row(data_dir, "harbor")
    assert registered is not None
    assert (registered.classification, registered.remote, registered.root) == (
        "unclassified",
        None,
        None,
    )
    assert datetime.fromisoformat(registered.created_at).tzinfo is not None
    assert datetime.fromisoformat(registered.created_at) <= datetime.now(UTC)
