"""Which project a checkout belongs to.

A project is a repository, not a folder. Claude Code and OpenResearch run every agent session
in its own linked worktree, in a folder named for the session; naming the project after that
folder gave each session a project of its own, so nothing one session remembered reached the
next.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from morgan_brain.surfaces.cli.project import detect_project


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _repository(path: Path) -> Path:
    path.mkdir()
    _git("init", "-q", cwd=path)
    (path / "README.md").write_text("x", encoding="utf-8")
    _git("add", "README.md", cwd=path)
    _git("commit", "-q", "-m", "init", cwd=path)
    return path


def test_a_linked_worktree_belongs_to_its_repository(tmp_path):
    repo = _repository(tmp_path / "harbor")
    worktree = tmp_path / "session-8e1008"
    _git("worktree", "add", "-q", "-b", "session", str(worktree), cwd=repo)

    assert detect_project(worktree) == "harbor"


def test_a_folder_inside_a_linked_worktree_belongs_to_its_repository(tmp_path):
    repo = _repository(tmp_path / "harbor")
    worktree = tmp_path / "session-8e1008"
    _git("worktree", "add", "-q", "-b", "session", str(worktree), cwd=repo)
    nested = worktree / "src" / "deep"
    nested.mkdir(parents=True)

    assert detect_project(nested) == "harbor"


def test_a_submodule_is_its_own_project(tmp_path):
    library = _repository(tmp_path / "library")
    app = _repository(tmp_path / "app")
    _git(
        "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(library), "vendored",
        cwd=app,
    )  # fmt: skip

    assert detect_project(app / "vendored") == "vendored"
