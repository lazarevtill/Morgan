"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

from pathlib import Path

from morgan_brain.models import DEFAULT_PROJECT


def detect_project(cwd: Path | None = None) -> str:
    """Return the enclosing git repository's name, or DEFAULT_PROJECT outside one.

    Walks up looking for ``.git`` rather than shelling out to ``git rev-parse --show-toplevel``.
    Same answer, and it drops a process spawn from every single CLI invocation -- ``morgan
    recall`` is meant to feel like a shell builtin. It also means the CLI works with no ``git``
    on PATH, which matters inside slim containers.

    A ``.git`` directory marks a repository named by its folder. A ``.git`` *file* is a pointer,
    and what it points at decides the name: see ``_repository_behind``.
    """
    start = (cwd or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        marker = candidate / ".git"
        if marker.is_dir():
            return candidate.name or DEFAULT_PROJECT
        if marker.is_file():
            return _repository_behind(candidate, marker) or DEFAULT_PROJECT
    return DEFAULT_PROJECT


def _repository_behind(checkout: Path, pointer: Path) -> str:
    """The project a checkout whose ``.git`` is a pointer file belongs to.

    A linked worktree points into ``<repository>/.git/worktrees/<name>``, and that directory
    holds a ``commondir`` file leading back to the repository's own ``.git``. The worktree
    belongs to that repository: Claude Code and OpenResearch give every agent session a
    worktree in a folder named for the session, and naming the project after the folder gave
    every session a project of its own.

    A submodule points into its superproject's ``.git/modules/<name>``, which has no
    ``commondir``. A submodule is a repository in its own right, named by its own folder.
    """
    text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return checkout.name
    gitdir = Path(text.removeprefix("gitdir:").strip())
    if not gitdir.is_absolute():
        gitdir = checkout / gitdir
    commondir = gitdir / "commondir"
    if not commondir.is_file():
        return checkout.name
    common = Path(commondir.read_text(encoding="utf-8", errors="replace").strip())
    if not common.is_absolute():
        common = gitdir / common
    common = common.resolve()
    # `<repository>/.git` for an ordinary repository; a bare one is the directory itself.
    return common.parent.name if common.name == ".git" else common.name.removesuffix(".git")
