"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

from pathlib import Path


def detect_project(cwd: Path | None = None) -> str | None:
    """Return the enclosing git repository's name, or ``None`` outside one.

    ``None``, never ``PERSONAL_PROJECT``: this function only detects, it does not resolve. A
    caller that finds no repository decides for itself whether that means "fall back to the
    personal project" -- conflating the two here once made a repository literally named
    ``personal`` indistinguishable from no repository at all, since both produced the same
    string. Resolving ``None`` to the personal project, and reporting that the caller named
    nothing, happens at each surface's one call site instead (``surfaces.cli.__main__.main``;
    an MCP tool never calls this at all -- see ``surfaces.mcp_server``'s module docstring).

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
            return candidate.name or None
        if marker.is_file():
            return _repository_behind(candidate, marker) or None
    return None


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
