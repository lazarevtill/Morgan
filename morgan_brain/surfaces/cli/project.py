"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

from pathlib import Path

from morgan_brain.models import DEFAULT_PROJECT


def detect_project(cwd: Path | None = None) -> str:
    """Return the enclosing git repository's directory name, or DEFAULT_PROJECT outside one.

    Walks up looking for ``.git`` rather than shelling out to ``git rev-parse --show-toplevel``.
    Same answer, and it drops a process spawn from every single CLI invocation -- ``morgan
    recall`` is meant to feel like a shell builtin. It also means the CLI works with no ``git``
    on PATH, which matters inside slim containers.

    ``.git`` is matched as either a directory or a file: a linked worktree and a submodule both
    record a gitdir pointer in a regular file, and both are still repositories whose directory
    name is the project.
    """
    start = (cwd or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate.name or DEFAULT_PROJECT
    return DEFAULT_PROJECT
