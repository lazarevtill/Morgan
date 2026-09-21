"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urlsplit


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


#: ``classify``'s answer when nothing given matches -- SPEC-phase0 §3.7's third label.
_UNCLASSIFIED = "unclassified"
_WORK = "work"
_PERSONAL = "personal"


def _host_of(remote_url: str) -> str | None:
    """The host git would connect to, from a URL-style or SCP-style remote.

    ``scheme://[user@]host[:port]/path`` is parsed with ``urlsplit``. ``[user@]host:path`` --
    SCP-style, ``git@gitlab.work.example:team/service.git`` -- has no scheme, so the host is
    whatever precedes the first ``:``, with a leading ``user@`` stripped.
    """
    if "://" in remote_url:
        return urlsplit(remote_url).hostname
    host_part = remote_url.split(":", 1)[0]
    if "@" in host_part:
        host_part = host_part.rsplit("@", 1)[1]
    return host_part or None


def classify(remote_url: str | None, work_globs: Sequence[str]) -> str:
    """``work`` on a ``work_globs`` match, ``personal`` otherwise, ``unclassified`` with no
    remote. The label restricts nothing: recorded, printed, later used for labelling and
    sharing -- never for hiding a project from recall or exempting it from consolidation.

    A bare glob (no ``*``) must equal the remote's **host** exactly. Matched against the whole
    URL instead, a glob naming a host would also match that same text sitting in a path
    component -- ``gitlab.work.example`` inside
    ``https://example.invalid/gitlab.work.example/spoof.git``, a URL whose real host is
    ``example.invalid``. A glob with a wildcard is tried with ``fnmatch`` against the host
    first, then the whole URL, so an operator can write either ``*.example.org`` (host-shaped)
    or ``*acme*`` (matches a path or org name anywhere in the URL).
    """
    if not remote_url:
        return _UNCLASSIFIED
    host = _host_of(remote_url)
    for glob in work_globs:
        if "*" not in glob:
            if host is not None and glob == host:
                return _WORK
        elif (host is not None and fnmatch(host, glob)) or fnmatch(remote_url, glob):
            return _WORK
    return _PERSONAL
