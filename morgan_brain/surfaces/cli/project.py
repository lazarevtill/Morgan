"""Which project a command belongs to, per the reshape spec §4.3, and what repository that
project is: its root, its remote and the label ``classify`` gives it.

One resolution answers all three (``_repository_at``), so the name a command is scoped to and
the remote recorded against it can never come from two different repositories.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Repository:
    """The enclosing repository as ``morgan`` sees it: where it is on disk, and the remote its
    classification is derived from.

    Three answers, not two, because "this repository has no remote" and "Morgan could not read
    its config" are different facts and only the first is a classification:

    - ``remote`` a URL, ``remote_readable`` true -- the label follows from it;
    - ``remote`` ``None``, ``remote_readable`` true -- the config was read and names no remote
      to single out, which is ``unclassified``;
    - ``remote`` ``None``, ``remote_readable`` false -- the config could not be read or parsed.
      A caller records nothing then: the label would be a fact about the read, not about the
      repository, and it would overwrite what an earlier readable config recorded.
    """

    root: Path
    remote: str | None
    remote_readable: bool


@dataclass(frozen=True)
class _Checkout:
    """What the walk up from a working directory found.

    *name* is the project, *root* the repository's own directory, and *config* the git config
    file that holds its remotes -- ``None`` when the ``.git`` pointer leads somewhere this
    code does not recognise.
    """

    name: str
    root: Path
    config: Path | None


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
    """
    found = _repository_at(cwd)
    return found.name or None if found is not None else None


def read_repository(cwd: Path | None = None) -> Repository | None:
    """The enclosing repository's root and remote URL, or ``None`` outside one.

    The same resolution ``detect_project`` makes -- a linked worktree is the repository it was
    created from, a submodule is its own -- and then that repository's config file, parsed
    here rather than asked of ``git``: a CLI invocation spawns no process, and the CLI still
    works where ``git`` is not installed.

    ``origin`` is the remote, else the sole remote when a repository has exactly one, else
    none: with two remotes and no ``origin`` there is nothing to prefer, and a guess would
    label the project from the wrong one. No remote at all classifies as ``unclassified``.

    It never raises into the command that called it, and it distinguishes the two ways it can
    come back empty (see ``Repository``):

    - a config file that cannot be read, decoded or parsed, and a ``.git`` pointer that leads
      somewhere this code does not recognise, answer a repository whose ``remote_readable`` is
      false -- where it is on disk is known, what it is is not;
    - a ``.git`` pointer that cannot be read at all answers ``None``, the same as standing
      outside a repository, because nothing about the checkout could be established.
    """
    try:
        found = _repository_at(cwd)
    except OSError:
        return None
    if found is None:
        return None
    remote, readable = _remote_of(found.config)
    return Repository(root=found.root, remote=remote, remote_readable=readable)


def _repository_at(cwd: Path | None) -> _Checkout | None:
    """Walk up from *cwd* to the first ``.git``, and say what repository it belongs to.

    A ``.git`` directory marks a repository named by its folder. A ``.git`` *file* is a
    pointer, and what it points at decides the answer: see ``_behind_the_pointer``.
    """
    start = (cwd or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        marker = candidate / ".git"
        if marker.is_dir():
            return _Checkout(candidate.name, candidate, marker / "config")
        if marker.is_file():
            return _behind_the_pointer(candidate, marker)
    return None


def _behind_the_pointer(checkout: Path, pointer: Path) -> _Checkout:
    """The repository a checkout whose ``.git`` is a pointer file belongs to.

    A linked worktree points into ``<repository>/.git/worktrees/<name>``, and that directory
    holds a ``commondir`` file leading back to the repository's own ``.git``. The worktree
    belongs to that repository: Claude Code and OpenResearch give every agent session a
    worktree in a folder named for the session, and naming the project after the folder gave
    every session a project of its own. Its remotes live in that repository's config, which
    is the one git itself reads from inside a worktree.

    A submodule points into its superproject's ``.git/modules/<name>``, which has no
    ``commondir``. A submodule is a repository in its own right, named by its own folder, with
    its own remotes in the config file that pointer leads to.
    """
    text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return _Checkout(checkout.name, checkout, None)
    gitdir = Path(text.removeprefix("gitdir:").strip())
    if not gitdir.is_absolute():
        gitdir = checkout / gitdir
    commondir = gitdir / "commondir"
    if not commondir.is_file():
        return _Checkout(checkout.name, checkout, gitdir / "config")
    common = Path(commondir.read_text(encoding="utf-8", errors="replace").strip())
    if not common.is_absolute():
        common = gitdir / common
    common = common.resolve()
    # `<repository>/.git` for an ordinary repository; a bare one is the directory itself.
    if common.name == ".git":
        return _Checkout(common.parent.name, common.parent, common / "config")
    return _Checkout(common.name.removesuffix(".git"), common, common / "config")


class _Unparseable(ValueError):
    """A git config file this code will not guess at. Never leaves this module."""


#: ``[section]``, ``[section "subsection"]`` or the deprecated ``[section.subsection]``.
_HEADER = re.compile(r'\[\s*([A-Za-z0-9.-]+)\s*(?:"((?:[^"\\]|\\.)*)")?\s*\]')
#: A variable name: a letter, then letters, digits and dashes. Git's own rule.
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
#: The escapes git recognises inside a value. Any other one is an error, not a literal.
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "b": "\b"}


def _remote_of(config: Path | None) -> tuple[str | None, bool]:
    """``origin``'s URL, else the sole remote's, else ``None`` -- from *config* alone -- and
    whether that answer came from a config this code could read at all.

    A false second value is not "no remote": git reads its config as bytes and is unaffected
    by a name or an editor path written in a legacy code page, while this decodes text, and a
    line this parser will not guess at fails the whole file. The caller must not turn either
    into a classification -- see ``Repository``.
    """
    if config is None:
        return None, False
    try:
        # utf-8-sig: git accepts a byte-order mark at the top of a config file.
        urls = _remote_urls(config.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        # ValueError covers both a file that is not UTF-8 and one this code will not guess at.
        return None, False
    if "origin" in urls:
        return urls["origin"], True
    return (next(iter(urls.values())) if len(urls) == 1 else None), True


def _remote_urls(text: str) -> dict[str, str]:
    """Every remote's fetch URL in *text*, by remote name.

    Git's own reading of its format, because the URL recorded must be the one git would push
    to: section names are case-insensitive and quoted subsection names are not, a variable's
    first value is the one a fetch uses, quotes and ``#``/``;`` comments are stripped. A
    ``[remote "x"]`` with no ``url`` is not a remote here -- it has no URL to classify, and
    counting it would turn a one-remote repository into an ambiguous one.
    """
    urls: dict[str, str] = {}
    section: str | None = None
    subsection: str | None = None
    lines = iter(text.splitlines())
    for raw in lines:
        line = raw.strip()
        if line.startswith("["):
            section, subsection, line = _header(line)
            line = line.strip()
        if not line or line[0] in "#;":
            continue
        if section is None:
            # Git refuses a variable before any section header; so does this, rather than
            # reading the rest of a file it has already misunderstood.
            raise _Unparseable(f"a variable outside any section: {line!r}")
        name, value = _variable(line, lines)
        if section == "remote" and subsection and name == "url" and value:
            urls.setdefault(subsection, value)
    return urls


def _header(line: str) -> tuple[str, str | None, str]:
    """The section and subsection *line* opens, and whatever follows the ``]``."""
    match = _HEADER.match(line)
    if match is None:
        raise _Unparseable(f"not a section header: {line!r}")
    name, quoted = match.group(1), match.group(2)
    rest = line[match.end() :]
    if quoted is not None:
        # A quoted subsection is case-sensitive and may escape a quote or a backslash.
        return name.lower(), re.sub(r"\\(.)", r"\1", quoted), rest
    head, _, tail = name.partition(".")
    return head.lower(), tail.lower() or None, rest


def _variable(line: str, lines: Iterator[str]) -> tuple[str, str]:
    """One ``name = value`` (or a bare name, which is a boolean, never a URL)."""
    name, equals, rest = line.partition("=")
    if not equals:
        name = name.split("#")[0].split(";")[0]
    key = name.strip()
    if _NAME.fullmatch(key) is None:
        raise _Unparseable(f"not a variable name: {key!r}")
    return key.lower(), _value(rest, lines) if equals else ""


def _value(rest: str, lines: Iterator[str]) -> str:
    """A variable's value: quotes removed, escapes resolved, a comment and the whitespace
    around the value dropped, and a trailing backslash continuing onto the next line."""
    out: list[str] = []
    pending = ""  # whitespace outside quotes, kept only once more value follows it
    quoted = False
    index = 0
    while True:
        if index >= len(rest):
            if quoted:
                raise _Unparseable("a quoted value is never closed")
            return "".join(out)
        char = rest[index]
        index += 1
        if char == "\\":
            if index == len(rest):
                following = next(lines, None)
                if following is None:
                    raise _Unparseable("a value is continued past the end of the file")
                rest, index = following, 0
                continue
            escape = rest[index]
            index += 1
            if escape not in _ESCAPES:
                raise _Unparseable(f"not an escape git recognises: \\{escape}")
            out.append(pending + _ESCAPES[escape])
            pending = ""
        elif char == '"':
            quoted = not quoted
        elif not quoted and char in "#;":
            return "".join(out)
        elif not quoted and char.isspace():
            if out:
                pending += char
        else:
            out.append(pending + char)
            pending = ""


#: ``classify``'s answer when nothing given matches -- the third label, beside the two below.
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
