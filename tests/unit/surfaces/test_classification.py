"""A project's classification is recorded from its git remote, and restricts nothing.

Work is whatever MORGAN_WORK_REMOTE_GLOBS matches, everything else personal, a folder with no
remote unclassified. The label is recorded and printed; it does not hide a project from recall
and does not exempt it from consolidation.

The remote itself is read from the repository's own config file, never by spawning git: the
same resolution that names the project (a linked worktree belongs to the repository it came
from, a submodule is its own), then ``origin``, else the sole remote, else none.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from morgan_brain.surfaces.cli.project import classify, read_repository

GLOBS = ("gitlab.work.example", "*acme*")


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://gitlab.work.example/team/service.git", "work"),
        ("git@gitlab.work.example:team/service.git", "work"),
        ("https://github.com/someone/acme-api.git", "work"),
        ("https://github.com/someone/notes.git", "personal"),
        ("https://example.invalid/gitlab.work.example/spoof.git", "personal"),
        (None, "unclassified"),
    ],
)
def test_a_remote_is_classified_by_host_then_by_url(remote, expected):
    assert classify(remote, GLOBS) == expected


def test_no_globs_means_everything_is_personal():
    assert classify("https://gitlab.work.example/a.git", ()) == "personal"


#: The brief's remote cases are built with real ``git``, the way ``test_project_detection.py``
#: builds its checkouts -- a hand-written ``.git`` is only used where the point is a config git
#: itself would never write.
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


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


def _a_git_dir(path: Path, config: str | None) -> Path:
    """A checkout whose ``.git`` holds nothing but *config* -- enough for the reader, and the
    only way to write a config file git's own porcelain never produces."""
    (path / ".git").mkdir(parents=True)
    if config is not None:
        (path / ".git" / "config").write_text(config, encoding="utf-8")
    return path


@_needs_git
def test_origin_is_the_remote_even_beside_another(tmp_path):
    repo = _repository(tmp_path / "harbor")
    _git("remote", "add", "upstream", "https://example.com/someone/harbor.git", cwd=repo)
    _git("remote", "add", "origin", "https://gitlab.work.example/team/harbor.git", cwd=repo)

    found = read_repository(repo)

    assert found is not None
    assert found.remote == "https://gitlab.work.example/team/harbor.git"
    assert found.root == repo.resolve()


@_needs_git
def test_the_sole_remote_is_read_when_it_is_not_named_origin(tmp_path):
    repo = _repository(tmp_path / "harbor")
    _git("remote", "add", "upstream", "https://gitlab.work.example/team/harbor.git", cwd=repo)

    found = read_repository(repo)

    assert found is not None
    assert found.remote == "https://gitlab.work.example/team/harbor.git"


@_needs_git
def test_two_remotes_and_no_origin_name_no_remote(tmp_path):
    """Which of the two the project belongs to is the owner's call, not a guess: with no
    ``origin`` to break the tie the project is left ``unclassified``."""
    repo = _repository(tmp_path / "harbor")
    _git("remote", "add", "upstream", "https://gitlab.work.example/team/harbor.git", cwd=repo)
    _git("remote", "add", "fork", "https://example.com/someone/harbor.git", cwd=repo)

    found = read_repository(repo)

    assert found is not None
    assert found.remote is None
    assert classify(found.remote, GLOBS) == "unclassified"


@_needs_git
def test_a_repository_with_no_remote_has_none(tmp_path):
    repo = _repository(tmp_path / "harbor")

    found = read_repository(repo)

    assert found is not None
    assert (found.root, found.remote) == (repo.resolve(), None)
    assert classify(found.remote, GLOBS) == "unclassified"


@_needs_git
def test_a_linked_worktree_reads_the_repository_it_came_from(tmp_path):
    """The same resolution ``detect_project`` makes: an agent session's worktree is the
    repository it was created from, so it is that repository's remote and root that are
    recorded, not the session folder's."""
    repo = _repository(tmp_path / "harbor")
    _git("remote", "add", "origin", "https://gitlab.work.example/team/harbor.git", cwd=repo)
    worktree = tmp_path / "session-8e1008"
    _git("worktree", "add", "-q", "-b", "session", str(worktree), cwd=repo)

    nested = worktree / "src" / "deep"
    nested.mkdir(parents=True)

    found = read_repository(nested)

    assert found is not None
    assert found.remote == "https://gitlab.work.example/team/harbor.git"
    assert found.root == repo.resolve()


@_needs_git
def test_a_submodule_reads_its_own_config(tmp_path):
    """A submodule is its own project, so it is its own remote that classifies it -- never
    the superproject's."""
    library = _repository(tmp_path / "library")
    app = _repository(tmp_path / "app")
    _git("remote", "add", "origin", "https://gitlab.work.example/team/app.git", cwd=app)
    _git(
        "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(library), "vendored",
        cwd=app,
    )  # fmt: skip
    vendored = app / "vendored"
    _git("remote", "set-url", "origin", "https://example.com/someone/library.git", cwd=vendored)

    found = read_repository(vendored)

    assert found is not None
    assert found.remote == "https://example.com/someone/library.git"
    assert found.root == vendored.resolve()


def test_a_checkout_outside_any_repository_is_none(tmp_path):
    assert read_repository(tmp_path) is None


@pytest.mark.parametrize(
    "config",
    [
        pytest.param('[remote "origin"\n\turl = https://example.com/a.git\n', id="unterminated"),
        pytest.param("url = https://example.com/a.git\n", id="no-section"),
        pytest.param('[remote "origin"]\n\turl = "https://example.com/a.git\n', id="open-quote"),
    ],
)
def test_a_config_that_cannot_be_parsed_names_no_remote(tmp_path, config):
    """A config Morgan cannot read is a project it cannot classify -- never an error thrown
    into a command the owner ran to remember something."""
    repo = _a_git_dir(tmp_path / "harbor", config)

    found = read_repository(repo)

    assert found is not None
    assert (found.root, found.remote) == (repo.resolve(), None)


def test_a_config_that_cannot_be_read_names_no_remote(tmp_path):
    repo = _a_git_dir(tmp_path / "harbor", None)
    (repo / ".git" / "config").mkdir()

    found = read_repository(repo)

    assert found is not None
    assert found.remote is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            '[remote "origin"]\n'
            "\turl = https://gitlab.work.example/team/harbor.git\n"
            "\turl = https://example.com/someone/harbor.git\n",
            "https://gitlab.work.example/team/harbor.git",
            id="the-first-url-is-the-one-git-fetches-from",
        ),
        pytest.param(
            '[remote "origin"]\n'
            '\turl = "https://gitlab.work.example/team/harbor.git" ; the work mirror\n',
            "https://gitlab.work.example/team/harbor.git",
            id="quoted-value-and-a-comment",
        ),
        pytest.param(
            "[remote.origin]\n\tURL = https://gitlab.work.example/team/harbor.git\n",
            "https://gitlab.work.example/team/harbor.git",
            id="the-deprecated-header-and-a-cased-key",
        ),
        pytest.param(
            '[remote "Origin"]\n\turl = https://example.com/someone/harbor.git\n'
            '[remote "origin"]\n\turl = https://gitlab.work.example/team/harbor.git\n',
            "https://gitlab.work.example/team/harbor.git",
            id="a-subsection-name-is-case-sensitive",
        ),
        pytest.param(
            '[remote "pushonly"]\n\tpushurl = https://example.com/someone/harbor.git\n'
            '[remote "origin"]\n\turl = https://gitlab.work.example/team/harbor.git\n',
            "https://gitlab.work.example/team/harbor.git",
            id="a-remote-with-no-url-is-not-a-remote",
        ),
        pytest.param("[core]\n\tbare = false\n", None, id="no-remote-section"),
    ],
)
def test_the_config_is_read_the_way_git_reads_it(tmp_path, config, expected):
    """Git's own rules, because the URL recorded must be the one git would push to: the first
    ``url`` of a remote, section names case-insensitive and subsection names case-sensitive,
    quotes and comments stripped."""
    repo = _a_git_dir(tmp_path / "harbor", config)

    found = read_repository(repo)

    assert found is not None
    assert found.remote == expected
