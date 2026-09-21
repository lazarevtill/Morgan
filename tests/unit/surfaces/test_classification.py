"""A project's classification is recorded from its git remote, and restricts nothing.

Work is whatever MORGAN_WORK_REMOTE_GLOBS matches, everything else personal, a folder with no
remote unclassified. The label is recorded and printed; it does not hide a project from recall
and does not exempt it from consolidation.
"""

from __future__ import annotations

import pytest

from morgan_brain.surfaces.cli.project import classify

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
