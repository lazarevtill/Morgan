"""Context commands must not inspect the enclosing repository."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("named", [None, "specified-project"])
def test_context_main_skips_unreadable_git_pointer(tmp_path, monkeypatch, named):
    cli = importlib.import_module("morgan_brain.surfaces.cli.__main__")
    repo = tmp_path / "unrelated"
    repo.mkdir()
    pointer = repo / ".git"
    pointer.write_text("gitdir: missing\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    original_read = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == pointer:
            raise PermissionError("synthetic unreadable git pointer")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    monkeypatch.setattr(cli, "settings_for", lambda surface: object())
    seen = {}

    async def dispatch(args, settings, project, **kwargs):
        seen.update(project=project, **kwargs)
        return 0

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    argv = ["context", "list"]
    if named is not None:
        argv.extend(["--project", named])
    assert cli.main(argv) == 0
    assert seen["project"] == (named or "personal")
    assert seen["repository"] is None
    assert seen["remember_project"] == named
