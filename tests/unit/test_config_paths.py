"""Configuration has a home that does not move with the working directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from morgan_brain.config import Settings, default_data_dir, user_config_file


def test_data_dir_defaults_under_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.delenv("MORGAN_DATA_DIR", raising=False)
    s = Settings()
    assert s.data_dir == str(tmp_path / "share" / "morgan")
    # as_posix(): data_dir is a native path, the database URL is a URL. Identical on a
    # POSIX host; on Windows only this spelling states the contract the code now keeps.
    expected_db = (tmp_path / "share" / "morgan" / "morgan.db").as_posix()
    assert s.temporal_db_url == f"sqlite:///{expected_db}"


def test_data_dir_defaults_under_home_without_xdg(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert default_data_dir() == str(Path.home() / ".local" / "share" / "morgan")


def test_data_dir_expands_tilde(monkeypatch: pytest.MonkeyPatch):
    s = Settings(data_dir="~/brain")
    assert s.data_dir == str(Path.home() / "brain")
    assert "~" not in s.temporal_db_url


def test_data_dir_is_never_relative_to_the_working_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MORGAN_DATA_DIR", raising=False)
    assert Path(Settings().data_dir).is_absolute()


def test_user_config_file_follows_xdg_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_config_file() == tmp_path / "morgan" / ".env"


def test_database_url_uses_url_separators_on_every_platform(tmp_path: Path):
    """``temporal_db_url`` is a URL, so its path component uses forward slashes regardless
    of the platform's separator. Interpolating a native path mixed the two on Windows: a
    native separator for the directory, a forward slash before the file name. No reader of
    the setting can predict or match that.
    """
    s = Settings(data_dir=str(tmp_path))

    assert "\\" not in s.temporal_db_url
    assert s.temporal_db_url == f"sqlite:///{(tmp_path / 'morgan.db').as_posix()}"
