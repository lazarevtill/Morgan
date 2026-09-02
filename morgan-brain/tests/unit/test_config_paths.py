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
    assert s.temporal_db_url == f"sqlite:///{tmp_path / 'share' / 'morgan' / 'morgan.db'}"


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
