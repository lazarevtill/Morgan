"""Tests for learning_lifecycle.factory — backend selection + telemetry-off enforcement."""

import os
import sqlite3

import pytest

from morgan_brain.config import Settings
from morgan_brain.learning_lifecycle.factory import build_registry
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry


# ------------------------------------------------------------------
# local backend (default)
# ------------------------------------------------------------------


def test_build_registry_local_returns_local_prompt_registry() -> None:
    reg = build_registry(Settings(learning_backend="local"))
    assert isinstance(reg, LocalPromptRegistry)


def test_build_registry_local_satisfies_protocol() -> None:
    reg = build_registry(Settings(learning_backend="local"))
    assert isinstance(reg, PromptRegistry)


def test_build_registry_local_derives_path_from_data_dir(tmp_path) -> None:
    """The prompt registry must live under settings.data_dir, not a hardcoded
    CWD-relative "./data" -- callers that build the app context from an arbitrary
    working directory (the CLI being the first) must not depend on "./data" existing."""
    data_dir = tmp_path / "nested" / "data"
    build_registry(Settings(learning_backend="local", data_dir=str(data_dir)))
    assert (data_dir / "prompts.db").exists()


def test_build_registry_local_creates_missing_parent_directories(tmp_path) -> None:
    """A data_dir several levels deep that doesn't exist yet must not raise
    "unable to open database file" -- the parent directories are created."""
    data_dir = tmp_path / "a" / "b" / "c"
    assert not data_dir.exists()
    reg = build_registry(Settings(learning_backend="local", data_dir=str(data_dir)))
    assert isinstance(reg, LocalPromptRegistry)
    assert data_dir.is_dir()


def test_build_registry_local_shares_a_given_connection(tmp_path) -> None:
    """Task 13A's one-database invariant: every production caller (build_app_context,
    build_worker_context) has a shared memory-database connection and must get a registry
    that uses it, instead of a second .db file -- so forget()'s single-transaction, and the
    single-file backup/encryption story, still cover the champion registry."""
    conn = sqlite3.connect(str(tmp_path / "morgan.db"), check_same_thread=False)
    reg = build_registry(Settings(learning_backend="local"), conn=conn)
    assert isinstance(reg, LocalPromptRegistry)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "prompt_versions" in tables
    assert not (tmp_path / "prompts.db").exists()


# ------------------------------------------------------------------
# mlflow backend — telemetry env vars MUST be set before any failure
# ------------------------------------------------------------------


def test_build_registry_mlflow_sets_telemetry_env_vars() -> None:
    """Regardless of whether mlflow is installed, the two env vars must be set."""
    # Clean up before test so we're not reading a pre-existing value.
    os.environ.pop("MLFLOW_DISABLE_TELEMETRY", None)
    os.environ.pop("DO_NOT_TRACK", None)

    try:
        build_registry(Settings(learning_backend="mlflow"))
    except NotImplementedError:
        pass  # Expected — Wave 1/5 not yet implemented

    assert os.environ.get("MLFLOW_DISABLE_TELEMETRY") == "true"
    assert os.environ.get("DO_NOT_TRACK") == "true"


def test_build_registry_mlflow_raises_not_implemented() -> None:
    """The mlflow backend always raises NotImplementedError until Wave 1/5."""
    with pytest.raises(NotImplementedError, match="Wave 1/5"):
        build_registry(Settings(learning_backend="mlflow"))
