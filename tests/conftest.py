"""Root conftest."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from morgan_brain.config import Settings
from morgan_brain.memory import checked_embedder


@pytest.fixture(scope="session")
def _empty_config_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("config-home")


@pytest.fixture(autouse=True)
def _no_env_files(monkeypatch: pytest.MonkeyPatch, _empty_config_home: Path) -> None:
    """Keep the suite independent of the developer's own configuration.

    ``Settings`` reads ``~/.config/morgan/.env`` and then ``./.env`` -- exactly what the
    CLI needs, and exactly what a test must never see: a result that changes with the
    contents of the developer's home directory depends on install state, not on the code.
    A CLI or MCP server the suite starts as a subprocess builds its own settings, so the
    configuration directory it inherits is an empty one.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_empty_config_home))


@pytest.fixture(autouse=True)
def _structlog_config_restored() -> Iterator[None]:
    """Leave structlog configured as the test found it.

    A CLI or MCP entrypoint run in-process calls ``configure_logging``, which points structlog
    at the ``sys.stderr`` it sees -- under pytest, that test's capture stream, closed when the
    test ends. Left in place, the next test whose code logs anything writes to a closed file.
    """
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


@pytest.fixture(autouse=True)
def _a_fresh_process() -> Iterator[None]:
    """Each test is a new process as far as the embedding-space check goes.

    ``checked_embedder`` records the spaces a process has proven at module level, keyed by
    endpoint, model, width and space id -- and every test database's first space has id 1, at
    the same default model and width. Left in place, a test after the first would skip the
    check its own database needs, and pass or fail by the order the suite ran in.
    """
    checked_embedder._checked.clear()
    checked_embedder._unregistered.clear()
    yield
    checked_embedder._checked.clear()
    checked_embedder._unregistered.clear()


@pytest.fixture
def settings_for_tmp(tmp_path: Path) -> Settings:
    """A ``Settings`` that opens its database under ``tmp_path`` with the hash embedding
    backend: no live model server, and never the owner's own data directory. Shared by any
    test that needs a real (but scratch) database rather than an in-memory store."""
    return Settings(data_dir=str(tmp_path), embedding_backend="hash")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that need a reachable model server (marked `live`).",
    )
    # The three below serve exactly one test, tests/memory_quality/test_holdout_floor_sweep.py.
    # They default to unset so a bare `pytest --live` still runs every other live test; that
    # one test's own fixtures skip (not fail) when its option is missing.
    parser.addoption(
        "--snapshot-db",
        action="store",
        default=None,
        help="Path to a copy-source morgan.db for the holdout floor sweep. The test copies "
        "it before opening anything; the path itself is never written.",
    )
    parser.addoption(
        "--holdout-probes",
        action="store",
        default=None,
        help="Path to the 128-label holdout probe set (JSON) for the holdout floor sweep.",
    )
    parser.addoption(
        "--floor-sweep",
        action="store",
        default=None,
        help="Comma-separated relevance-floor margins to sweep, e.g. 0.00,0.04,0.08.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: needs a reachable model server; skipped by default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs a model server: pass --live to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture
def snapshot_db(request: pytest.FixtureRequest) -> Path:
    """The ``--snapshot-db`` path. Absent is a normal way to run ``pytest --live`` broadly,
    so this skips rather than fails -- a wrong path, once given, still fails loudly below."""
    value = request.config.getoption("--snapshot-db")
    if not value:
        pytest.skip("holdout floor sweep needs --snapshot-db PATH (a copy-source morgan.db)")
    return Path(value)


@pytest.fixture
def holdout_probes(request: pytest.FixtureRequest) -> Path:
    """The ``--holdout-probes`` path. See ``snapshot_db`` for why absence skips."""
    value = request.config.getoption("--holdout-probes")
    if not value:
        pytest.skip("holdout floor sweep needs --holdout-probes PATH (the 128-label set)")
    return Path(value)


@pytest.fixture
def floor_sweep(request: pytest.FixtureRequest) -> list[float]:
    """The ``--floor-sweep`` margins, comma-separated. See ``snapshot_db`` for why absence
    skips rather than fails."""
    value = request.config.getoption("--floor-sweep")
    if not value:
        pytest.skip("holdout floor sweep needs --floor-sweep A,B,C (relevance-floor margins)")
    return [float(part) for part in value.split(",")]
