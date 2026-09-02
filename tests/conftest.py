"""Root conftest."""

from __future__ import annotations

import pytest

from morgan_brain.config import Settings


@pytest.fixture(autouse=True)
def _no_env_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite independent of the developer's own configuration.

    ``Settings`` reads ``~/.config/morgan/.env`` and then ``./.env`` -- exactly what the
    CLI needs, and exactly what a test must never see: a result that changes with the
    contents of the developer's home directory depends on install state, not on the code.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that need a reachable model server (marked `live`).",
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
