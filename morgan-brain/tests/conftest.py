"""Root conftest: project-wide pytest fixtures and marks.

live tests
----------
Tests marked ``@pytest.mark.live`` require real external services (Qdrant, Redis,
etc.). They are **skipped by default** and only run when ``--live`` is passed on
the command line::

    python -m pytest --live tests/unit/test_qdrant_vector_index.py
"""

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
        help="Run tests that require live external services (Qdrant, Redis, …).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return  # run everything, including live tests
    skip_live = pytest.mark.skip(reason="Live service test: pass --live to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)
