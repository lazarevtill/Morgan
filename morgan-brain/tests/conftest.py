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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that require live external services (Qdrant, Redis, …).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live"):
        return  # run everything, including live tests
    skip_live = pytest.mark.skip(reason="Live service test: pass --live to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)
