"""Pytest entrypoint for the text E2E benchmark.

Runs every benchmark scenario as an individual test. Deterministic mode is the
default and runs with zero external services. Live mode is opt-in via
``MORGAN_BENCH_LIVE=1`` and skips gracefully when the configured services are
absent.

    pytest -q tests/e2e
    MORGAN_BENCH_LIVE=1 pytest -q tests/e2e
"""

from __future__ import annotations

import os

import pytest

from tests.e2e.harness import (
    SCENARIOS,
    ScenarioFn,
    _live_settings,
    probe_live,
    run_all,
)
from tests.e2e.report import to_dict


def _is_live() -> bool:
    return os.environ.get("MORGAN_BENCH_LIVE", "").strip() in ("1", "true", "yes", "on")


_LIVE = _is_live()


@pytest.fixture(scope="session")
def _live_available() -> bool:
    """Probe live services once per session; cache the verdict."""
    if not _LIVE:
        return False
    import asyncio

    probe = asyncio.run(probe_live(_live_settings()))
    return probe.llm_ok and probe.qdrant_ok


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[f.__name__ for f in SCENARIOS])
async def test_scenario(scenario: ScenarioFn, _live_available: bool) -> None:
    if _LIVE and not _live_available:
        pytest.skip("MORGAN_BENCH_LIVE set but configured services are unreachable")

    result = await scenario(_LIVE)
    assert result.passed, f"[{result.category}] {result.name}: {result.detail}"
    # Latency must have been measured for at least one turn.
    assert result.turn_latencies_ms, "no per-turn latency captured"


@pytest.mark.asyncio
async def test_full_report_smoke() -> None:
    """The aggregate report renders and, in deterministic mode, every scenario passes."""
    report = await run_all(live=_LIVE)
    payload = to_dict(report)
    assert payload["mode"] == ("live" if _LIVE else "deterministic")
    assert payload["scenarios"], "no scenarios in report"

    if not _LIVE:
        # Deterministic mode is a wiring gate: nothing may fail or skip.
        assert report.failed_count == 0, [
            (r.name, r.detail) for r in report.results if not r.passed
        ]
        assert report.skipped_count == 0
        assert report.recall_accuracy() == 1.0
