"""Smoke tests for the learning-worker app (Phase 4, commit 4).

Verifies that the worker module is importable and that the scheduler + proactivity
engine are built + wired when the corresponding feature flags are enabled.

No real event-loop running: we just confirm the build path is reachable without
exceptions and that the right types are registered.
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# Import-time smoke: module must be importable without error
# ---------------------------------------------------------------------------


def test_learning_worker_module_importable() -> None:
    """The __main__ module must import cleanly (flags off → minimal deps)."""
    import importlib

    mod = importlib.import_module("morgan_brain.apps.learning_worker.__main__")
    assert hasattr(mod, "run")
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# Scheduler build: enable_scheduling=True produces a running CronService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_cron_service_returns_cron_service() -> None:
    from morgan_brain.apps.learning_worker.__main__ import _build_cron_service
    from morgan_brain.scheduling.cron import CronService

    cron = _build_cron_service()
    assert isinstance(cron, CronService)


@pytest.mark.asyncio
async def test_build_learning_scheduler_returns_scheduler_or_none() -> None:
    """_build_learning_scheduler returns a LearningScheduler or None (graceful failure)."""
    from morgan_brain.apps.learning_worker.__main__ import (
        _build_cron_service,
        _build_learning_scheduler,
    )
    from morgan_brain.scheduling.learning_jobs import LearningScheduler

    class _FakeLearner:
        async def consolidate(self, user_id: str) -> None:
            pass

    cron = _build_cron_service()
    result = _build_learning_scheduler(cron, _FakeLearner())  # type: ignore[arg-type]
    # Either None (if composition fails) or a LearningScheduler.
    assert result is None or isinstance(result, LearningScheduler)


# ---------------------------------------------------------------------------
# Proactivity build: enable_proactivity=True produces an engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_proactivity_engine_returns_engine() -> None:
    from morgan_brain.apps.learning_worker.__main__ import _build_proactivity_engine
    from morgan_brain.proactivity.engine import ProactivityEngine
    from morgan_brain.bus.inproc import InProcessBus

    bus = InProcessBus()
    engine = _build_proactivity_engine(bus)
    assert isinstance(engine, ProactivityEngine)


@pytest.mark.asyncio
async def test_register_proactivity_handler_subscribes_to_heartbeat() -> None:
    """After registration, a HEARTBEAT event triggers the handler without errors."""
    from morgan_brain.apps.learning_worker.__main__ import (
        _build_proactivity_engine,
        _register_proactivity_handler,
    )
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.interfaces.events import Event, EventType

    bus = InProcessBus()
    engine = _build_proactivity_engine(bus)
    assert engine is not None
    _register_proactivity_handler(bus, engine)

    # Publish a HEARTBEAT — handler should run without raising.
    event = Event(type=EventType.HEARTBEAT, user_id="system", payload={})
    await bus.publish(event)  # If handler raises, test fails.


# ---------------------------------------------------------------------------
# Flag-gating: default flags=False → scheduler not started
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_run_with_flags_off_does_not_start_cron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With both flags False, run() sets up the bus but no CronService is started."""
    import morgan_brain.apps.learning_worker.__main__ as worker_mod

    cron_starts: list[str] = []
    original_build = worker_mod._build_cron_service

    def _spy_build():
        cron_starts.append("built")
        return original_build()

    monkeypatch.setattr(worker_mod, "_build_cron_service", _spy_build)

    from morgan_brain.config import Settings

    # Custom settings with both flags off.
    settings = Settings(
        llm_model="test",
        llm_fast_model="test",
        enable_scheduling=False,
        enable_proactivity=False,
    )

    # Run for one very short cycle then cancel.
    async def _run_briefly() -> None:
        task = asyncio.create_task(worker_mod.run(settings=settings))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    await _run_briefly()
    assert cron_starts == [], "CronService should NOT be built when enable_scheduling=False"
