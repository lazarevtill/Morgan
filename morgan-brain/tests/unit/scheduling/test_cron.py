"""Unit tests for CronService + InProcessScheduler (Phase 4, commit 1).

All tests are deterministic:
  - Injected fake clock (fixed datetime).
  - tick() called directly — no real sleeping, no APScheduler dependency.
  - Job callables are simple counters (sync and async).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.scheduling.cron import (
    CronService,
    InProcessScheduler,
    Job,
    Scheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _fixed_clock(t: datetime):  # type: ignore[return]
    def clock() -> datetime:
        return t

    return clock


class _Counter:
    """Simple callable that counts invocations (sync)."""

    def __init__(self) -> None:
        self.calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


class _AsyncCounter:
    """Simple async callable that counts invocations."""

    def __init__(self) -> None:
        self.calls: int = 0

    async def __call__(self) -> None:  # type: ignore[override]
        self.calls += 1


# ---------------------------------------------------------------------------
# InProcessScheduler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_job_registers_and_list_jobs() -> None:
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("j1", lambda: None, interval_seconds=60)
    sched.add_job("j2", lambda: None, interval_seconds=120)
    assert set(sched.list_jobs()) == {"j1", "j2"}


@pytest.mark.asyncio
async def test_tick_runs_new_job_immediately() -> None:
    """A job with last_run=None should run on the very first tick."""
    counter = _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("j", counter, interval_seconds=60)
    await sched.tick(T0)
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_tick_does_not_run_before_interval() -> None:
    """After the first run, tick at T0+30s should NOT re-run a 60s-interval job."""
    counter = _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("j", counter, interval_seconds=60)
    await sched.tick(T0)  # first run — due (last_run=None)
    await sched.tick(T0 + timedelta(seconds=30))  # 30 s later — not due
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_tick_runs_after_interval() -> None:
    """After first run, tick at T0+60s should re-run the job."""
    counter = _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("j", counter, interval_seconds=60)
    await sched.tick(T0)
    await sched.tick(T0 + timedelta(seconds=60))
    assert counter.calls == 2


@pytest.mark.asyncio
async def test_tick_runs_async_job() -> None:
    """Async callables are awaited correctly."""
    counter = _AsyncCounter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("async_j", counter, interval_seconds=10)
    await sched.tick(T0)
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_tick_does_not_run_job_not_yet_registered() -> None:
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    assert sched.list_jobs() == []
    await sched.tick(T0)  # no error, nothing to run


@pytest.mark.asyncio
async def test_add_job_cron_only_uses_default_interval() -> None:
    """Cron-only jobs are approximated with 3600s interval."""
    counter = _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("cron_j", counter, cron="0 * * * *")
    await sched.tick(T0)  # first tick → runs (last_run=None)
    await sched.tick(T0 + timedelta(seconds=1800))  # 30 min later → not due
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_add_job_requires_interval_or_cron() -> None:
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    with pytest.raises(ValueError, match="interval_seconds or cron"):
        sched.add_job("bad", lambda: None)


@pytest.mark.asyncio
async def test_multiple_jobs_run_independently() -> None:
    c1, c2 = _Counter(), _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("j1", c1, interval_seconds=60)
    sched.add_job("j2", c2, interval_seconds=120)
    await sched.tick(T0)
    await sched.tick(T0 + timedelta(seconds=60))
    # j1: ran at T0 + T0+60s = 2 times; j2: ran only at T0 = 1 time
    assert c1.calls == 2
    assert c2.calls == 1


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    await sched.start()
    await sched.stop()  # Should not raise


@pytest.mark.asyncio
async def test_job_exception_does_not_crash_tick() -> None:
    """A failing job should be logged but not stop the tick loop."""

    def bad_fn() -> None:
        raise RuntimeError("boom")

    good = _Counter()
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    sched.add_job("bad", bad_fn, interval_seconds=10)
    sched.add_job("good", good, interval_seconds=10)
    await sched.tick(T0)  # bad_fn raises; good still runs
    assert good.calls == 1


# ---------------------------------------------------------------------------
# Scheduler Protocol conformance
# ---------------------------------------------------------------------------


def test_inprocess_scheduler_satisfies_protocol() -> None:
    sched = InProcessScheduler(clock=_fixed_clock(T0))
    assert isinstance(sched, Scheduler)


# ---------------------------------------------------------------------------
# CronService tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_service_registers_and_lists() -> None:
    svc = CronService(clock=_fixed_clock(T0))
    svc.register(Job(name="heartbeat", interval_seconds=300), lambda: None)
    assert "heartbeat" in svc.list()


@pytest.mark.asyncio
async def test_cron_service_falls_back_to_inprocess_without_apscheduler() -> None:
    """Without an explicit scheduler, CronService uses InProcessScheduler."""
    svc = CronService(clock=_fixed_clock(T0))
    # The internal scheduler is an InProcessScheduler instance.
    assert isinstance(svc.scheduler, InProcessScheduler)


@pytest.mark.asyncio
async def test_cron_service_accepts_explicit_scheduler() -> None:
    explicit = InProcessScheduler(clock=_fixed_clock(T0))
    svc = CronService(clock=_fixed_clock(T0), scheduler=explicit)
    assert svc.scheduler is explicit


@pytest.mark.asyncio
async def test_cron_service_start_stop() -> None:
    svc = CronService(clock=_fixed_clock(T0))
    await svc.start()
    await svc.stop()


@pytest.mark.asyncio
async def test_cron_service_job_runs_via_tick() -> None:
    counter = _Counter()
    svc = CronService(clock=_fixed_clock(T0))
    svc.register(Job(name="j", interval_seconds=60), counter)
    await svc.scheduler.tick(T0)  # type: ignore[attr-defined]
    assert counter.calls == 1
