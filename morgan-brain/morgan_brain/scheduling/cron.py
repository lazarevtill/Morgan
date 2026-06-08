"""CronService + InProcessScheduler — deterministic, APScheduler-optional.

Design principles
-----------------
* ``InProcessScheduler`` is the dependency-light default AND the test substrate.
  It exposes ``async tick(now)`` so tests can advance time without real sleeping.
* ``CronService`` is a thin facade.  If APScheduler is importable **and** the
  caller explicitly enables it, a future APScheduler-backed implementation could
  be plugged in; for now, the facade always uses ``InProcessScheduler``.
* No ``datetime.now()`` calls inside any class — always receive the clock from
  the outside (``clock: Callable[[], datetime]``).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class Job(BaseModel):
    """Descriptor for a scheduled job.

    Either ``interval_seconds`` or ``cron`` must be set (not mutually
    exclusive — cron is accepted but ``InProcessScheduler`` approximates it
    via interval_seconds; a real APScheduler-backed implementation would use
    the cron expression directly).
    """

    name: str
    interval_seconds: float | None = None
    cron: str | None = None


# A job callable may be sync or async; we normalise in the scheduler.
JobFn = Callable[[], Any]


# ---------------------------------------------------------------------------
# Scheduler Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Scheduler(Protocol):
    """Minimal interface for a job scheduler."""

    def add_job(
        self,
        name: str,
        fn: JobFn,
        *,
        interval_seconds: float | None = None,
        cron: str | None = None,
    ) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def list_jobs(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# InProcessScheduler — deterministic; tests call tick() directly
# ---------------------------------------------------------------------------


class _JobEntry:
    """Internal record for a registered interval job."""

    def __init__(self, name: str, fn: JobFn, interval_seconds: float) -> None:
        self.name = name
        self.fn = fn
        self.interval_seconds = interval_seconds
        self.last_run: datetime | None = None


class InProcessScheduler:
    """Lightweight scheduler that runs interval jobs when ``tick`` is called.

    This is both the production fallback (no APScheduler dependency) and the
    test substrate.  Tests call ``await scheduler.tick(some_datetime)`` to
    advance the logical clock and verify that due jobs are invoked.

    Cron expressions are accepted (stored in the ``Job`` descriptor) but
    currently **approximated** by treating ``interval_seconds`` as the period.
    A full cron parser is not bundled; use APScheduler if exact cron semantics
    are required.
    """

    def __init__(self, *, clock: Callable[[], datetime]) -> None:
        self._clock = clock
        self._jobs: dict[str, _JobEntry] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Scheduler Protocol
    # ------------------------------------------------------------------

    def add_job(
        self,
        name: str,
        fn: JobFn,
        *,
        interval_seconds: float | None = None,
        cron: str | None = None,
    ) -> None:
        """Register a job.

        At least one of *interval_seconds* or *cron* must be provided.  When
        only *cron* is given, the expression is stored for reference but the
        scheduler uses a default interval of 3600 s as a conservative
        approximation (override by also passing ``interval_seconds``).
        """
        if interval_seconds is None and cron is None:
            raise ValueError(f"Job {name!r}: must specify interval_seconds or cron")

        effective_interval: float
        if interval_seconds is not None:
            effective_interval = interval_seconds
        else:
            # Approximate: cron-only job fires every hour by default.
            effective_interval = 3600.0

        self._jobs[name] = _JobEntry(name, fn, effective_interval)
        logger.debug("InProcessScheduler: registered job %r (interval=%.0fs)", name, effective_interval)

    async def start(self) -> None:
        self._running = True
        logger.debug("InProcessScheduler: started")

    async def stop(self) -> None:
        self._running = False
        logger.debug("InProcessScheduler: stopped")

    def list_jobs(self) -> list[str]:
        return list(self._jobs.keys())

    # ------------------------------------------------------------------
    # Deterministic tick — called by tests and the optional real loop
    # ------------------------------------------------------------------

    async def tick(self, now: datetime) -> None:
        """Evaluate all jobs against *now* and run those that are due.

        A job is **due** when:
        - It has never run (``last_run is None``), OR
        - ``(now - last_run).total_seconds() >= interval_seconds``.

        Jobs are run sequentially in registration order.  Both sync and async
        callables are supported.
        """
        for entry in self._jobs.values():
            due = entry.last_run is None or (
                (now - entry.last_run).total_seconds() >= entry.interval_seconds
            )
            if not due:
                continue
            logger.debug("InProcessScheduler: running job %r at %s", entry.name, now)
            entry.last_run = now
            try:
                result = entry.fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("InProcessScheduler: job %r raised", entry.name)


# ---------------------------------------------------------------------------
# CronService — thin facade over a Scheduler implementation
# ---------------------------------------------------------------------------


class CronService:
    """Thin facade that provides a stable API regardless of the backend.

    Backend selection (in order):
    1. If *scheduler* is explicitly passed → use it directly.
    2. ``InProcessScheduler`` (default; no optional deps required).

    A future APScheduler integration can be added behind the ``enable_scheduling``
    flag without changing callers.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        scheduler: Scheduler | None = None,
    ) -> None:
        self._clock = clock
        if scheduler is not None:
            self._scheduler: Scheduler = scheduler
        else:
            self._scheduler = InProcessScheduler(clock=clock)

    # ------------------------------------------------------------------
    # Public API (mirrors Scheduler Protocol, not re-implementing the
    # Protocol since CronService adds higher-level register() method)
    # ------------------------------------------------------------------

    def register(
        self,
        job: Job,
        fn: JobFn,
    ) -> None:
        """Register *job* with the underlying scheduler."""
        self._scheduler.add_job(
            job.name,
            fn,
            interval_seconds=job.interval_seconds,
            cron=job.cron,
        )

    async def start(self) -> None:
        await self._scheduler.start()

    async def stop(self) -> None:
        await self._scheduler.stop()

    def list(self) -> list[str]:
        """Return the names of all registered jobs."""
        return self._scheduler.list_jobs()

    async def tick(self, now: datetime) -> None:
        """Drive a tick on the underlying scheduler (no-op if not InProcessScheduler).

        This provides a type-safe way for the run loop and tests to advance the
        scheduler clock without casting to InProcessScheduler.
        """
        if isinstance(self._scheduler, InProcessScheduler):
            await self._scheduler.tick(now)

    # Expose the underlying scheduler for tests that need tick() directly.
    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler
