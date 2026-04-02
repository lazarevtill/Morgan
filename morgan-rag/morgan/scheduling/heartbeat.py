"""
HeartbeatManager — approximate-interval scheduling with jitter and batching.

Unlike cron (exact timing, isolated), heartbeat checks are lightweight,
conversational, and run in batches with intentional timing jitter.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from typing import Any, Callable

from morgan.scheduling.jobs import HeartbeatCheck

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Manages periodic heartbeat checks.

    Parameters:
        interval_minutes: Base interval between beats (subject to jitter).
        checks_per_beat:  Maximum number of checks executed per beat.
    """

    def __init__(
        self,
        interval_minutes: float = 30,
        checks_per_beat: int = 3,
    ) -> None:
        self._interval_minutes = interval_minutes
        self._checks_per_beat = checks_per_beat
        self._checks: dict[str, HeartbeatCheck] = {}
        self._result_handler: Callable[..., Any] | None = None
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None

    # -- Registration -------------------------------------------------------

    def register_check(
        self,
        name: str,
        *,
        fn: Callable[..., Any],
        priority: int = 0,
    ) -> None:
        """Register (or replace) a heartbeat check.

        Parameters:
            name:     Unique name for the check.
            fn:       Sync or async callable returning a result.
            priority: Higher value = higher scheduling priority.
        """
        self._checks[name] = HeartbeatCheck(name=name, fn=fn, priority=priority)

    def set_result_handler(self, handler: Callable[..., Any]) -> None:
        """Set a callback invoked after each beat with the list of results.

        The handler receives a single argument: ``list[dict]`` where each dict
        has keys ``name``, ``result``, and optionally ``error``.
        """
        self._result_handler = handler

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat loop as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # -- Core logic ---------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Run beats at jittered intervals until stopped."""
        while self._running:
            try:
                await self._run_beat()
            except Exception:
                logger.exception("Heartbeat beat failed")

            sleep_seconds = self._interval_minutes * 60 * self._jitter()
            try:
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                break

    async def _run_beat(self) -> list[dict[str, Any]]:
        """Execute one heartbeat cycle.

        1. Sort checks by ``(last_run, -priority)`` — least-recently-run and
           highest-priority first.
        2. Pick the top ``checks_per_beat``.
        3. Run each check (sync or async), record ``last_run``.
        4. Collect results and call the result handler if set.

        Returns:
            List of result dicts: ``[{"name": ..., "result": ..., "error": ...}]``
        """
        if not self._checks:
            return []

        ordered = sorted(
            self._checks.values(),
            key=lambda c: (c.last_run, -c.priority),
        )
        batch = ordered[: self._checks_per_beat]

        results: list[dict[str, Any]] = []
        for check in batch:
            entry: dict[str, Any] = {"name": check.name}
            try:
                if inspect.iscoroutinefunction(check.fn):
                    entry["result"] = await check.fn()
                else:
                    entry["result"] = check.fn()
            except Exception as exc:
                entry["error"] = str(exc)
                logger.warning("Heartbeat check '%s' failed: %s", check.name, exc)
            check.last_run = time.time()
            results.append(entry)

        if self._result_handler is not None:
            if inspect.iscoroutinefunction(self._result_handler):
                await self._result_handler(results)
            else:
                self._result_handler(results)

        return results

    # -- Helpers ------------------------------------------------------------

    def _jitter(self) -> float:
        """Return a random multiplier in [0.8, 1.2] for interval jitter."""
        return random.uniform(0.8, 1.2)
