"""HeartbeatManager — periodic HEARTBEAT events on the event bus.

Design principles
-----------------
* Deterministic jitter: jitter is derived from a cycling sequence of offsets
  (injected or defaulting to ``[0, 15, -10, 20, -5, 30, -15]`` seconds) rather
  than ``random``.  This keeps tests reproducible.
* Clock-injected: ``next_delay()`` never calls ``time.time()`` or
  ``datetime.now()`` — all time is supplied by the injected *clock*.
* The bus is called with a fully-constructed ``Event``; no knowledge of channel
  routing or delivery (that is Phase 5).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import datetime

from morgan_brain.interfaces.events import Event, EventBus, EventType

# Default jitter offsets in seconds (deterministic sequence, cycling).
_DEFAULT_JITTER_OFFSETS: tuple[float, ...] = (0.0, 15.0, -10.0, 20.0, -5.0, 30.0, -15.0)


class HeartbeatManager:
    """Publishes HEARTBEAT events on a jittered schedule.

    Parameters
    ----------
    bus:
        The :class:`EventBus` to publish events to.
    clock:
        Zero-argument callable returning the current :class:`datetime`.
    interval_seconds:
        Base interval between heartbeats (default: 300 s / 5 min).
    jitter_seconds:
        Maximum absolute jitter added to the interval (default: 30 s).
    jitter_offsets:
        Explicit sequence of jitter offsets (in seconds) that cycles
        deterministically.  When provided, ``jitter_seconds`` is ignored
        and the cycle values are used directly.  Intended for tests.
    user_id:
        The user_id to embed in emitted events (default: ``"system"``).
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        clock: Callable[[], datetime],
        interval_seconds: float = 300.0,
        jitter_seconds: float = 30.0,
        jitter_offsets: tuple[float, ...] | list[float] | None = None,
        user_id: str = "system",
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._interval_seconds = interval_seconds
        self._jitter_seconds = jitter_seconds
        self._user_id = user_id

        # Build a cycling iterator of jitter offsets.
        if jitter_offsets is not None:
            offsets: tuple[float, ...] = tuple(jitter_offsets)
        else:
            # Scale the default offsets to respect jitter_seconds.
            scale = jitter_seconds / 30.0  # default offsets designed for 30 s max
            offsets = tuple(o * scale for o in _DEFAULT_JITTER_OFFSETS)

        self._jitter_cycle = itertools.cycle(offsets)
        self._beat_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def beat(self) -> None:
        """Publish one HEARTBEAT event on the bus."""
        now = self._clock()
        self._beat_count += 1
        event = Event(
            type=EventType.HEARTBEAT,
            user_id=self._user_id,
            payload={"beat": self._beat_count, "ts": now.isoformat()},
        )
        await self._bus.publish(event)

    def next_delay(self) -> float:
        """Return the number of seconds to wait before the next beat.

        The delay is ``interval_seconds + jitter`` where jitter cycles
        deterministically through ``jitter_offsets``.  The returned value is
        always >= 1 s to avoid busy-loops.
        """
        jitter = next(self._jitter_cycle)
        delay = self._interval_seconds + jitter
        return max(1.0, delay)
