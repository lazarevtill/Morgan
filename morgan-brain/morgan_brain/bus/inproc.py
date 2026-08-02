"""In-process event bus. Handlers run in the same process — used when all modules share
brain-api. Satisfies the EventBus Protocol exactly like the Redis backend.

``publish()`` enqueues the event and returns immediately; a background drain task dispatches
queued events to their subscribers one at a time. This keeps the request path (hot path) free
of cold-path work such as consolidation — the "hot path reads, cold path writes" invariant from
the design doc holds only if ``publish()`` never awaits a handler inline.

The queue is bounded: if it fills up (the drain task is stalled or overwhelmed), ``publish()``
drops the event and increments ``dropped`` rather than blocking the caller. Back-pressure must
never propagate into the request path.

Crash recovery is deliberately NOT built here: queued work is derived from durable signal rows
written synchronously by ``Orchestrator._persist_turn`` before the event is ever published, so a
process crash with events still queued loses scheduling (the queued dispatch), not data — the
underlying rows are already on disk and get picked up by the next consolidation pass.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from morgan_brain.interfaces.events import Event, EventType, Handler

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 1000


class InProcessBus:
    def __init__(self, queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._drain_task: asyncio.Task[None] | None = None
        self.dropped = 0

    @property
    def is_running(self) -> bool:
        """True between ``start()`` and ``stop()`` — lets the app lifespan (and its tests)
        confirm the bus was actually brought up, not just constructed."""
        return self._running

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Enqueue the event and return. Never awaits a handler — that's the drain task's job."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning(
                "inproc bus queue full, dropping event type=%s user_id=%s dropped_total=%d",
                event.type.value,
                event.user_id,
                self.dropped,
            )

    async def _drain_loop(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                for handler in self._handlers.get(event.type, []):
                    await handler(event)
            except Exception:
                logger.exception(
                    "inproc bus handler failed type=%s user_id=%s",
                    event.type.value,
                    event.user_id,
                )
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        """Block until every currently-queued event has been dispatched. Tests use this for
        deterministic assertions instead of racing the background drain task."""
        await self._queue.join()

    async def start(self) -> None:
        self._running = True
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        self._running = False
        if self._drain_task is not None:
            await self.drain()
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
