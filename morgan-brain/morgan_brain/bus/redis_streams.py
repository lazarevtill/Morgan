"""Redis Streams event bus — cross-service backend (brain-api ↔ learning-worker ↔
perception-gpu). Same EventBus Protocol as the in-process bus.

Phase 0 stub: the consumer loop is sketched, not wired. Implemented in Phase 1+ when the
learning-worker becomes a separate process.
"""
from __future__ import annotations

from collections import defaultdict

from morgan_brain.interfaces.events import Event, EventType, Handler

_STREAM = "morgan:events"


class RedisStreamsBus:
    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        raise NotImplementedError("RedisStreamsBus.publish — wired in Phase 1")

    async def start(self) -> None:
        raise NotImplementedError("RedisStreamsBus consumer loop — wired in Phase 1")

    async def stop(self) -> None:
        self._running = False
