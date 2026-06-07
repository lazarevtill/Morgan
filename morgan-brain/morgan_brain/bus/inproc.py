"""In-process event bus. Handlers run in the same process — used when all modules share
brain-api. Satisfies the EventBus Protocol exactly like the Redis backend.
"""
from __future__ import annotations

from collections import defaultdict

from morgan_brain.interfaces.events import Event, EventType, Handler


class InProcessBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            await handler(event)

    async def start(self) -> None:  # no-op for in-process
        return None

    async def stop(self) -> None:
        return None
