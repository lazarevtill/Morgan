"""Run the learning-worker: ``python -m morgan_brain.apps.learning_worker``.

Phase 0: subscribes to RESPONSE_GENERATED / SESSION_END and logs. Phase 2+ replaces the handler
body with extraction → UserModel update → consolidation, all off the request path.
"""
from __future__ import annotations

import asyncio

import structlog

from morgan_brain.bus import get_event_bus
from morgan_brain.interfaces.events import Event, EventType

log = structlog.get_logger("learning-worker")


async def _on_response(event: Event) -> None:
    # Phase 2: queue the session for trait/preference/behavior extraction.
    log.info("response.observed", user_id=event.user_id)


async def run() -> None:
    bus = get_event_bus()
    bus.subscribe(EventType.RESPONSE_GENERATED, _on_response)
    bus.subscribe(EventType.SESSION_END, _on_response)
    await bus.start()
    log.info("learning-worker.started")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await bus.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
