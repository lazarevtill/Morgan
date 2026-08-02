"""The in-process bus must dispatch subscribers off the request path.

`publish()` enqueues and returns immediately; handlers run on a background drain task.
Against the old inline bus (pre-Task-15) `publish` awaited every handler directly, so the
request path blocked on cold-path work like consolidation. These tests prove it no longer does.
"""

import asyncio

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import Event, EventType


async def test_publish_returns_while_the_handler_is_still_running():
    """Against the old inline bus, `publish` would not return until `slow` finished."""
    bus = InProcessBus()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_event):
        started.set()
        await release.wait()

    bus.subscribe(EventType.RESPONSE_GENERATED, slow)
    await bus.start()

    # A timeout, not a bare assert: the old bus HANGS here rather than failing cleanly.
    await asyncio.wait_for(
        bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={})),
        timeout=1.0,
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not release.is_set()  # handler is genuinely still blocked
    release.set()
    await bus.drain()
    await bus.stop()


async def test_drain_runs_every_queued_handler():
    bus = InProcessBus()
    seen: list[Event] = []

    async def collect(event):
        seen.append(event)

    bus.subscribe(EventType.RESPONSE_GENERATED, collect)
    await bus.start()
    for _ in range(3):
        await bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={}))
    await bus.drain()
    assert len(seen) == 3
    await bus.stop()


async def test_full_queue_drops_rather_than_blocks():
    """Back-pressure must never block the request path, even when the drain stalls."""
    bus = InProcessBus(queue_size=1)
    gate = asyncio.Event()

    async def stalled(_event):
        await gate.wait()

    bus.subscribe(EventType.RESPONSE_GENERATED, stalled)
    await bus.start()

    for _ in range(5):
        await asyncio.wait_for(
            bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={})),
            timeout=1.0,
        )
    assert bus.dropped > 0, "a full queue must drop and count, not block"

    gate.set()
    await bus.drain()
    await bus.stop()
