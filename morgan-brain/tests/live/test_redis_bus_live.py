"""Live end-to-end test for RedisStreamsBus.

Requires a real Redis instance accessible at ``MORGAN_REDIS_URL``
(default: ``redis://localhost:6379/0``).

Run with::

    python -m pytest --live tests/live/test_redis_bus_live.py -v

All tests are marked ``@pytest.mark.live`` and are SKIPPED by default.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from morgan_brain.bus.redis_streams import RedisStreamsBus
from morgan_brain.interfaces.events import Event, EventType

pytestmark = pytest.mark.live

_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_TIMEOUT = 10.0  # seconds to wait for a message to arrive


def _redis_url() -> str:
    return os.environ.get("MORGAN_REDIS_URL", _DEFAULT_REDIS_URL)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def publisher() -> RedisStreamsBus:  # type: ignore[misc]
    """A RedisStreamsBus used as the publisher (no start/stop needed for publish)."""
    bus = RedisStreamsBus(
        _redis_url(),
        stream="test:events:live",
        group="test-group",
        consumer="publisher",
    )
    yield bus
    await bus.stop()


@pytest.fixture()
async def subscriber() -> RedisStreamsBus:  # type: ignore[misc]
    """A RedisStreamsBus with a started consumer group for receiving messages."""
    bus = RedisStreamsBus(
        _redis_url(),
        stream="test:events:live",
        group="test-group",
        consumer="subscriber",
    )
    yield bus
    await bus.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_received_by_subscriber(
    publisher: RedisStreamsBus,
    subscriber: RedisStreamsBus,
) -> None:
    """An event published by one bus instance is received by a second instance.

    This validates the two-process path: brain-api (publisher) and
    learning-worker (subscriber) sharing the same Redis stream + group.
    """
    received: list[Event] = []
    ready = asyncio.Event()

    async def handler(event: Event) -> None:
        received.append(event)
        ready.set()

    subscriber.subscribe(EventType.MESSAGE_RECEIVED, handler)
    await subscriber.start()

    # Give the consume loop a moment to begin reading.
    await asyncio.sleep(0.1)

    original = Event(
        type=EventType.MESSAGE_RECEIVED,
        user_id="live-test-user",
        payload={"text": "hello from live test"},
    )
    await publisher.publish(original)

    # Wait up to _TIMEOUT seconds for the handler to fire.
    try:
        await asyncio.wait_for(ready.wait(), timeout=_TIMEOUT)
    except TimeoutError:
        pytest.fail(
            f"Subscriber did not receive the event within {_TIMEOUT}s. "
            "Is Redis running at %s?" % _redis_url()
        )

    assert len(received) == 1
    assert received[0].type == EventType.MESSAGE_RECEIVED
    assert received[0].user_id == "live-test-user"
    assert received[0].payload["text"] == "hello from live test"


@pytest.mark.asyncio
async def test_multiple_event_types_routed_correctly(
    publisher: RedisStreamsBus,
    subscriber: RedisStreamsBus,
) -> None:
    """Multiple event types are each delivered only to the correct handler."""
    heartbeats: list[Event] = []
    sessions: list[Event] = []
    both_ready = asyncio.Event()

    async def on_heartbeat(event: Event) -> None:
        heartbeats.append(event)
        if heartbeats and sessions:
            both_ready.set()

    async def on_session(event: Event) -> None:
        sessions.append(event)
        if heartbeats and sessions:
            both_ready.set()

    subscriber.subscribe(EventType.HEARTBEAT, on_heartbeat)
    subscriber.subscribe(EventType.SESSION_START, on_session)
    await subscriber.start()
    await asyncio.sleep(0.1)

    await publisher.publish(Event(type=EventType.HEARTBEAT, user_id="u1", payload={"tick": 1}))
    await publisher.publish(
        Event(type=EventType.SESSION_START, user_id="u2", payload={"session": "s1"})
    )

    try:
        await asyncio.wait_for(both_ready.wait(), timeout=_TIMEOUT)
    except TimeoutError:
        pytest.fail("Did not receive both event types within timeout.")

    assert len(heartbeats) >= 1
    assert all(e.type == EventType.HEARTBEAT for e in heartbeats)
    assert len(sessions) >= 1
    assert all(e.type == EventType.SESSION_START for e in sessions)
