"""Unit tests for RedisStreamsBus.

All tests use a deterministic FakeRedis injected via ``client=``; no live Redis is
required or contacted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from morgan_brain.bus.redis_streams import RedisStreamsBus
from morgan_brain.interfaces.events import Event, EventType


# ---------------------------------------------------------------------------
# Fake async redis client
# ---------------------------------------------------------------------------


class FakeRedis:
    """Minimal async redis stub that records calls and serves scripted responses."""

    def __init__(self) -> None:
        self.xadd_calls: list[dict[str, Any]] = []
        self.xgroup_create_calls: list[dict[str, Any]] = []
        self.xreadgroup_calls: list[dict[str, Any]] = []
        self.xack_calls: list[dict[str, Any]] = []
        # Scripted xreadgroup responses: list of return values (consumed in order).
        # Each entry is what xreadgroup should return.
        self._xreadgroup_responses: list[list[Any]] = []
        self.closed = False

    def queue_messages(self, *batches: list[Any]) -> None:
        """Queue xreadgroup return values to be consumed one call at a time."""
        self._xreadgroup_responses.extend(batches)

    async def xadd(self, stream: str, fields: dict[str, str]) -> str:
        self.xadd_calls.append({"stream": stream, "fields": fields})
        return "0-1"

    async def xgroup_create(
        self, stream: str, group: str, id: str = "$", mkstream: bool = False
    ) -> None:
        self.xgroup_create_calls.append(
            {"stream": stream, "group": group, "id": id, "mkstream": mkstream}
        )

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        count: int = 10,
        block: int | None = None,
    ) -> list[Any]:
        self.xreadgroup_calls.append(
            {"group": group, "consumer": consumer, "streams": streams, "count": count}
        )
        if self._xreadgroup_responses:
            return self._xreadgroup_responses.pop(0)
        return []

    async def xack(self, stream: str, group: str, *msg_ids: str) -> int:
        for mid in msg_ids:
            self.xack_calls.append({"stream": stream, "group": group, "msg_id": mid})
        return len(msg_ids)

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(
    event_type: EventType = EventType.MESSAGE_RECEIVED,
    user_id: str = "u1",
    **payload: Any,
) -> Event:
    return Event(type=event_type, user_id=user_id, payload=dict(payload))


def _raw_message(event: Event, msg_id: str = "1-0") -> tuple[str, list[tuple[str, dict[str, str]]]]:
    """Return a (stream_name, messages) tuple as xreadgroup would deliver."""
    return ("morgan:events", [(msg_id, {"data": event.model_dump_json()})])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_does_not_contact_redis() -> None:
    """Constructing RedisStreamsBus(url) must not open any connection."""
    # If this test just completes without raising, no connection was made.
    bus = RedisStreamsBus("redis://localhost:6379/0")
    assert bus._client is None  # lazy — not yet created


def test_construction_with_injected_client() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    assert bus._client is fake
    # We did NOT create it so we should NOT close it on stop.
    assert bus._client_owned is False


# ---------------------------------------------------------------------------
# subscribe
# ---------------------------------------------------------------------------


def test_subscribe_registers_handler() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    handler = AsyncMock()
    bus.subscribe(EventType.MESSAGE_RECEIVED, handler)
    assert handler in bus._handlers[EventType.MESSAGE_RECEIVED]


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_calls_xadd_with_stream_and_json() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake, stream="morgan:events")
    event = make_event(EventType.MESSAGE_RECEIVED, "alice", text="hello")

    await bus.publish(event)

    assert len(fake.xadd_calls) == 1
    call = fake.xadd_calls[0]
    assert call["stream"] == "morgan:events"
    assert "data" in call["fields"]


@pytest.mark.asyncio
async def test_publish_roundtrip() -> None:
    """Data captured by xadd must round-trip back to the original Event."""
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    original = make_event(EventType.HEARTBEAT, "bob", tick=42)

    await bus.publish(original)

    captured_json = fake.xadd_calls[0]["fields"]["data"]
    recovered = Event.model_validate_json(captured_json)
    assert recovered == original


@pytest.mark.asyncio
async def test_publish_uses_custom_stream_name() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake, stream="custom:stream")
    await bus.publish(make_event())
    assert fake.xadd_calls[0]["stream"] == "custom:stream"


# ---------------------------------------------------------------------------
# _handle_message dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_calls_subscribed_handler() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    event = make_event(EventType.SESSION_START, "carol")
    bus.subscribe(EventType.SESSION_START, handler)

    await bus._handle_message("1-0", {"data": event.model_dump_json()})

    assert len(received) == 1
    assert received[0] == event


@pytest.mark.asyncio
async def test_handle_message_does_not_call_handler_for_wrong_type() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    wrong_type_called = False

    async def wrong_handler(event: Event) -> None:
        nonlocal wrong_type_called
        wrong_type_called = True

    bus.subscribe(EventType.MEMORY_STORED, wrong_handler)
    event = make_event(EventType.SESSION_START, "dave")

    await bus._handle_message("1-1", {"data": event.model_dump_json()})

    assert not wrong_type_called


@pytest.mark.asyncio
async def test_handle_message_calls_multiple_handlers_for_type() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    calls: list[str] = []

    async def h1(event: Event) -> None:
        calls.append("h1")

    async def h2(event: Event) -> None:
        calls.append("h2")

    bus.subscribe(EventType.HEARTBEAT, h1)
    bus.subscribe(EventType.HEARTBEAT, h2)
    event = make_event(EventType.HEARTBEAT, "eve")

    await bus._handle_message("2-0", {"data": event.model_dump_json()})

    assert calls == ["h1", "h2"]


@pytest.mark.asyncio
async def test_handle_message_bad_handler_does_not_propagate() -> None:
    """A handler that raises must NOT cause _handle_message to raise."""
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    good_received: list[Event] = []

    async def bad_handler(event: Event) -> None:
        raise RuntimeError("handler blew up")

    async def good_handler(event: Event) -> None:
        good_received.append(event)

    bus.subscribe(EventType.HEARTBEAT, bad_handler)
    bus.subscribe(EventType.HEARTBEAT, good_handler)
    event = make_event(EventType.HEARTBEAT, "frank")

    # Should NOT raise.
    await bus._handle_message("3-0", {"data": event.model_dump_json()})

    # The good handler still runs.
    assert len(good_received) == 1


@pytest.mark.asyncio
async def test_handle_message_xack_is_called() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake, stream="s:test", group="grp")
    event = make_event(EventType.HEARTBEAT, "gina")

    await bus._handle_message("5-0", {"data": event.model_dump_json()})

    assert len(fake.xack_calls) == 1
    ack = fake.xack_calls[0]
    assert ack["stream"] == "s:test"
    assert ack["group"] == "grp"
    assert ack["msg_id"] == "5-0"


@pytest.mark.asyncio
async def test_handle_message_xack_called_even_when_handler_raises() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)

    async def bad(event: Event) -> None:
        raise ValueError("boom")

    bus.subscribe(EventType.HEARTBEAT, bad)
    event = make_event(EventType.HEARTBEAT, "hank")

    await bus._handle_message("6-0", {"data": event.model_dump_json()})

    # xack must have been called despite the handler error.
    assert len(fake.xack_calls) == 1


@pytest.mark.asyncio
async def test_handle_message_xack_called_for_invalid_json() -> None:
    """A malformed message should be acked (not re-delivered forever)."""
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)

    await bus._handle_message("7-0", {"data": "{not valid json}"})

    assert len(fake.xack_calls) == 1
    assert fake.xack_calls[0]["msg_id"] == "7-0"


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_creates_consumer_group() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake, stream="s", group="g")

    await bus.start()
    await bus.stop()

    assert len(fake.xgroup_create_calls) == 1
    gc = fake.xgroup_create_calls[0]
    assert gc["stream"] == "s"
    assert gc["group"] == "g"
    assert gc["mkstream"] is True


@pytest.mark.asyncio
async def test_start_swallows_busygroup_error() -> None:
    """BUSYGROUP means the group already exists — that's fine."""

    class BusyRedis(FakeRedis):
        async def xgroup_create(self, *args: Any, **kwargs: Any) -> None:
            raise Exception("BUSYGROUP Consumer Group name already exists")

    bus = RedisStreamsBus("redis://unused", client=BusyRedis())
    await bus.start()  # must NOT raise
    await bus.stop()


@pytest.mark.asyncio
async def test_start_re_raises_non_busygroup_error() -> None:
    class BrokenRedis(FakeRedis):
        async def xgroup_create(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("WRONGTYPE Operation against a key holding wrong kind")

    bus = RedisStreamsBus("redis://unused", client=BrokenRedis())
    with pytest.raises(RuntimeError):
        await bus.start()


@pytest.mark.asyncio
async def test_stop_sets_running_false() -> None:
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    await bus.start()
    assert bus._running is True
    await bus.stop()
    assert bus._running is False


@pytest.mark.asyncio
async def test_stop_closes_owned_client() -> None:
    fake = FakeRedis()
    # client=None → bus owns future client; inject post-construction for test simplicity.
    bus = RedisStreamsBus("redis://unused")
    bus._client = fake  # inject after construction (ownership flag is True)
    assert bus._client_owned is True

    await bus.stop()

    assert fake.closed is True


@pytest.mark.asyncio
async def test_stop_does_not_close_injected_client() -> None:
    """When the caller injected the client, stop() must not close it."""
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    assert bus._client_owned is False

    await bus.stop()

    assert fake.closed is False


# ---------------------------------------------------------------------------
# Consumer group name / consumer name
# ---------------------------------------------------------------------------


def test_default_group_and_stream() -> None:
    bus = RedisStreamsBus("redis://unused")
    assert bus._stream == "morgan:events"
    assert bus._group == "morgan"


def test_custom_consumer_name_used() -> None:
    bus = RedisStreamsBus("redis://unused", consumer="worker-1")
    assert bus._consumer == "worker-1"


def test_auto_consumer_name_is_unique() -> None:
    bus1 = RedisStreamsBus("redis://unused")
    bus2 = RedisStreamsBus("redis://unused")
    assert bus1._consumer != bus2._consumer


# ---------------------------------------------------------------------------
# End-to-end: publish → _handle_message (fake full round-trip without loop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_then_handle_message_delivers_event() -> None:
    """Publish writes JSON; re-feeding that JSON through _handle_message calls handler."""
    fake = FakeRedis()
    bus = RedisStreamsBus("redis://unused", client=fake)
    delivered: list[Event] = []

    async def capture(event: Event) -> None:
        delivered.append(event)

    bus.subscribe(EventType.RESPONSE_GENERATED, capture)
    original = make_event(EventType.RESPONSE_GENERATED, "iris", answer="42")

    await bus.publish(original)

    # Simulate the consume-loop path: take the xadd payload and feed it directly.
    captured_fields = fake.xadd_calls[0]["fields"]
    await bus._handle_message("10-0", captured_fields)

    assert len(delivered) == 1
    assert delivered[0] == original
