"""Unit tests for HeartbeatManager (Phase 4, commit 1).

All tests are deterministic:
  - Injected fake clock.
  - Injected jitter_offsets sequence for reproducible next_delay values.
  - InProcessBus captures emitted events.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.scheduling.heartbeat import HeartbeatManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 0, 0, 0)

FIXED_OFFSETS = (0.0, 10.0, -10.0)  # simple deterministic sequence


def _make_bus_and_collector():
    """Return (bus, list_that_accumulates_events)."""
    bus = InProcessBus()
    collected: list[Event] = []

    async def _collect(event: Event) -> None:
        collected.append(event)

    bus.subscribe(EventType.HEARTBEAT, _collect)
    return bus, collected


# ---------------------------------------------------------------------------
# beat() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_beat_emits_heartbeat_event() -> None:
    bus, events = _make_bus_and_collector()
    mgr = HeartbeatManager(
        bus=bus,
        clock=lambda: T0,
        jitter_offsets=FIXED_OFFSETS,
    )
    await mgr.beat()
    assert len(events) == 1
    assert events[0].type == EventType.HEARTBEAT


@pytest.mark.asyncio
async def test_beat_event_has_correct_user_id() -> None:
    bus, events = _make_bus_and_collector()
    mgr = HeartbeatManager(
        bus=bus,
        clock=lambda: T0,
        user_id="alice",
        jitter_offsets=FIXED_OFFSETS,
    )
    await mgr.beat()
    assert events[0].user_id == "alice"


@pytest.mark.asyncio
async def test_beat_increments_beat_count_in_payload() -> None:
    bus, events = _make_bus_and_collector()
    mgr = HeartbeatManager(bus=bus, clock=lambda: T0, jitter_offsets=FIXED_OFFSETS)
    await mgr.beat()
    await mgr.beat()
    assert events[0].payload["beat"] == 1
    assert events[1].payload["beat"] == 2


@pytest.mark.asyncio
async def test_beat_payload_contains_timestamp() -> None:
    bus, events = _make_bus_and_collector()
    mgr = HeartbeatManager(bus=bus, clock=lambda: T0, jitter_offsets=FIXED_OFFSETS)
    await mgr.beat()
    assert "ts" in events[0].payload
    assert events[0].payload["ts"] == T0.isoformat()


@pytest.mark.asyncio
async def test_beat_multiple_emits_multiple_events() -> None:
    bus, events = _make_bus_and_collector()
    mgr = HeartbeatManager(bus=bus, clock=lambda: T0, jitter_offsets=FIXED_OFFSETS)
    for _ in range(5):
        await mgr.beat()
    assert len(events) == 5


# ---------------------------------------------------------------------------
# next_delay() tests
# ---------------------------------------------------------------------------


def test_next_delay_deterministic_sequence() -> None:
    """next_delay cycles through jitter_offsets deterministically."""
    bus, _ = _make_bus_and_collector()
    # offsets = (0, 10, -10); interval = 300
    mgr = HeartbeatManager(
        bus=bus,
        clock=lambda: T0,
        interval_seconds=300.0,
        jitter_offsets=FIXED_OFFSETS,
    )
    delays = [mgr.next_delay() for _ in range(6)]
    # Expect: 300, 310, 290, 300, 310, 290 (cycling)
    assert delays == [300.0, 310.0, 290.0, 300.0, 310.0, 290.0]


def test_next_delay_minimum_is_one_second() -> None:
    """next_delay is clamped to at least 1 s."""
    bus, _ = _make_bus_and_collector()
    mgr = HeartbeatManager(
        bus=bus,
        clock=lambda: T0,
        interval_seconds=0.0,
        jitter_offsets=(-100.0,),
    )
    assert mgr.next_delay() >= 1.0


def test_next_delay_within_expected_bounds() -> None:
    """Without explicit offsets, delays stay within interval ± jitter_seconds."""
    bus, _ = _make_bus_and_collector()
    mgr = HeartbeatManager(
        bus=bus,
        clock=lambda: T0,
        interval_seconds=300.0,
        jitter_seconds=30.0,
    )
    for _ in range(20):
        d = mgr.next_delay()
        assert 270.0 <= d <= 330.0, f"delay {d} out of expected range"


def test_next_delay_independent_of_clock() -> None:
    """next_delay does not depend on the clock — it is purely offset-based."""
    bus1, _ = _make_bus_and_collector()
    bus2, _ = _make_bus_and_collector()
    mgr1 = HeartbeatManager(
        bus=bus1,
        clock=lambda: datetime(2025, 1, 1),
        interval_seconds=60.0,
        jitter_offsets=(5.0,),
    )
    mgr2 = HeartbeatManager(
        bus=bus2,
        clock=lambda: datetime(2030, 6, 15),
        interval_seconds=60.0,
        jitter_offsets=(5.0,),
    )
    assert mgr1.next_delay() == mgr2.next_delay()
