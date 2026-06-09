"""Unit tests for InteractionSignal, SignalStore (TDD, Phase 2 Increment A)."""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.learning.signals import InteractionSignal, SignalStore, Thumb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLOCK = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731


@pytest.fixture
def store() -> SignalStore:
    return SignalStore(clock=CLOCK)


def _signal(user_id: str = "u1", turn_id: str = "t1", **overrides: object) -> InteractionSignal:
    defaults: dict[str, object] = {
        "user_id": user_id,
        "session_id": "s1",
        "turn_id": turn_id,
        "query": "hello",
        "original_reply": "hi",
    }
    defaults.update(overrides)
    return InteractionSignal(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# value_rank ordering
# ---------------------------------------------------------------------------


def test_value_rank_edit() -> None:
    sig = _signal(user_edit="better reply")
    assert sig.value_rank == 3


def test_value_rank_retry() -> None:
    sig = _signal(retried=True)
    assert sig.value_rank == 2


def test_value_rank_thumb_down() -> None:
    sig = _signal(thumb=Thumb.DOWN)
    assert sig.value_rank == 2


def test_value_rank_thumb_up() -> None:
    sig = _signal(thumb=Thumb.UP)
    assert sig.value_rank == 1


def test_value_rank_none() -> None:
    sig = _signal()
    assert sig.value_rank == 0


def test_edit_rank_dominates_thumb_up() -> None:
    """Edit (3) should beat thumb-up (1) even when both set."""
    sig = _signal(user_edit="foo", thumb=Thumb.UP)
    assert sig.value_rank == 3


# ---------------------------------------------------------------------------
# record + for_user roundtrip (newest first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_and_for_user(store: SignalStore) -> None:
    sig1 = _signal(turn_id="t1")
    sig2 = _signal(turn_id="t2")
    await store.record(sig1)
    await store.record(sig2)
    results = await store.for_user("u1")
    assert len(results) == 2
    # newest first — t2 was inserted last, so its id should come first
    assert results[0].turn_id == "t2"
    assert results[1].turn_id == "t1"


@pytest.mark.asyncio
async def test_for_user_respects_limit(store: SignalStore) -> None:
    for i in range(5):
        await store.record(_signal(turn_id=f"t{i}"))
    results = await store.for_user("u1", limit=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# user scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_scoped(store: SignalStore) -> None:
    await store.record(_signal(user_id="u1", turn_id="t1"))
    await store.record(_signal(user_id="u2", turn_id="t2"))
    u1_results = await store.for_user("u1")
    assert len(u1_results) == 1
    assert u1_results[0].user_id == "u1"
    u2_results = await store.for_user("u2")
    assert len(u2_results) == 1
    assert u2_results[0].user_id == "u2"


# ---------------------------------------------------------------------------
# high_value filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_value_filters_rank(store: SignalStore) -> None:
    await store.record(_signal(turn_id="t0"))  # rank 0
    await store.record(_signal(turn_id="t1", thumb=Thumb.UP))  # rank 1
    await store.record(_signal(turn_id="t2", retried=True))  # rank 2
    await store.record(_signal(turn_id="t3", user_edit="edited"))  # rank 3

    high = await store.high_value("u1", min_rank=2)
    turn_ids = {s.turn_id for s in high}
    assert "t2" in turn_ids
    assert "t3" in turn_ids
    assert "t0" not in turn_ids
    assert "t1" not in turn_ids


@pytest.mark.asyncio
async def test_high_value_user_scoped(store: SignalStore) -> None:
    await store.record(_signal(user_id="u1", turn_id="ta", user_edit="e"))
    await store.record(_signal(user_id="u2", turn_id="tb", user_edit="e"))
    high = await store.high_value("u1")
    assert all(s.user_id == "u1" for s in high)


# ---------------------------------------------------------------------------
# unconsumed / mark_consumed flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unconsumed_initially_all(store: SignalStore) -> None:
    await store.record(_signal(turn_id="t1"))
    await store.record(_signal(turn_id="t2"))
    unconsumed = await store.unconsumed("u1")
    assert len(unconsumed) == 2


@pytest.mark.asyncio
async def test_mark_consumed_filters(store: SignalStore) -> None:
    id1 = await store.record(_signal(turn_id="t1"))
    id2 = await store.record(_signal(turn_id="t2"))
    await store.mark_consumed([id1])
    unconsumed = await store.unconsumed("u1")
    assert len(unconsumed) == 1
    assert unconsumed[0].turn_id == "t2"
    _ = id2  # used implicitly via store


@pytest.mark.asyncio
async def test_mark_consumed_empty_list(store: SignalStore) -> None:
    await store.record(_signal(turn_id="t1"))
    await store.mark_consumed([])  # should not raise
    unconsumed = await store.unconsumed("u1")
    assert len(unconsumed) == 1


# ---------------------------------------------------------------------------
# Thumb enum roundtrip through SQLite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thumb_enum_roundtrip_down(store: SignalStore) -> None:
    sig = _signal(turn_id="t1", thumb=Thumb.DOWN)
    await store.record(sig)
    results = await store.for_user("u1")
    assert results[0].thumb is Thumb.DOWN


@pytest.mark.asyncio
async def test_thumb_enum_roundtrip_up(store: SignalStore) -> None:
    sig = _signal(turn_id="t1", thumb=Thumb.UP)
    await store.record(sig)
    results = await store.for_user("u1")
    assert results[0].thumb is Thumb.UP


@pytest.mark.asyncio
async def test_thumb_none_roundtrip(store: SignalStore) -> None:
    sig = _signal(turn_id="t1")
    await store.record(sig)
    results = await store.for_user("u1")
    assert results[0].thumb is None


# ---------------------------------------------------------------------------
# created_at is set from clock when None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_created_at_set_from_clock(store: SignalStore) -> None:
    sig = _signal(turn_id="t1")
    assert sig.created_at is None
    await store.record(sig)
    results = await store.for_user("u1")
    assert results[0].created_at == CLOCK()
