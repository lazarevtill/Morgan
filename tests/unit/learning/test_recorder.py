"""Unit tests for SignalRecorder facade (TDD, Phase 2 Increment A)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.learning.recorder import SignalRecorder
from morgan_brain.learning.signals import SignalStore, Thumb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731


@pytest.fixture
def store() -> SignalStore:
    return SignalStore(clock=FIXED_CLOCK)


@pytest.fixture
def recorder(store: SignalStore) -> SignalRecorder:
    return SignalRecorder(store=store, clock=FIXED_CLOCK)


# ---------------------------------------------------------------------------
# record_turn → add_edit updates the SAME row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edit_updates_existing_row(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1",
        project="p",
        session_id="s1",
        turn_id="t1",
        query="what time is it?",
        reply="It is noon.",
    )
    await recorder.add_edit(turn_id="t1", user_id="u1", edited_reply="It is 12:00.")

    results = await store.for_user("u1")
    assert len(results) == 1  # same row, not a new one
    sig = results[0]
    assert sig.user_edit == "It is 12:00."
    assert sig.value_rank == 3


# ---------------------------------------------------------------------------
# add_thumb sets thumb
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_thumb_up(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1", project="p", session_id="s1", turn_id="t1", query="q", reply="r"
    )
    await recorder.add_thumb(turn_id="t1", user_id="u1", thumb=Thumb.UP)

    results = await store.for_user("u1")
    assert len(results) == 1
    assert results[0].thumb is Thumb.UP
    assert results[0].value_rank == 1  # thumb-up = low-trust rank 1


@pytest.mark.asyncio
async def test_add_thumb_down(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1", project="p", session_id="s1", turn_id="t1", query="q", reply="r"
    )
    await recorder.add_thumb(turn_id="t1", user_id="u1", thumb=Thumb.DOWN)

    results = await store.for_user("u1")
    assert results[0].thumb is Thumb.DOWN
    assert results[0].value_rank == 2


# ---------------------------------------------------------------------------
# add_retry sets retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_retry(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1", project="p", session_id="s1", turn_id="t1", query="q", reply="r"
    )
    await recorder.add_retry(turn_id="t1", user_id="u1")

    results = await store.for_user("u1")
    assert len(results) == 1
    assert results[0].retried is True
    assert results[0].value_rank == 2


# ---------------------------------------------------------------------------
# add_edit when no base signal creates one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edit_creates_signal_when_missing(
    recorder: SignalRecorder, store: SignalStore
) -> None:
    await recorder.add_edit(turn_id="orphan", user_id="u1", edited_reply="better")

    results = await store.for_user("u1")
    assert len(results) == 1
    sig = results[0]
    assert sig.turn_id == "orphan"
    assert sig.user_edit == "better"
    assert sig.value_rank == 3
    assert sig.created_at == FIXED_CLOCK()


# ---------------------------------------------------------------------------
# add_thumb when no base signal creates one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_thumb_creates_signal_when_missing(
    recorder: SignalRecorder, store: SignalStore
) -> None:
    await recorder.add_thumb(turn_id="orphan", user_id="u1", thumb=Thumb.DOWN)

    results = await store.for_user("u1")
    assert len(results) == 1
    assert results[0].thumb is Thumb.DOWN


# ---------------------------------------------------------------------------
# add_retry when no base signal creates one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_retry_creates_signal_when_missing(
    recorder: SignalRecorder, store: SignalStore
) -> None:
    await recorder.add_retry(turn_id="orphan", user_id="u1")

    results = await store.for_user("u1")
    assert len(results) == 1
    assert results[0].retried is True


# ---------------------------------------------------------------------------
# context_summary is preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_summary_preserved(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1",
        project="p",
        session_id="s1",
        turn_id="t1",
        query="q",
        reply="r",
        context_summary="some context",
    )
    results = await store.for_user("u1")
    assert results[0].context_summary == "some context"


# ---------------------------------------------------------------------------
# Multiple turns are separate signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_turns_are_separate(recorder: SignalRecorder, store: SignalStore) -> None:
    await recorder.record_turn(
        user_id="u1", project="p", session_id="s1", turn_id="t1", query="q1", reply="r1"
    )
    await recorder.record_turn(
        user_id="u1", project="p", session_id="s1", turn_id="t2", query="q2", reply="r2"
    )
    results = await store.for_user("u1")
    assert len(results) == 2
