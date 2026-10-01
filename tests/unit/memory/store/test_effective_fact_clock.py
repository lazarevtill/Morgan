"""Effective reads compare real instants across legacy and offset timestamps."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import TemporalFact


async def test_effective_clock_normalizes_offsets_and_legacy_naive_boundaries():
    store = SqliteTemporalStore()
    boundary = datetime(2026, 6, 1, tzinfo=UTC)
    offset_boundary = datetime(2026, 6, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    try:
        await store.upsert_fact(
            TemporalFact(
                id="old", user_id="owner", subject="user", predicate="drink", object="tea"
            ),
            now=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        )
        await store.upsert_fact(
            TemporalFact(
                id="new", user_id="owner", subject="user", predicate="drink", object="water"
            ),
            now=offset_boundary,
        )
        before = await store.current_facts(user_id="owner", at=boundary - timedelta(seconds=1))
        assert [fact.id for fact in before] == ["old"]
        for clock in (boundary, offset_boundary, boundary.replace(tzinfo=None)):
            assert [fact.id for fact in await store.current_facts(user_id="owner", at=clock)] == [
                "new"
            ]
        assert await store.current_facts(user_id="owner", at=datetime(2025, 1, 1, tzinfo=UTC)) == []
        assert await store.current_facts(user_id="intruder", at=boundary) == []
        assert await store.current_facts(user_id="owner", project="work", at=boundary) == []
        assert await store.current_facts(user_id="owner", subject="other", at=boundary) == []
    finally:
        store._conn.close()


async def test_ordinary_writes_with_reversed_captured_clocks_serialize_without_negative_interval():
    store = SqliteTemporalStore()
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = earlier + timedelta(microseconds=1)
    try:
        for identity, clock in (("first", later), ("second", earlier), ("third", earlier)):
            await store.upsert_fact(
                TemporalFact(
                    id=identity, user_id="u", subject="user", predicate="drink", object=identity
                ),
                now=clock,
            )
        rows = await store.history(user_id="u", subject="user", predicate="drink")
        assert len(rows) == 3
        assert all(fact.valid_from == later for fact in rows)
        assert all(fact.valid_to is None or fact.valid_to >= fact.valid_from for fact in rows)
        assert [fact.id for fact in await store.current_facts(user_id="u", at=later)] == ["third"]
    finally:
        store._conn.close()


async def test_past_schedule_does_not_block_existing_backdated_import_semantics():
    store = SqliteTemporalStore()
    try:
        await store.upsert_fact(
            TemporalFact(
                id="scheduled",
                user_id="u",
                subject="user",
                predicate="drink",
                object="tea",
                valid_from=datetime(2027, 1, 1, tzinfo=UTC),
            ),
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
        await store.upsert_fact(
            TemporalFact(
                id="late",
                user_id="u",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=datetime(2026, 6, 1, tzinfo=UTC),
            ),
            now=datetime(2028, 1, 1, tzinfo=UTC),
        )
        rows = await store.history(user_id="u", subject="user", predicate="drink")
        scheduled = next(fact for fact in rows if fact.id == "scheduled")
        assert scheduled.valid_to == datetime(2028, 1, 1, tzinfo=UTC)
        assert [fact.id for fact in await store.current_facts(user_id="u")] == ["late"]
    finally:
        store._conn.close()


@pytest.mark.parametrize("legacy_naive", [False, True])
@pytest.mark.parametrize("immediate", [False, True])
async def test_scheduled_write_keeps_present_fact_and_rejects_reverse_order(
    legacy_naive, immediate
):
    store = SqliteTemporalStore()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    future = datetime(2027, 1, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    if legacy_naive:
        now = now.replace(tzinfo=None)
    try:
        await store.upsert_fact(
            TemporalFact(
                id="present", user_id="u", subject="user", predicate="drink", object="tea"
            ),
            now=now,
        )
        await store.upsert_fact(
            TemporalFact(
                id="future",
                user_id="u",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=future,
            ),
            now=now,
        )
        before = await store.history(user_id="u", subject="user", predicate="drink")
        with pytest.raises(ValueError, match="scheduled"):
            await store.upsert_fact(
                TemporalFact(
                    id="reverse",
                    user_id="u",
                    subject="user",
                    predicate="drink",
                    object="coffee",
                    valid_from=None if immediate else datetime(2026, 6, 1, tzinfo=UTC),
                ),
                now=now,
            )
        assert await store.history(user_id="u", subject="user", predicate="drink") == before
        assert [fact.id for fact in await store.current_facts(user_id="u", at=now)] == ["present"]
        assert [fact.id for fact in await store.current_facts(user_id="u", at=future)] == ["future"]
    finally:
        store._conn.close()


@pytest.mark.parametrize("legacy_naive", [False, True])
async def test_delete_effective_predecessor_preserves_future_successor(legacy_naive):
    store = SqliteTemporalStore()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    delete_at = datetime(2026, 6, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    future = datetime(2027, 1, 1, tzinfo=UTC)
    if legacy_naive:
        start = start.replace(tzinfo=None)
    try:
        await store.upsert_fact(
            TemporalFact(
                id="present", user_id="u", subject="user", predicate="drink", object="tea"
            ),
            now=start,
        )
        await store.upsert_fact(
            TemporalFact(
                id="future",
                user_id="u",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=future,
            ),
            now=start,
        )
        await store.close_fact("present", user_id="u", project="personal", now=delete_at)
        assert await store.current_facts(user_id="u", at=delete_at) == []
        assert [f.id for f in await store.current_facts(user_id="u", at=future)] == ["future"]
        [predecessor, successor] = await store.history(
            user_id="u", subject="user", predicate="drink"
        )
        assert predecessor.valid_to == delete_at and predecessor.superseded_by == "future"
        assert successor.valid_from == future and successor.valid_to is None
        before = store._conn.serialize()
        await store.close_fact("present", user_id="u", project="personal", now=future)
        assert store._conn.serialize() == before
    finally:
        store._conn.close()


async def test_delete_future_fact_cancels_without_negative_interval_or_predecessor_resurrection():
    store = SqliteTemporalStore()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    future = datetime(2027, 1, 1, tzinfo=UTC)
    try:
        await store.upsert_fact(
            TemporalFact(
                id="present", user_id="u", subject="user", predicate="drink", object="tea"
            ),
            now=now,
        )
        await store.upsert_fact(
            TemporalFact(
                id="future",
                user_id="u",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=future,
            ),
            now=now,
        )
        before = store._conn.serialize()
        await store.close_fact("future", user_id="intruder", project="personal", now=now)
        await store.close_fact("future", user_id="u", project="work", now=now)
        assert store._conn.serialize() == before
        await store.close_fact("future", user_id="u", project="personal", now=now)
        rows = await store.history(user_id="u", subject="user", predicate="drink")
        cancelled = next(f for f in rows if f.id == "future")
        assert cancelled.valid_to == cancelled.valid_from == future
        assert [f.id for f in await store.current_facts(user_id="u", at=now)] == ["present"]
        assert await store.current_facts(user_id="u", at=future) == []
    finally:
        store._conn.close()


@pytest.mark.parametrize("legacy_naive", [False, True])
async def test_delete_never_extends_ended_history_across_offset_boundaries(legacy_naive):
    store = SqliteTemporalStore()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 6, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    if legacy_naive:
        end = datetime(2026, 6, 1, tzinfo=UTC).replace(tzinfo=None)
    try:
        await store.upsert_fact(
            TemporalFact(
                id="ended",
                user_id="u",
                subject="user",
                predicate="drink",
                object="tea",
                valid_to=end,
            ),
            now=start,
        )
        before = store._conn.serialize()
        await store.close_fact(
            "ended", user_id="u", project="personal", now=datetime(2026, 6, 1, 0, 30, tzinfo=UTC)
        )
        assert store._conn.serialize() == before
    finally:
        store._conn.close()


async def test_ordinary_update_after_cancelled_schedule_closes_effective_predecessor():
    store = SqliteTemporalStore()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    replacement_at = datetime(2026, 6, 1, tzinfo=UTC)
    future = datetime(2027, 1, 1, tzinfo=UTC)
    try:
        await store.upsert_fact(
            TemporalFact(id="tea", user_id="u", subject="user", predicate="drink", object="tea"),
            now=start,
        )
        await store.upsert_fact(
            TemporalFact(
                id="water",
                user_id="u",
                subject="user",
                predicate="drink",
                object="water",
                valid_from=future,
            ),
            now=start,
        )
        await store.close_fact("water", user_id="u", project="personal", now=start)
        await store.upsert_fact(
            TemporalFact(
                id="coffee", user_id="u", subject="user", predicate="drink", object="coffee"
            ),
            now=replacement_at,
        )
        assert [f.id for f in await store.current_facts(user_id="u", at=replacement_at)] == [
            "coffee"
        ]
        assert [f.id for f in await store.current_facts(user_id="u", at=future)] == ["coffee"]
        assert [f.id for f in await store.current_facts(user_id="u", at=start)] == ["tea"]
        rows = {
            f.id: f for f in await store.history(user_id="u", subject="user", predicate="drink")
        }
        assert rows["tea"].valid_to == replacement_at and rows["tea"].superseded_by == "coffee"
        assert rows["water"].valid_from == rows["water"].valid_to == future
    finally:
        store._conn.close()


@pytest.mark.parametrize("replacement_month", [6, 12])
async def test_reschedule_after_cancellation_never_overlaps_or_extends_predecessor(
    replacement_month,
):
    store = SqliteTemporalStore()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    original_end = datetime(2026, 9, 1, tzinfo=UTC)
    replacement_start = datetime(2026, replacement_month, 1, tzinfo=UTC)
    try:
        for identity, effective in (("tea", None), ("water", original_end)):
            await store.upsert_fact(
                TemporalFact(
                    id=identity,
                    user_id="u",
                    subject="user",
                    predicate="drink",
                    object=identity,
                    valid_from=effective,
                ),
                now=now,
            )
        await store.close_fact("water", user_id="u", project="personal", now=now)
        await store.upsert_fact(
            TemporalFact(
                id="coffee",
                user_id="u",
                subject="user",
                predicate="drink",
                object="coffee",
                valid_from=replacement_start,
            ),
            now=now,
        )
        assert [f.id for f in await store.current_facts(user_id="u", at=replacement_start)] == [
            "coffee"
        ]
        rows = {
            f.id: f for f in await store.history(user_id="u", subject="user", predicate="drink")
        }
        assert rows["tea"].valid_to == min(original_end, replacement_start)
        assert rows["water"].valid_from == rows["water"].valid_to == original_end
        if replacement_start > original_end:
            assert await store.current_facts(user_id="u", at=original_end) == []
    finally:
        store._conn.close()
