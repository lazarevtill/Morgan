"""Effective reads compare real instants across legacy and offset timestamps."""

from datetime import UTC, datetime, timedelta, timezone

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
