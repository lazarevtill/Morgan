"""SQL is a conservative candidate filter; Python decides exact validity."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from morgan_brain.memory.store.temporal import SqliteTemporalStore


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("clock_kind", ["utc", "naive", "offset"])
async def test_candidates_retain_exact_boundaries_and_python_only_timestamps(delta, clock_kind):
    store = SqliteTemporalStore()
    conn = store._conn
    boundary = datetime(2026, 6, 1, 0, 0, 0, 123456, tzinfo=UTC)
    unusual_offset = boundary.astimezone(timezone(timedelta(hours=3, seconds=30)))
    rows = [
        ("open", None, None),
        ("expired", None, (boundary - timedelta(days=1)).isoformat()),
        ("future", (boundary + timedelta(days=1)).isoformat(), None),
        ("starts", boundary.isoformat(), None),
        ("ends", None, boundary.isoformat()),
        ("legacy", boundary.replace(tzinfo=None).isoformat(), None),
        ("offset_seconds", unusual_offset.isoformat(), None),
        ("basic", "20260601T000000", None),
    ]
    try:
        for identity, start, end in rows:
            conn.execute(
                "INSERT INTO facts (id,user_id,project,subject,predicate,object,source,"
                "confidence,valid_from,valid_to) VALUES (?, 'owner', 'personal', ?,"
                "'p', 'synthetic', 'user_stated', 1.0, ?, ?)",
                (identity, identity, start, end),
            )
        # These are valid Python datetimes SQLite cannot parse: they must survive
        # the SQL candidate filter rather than disappearing silently.
        for text in (unusual_offset.isoformat(), "20260601T000000"):
            assert datetime.fromisoformat(text) is not None
            assert conn.execute("SELECT julianday(?)", (text,)).fetchone()[0] is None
        clock = boundary + timedelta(microseconds=delta)
        if clock_kind == "naive":
            clock = clock.replace(tzinfo=None)
        elif clock_kind == "offset":
            clock = clock.astimezone(timezone(timedelta(hours=-4)))
        expected = (
            {"open", "basic", "ends"}
            if delta < 0
            else {"open", "basic", "starts", "legacy", "offset_seconds"}
        )
        assert {
            fact.id for fact in await store.current_facts(user_id="owner", at=clock)
        } == expected
        assert await store.current_facts(user_id="other", at=clock) == []
        assert await store.current_facts(user_id="owner", project="work", at=clock) == []
        assert [
            fact.id
            for fact in await store.current_facts(user_id="owner", subject="basic", at=clock)
        ] == ["basic"]
        assert {fact.id for fact in await store.current_facts(user_id="owner")} == {
            "open",
            "future",
            "starts",
            "legacy",
            "offset_seconds",
            "basic",
        }
    finally:
        conn.close()
