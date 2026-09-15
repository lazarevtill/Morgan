import sqlite3
from datetime import UTC, datetime

import pytest

from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import TemporalFact


def _fact(obj: str, **kw) -> TemporalFact:
    return TemporalFact(user_id="u1", subject="user", predicate="lives_in", object=obj, **kw)


async def test_upsert_then_current_returns_fact():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1, tzinfo=UTC))
    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Berlin"
    assert current[0].valid_to is None


async def test_conflicting_fact_supersedes_not_overwrites():
    store = SqliteTemporalStore(":memory:")
    first_id = await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1, tzinfo=UTC))
    await store.upsert_fact(_fact("Munich"), now=datetime(2026, 6, 1, tzinfo=UTC))

    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Munich"

    history = await store.history(user_id="u1", subject="user", predicate="lives_in")
    assert len(history) == 2
    old = next(f for f in history if f.id == first_id)
    assert old.valid_to == datetime(2026, 6, 1, tzinfo=UTC)
    assert old.superseded_by is not None


async def test_user_scoped():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1, tzinfo=UTC))
    assert await store.current_facts(user_id="u2") == []


async def test_upsert_does_not_mutate_caller_object():
    store = SqliteTemporalStore(":memory:")
    f = _fact("Berlin")
    await store.upsert_fact(f, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert f.valid_from is None and f.last_confirmed is None


def _database_written_before_one_current_fact_was_enforced(path: str) -> sqlite3.Connection:
    """The facts table as it was before its current facts were unique per key, with a key that a
    race between two processes left holding two current facts, and one untouched key."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
            subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
            source TEXT NOT NULL, confidence REAL NOT NULL, valid_from TEXT, valid_to TEXT,
            superseded_by TEXT, last_confirmed TEXT
        );
        CREATE INDEX idx_facts_current
            ON facts (user_id, project, subject, predicate) WHERE valid_to IS NULL;
        INSERT INTO facts VALUES
            ('seed', 'u1', 'p', 'user', 'lives_in', 'Paris', 'user_stated', 1.0,
             '2026-01-01T00:00:00+00:00', '2026-02-01T00:00:00+00:00', 'berlin', NULL),
            ('berlin', 'u1', 'p', 'user', 'lives_in', 'Berlin', 'user_stated', 1.0,
             '2026-02-01T00:00:00+00:00', NULL, NULL, NULL),
            ('munich', 'u1', 'p', 'user', 'lives_in', 'Munich', 'user_stated', 1.0,
             '2026-02-01T00:00:05+00:00', NULL, NULL, NULL),
            ('job', 'u1', 'p', 'user', 'works_at', 'Acme', 'user_stated', 1.0,
             '2026-01-01T00:00:00+00:00', NULL, NULL, NULL);
        """
    )
    conn.commit()
    return conn


def test_a_key_left_with_two_current_facts_keeps_the_newest_on_open(tmp_path):
    """The older one is closed when the newer became valid and points at it, as a serial run
    would have written it. Nothing is deleted."""
    path = str(tmp_path / "m.db")
    _database_written_before_one_current_fact_was_enforced(path).close()

    store = SqliteTemporalStore(path)

    rows = {
        r["id"]: (r["valid_to"], r["superseded_by"])
        for r in store._conn.execute("SELECT id, valid_to, superseded_by FROM facts")
    }
    assert rows == {
        "seed": ("2026-02-01T00:00:00+00:00", "berlin"),
        "berlin": ("2026-02-01T00:00:05+00:00", "munich"),
        "munich": (None, None),
        "job": (None, None),
    }


def test_the_database_refuses_a_second_current_fact_for_a_key(tmp_path):
    path = str(tmp_path / "m.db")
    _database_written_before_one_current_fact_was_enforced(path).close()
    store = SqliteTemporalStore(path)

    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO facts VALUES ('rome', 'u1', 'p', 'user', 'lives_in', 'Rome', "
            "'user_stated', 1.0, '2026-03-01T00:00:00+00:00', NULL, NULL, NULL)"
        )


async def test_superseding_still_works_under_the_one_current_fact_rule(tmp_path):
    store = SqliteTemporalStore(str(tmp_path / "m.db"))
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1, tzinfo=UTC))
    await store.upsert_fact(_fact("Munich"), now=datetime(2026, 6, 1, tzinfo=UTC))
    await store.upsert_fact(_fact("Rome"), now=datetime(2026, 9, 1, tzinfo=UTC))

    current = await store.current_facts(user_id="u1")
    assert [f.object for f in current] == ["Rome"]
    history = await store.history(user_id="u1", subject="user", predicate="lives_in")
    assert [f.object for f in history] == ["Berlin", "Munich", "Rome"]
