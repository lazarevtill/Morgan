from datetime import datetime

from morgan_brain.models.memory import TemporalFact
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore


def _fact(obj: str, **kw) -> TemporalFact:
    return TemporalFact(user_id="u1", subject="user", predicate="lives_in", object=obj, **kw)


async def test_upsert_then_current_returns_fact():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Berlin"
    assert current[0].valid_to is None


async def test_conflicting_fact_supersedes_not_overwrites():
    store = SqliteTemporalStore(":memory:")
    first_id = await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    await store.upsert_fact(_fact("Munich"), now=datetime(2026, 6, 1))

    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Munich"

    history = await store.history(user_id="u1", subject="user", predicate="lives_in")
    assert len(history) == 2
    old = next(f for f in history if f.id == first_id)
    assert old.valid_to == datetime(2026, 6, 1)
    assert old.superseded_by is not None


async def test_user_scoped():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    assert await store.current_facts(user_id="u2") == []


async def test_upsert_does_not_mutate_caller_object():
    store = SqliteTemporalStore(":memory:")
    f = _fact("Berlin")
    await store.upsert_fact(f, now=datetime(2026, 1, 1))
    assert f.valid_from is None and f.last_confirmed is None
