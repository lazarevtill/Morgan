"""Recall-quality regression. Keep these GREEN; if a memory change drops a score, that's the
signal the change hurt recall. Thresholds are intentionally strict for the fake embedder
(deterministic), and document the categories that matter."""

from __future__ import annotations

from morgan_brain.models.memory import MemoryQuery, TemporalFact
from tests.memory_quality.conftest import recall_at_k, seed


async def test_single_hop_recall(memory):
    await seed(
        memory,
        "u1",
        [
            "User's favorite programming language is Python",
            "User enjoys mountain biking on weekends",
            "User works as a data engineer",
        ],
    )
    results = await memory.recall(
        MemoryQuery(user_id="u1", text="favorite programming language", top_k=3)
    )
    assert recall_at_k(results, "Python", k=3) == 1.0


async def test_knowledge_update_latest_fact_wins(memory):
    await memory.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    await memory.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Munich")
    )
    current = await memory.current_facts(user_id="u1", subject="user")
    objs = {f.object for f in current}
    assert objs == {"Munich"}
    assert "Berlin" not in objs


async def test_temporal_history_is_queryable(memory):
    await memory.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    )
    await memory.upsert_fact(
        TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Munich")
    )
    history = await memory._temporal.history(user_id="u1", subject="user", predicate="lives_in")
    assert {f.object for f in history} == {"Berlin", "Munich"}
