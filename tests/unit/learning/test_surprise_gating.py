"""Surprise-gated consolidation: episodics already predicted by the current fact base are
dropped before the (expensive) LLM consolidation call; novel ones are kept, most-surprising
first. Neuro-grounded prediction-error gating — deterministic and zero extra LLM cost.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.consolidation import _surprise_filter
from morgan_brain.models import Memory, MemoryKind, MemorySource, TemporalFact

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _ep(content: str) -> Memory:
    return Memory(
        user_id="u1", kind=MemoryKind.EPISODIC, content=content, source=MemorySource.USER_STATED
    )


def _fact(subject: str, predicate: str, obj: str) -> TemporalFact:
    return TemporalFact(
        user_id="u1",
        subject=subject,
        predicate=predicate,
        object=obj,
        source=MemorySource.USER_STATED,
        created_at=T0,
    )


def test_known_episodics_dropped_novel_kept() -> None:
    facts = [_fact("user", "lives_in", "Berlin")]
    episodics = [
        _ep("user lives in Berlin"),  # fully covered by the fact → dropped
        _ep("bought a Tesla Model 3 yesterday"),  # all new → kept
    ]
    kept = [m.content for m in _surprise_filter(episodics, facts)]
    assert any("Tesla" in c for c in kept)
    assert not any("lives in Berlin" in c for c in kept)


def test_cold_start_keeps_everything() -> None:
    # No facts yet → every episodic is fully novel → nothing is dropped.
    episodics = [_ep("anything"), _ep("something else entirely")]
    kept = _surprise_filter(episodics, [])
    assert len(kept) == 2


def test_results_ordered_most_surprising_first_and_capped() -> None:
    facts = [_fact("user", "likes", "coffee")]
    episodics = [_ep(f"novel statement number {i} about topic {i}") for i in range(40)]
    kept = _surprise_filter(episodics, facts, max_keep=10)
    assert len(kept) == 10  # capped
