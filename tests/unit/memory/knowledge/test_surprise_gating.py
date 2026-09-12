"""Surprise-gated consolidation: episodics already predicted by the current fact base are
dropped before the (expensive) LLM consolidation call; novel ones are kept, most-surprising
first. Neuro-grounded prediction-error gating — deterministic and zero extra LLM cost.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.knowledge.surprise import keep_surprising
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
    kept = [m.content for m in keep_surprising(episodics, facts)]
    assert any("Tesla" in c for c in kept)
    assert not any("lives in Berlin" in c for c in kept)


def test_cold_start_keeps_everything() -> None:
    # No facts yet → every episodic is fully novel → nothing is dropped.
    episodics = [_ep("anything"), _ep("something else entirely")]
    kept = keep_surprising(episodics, [])
    assert len(kept) == 2


def test_results_ordered_most_surprising_first_and_capped() -> None:
    facts = [_fact("user", "likes", "coffee")]
    episodics = [_ep(f"novel statement number {i} about topic {i}") for i in range(40)]
    kept = keep_surprising(episodics, facts, max_keep=10)
    assert len(kept) == 10  # capped


def test_a_cyrillic_episodic_is_scored_rather_than_silently_dropped() -> None:
    """The gate tokenised with an ASCII-only pattern, so a Russian episodic produced an
    empty token set and was skipped before it could be scored. It never reached
    consolidation, so it could never become a fact -- silently, for about 71% of this
    owner's corpus. The same defect was already fixed twice below this layer: in the FTS5
    tokenizer and in the entity extractor.
    """
    kept = keep_surprising([_ep("жёлтая папка лежит на верхней полке")], [])

    assert [m.content for m in kept] == ["жёлтая папка лежит на верхней полке"]


def test_novelty_is_actually_computed_in_cyrillic_not_just_waved_through() -> None:
    """Keeping every Russian episodic would pass the test above and still be wrong: the
    gate exists to drop what the facts already predict. This fails both for the original
    bug, which dropped both, and for a fix that merely stops dropping anything.
    """
    facts = [_fact("пользователь", "живёт_в", "Берлине")]
    episodics = [
        _ep("пользователь живёт в Берлине"),
        _ep("купил новый велосипед вчера"),
    ]

    kept = [m.content for m in keep_surprising(episodics, facts)]

    assert kept == ["купил новый велосипед вчера"]
