"""Unit tests for ChampionTrainer (Phase 3 Increment D — TDD).

All tests are deterministic:
  - LocalPromptRegistry(":memory:") — no disk I/O.
  - Injected scorer maps body → score.
  - Fake clock.
  - No network, no LLM, no MLflow.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.learning.champion_trainer import ChampionTrainer
from morgan_brain.learning.optimizer import Example
from morgan_brain.learning_lifecycle.interfaces import PromptVersion
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLOCK = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731

TRAIN = [Example(query="q1", good_output="a1"), Example(query="q2", good_output="a2")]


class _ScriptedOptimizer:
    """Fake Optimizer that returns a fixed body + score."""

    def __init__(self, body: str, score: float) -> None:
        self._body = body
        self._score = score

    async def optimize(
        self,
        name: str,
        *,
        train: list[object],
        scorer: object,
        max_calls: int = 6,
        current_body: str = "",
    ) -> PromptVersion:
        return PromptVersion(
            name=name,
            version=0,
            body=self._body,
            commit_message="scripted",
            metrics={"score": self._score},
        )


def _make_scorer(scores: dict[str, float]):  # type: ignore[type-arg]
    async def scorer(body: str) -> float:
        return scores.get(body, 0.0)

    return scorer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_no_champion_accepts_any_candidate() -> None:
    """No existing champion → any candidate is accepted (strict improvement over nothing)."""
    registry = LocalPromptRegistry(clock=CLOCK)
    optimizer = _ScriptedOptimizer(body="First body", score=0.6)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"First body": 0.6})
    promoted = await trainer.train("system-prompt", train=TRAIN, scorer=scorer)

    assert promoted is True
    champion = await registry.champion("system-prompt")
    assert champion is not None
    assert champion.body == "First body"


@pytest.mark.asyncio
async def test_train_better_candidate_replaces_champion() -> None:
    """A candidate strictly better than the current champion is promoted."""
    registry = LocalPromptRegistry(clock=CLOCK)
    # Set up an existing champion with score 0.5
    v = await registry.register("p", "Old body", metrics={"score": 0.5})
    await registry.set_champion("p", v.version)

    optimizer = _ScriptedOptimizer(body="Better body", score=0.8)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"Old body": 0.5, "Better body": 0.8})
    promoted = await trainer.train("p", train=TRAIN, scorer=scorer)

    assert promoted is True
    champion = await registry.champion("p")
    assert champion is not None
    assert champion.body == "Better body"


@pytest.mark.asyncio
async def test_train_worse_candidate_does_not_replace_champion() -> None:
    """Gate: a candidate scoring WORSE than the champion must NOT be promoted."""
    registry = LocalPromptRegistry(clock=CLOCK)
    v = await registry.register("p", "Good body", metrics={"score": 0.9})
    await registry.set_champion("p", v.version)

    optimizer = _ScriptedOptimizer(body="Worse body", score=0.4)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"Good body": 0.9, "Worse body": 0.4})
    promoted = await trainer.train("p", train=TRAIN, scorer=scorer)

    assert promoted is False
    champion = await registry.champion("p")
    assert champion is not None
    assert champion.body == "Good body"  # unchanged


@pytest.mark.asyncio
async def test_train_equal_score_does_not_replace_champion() -> None:
    """A tie does NOT trigger promotion (strict improvement required)."""
    registry = LocalPromptRegistry(clock=CLOCK)
    v = await registry.register("p", "Champion body", metrics={"score": 0.7})
    await registry.set_champion("p", v.version)

    optimizer = _ScriptedOptimizer(body="Tie body", score=0.7)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"Champion body": 0.7, "Tie body": 0.7})
    promoted = await trainer.train("p", train=TRAIN, scorer=scorer)

    assert promoted is False
    champion = await registry.champion("p")
    assert champion is not None
    assert champion.body == "Champion body"


@pytest.mark.asyncio
async def test_train_promoted_champion_has_new_version() -> None:
    """After promotion the new champion has a higher version number than the old one."""
    registry = LocalPromptRegistry(clock=CLOCK)
    v1 = await registry.register("p", "V1 body", metrics={"score": 0.3})
    await registry.set_champion("p", v1.version)

    optimizer = _ScriptedOptimizer(body="V2 body", score=0.75)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"V1 body": 0.3, "V2 body": 0.75})
    promoted = await trainer.train("p", train=TRAIN, scorer=scorer)

    assert promoted is True
    champion = await registry.champion("p")
    assert champion is not None
    assert champion.version > v1.version


@pytest.mark.asyncio
async def test_train_history_preserved_on_promotion() -> None:
    """Old versions remain in the registry (for rollback) after promotion."""
    registry = LocalPromptRegistry(clock=CLOCK)
    v1 = await registry.register("p", "Old body", metrics={"score": 0.3})
    await registry.set_champion("p", v1.version)

    optimizer = _ScriptedOptimizer(body="New body", score=0.8)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"Old body": 0.3, "New body": 0.8})
    await trainer.train("p", train=TRAIN, scorer=scorer)

    versions = await registry.list_versions("p")
    bodies = [v.body for v in versions]
    assert "Old body" in bodies
    assert "New body" in bodies


@pytest.mark.asyncio
async def test_train_metrics_stored_on_new_version() -> None:
    """The registered version has a 'score' key in its metrics."""
    registry = LocalPromptRegistry(clock=CLOCK)

    optimizer = _ScriptedOptimizer(body="Body with metrics", score=0.65)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    scorer = _make_scorer({"Body with metrics": 0.65})
    await trainer.train("p", train=TRAIN, scorer=scorer)

    champion = await registry.champion("p")
    assert champion is not None
    assert "score" in champion.metrics
