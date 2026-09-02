"""Tests: optimize job uses mine_examples(signal_store, user_id, limit=50) (commit 4).

Verifies that LearningScheduler._register_optimize_job calls the async free function
mine_examples(store, user_id, limit=...) from learning.optimizer — NOT a sync
store.mine_examples() method. A real SignalStore is used so the contract is concrete.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from morgan_brain.learning.signals import InteractionSignal, SignalStore, Thumb
from morgan_brain.scheduling.cron import CronService, InProcessScheduler
from morgan_brain.scheduling.learning_jobs import LearningScheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
OPTIMIZE_INTERVAL = 50.0
CONSOLIDATE_INTERVAL = 1000.0  # large so only the optimize job fires in these tests


class _FakeLearner:
    async def consolidate(self, user_id: str, *, project: str = "default") -> None:
        pass

    async def projects_for_user(self, user_id: str) -> list[str]:
        return ["default"]


class _SpyTrainer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def train(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: Any,
        max_calls: int = 6,
    ) -> bool:
        self.calls.append({"name": name, "train": train})
        return True


async def _make_store_with_thumb_up(user_id: str = "u1") -> SignalStore:
    """Return a SignalStore containing one thumb-up signal for *user_id*."""
    store = SignalStore(clock=lambda: T0)
    sig = InteractionSignal(
        user_id=user_id,
        session_id="s1",
        turn_id="t1",
        query="What is 2+2?",
        original_reply="4",
        thumb=Thumb.UP,
        created_at=T0,
    )
    await store.record(sig)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_job_uses_mine_examples_free_function() -> None:
    """The optimizer job must mine examples via the async free function mine_examples()."""
    store = await _make_store_with_thumb_up("alice")
    trainer = _SpyTrainer()
    learner = _FakeLearner()

    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=store,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
        optimize_interval_seconds=OPTIMIZE_INTERVAL,
    )
    ls.register_default_jobs("alice")

    # Tick past the optimize interval (first tick always runs if last_run=None).
    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # The trainer must have been called.
    assert len(trainer.calls) == 1, "train() should have been called once"


@pytest.mark.asyncio
async def test_optimize_job_skips_when_store_empty() -> None:
    """When the store has no high-value signals, train() is NOT called."""
    store = SignalStore(clock=lambda: T0)  # empty
    trainer = _SpyTrainer()
    learner = _FakeLearner()

    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=store,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
        optimize_interval_seconds=OPTIMIZE_INTERVAL,
    )
    ls.register_default_jobs("bob")

    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert trainer.calls == [], "train() should NOT be called when store has no examples"


@pytest.mark.asyncio
async def test_signal_store_has_no_mine_examples_method() -> None:
    """SignalStore does NOT have a mine_examples method (the free function is used)."""
    store = SignalStore(clock=lambda: T0)
    assert not hasattr(store, "mine_examples"), (
        "SignalStore must NOT have a mine_examples() method; use the free function from optimizer"
    )
