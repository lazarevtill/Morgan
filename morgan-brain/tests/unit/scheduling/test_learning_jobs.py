"""Unit tests for LearningScheduler (Phase 4, commit 2).

All tests are deterministic:
  - Spy learner + spy trainer + spy signal_store.
  - Fake clock.
  - tick() on InProcessScheduler to advance logical time.
  - No real sleeping, no APScheduler, no LLM calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from morgan_brain.learning.signals import Thumb
from morgan_brain.scheduling.cron import CronService, InProcessScheduler
from morgan_brain.scheduling.learning_jobs import LearningScheduler

# ---------------------------------------------------------------------------
# Helpers & Fakes
# ---------------------------------------------------------------------------


class _FakeInteractionSignal:
    """Minimal InteractionSignal-compatible object for _FakeSignalStore.

    mine_examples checks .user_edit and .thumb; providing thumb=UP means it
    is included as a positive example (value_rank=1).
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        turn_id: str,
        query: str,
        original_reply: str,
        *,
        user_edit: str | None = None,
        thumb: Thumb | None = Thumb.UP,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.query = query
        self.original_reply = original_reply
        self.context_summary = ""
        self.user_edit = user_edit
        self.thumb = thumb
        self.retried = False

T0 = datetime(2026, 1, 1, 0, 0, 0)
CONSOLIDATE_INTERVAL = 100.0  # short interval so tests can tick past it easily
OPTIMIZE_INTERVAL = 50.0


class _FakeLearner:
    """Spy consolidation learner."""

    def __init__(self) -> None:
        self.consolidate_calls: list[str] = []

    async def consolidate(self, user_id: str) -> None:
        self.consolidate_calls.append(user_id)


class _FakeTrainer:
    """Spy champion trainer."""

    def __init__(self, *, promote: bool = True) -> None:
        self._promote = promote
        self.train_calls: list[dict[str, Any]] = []

    async def train(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: Any,
        max_calls: int = 6,
    ) -> bool:
        self.train_calls.append({"name": name, "train": train, "max_calls": max_calls})
        return self._promote


class _FakeSignalStore:
    """Spy signal store compatible with _SignalStore protocol (high_value interface).

    The optimizer job calls mine_examples(store, user_id) which internally calls
    store.high_value(user_id, min_rank=1, limit=N).  This fake honours that path.

    When constructed without arguments, high_value returns one thumb-up signal for
    ANY user_id so the optimizer job has an example to work with.  Pass
    ``examples=[]`` to simulate an empty store.
    """

    def __init__(self, examples: list[Any] | None = None) -> None:
        self._has_default = examples is None
        # Store is empty iff explicitly passed examples=[].
        self._static_examples: list[Any] = [] if examples is not None else []
        # Sentinel: None means "auto-generate for requested user_id"
        self._explicit_examples: list[Any] | None = examples
        self.high_value_calls: list[dict[str, Any]] = []

    async def high_value(
        self, user_id: str, *, min_rank: int = 1, limit: int = 50
    ) -> list[Any]:
        self.high_value_calls.append({"user_id": user_id, "min_rank": min_rank, "limit": limit})
        if self._explicit_examples is not None:
            # Explicit list (possibly empty): return as-is scoped to user_id.
            return [
                s for s in self._explicit_examples if getattr(s, "user_id", None) == user_id
            ]
        # Auto mode: return one thumb-up signal for the requested user so train() is called.
        return [
            _FakeInteractionSignal(
                user_id=user_id,
                session_id="s1",
                turn_id="t1",
                query="q1",
                original_reply="a1",
            )
        ]


def _make_svc(clock_t: datetime = T0) -> tuple[CronService, InProcessScheduler]:
    sched = InProcessScheduler(clock=lambda: clock_t)
    svc = CronService(clock=lambda: clock_t, scheduler=sched)
    return svc, sched


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_default_jobs_consolidation_only() -> None:
    """Without trainer/store, only the consolidation job is registered."""
    learner = _FakeLearner()
    svc, _ = _make_svc()
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("alice")
    assert "consolidate:alice" in svc.list()
    assert "optimize:alice" not in svc.list()


def test_register_default_jobs_both_when_trainer_and_store_provided() -> None:
    """With trainer + store, both consolidation and optimizer jobs are registered."""
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    store = _FakeSignalStore()
    svc, _ = _make_svc()
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
    assert "consolidate:bob" in svc.list()
    assert "optimize:bob" in svc.list()


def test_register_default_jobs_trainer_without_store_skips_optimizer() -> None:
    """trainer without signal_store → no optimizer job."""
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    svc, _ = _make_svc()
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=None,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("charlie")
    assert "consolidate:charlie" in svc.list()
    assert "optimize:charlie" not in svc.list()


# ---------------------------------------------------------------------------
# Tick tests — consolidation job fires the learner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_past_consolidate_interval_calls_learner() -> None:
    """Ticking past the consolidation interval calls learner.consolidate(user_id)."""
    learner = _FakeLearner()
    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("alice")

    # First tick at T0 → job runs (last_run=None).
    await sched.tick(T0)
    # Let the event loop drain so any ensure_future coroutines complete.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "alice" in learner.consolidate_calls


@pytest.mark.asyncio
async def test_tick_before_consolidate_interval_does_not_call_learner() -> None:
    """Ticking before the interval expires → learner.consolidate NOT called."""
    learner = _FakeLearner()
    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("alice")

    # First tick — runs once (last_run=None).
    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Second tick — only 10 s later, interval is 100 s → not due.
    await sched.tick(T0 + timedelta(seconds=10))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert learner.consolidate_calls.count("alice") == 1


@pytest.mark.asyncio
async def test_tick_twice_past_interval_calls_learner_twice() -> None:
    learner = _FakeLearner()
    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("alice")

    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await sched.tick(T0 + timedelta(seconds=CONSOLIDATE_INTERVAL))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert learner.consolidate_calls.count("alice") == 2


# ---------------------------------------------------------------------------
# Tick tests — optimizer job fires the trainer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_past_optimize_interval_calls_trainer() -> None:
    """Ticking past the optimizer interval calls champion_trainer.train."""
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    store = _FakeSignalStore()
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
    await asyncio.sleep(0)  # extra drain for nested futures

    assert len(trainer.train_calls) == 1
    assert trainer.train_calls[0]["name"] == "system-prompt"


@pytest.mark.asyncio
async def test_optimizer_skips_when_no_examples() -> None:
    """When signal_store returns no examples, train() is NOT called."""
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    store = _FakeSignalStore(examples=[])  # empty → skip
    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=store,
        clock=lambda: T0,
        optimize_interval_seconds=OPTIMIZE_INTERVAL,
    )
    ls.register_default_jobs("bob")

    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert trainer.train_calls == []


@pytest.mark.asyncio
async def test_optimize_uses_custom_prompt_name() -> None:
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    store = _FakeSignalStore()
    sched = InProcessScheduler(clock=lambda: T0)
    svc = CronService(clock=lambda: T0, scheduler=sched)
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=store,
        clock=lambda: T0,
        optimize_interval_seconds=OPTIMIZE_INTERVAL,
        prompt_name="custom-prompt",
    )
    ls.register_default_jobs("carol")

    await sched.tick(T0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert trainer.train_calls[0]["name"] == "custom-prompt"


# ---------------------------------------------------------------------------
# Off-request-path check: jobs are scheduled callables, not executed at registration
# ---------------------------------------------------------------------------


def test_registration_does_not_invoke_learner() -> None:
    """Registering jobs must NOT call learner.consolidate immediately."""
    learner = _FakeLearner()
    svc, _ = _make_svc()
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
    )
    ls.register_default_jobs("dave")
    assert learner.consolidate_calls == []


def test_registration_does_not_invoke_trainer() -> None:
    """Registering jobs must NOT call champion_trainer.train immediately."""
    learner = _FakeLearner()
    trainer = _FakeTrainer()
    store = _FakeSignalStore()
    svc, _ = _make_svc()
    ls = LearningScheduler(
        cron=svc,
        learner=learner,
        champion_trainer=trainer,
        signal_store=store,
        clock=lambda: T0,
        consolidate_interval_seconds=CONSOLIDATE_INTERVAL,
        optimize_interval_seconds=OPTIMIZE_INTERVAL,
    )
    ls.register_default_jobs("dave")
    assert trainer.train_calls == []
