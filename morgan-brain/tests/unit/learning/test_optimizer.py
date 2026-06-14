"""Unit tests for mine_examples + ReflectiveOptimizer (Phase 3 Increment D — TDD).

All tests are fully deterministic:
  - SignalStore + FakeChatClient → no network, no LLM, no MLflow.
  - Injected clock pins time.
  - Injected scorer maps body → score.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.learning.optimizer import Example, ReflectiveOptimizer, mine_examples
from morgan_brain.learning.signals import InteractionSignal, SignalStore, Thumb
from morgan_brain.learning_lifecycle.interfaces import PromptVersion
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLOCK = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731


def _store() -> SignalStore:
    return SignalStore(clock=CLOCK)


def _signal(
    user_id: str = "u1",
    turn_id: str = "t1",
    *,
    query: str = "q?",
    original_reply: str = "orig",
    context_summary: str = "ctx",
    user_edit: str | None = None,
    thumb: Thumb | None = None,
    retried: bool = False,
) -> InteractionSignal:
    return InteractionSignal(
        user_id=user_id,
        session_id="s1",
        turn_id=turn_id,
        query=query,
        original_reply=original_reply,
        context_summary=context_summary,
        user_edit=user_edit,
        thumb=thumb,
        retried=retried,
    )


def _router(replies: list[str]) -> RoleRouter:
    """Build a minimal RoleRouter whose 'reflection' role uses FakeChatClient."""
    client = FakeChatClient(replies=replies)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/reflection-model": {
                "context_window": 8192,
                "supports_tools": False,
            }
        }
    )
    return RoleRouter(
        reg=reg,
        bindings={
            "reflection": [Binding("fake", "reflection-model", client)],
            "strong": [Binding("fake", "reflection-model", client)],
        },
    )


# ---------------------------------------------------------------------------
# mine_examples
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_examples_edited_signal_uses_user_edit() -> None:
    """Edited signal → good_output = user_edit."""
    store = _store()
    await store.record(
        _signal(
            query="How are you?",
            original_reply="Fine.",
            context_summary="morning chat",
            user_edit="I'm doing well, thanks for asking!",
        )
    )
    examples = await mine_examples(store, "u1")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.query == "How are you?"
    assert ex.good_output == "I'm doing well, thanks for asking!"
    assert ex.context == "morning chat"


@pytest.mark.asyncio
async def test_mine_examples_thumb_up_uses_original_reply() -> None:
    """Thumb-up (no edit) → good_output = original_reply."""
    store = _store()
    await store.record(
        _signal(
            turn_id="t2",
            query="Tell me a joke",
            original_reply="Why did the chicken cross the road?",
            thumb=Thumb.UP,
        )
    )
    examples = await mine_examples(store, "u1")
    assert len(examples) == 1
    assert examples[0].good_output == "Why did the chicken cross the road?"


@pytest.mark.asyncio
async def test_mine_examples_thumb_down_without_edit_skipped() -> None:
    """Thumb-down with no edit → no positive target → skipped."""
    store = _store()
    await store.record(_signal(turn_id="t3", thumb=Thumb.DOWN))
    examples = await mine_examples(store, "u1")
    assert examples == []


@pytest.mark.asyncio
async def test_mine_examples_retry_without_edit_skipped() -> None:
    """Retry with no edit → no positive target → skipped."""
    store = _store()
    await store.record(_signal(turn_id="t4", retried=True))
    examples = await mine_examples(store, "u1")
    assert examples == []


@pytest.mark.asyncio
async def test_mine_examples_mixed_signals() -> None:
    """Only edit and thumb-up signals produce Examples; others are filtered."""
    store = _store()
    # edit — rank 3 → included
    await store.record(_signal(turn_id="t1", user_edit="edited"))
    # thumb-up — rank 1 → included
    await store.record(_signal(turn_id="t2", thumb=Thumb.UP))
    # thumb-down — rank 2, no edit → excluded
    await store.record(_signal(turn_id="t3", thumb=Thumb.DOWN))
    # retry no edit → excluded
    await store.record(_signal(turn_id="t4", retried=True))

    examples = await mine_examples(store, "u1")
    assert len(examples) == 2


@pytest.mark.asyncio
async def test_mine_examples_user_scoped() -> None:
    """Examples are scoped to the given user_id."""
    store = _store()
    await store.record(_signal(user_id="u1", turn_id="ta", user_edit="e1"))
    await store.record(_signal(user_id="u2", turn_id="tb", user_edit="e2"))
    u1 = await mine_examples(store, "u1")
    assert len(u1) == 1
    assert u1[0].good_output == "e1"


@pytest.mark.asyncio
async def test_mine_examples_limit_respected() -> None:
    """Limit parameter caps the number of returned examples."""
    store = _store()
    for i in range(20):
        await store.record(_signal(turn_id=f"t{i}", user_edit=f"edit{i}"))
    examples = await mine_examples(store, "u1", limit=5)
    assert len(examples) == 5


# ---------------------------------------------------------------------------
# ReflectiveOptimizer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflective_optimizer_returns_best_scoring_body() -> None:
    """The optimizer keeps whichever candidate body has the highest score."""
    # Fake LLM proposes two candidates; the second has a higher score.
    candidate_a = "Candidate body A"
    candidate_b = "Candidate body B — better"

    router = _router(replies=[candidate_a, candidate_b])
    optimizer = ReflectiveOptimizer(router=router, role="reflection", char_budget=2000)

    # Curate-contract: the proposed bullets are merged into the playbook, so the champion body
    # CONTAINS the chosen delta rather than equalling it. Score by containment.
    async def scorer(body: str) -> float:
        if "Candidate body B" in body:
            return 0.8
        if "Candidate body A" in body:
            return 0.5
        return 0.3

    train = [
        Example(query="q1", good_output="answer1"),
        Example(query="q2", good_output="answer2"),
    ]

    result = await optimizer.optimize(
        "test-prompt",
        train=train,
        scorer=scorer,
        max_calls=2,
        current_body="",
    )

    assert isinstance(result, PromptVersion)
    assert "Candidate body B" in result.body  # best delta merged into the playbook
    assert result.metrics["score"] == pytest.approx(0.8)
    assert result.version == 0  # candidate, not yet registered


@pytest.mark.asyncio
async def test_reflective_optimizer_keeps_current_if_no_improvement() -> None:
    """If the reflection model's proposals don't beat the current body, current wins."""
    current = "Current champion body"
    worse_candidate = "Worse body"

    router = _router(replies=[worse_candidate])
    optimizer = ReflectiveOptimizer(router=router, role="reflection", char_budget=2000)

    scores: dict[str, float] = {
        current: 0.9,
        worse_candidate: 0.4,
    }

    async def scorer(body: str) -> float:
        return scores.get(body, 0.0)

    result = await optimizer.optimize(
        "test-prompt",
        train=[Example(query="q", good_output="a")],
        scorer=scorer,
        max_calls=1,
        current_body=current,
    )

    assert result.body == current
    assert result.metrics["score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_reflective_optimizer_rejects_over_budget_proposal() -> None:
    """A proposal that exceeds char_budget is rejected and not chosen."""
    over_budget = "X" * 2001  # exceeds char_budget=2000
    good_candidate = "A fine short body"

    router = _router(replies=[over_budget, good_candidate])
    optimizer = ReflectiveOptimizer(router=router, role="reflection", char_budget=2000)

    scores: dict[str, float] = {
        "": 0.0,
        good_candidate: 0.7,
        # over_budget should never be scored (rejected on size)
    }

    scored_bodies: list[str] = []

    async def scorer(body: str) -> float:
        scored_bodies.append(body)
        return scores.get(body, 0.0)

    result = await optimizer.optimize(
        "budget-test",
        train=[Example(query="q", good_output="a")],
        scorer=scorer,
        max_calls=2,
        current_body="",
    )

    # Over-budget body must never have been scored.
    assert over_budget not in scored_bodies
    # Good candidate or empty wins (good candidate was scored).
    assert result.body != over_budget


@pytest.mark.asyncio
async def test_reflective_optimizer_returns_prompt_version_candidate() -> None:
    """Return value has name=<name>, version=0, commit_message='reflective optimize'."""
    router = _router(replies=["Improved body"])
    optimizer = ReflectiveOptimizer(router=router, role="reflection", char_budget=2000)

    async def scorer(body: str) -> float:
        return 0.6

    result = await optimizer.optimize(
        "my-prompt",
        train=[Example(query="q", good_output="a")],
        scorer=scorer,
        max_calls=1,
        current_body="",
    )

    assert result.name == "my-prompt"
    assert result.version == 0
    assert result.commit_message == "reflective optimize"


@pytest.mark.asyncio
async def test_reflective_optimizer_fallback_role() -> None:
    """If 'reflection' role is missing, optimizer falls back to 'strong'."""
    client = FakeChatClient(replies=["Better body from strong"])
    reg = CapabilityRegistry.from_seed({"fake/strong": {"context_window": 8192}})
    # Only 'strong' role registered, no 'reflection' role.
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "strong", client)]},
    )
    optimizer = ReflectiveOptimizer(
        router=router, role="reflection", fallback_role="strong", char_budget=2000
    )

    async def scorer(body: str) -> float:
        # Current body is empty (score 0.0); the curated proposal scores higher.
        return 0.8 if "Better body from strong" in body else 0.0

    result = await optimizer.optimize(
        "fallback-test",
        train=[Example(query="q", good_output="a")],
        scorer=scorer,
        max_calls=1,
        current_body="",
    )

    assert "Better body from strong" in result.body


@pytest.mark.asyncio
async def test_reflective_optimizer_sync_scorer() -> None:
    """Optimizer handles a sync scorer (not a coroutine)."""
    router = _router(replies=["Nice body"])
    optimizer = ReflectiveOptimizer(router=router, role="reflection", char_budget=2000)

    def sync_scorer(body: str) -> float:
        return 0.7 if "Nice body" in body else 0.0

    result = await optimizer.optimize(
        "sync-test",
        train=[Example(query="q", good_output="a")],
        scorer=sync_scorer,  # type: ignore[arg-type]
        max_calls=1,
        current_body="",
    )

    assert "Nice body" in result.body
