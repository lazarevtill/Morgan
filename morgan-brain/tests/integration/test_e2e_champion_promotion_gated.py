"""E2E test: eval-gated champion promotion.

A better candidate preprompt (eval scorer returns higher score) is promoted in
the registry; a worse one is rejected. Uses fakes for the assistant + judge.

This test consolidates and extends test_champion_promotion_wired.py with explicit
E2E wiring assertions (scorer called, registry updated, strict gate enforced).

Not a duplicate — test_champion_promotion_wired.py covers the unit isolation of
ChampionTrainer internals; this file covers the full integration: faked judge
router → EvalHarness → eval_scorer → ChampionTrainer → PromptRegistry.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.eval.golden import GoldenItem, ProbeType
from morgan_brain.eval.harness import EvalHarness
from morgan_brain.eval.judge import LLMJudge
from morgan_brain.eval.runner import make_eval_scorer, make_predict_fn
from morgan_brain.learning.champion_trainer import ChampionTrainer
from morgan_brain.learning.optimizer import ReflectiveOptimizer
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry, JsonMode
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


def _verdict_json(score: float, passed: bool, rationale: str = "ok") -> str:
    return json.dumps({"score": score, "passed": passed, "rationale": rationale})


def _make_judge_router(verdicts: list[bool]) -> tuple[RoleRouter, FakeChatClient]:
    replies = [_verdict_json(1.0 if v else 0.0, v) for v in verdicts]
    client = FakeChatClient(replies=replies)
    reg = CapabilityRegistry.from_seed({"fake/judge-m": {"json_mode": JsonMode.NONE}})
    router = RoleRouter(reg=reg, bindings={"judge": [Binding("fake", "judge-m", client)]})
    return router, client


def _make_orch_and_router(
    optimizer_reply: str,
    assistant_reply: str,
) -> tuple[object, RoleRouter]:
    reflection_client = FakeChatClient(reply=optimizer_reply)
    assistant_client = FakeChatClient(reply=assistant_reply)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/strong-m": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            },
            "fake/reflect-m": {
                "supports_tools": False,
                "json_mode": "json_schema",
                "context_window": 32768,
            },
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={
            "strong": [Binding("fake", "strong-m", assistant_client)],
            "reflection": [Binding("fake", "reflect-m", reflection_client)],
        },
    )
    settings = Settings(llm_model="strong-m", llm_fast_model="strong-m")
    bus = InProcessBus()
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )
    return orch, router


def _one_golden_item() -> GoldenItem:
    return GoldenItem(
        id="e2e-promo-test",
        probe=ProbeType.EXPLICIT_RECALL,
        setup=["User name is TestUser"],
        query="what is my name?",
        expected="TestUser",
        should_inject=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_better_candidate_promoted_in_registry() -> None:
    """A candidate with a higher eval score than the existing champion is promoted."""
    registry = LocalPromptRegistry(":memory:", clock=CLOCK)

    # Seed an existing champion with a low score.
    v1 = await registry.register("morgan-system", "old body", metrics={"score": 0.1})
    await registry.set_champion("morgan-system", v1.version)

    call_log: list[str] = []

    async def spy_scorer(body: str) -> float:
        call_log.append(body)
        return 0.9 if body == "improved body" else 0.1

    orch, router = _make_orch_and_router(
        optimizer_reply="improved body",
        assistant_reply="TestUser",
    )
    optimizer = ReflectiveOptimizer(router=router, char_budget=1000)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    promoted = await trainer.train("morgan-system", train=[], scorer=spy_scorer, max_calls=1)

    assert promoted is True
    champion = await registry.champion("morgan-system")
    assert champion is not None
    assert champion.body == "improved body"
    # Scorer must have been called at least twice (current + candidate).
    assert len(call_log) >= 2


@pytest.mark.asyncio
async def test_worse_candidate_rejected_registry_unchanged() -> None:
    """A candidate with a lower eval score than the existing champion is NOT promoted."""
    registry = LocalPromptRegistry(":memory:", clock=CLOCK)

    v1 = await registry.register("morgan-system", "good body", metrics={"score": 0.95})
    await registry.set_champion("morgan-system", v1.version)

    async def low_scorer(body: str) -> float:
        return 0.0  # always worse

    orch, router = _make_orch_and_router(
        optimizer_reply="worse body",
        assistant_reply="bad",
    )
    optimizer = ReflectiveOptimizer(router=router, char_budget=1000)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    promoted = await trainer.train("morgan-system", train=[], scorer=low_scorer, max_calls=1)

    assert promoted is False
    champion = await registry.champion("morgan-system")
    assert champion is not None
    assert champion.body == "good body"  # unchanged


@pytest.mark.asyncio
async def test_eval_scorer_gates_promotion_via_fake_judge() -> None:
    """The eval scorer (backed by a fake judge) gates champion promotion end-to-end."""
    registry = LocalPromptRegistry(":memory:", clock=CLOCK)

    # No existing champion → any candidate accepted.
    judge_router, _ = _make_judge_router([True])
    judge = LLMJudge(router=judge_router, role="judge")
    harness = EvalHarness(judge=judge)
    items = [_one_golden_item()]

    orch, router = _make_orch_and_router(
        optimizer_reply="eval gated body",
        assistant_reply="TestUser",
    )
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)  # type: ignore[arg-type]
    eval_scorer = make_eval_scorer(harness=harness, golden_items=items, predict_fn=predict_fn)
    optimizer = ReflectiveOptimizer(router=router, char_budget=1000)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    promoted = await trainer.train("morgan-system", train=[], scorer=eval_scorer, max_calls=1)

    # No champion existed → any score → promoted.
    assert promoted is True
    champion = await registry.champion("morgan-system")
    assert champion is not None


@pytest.mark.asyncio
async def test_tie_score_does_not_promote() -> None:
    """Equal scores do NOT trigger promotion (strict gate: must strictly improve)."""
    registry = LocalPromptRegistry(":memory:", clock=CLOCK)

    v1 = await registry.register("morgan-system", "existing body", metrics={"score": 0.5})
    await registry.set_champion("morgan-system", v1.version)

    async def const_scorer(body: str) -> float:
        return 0.5  # same score for both current and candidate

    orch, router = _make_orch_and_router(
        optimizer_reply="tied body",
        assistant_reply="ok",
    )
    optimizer = ReflectiveOptimizer(router=router, char_budget=1000)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=CLOCK)

    promoted = await trainer.train("morgan-system", train=[], scorer=const_scorer, max_calls=1)

    assert promoted is False
    champion = await registry.champion("morgan-system")
    assert champion is not None
    assert champion.body == "existing body"  # unchanged
