"""Tests for morgan_brain.eval.runner (commit 3).

All tests use FakeChatClient-backed orchestrator and judge — NO network.

Asserts:
- predict_fn returns the scripted answer for an item.
- predict_fn DOES seed setup facts (visible via memory recall inside the turn).
- predict_fn does NOT seed setup for should_inject=False items.
- The real orchestrator's memory is UNTOUCHED by eval (firewall).
- scorer returns a float in [0, 1].
- scorer returns correct value based on judge verdicts.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from morgan_brain.eval.golden import GoldenItem, ProbeType
from morgan_brain.eval.harness import EvalHarness
from morgan_brain.eval.judge import LLMJudge
from morgan_brain.eval.runner import EVAL_USER_ID, make_eval_scorer, make_predict_fn
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry, JsonMode
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict_json(score: float, passed: bool, rationale: str = "ok") -> str:
    return json.dumps({"score": score, "passed": passed, "rationale": rationale})


def _make_judge(verdicts: list[bool]) -> LLMJudge:
    replies = [_verdict_json(1.0 if v else 0.0, v) for v in verdicts]
    client = FakeChatClient(replies=replies)
    reg = CapabilityRegistry.from_seed({"fake/judge-m": {"json_mode": JsonMode.NONE}})
    router = RoleRouter(
        reg=reg,
        bindings={"judge": [Binding("fake", "judge-m", client)]},
    )
    return LLMJudge(router=router)


def _build_orch_with_client(
    reply: str,
) -> tuple[Any, FakeChatClient, Any]:
    """Build a minimal fake orchestrator + return the FakeChatClient and MemoryGate."""
    from morgan_brain.composition import _assemble
    from morgan_brain.config import Settings
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder

    fake_client = FakeChatClient(reply=reply)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()
    orch, memory_module, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )
    return orch, fake_client, orch._memory  # type: ignore[attr-defined]


def _item(
    *,
    probe: ProbeType = ProbeType.EXPLICIT_RECALL,
    setup: list[str] | None = None,
    query: str = "test query",
    expected: str = "expected answer",
    should_inject: bool = True,
) -> GoldenItem:
    return GoldenItem(
        id="test-item",
        probe=probe,
        setup=setup or [],
        query=query,
        expected=expected,
        should_inject=should_inject,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_fn_returns_orchestrator_reply() -> None:
    """predict_fn returns the scripted reply from the fake client."""
    orch, _, _ = _build_orch_with_client("scripted reply")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    item = _item(query="hello")
    result = await predict_fn(item, "")
    assert result == "scripted reply"


@pytest.mark.asyncio
async def test_predict_fn_seeds_setup_into_scratch_memory() -> None:
    """Setup facts are seeded; they must appear in the context for the turn.

    We verify indirectly: the FakeChatClient records the system message sent
    to the LLM. That message should include the seeded fact (built by build_messages).
    """
    orch, fake_client, _ = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    item = _item(
        setup=["User loves jazz music"],
        query="what music do I like?",
        should_inject=True,
    )
    await predict_fn(item, "")
    # The seeded fact must appear in the system message sent to the LLM.
    messages = fake_client.last_messages
    system_content = messages[0].content
    assert "jazz" in system_content, (
        f"Seeded fact 'jazz' not found in system message: {system_content!r}"
    )


@pytest.mark.asyncio
async def test_predict_fn_does_not_seed_when_should_inject_false() -> None:
    """For should_inject=False items, setup is NOT seeded into scratch memory."""
    orch, fake_client, _ = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    item = _item(
        probe=ProbeType.OVER_PERSONALIZATION_NEGATIVE,
        setup=["User likes very long essays about everything"],
        query="explain lists briefly",
        should_inject=False,
    )
    await predict_fn(item, "")
    messages = fake_client.last_messages
    system_content = messages[0].content
    # The stale preference must NOT appear in the system message.
    assert "long essays" not in system_content, (
        "Stale preference was seeded despite should_inject=False"
    )


@pytest.mark.asyncio
async def test_firewall_real_memory_untouched() -> None:
    """The real orchestrator's MemoryGate must not contain eval content after predict_fn."""
    orch, _, real_gate = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    item = _item(
        setup=["EVAL_SECRET_FACT_DO_NOT_LEAK"],
        query="what is the secret fact?",
        should_inject=True,
    )
    await predict_fn(item, "")

    # After the call, the real gate must be restored and must NOT contain eval memory.
    assert orch._memory is real_gate, "Real memory gate was not restored after predict_fn"  # type: ignore[attr-defined]

    # The real gate should have no memories for the eval user_id.
    recalled = await real_gate.recall(MemoryQuery(user_id=EVAL_USER_ID, text="secret fact"))
    # Filter to only memories whose content contains our sentinel.
    leaked = [m for m in recalled if "EVAL_SECRET_FACT_DO_NOT_LEAK" in m.content]
    assert not leaked, f"Eval content leaked into real memory: {leaked}"


@pytest.mark.asyncio
async def test_firewall_gate_restored_on_exception() -> None:
    """Real gate is restored even if orchestrator raises an exception."""
    from morgan_brain.core.orchestrator import Orchestrator
    from morgan_brain.composition import _assemble
    from morgan_brain.config import Settings
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder

    # Build orchestrator that will raise on handle_turn.
    fake_client = FakeChatClient(reply="ok")
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )

    real_gate = orch._memory  # type: ignore[attr-defined]

    # Patch handle_turn to raise.
    original_handle_turn = orch.handle_turn

    async def _raise(**kwargs: object) -> None:  # type: ignore[return]
        raise RuntimeError("simulated failure")

    orch.handle_turn = _raise  # type: ignore[method-assign]

    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    with pytest.raises(RuntimeError):
        await predict_fn(_item(), "")

    # Gate must be restored even after exception.
    assert orch._memory is real_gate  # type: ignore[attr-defined]

    orch.handle_turn = original_handle_turn  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_scorer_returns_float_in_0_1() -> None:
    """scorer(body) returns a float in [0, 1]."""
    orch, _, _ = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)

    item = _item()
    judge = _make_judge([True])
    harness = EvalHarness(judge=judge)
    scorer = make_eval_scorer(harness=harness, golden_items=[item], predict_fn=predict_fn)

    score = await scorer("")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_scorer_returns_1_0_when_all_pass() -> None:
    """scorer returns 1.0 when all items pass the judge."""
    orch, _, _ = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)

    items = [
        _item(probe=ProbeType.EXPLICIT_RECALL),
        _item(probe=ProbeType.ABSTENTION),
    ]
    judge = _make_judge([True, True])
    harness = EvalHarness(judge=judge)
    scorer = make_eval_scorer(harness=harness, golden_items=items, predict_fn=predict_fn)

    score = await scorer("some system override")
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_scorer_returns_0_0_when_all_fail() -> None:
    """scorer returns 0.0 when all items fail the judge."""
    orch, _, _ = _build_orch_with_client("wrong answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)

    items = [_item(probe=ProbeType.EXPLICIT_RECALL)]
    judge = _make_judge([False])
    harness = EvalHarness(judge=judge)
    scorer = make_eval_scorer(harness=harness, golden_items=items, predict_fn=predict_fn)

    score = await scorer("")
    assert score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_predict_fn_threads_system_override() -> None:
    """system_override is forwarded to orchestrator.handle_turn."""
    orch, fake_client, _ = _build_orch_with_client("answer")
    predict_fn = make_predict_fn(orchestrator=orch, clock=CLOCK)
    item = _item(query="hello")
    override = "TEST_OVERRIDE_SENTINEL"
    await predict_fn(item, override)

    system_msg = fake_client.last_messages[0]
    assert override in system_msg.content
