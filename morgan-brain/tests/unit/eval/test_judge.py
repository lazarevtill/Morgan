"""Tests for LLMJudge, order-invariance, and CalibratedJudge.

All tests use FakeChatClient (scripted replies) — NO network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from morgan_brain.eval.judge import CalibratedJudge, JudgeVerdict, LLMJudge, cohen_kappa
from morgan_brain.eval.scorers import cohen_kappa as scorers_kappa
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry, JsonMode
from morgan_brain.providers.router import Binding, RoleRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(replies: list[str]) -> RoleRouter:
    """Build a RoleRouter wired to a FakeChatClient with scripted replies."""
    client = FakeChatClient(replies=replies)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/judge-model": {
                "json_mode": JsonMode.NONE,
                "context_window": 8192,
                "max_output": 512,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"judge": [Binding(provider="fake", model="judge-model", client=client)]},
    )
    return router


def _verdict_json(score: float, passed: bool, rationale: str = "ok") -> str:
    return json.dumps({"score": score, "passed": passed, "rationale": rationale})


# ---------------------------------------------------------------------------
# JudgeVerdict model
# ---------------------------------------------------------------------------


class TestJudgeVerdict:
    def test_fields_and_defaults(self) -> None:
        v = JudgeVerdict(score=0.8, passed=True)
        assert v.score == 0.8
        assert v.passed is True
        assert v.rationale == ""

    def test_score_with_rationale(self) -> None:
        v = JudgeVerdict(score=0.3, passed=False, rationale="too vague")
        assert v.rationale == "too vague"


# ---------------------------------------------------------------------------
# LLMJudge.judge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_judge_returns_verdict_from_structured_output(self) -> None:
        reply = _verdict_json(0.9, True, "accurate and complete")
        router = _make_router([reply])
        judge = LLMJudge(router=router)
        verdict = await judge.judge(
            question="What is my favourite colour?",
            answer="Blue",
            expected="Blue",
        )
        assert verdict.passed is True
        assert abs(verdict.score - 0.9) < 1e-6

    @pytest.mark.asyncio
    async def test_judge_failing_verdict(self) -> None:
        reply = _verdict_json(0.2, False, "wrong answer")
        router = _make_router([reply])
        judge = LLMJudge(router=router)
        verdict = await judge.judge(
            question="What is my name?",
            answer="Bob",
            expected="Alice",
        )
        assert verdict.passed is False
        assert verdict.score < 0.5

    @pytest.mark.asyncio
    async def test_rubric_injected_into_prompt(self) -> None:
        """The rubric text must appear in the messages sent to the client."""
        reply = _verdict_json(1.0, True)
        client = FakeChatClient(replies=[reply])
        reg = CapabilityRegistry.from_seed({"fake/judge-model": {"json_mode": JsonMode.NONE}})
        router = RoleRouter(
            reg=reg,
            bindings={"judge": [Binding("fake", "judge-model", client)]},
        )
        judge = LLMJudge(router=router)
        rubric = "check for exact date match"
        await judge.judge(
            question="When is my birthday?",
            answer="March 3",
            expected="March 3",
            rubric=rubric,
        )
        # Rubric must appear somewhere in the messages sent.
        all_content = " ".join(m.content for m in client.last_messages)
        assert rubric in all_content

    @pytest.mark.asyncio
    async def test_both_question_and_expected_in_prompt(self) -> None:
        reply = _verdict_json(0.8, True)
        client = FakeChatClient(replies=[reply])
        reg = CapabilityRegistry.from_seed({"fake/judge-model": {"json_mode": JsonMode.NONE}})
        router = RoleRouter(
            reg=reg,
            bindings={"judge": [Binding("fake", "judge-model", client)]},
        )
        judge = LLMJudge(router=router)
        await judge.judge(
            question="my fav food?",
            answer="pizza",
            expected="pizza",
        )
        all_content = " ".join(m.content for m in client.last_messages)
        assert "my fav food?" in all_content
        assert "pizza" in all_content


# ---------------------------------------------------------------------------
# LLMJudge.judge_order_invariant — kills position bias
# ---------------------------------------------------------------------------


class TestOrderInvariant:
    @pytest.mark.asyncio
    async def test_both_orderings_pass_returns_passed_true(self) -> None:
        # Two replies: both passed=True → order-invariant result is passed=True.
        router = _make_router(
            [
                _verdict_json(0.9, True, "pass ordering 1"),
                _verdict_json(0.85, True, "pass ordering 2"),
            ]
        )
        judge = LLMJudge(router=router)
        verdict = await judge.judge_order_invariant(question="q", answer="a", expected="e")
        assert verdict.passed is True

    @pytest.mark.asyncio
    async def test_first_passes_second_fails_returns_passed_false(self) -> None:
        # Ordering 1 passes, ordering 2 fails → disagreement → passed=False.
        router = _make_router(
            [
                _verdict_json(0.9, True, "pass"),
                _verdict_json(0.1, False, "fail on swap"),
            ]
        )
        judge = LLMJudge(router=router)
        verdict = await judge.judge_order_invariant(question="q", answer="a", expected="e")
        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_first_fails_second_passes_returns_passed_false(self) -> None:
        router = _make_router(
            [
                _verdict_json(0.1, False, "fail ordering 1"),
                _verdict_json(0.9, True, "pass on swap"),
            ]
        )
        judge = LLMJudge(router=router)
        verdict = await judge.judge_order_invariant(question="q", answer="a", expected="e")
        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_both_fail_returns_passed_false(self) -> None:
        router = _make_router(
            [
                _verdict_json(0.2, False, "fail 1"),
                _verdict_json(0.1, False, "fail 2"),
            ]
        )
        judge = LLMJudge(router=router)
        verdict = await judge.judge_order_invariant(question="q", answer="a", expected="e")
        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_two_calls_are_made(self) -> None:
        """judge_order_invariant must call the LLM exactly twice."""
        client = FakeChatClient(
            replies=[
                _verdict_json(0.9, True),
                _verdict_json(0.8, True),
            ]
        )
        reg = CapabilityRegistry.from_seed({"fake/judge-model": {"json_mode": JsonMode.NONE}})
        router = RoleRouter(
            reg=reg,
            bindings={"judge": [Binding("fake", "judge-model", client)]},
        )
        judge = LLMJudge(router=router)
        await judge.judge_order_invariant(question="q", answer="a", expected="e")
        assert client.calls == 2


# ---------------------------------------------------------------------------
# CalibratedJudge.calibrate + is_trustworthy
# ---------------------------------------------------------------------------


class TestCalibratedJudge:
    def _make_calibrated_judge(self, judge_verdicts: list[bool]) -> CalibratedJudge:
        """Scripted judge that produces a sequence of passed booleans."""
        # We need one LLM reply per calibration item.
        replies = [_verdict_json(1.0 if v else 0.0, v) for v in judge_verdicts]
        router = _make_router(replies)
        inner = LLMJudge(router=router)
        return CalibratedJudge(inner=inner)

    @pytest.mark.asyncio
    async def test_calibrate_perfect_agreement_kappa_1(self) -> None:
        human_labels = [True, True, False, False, True, False]
        judge_labels = human_labels[:]
        cj = self._make_calibrated_judge(judge_labels)
        items: list[tuple[dict[str, Any], bool]] = [
            ({"question": "q", "answer": "a", "expected": "e"}, lbl) for lbl in human_labels
        ]
        kappa = await cj.calibrate(items)
        assert abs(kappa - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_calibrate_complete_disagreement_kappa_negative(self) -> None:
        human_labels = [True, True, False, False]
        judge_labels = [False, False, True, True]  # perfect flip
        cj = self._make_calibrated_judge(judge_labels)
        items: list[tuple[dict[str, Any], bool]] = [
            ({"question": "q", "answer": "a", "expected": "e"}, lbl) for lbl in human_labels
        ]
        kappa = await cj.calibrate(items)
        assert kappa <= 0.0

    @pytest.mark.asyncio
    async def test_calibrate_good_agreement_kappa_above_threshold(self) -> None:
        # 9 agree (of 10), 1 disagrees → kappa > 0.6
        human_labels = [True] * 5 + [False] * 5
        judge_labels = [True] * 5 + [False] * 4 + [True]  # last one flipped
        cj = self._make_calibrated_judge(judge_labels)
        items: list[tuple[dict[str, Any], bool]] = [
            ({"question": "q", "answer": "a", "expected": "e"}, lbl) for lbl in human_labels
        ]
        kappa = await cj.calibrate(items)
        assert kappa > 0.6

    def test_is_trustworthy_above_threshold(self) -> None:
        router = _make_router([_verdict_json(1.0, True)])
        cj = CalibratedJudge(inner=LLMJudge(router=router))
        assert cj.is_trustworthy(0.7) is True
        assert cj.is_trustworthy(0.6) is True
        assert cj.is_trustworthy(0.59) is False
        assert cj.is_trustworthy(0.0) is False

    def test_is_trustworthy_custom_threshold(self) -> None:
        router = _make_router([_verdict_json(1.0, True)])
        cj = CalibratedJudge(inner=LLMJudge(router=router))
        assert cj.is_trustworthy(0.5, threshold=0.4) is True
        assert cj.is_trustworthy(0.3, threshold=0.4) is False


# ---------------------------------------------------------------------------
# cohen_kappa re-export from judge module (same impl as scorers.py)
# ---------------------------------------------------------------------------


class TestCohenKappaJudgeModule:
    def test_same_result_as_scorers_module(self) -> None:
        a = [True, False, True, False, True]
        b = [True, True, False, False, True]
        assert abs(cohen_kappa(a, b) - scorers_kappa(a, b)) < 1e-12
