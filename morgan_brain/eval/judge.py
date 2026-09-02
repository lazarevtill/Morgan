"""Calibrated cross-family LLM judge (L2 — did it learn ME?).

Design constraints from the ADR:
- Different model family than the assistant (configurable role name, default "judge").
- Both answer orderings (order-invariance) to kill position bias.
- Rubric + CoT prompt; parse a structured JudgeVerdict.
- Calibrate once on ~50 hand-labeled items; Cohen's κ ≥ ~0.6 to auto-trust.

All LLM calls go through the RoleRouter seam (no direct model name) so the
judge model family is hot-swappable via config.

``cohen_kappa`` is re-exported from ``scorers`` for callers that import only
from ``judge``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from morgan_brain.eval.scorers import cohen_kappa as _cohen_kappa
from morgan_brain.providers.capability import CapabilityDescriptor
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage

# Re-export so callers can ``from morgan_brain.eval.judge import cohen_kappa``.
cohen_kappa = _cohen_kappa


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Structured output produced by the LLM judge.

    Attributes:
        score:     Continuous quality score in [0, 1].
        passed:    Binary pass/fail decision.
        rationale: Chain-of-thought explanation (optional; empty by default).
    """

    score: float
    passed: bool
    rationale: str = ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an impartial evaluator assessing whether an AI assistant's answer correctly \
addresses a question and matches a reference answer.

Scoring rubric:
- 1.0: Perfect match — all key facts present, nothing contradictory.
- 0.7–0.9: Mostly correct, minor omissions or slight wording differences.
- 0.4–0.6: Partially correct, important facts missing or partially wrong.
- 0.0–0.3: Wrong, irrelevant, or contradicts the reference.

Think step by step. Then output ONLY a JSON object with fields:
  score (float 0–1), passed (bool; true if score >= 0.7), rationale (string).
"""


def _build_judge_messages(
    question: str,
    answer: str,
    expected: str,
    rubric: str,
) -> list[ChatMessage]:
    """Build the message list for a single judge call."""
    system_content = _SYSTEM_PROMPT
    if rubric:
        system_content = system_content + f"\nAdditional rubric: {rubric}"

    user_content = (
        f"Question: {question}\n\nAnswer to evaluate: {answer}\n\nReference answer: {expected}"
    )
    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Calls the *judge* role via the RoleRouter and returns a structured JudgeVerdict.

    The judge role MUST be bound to a different model family than the assistant
    to satisfy the cross-family calibration requirement from the ADR.

    Args:
        router: ``RoleRouter`` instance; must have a "judge" (or custom *role*) binding.
        role:   Role name to look up in the router (default "judge").
    """

    def __init__(self, *, router: RoleRouter, role: str = "judge") -> None:
        self._router = router
        self._role = role

    def _descriptor_for_role(self) -> CapabilityDescriptor:
        """Return the CapabilityDescriptor for the first binding of this role."""
        bindings = self._router.bindings_for(self._role)
        if not bindings:
            raise LookupError(f"No bindings registered for judge role {self._role!r}")
        b = bindings[0]
        return self._router._reg.get(b.provider, b.model)

    async def judge(
        self,
        *,
        question: str,
        answer: str,
        expected: str,
        rubric: str = "",
    ) -> JudgeVerdict:
        """Ask the judge model to evaluate *answer* against *expected*.

        Args:
            question: The original question posed to the assistant.
            answer:   The assistant's answer to evaluate.
            expected: The reference / ground-truth answer.
            rubric:   Optional additional rubric text injected into the system prompt.

        Returns:
            A validated ``JudgeVerdict`` parsed from the model's structured output.
        """
        client, model = self._router.chat_for(self._role)
        descriptor = self._descriptor_for_role()
        messages = _build_judge_messages(question, answer, expected, rubric)
        return await generate_structured(
            client,
            messages,
            model=model,
            schema=JudgeVerdict,
            descriptor=descriptor,
        )

    async def judge_order_invariant(
        self,
        *,
        question: str,
        answer: str,
        expected: str,
        rubric: str = "",
    ) -> JudgeVerdict:
        """Evaluate in BOTH orderings; return passed=True only if both agree pass.

        This kills LLM position bias: the judge must consistently prefer the
        candidate answer regardless of which slot (A or B) it appears in.

        Calls the LLM exactly twice:
          1. answer vs expected  (canonical ordering)
          2. expected vs answer  (swapped ordering)

        Returns:
            A ``JudgeVerdict`` whose ``passed`` flag is True only when both
            orderings independently return passed=True.  The ``score`` is the
            average of the two scores and ``rationale`` is combined.
        """
        v1 = await self.judge(question=question, answer=answer, expected=expected, rubric=rubric)
        # Swapped ordering: answer ↔ expected
        v2 = await self.judge(question=question, answer=expected, expected=answer, rubric=rubric)
        both_pass = v1.passed and v2.passed
        avg_score = (v1.score + v2.score) / 2.0
        combined_rationale = f"[order1] {v1.rationale} | [order2] {v2.rationale}"
        return JudgeVerdict(score=avg_score, passed=both_pass, rationale=combined_rationale)


# ---------------------------------------------------------------------------
# CalibratedJudge
# ---------------------------------------------------------------------------


class CalibratedJudge:
    """Wraps ``LLMJudge`` with calibration logic.

    Calibration runs a set of hand-labeled items through the judge and computes
    Cohen's kappa against the human labels.  A kappa >= 0.6 is considered
    trustworthy (per the ADR: "κ ≥ ~0.6 to auto-trust").

    Args:
        inner: The ``LLMJudge`` instance to calibrate.
    """

    def __init__(self, *, inner: LLMJudge) -> None:
        self._inner = inner
        self._last_kappa: float | None = None

    async def calibrate(
        self,
        labeled: list[tuple[dict[str, Any], bool]],
    ) -> float:
        """Run *labeled* items through the judge; return Cohen's kappa vs human labels.

        Args:
            labeled: List of ``(inputs_dict, human_bool)`` pairs.  *inputs_dict* must
                     contain keys "question", "answer", "expected"; optionally "rubric".

        Returns:
            Cohen's kappa in [-1, 1].  Also stored in ``self._last_kappa``.
        """
        human: list[bool] = []
        judge: list[bool] = []
        for inputs, human_label in labeled:
            verdict = await self._inner.judge(
                question=inputs["question"],
                answer=inputs["answer"],
                expected=inputs["expected"],
                rubric=inputs.get("rubric", ""),
            )
            human.append(human_label)
            judge.append(verdict.passed)
        kappa = _cohen_kappa(human, judge)
        self._last_kappa = kappa
        return kappa

    def is_trustworthy(self, kappa: float, threshold: float = 0.6) -> bool:
        """Return True if *kappa* meets or exceeds *threshold* (default 0.6).

        Args:
            kappa:     The computed kappa value (from ``calibrate``).
            threshold: Minimum acceptable kappa (default 0.6 per ADR).
        """
        return kappa >= threshold

    # Delegate judge / judge_order_invariant so callers can use a CalibratedJudge
    # directly wherever an LLMJudge is expected.

    async def judge(
        self,
        *,
        question: str,
        answer: str,
        expected: str,
        rubric: str = "",
    ) -> JudgeVerdict:
        return await self._inner.judge(
            question=question, answer=answer, expected=expected, rubric=rubric
        )

    async def judge_order_invariant(
        self,
        *,
        question: str,
        answer: str,
        expected: str,
        rubric: str = "",
    ) -> JudgeVerdict:
        return await self._inner.judge_order_invariant(
            question=question, answer=answer, expected=expected, rubric=rubric
        )
