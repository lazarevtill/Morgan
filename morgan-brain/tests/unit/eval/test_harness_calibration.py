"""Harness calibration wiring: run_l2 collects per-item confidence and populates layer3.

Report-only: calibration is computed + surfaced on the Scorecard; the promotion gate
(beats_current) is unaffected by it.
"""

from __future__ import annotations

import pytest

from morgan_brain.eval.golden import GoldenItem, ProbeType
from morgan_brain.eval.harness import EvalHarness, Scorecard, beats_current
from morgan_brain.eval.judge import JudgeVerdict


class _FakeJudge:
    """Maps a (predicted) answer string to pass/fail; duck-typed for EvalHarness."""

    def __init__(self, passed_by_answer: dict[str, bool]) -> None:
        self._passed = passed_by_answer

    async def judge(
        self, *, question: str, answer: str, expected: str, rubric: str = ""
    ) -> JudgeVerdict:
        ok = self._passed.get(answer, False)
        return JudgeVerdict(score=1.0 if ok else 0.0, passed=ok)


def _item(item_id: str) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        probe=ProbeType.EXPLICIT_RECALL,
        setup=[],
        query=f"q-{item_id}",
        expected=f"e-{item_id}",
        should_inject=True,
    )


# Four items chosen so Brier and ECE are hand-computable:
#   A: conf 1.0, correct   -> (1.0, 1)
#   B: conf 1.0, incorrect -> (1.0, 0)
#   C: conf 0.0, incorrect -> (0.0, 0)
#   D: conf 0.0, correct   -> (0.0, 1)
# Brier = (0 + 1 + 0 + 1)/4 = 0.5 ; ECE = 0.5
_CONF = {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}
_CORRECT = {"A": True, "B": False, "C": False, "D": True}


@pytest.mark.asyncio
async def test_run_l2_populates_calibration_layer3() -> None:
    items = [_item(i) for i in ("A", "B", "C", "D")]
    harness = EvalHarness(judge=_FakeJudge(_CORRECT))  # type: ignore[arg-type]

    async def predict(item: GoldenItem) -> tuple[str, float]:
        return item.id, _CONF[item.id]

    card = await harness.run_l2(items, predict)

    assert card.layer3["brier"] == pytest.approx(0.5)
    assert card.layer3["ece"] == pytest.approx(0.5)
    assert len(card.reliability) == 10  # full reliability diagram
    # bins are serialisable floats
    assert all(isinstance(v, float) for row in card.reliability for v in row.values())


@pytest.mark.asyncio
async def test_str_predict_fn_leaves_calibration_empty() -> None:
    """A predict_fn that returns a bare str (no confidence) → no calibration (back-compat)."""
    items = [_item(i) for i in ("A", "B")]
    harness = EvalHarness(judge=_FakeJudge(_CORRECT))  # type: ignore[arg-type]

    async def predict(item: GoldenItem) -> str:
        return item.id

    card = await harness.run_l2(items, predict)

    assert card.layer3 == {}
    assert card.reliability == []
    # accuracy still computed as before
    assert card.layer2["overall_preference_following_accuracy"] == pytest.approx(0.5)


def test_calibration_is_report_only_not_gated() -> None:
    """beats_current must ignore calibration: a worse-calibrated candidate still promotes on
    equal/better accuracy (report-only phase)."""
    champion = Scorecard(
        layer2={"overall_preference_following_accuracy": 0.8},
        layer3={"brier": 0.1},
    )
    worse_calibrated_but_accurate = Scorecard(
        layer2={"overall_preference_following_accuracy": 0.8},
        layer3={"brier": 0.4},  # much worse calibration
    )
    assert beats_current(worse_calibrated_but_accurate, champion) is True
