"""make_predict_fn confidence wiring (calibration, report-only).

with_confidence=True → predict_fn returns (answer, confidence in [0,1]); default → bare str.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.composition import build_orchestrator_for_test
from morgan_brain.eval.golden import GoldenItem, ProbeType
from morgan_brain.eval.runner import make_predict_fn

T0 = datetime(2026, 1, 1)


def _item() -> GoldenItem:
    return GoldenItem(
        id="x",
        probe=ProbeType.EXPLICIT_RECALL,
        setup=["User drinks tea."],
        query="What does the user drink?",
        expected="tea",
        should_inject=True,
    )


@pytest.mark.asyncio
async def test_with_confidence_returns_answer_and_confidence() -> None:
    orch, _ = build_orchestrator_for_test(reply="tea", clock=lambda: T0)
    predict = make_predict_fn(orchestrator=orch, clock=lambda: T0, with_confidence=True)

    out = await predict(_item(), "")

    assert isinstance(out, tuple)
    answer, confidence = out
    assert isinstance(answer, str)
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_default_returns_bare_string() -> None:
    orch, _ = build_orchestrator_for_test(reply="tea", clock=lambda: T0)
    predict = make_predict_fn(orchestrator=orch, clock=lambda: T0)

    out = await predict(_item(), "")

    assert isinstance(out, str)
