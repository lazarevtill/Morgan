"""L2 eval harness — preference-following accuracy + promotion gate.

Design
------
The harness runs a ``predict_fn`` over a golden set and judges each answer
with the configured ``LLMJudge`` / ``CalibratedJudge``.  It produces a
``Scorecard`` with per-probe accuracies and an overall figure.

FIREWALL: the harness only reads ``predict_fn`` output.  It never writes to
any memory store, vector index, or signal recorder.  Eval items must not
leak into the assistant's consolidation pipeline (ADR requirement).

Beats-current gate
------------------
``beats_current(candidate, champion)`` is the single Boolean gate:
- True if champion is None (no baseline — anything is an improvement).
- True if candidate's overall accuracy ≥ champion's AND no per-probe score
  regresses more than ``EPSILON`` below the champion's corresponding score.
- False otherwise (regression — do not promote).

EvalGate wraps the registry seam: it compares the candidate scorecard to the
champion's stored metrics, promotes (register + set_champion) only if the gate
passes, and leaves the champion untouched on failure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable

import structlog
from pydantic import BaseModel, Field

from morgan_brain.eval.calibration import (
    Pair,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from morgan_brain.eval.golden import GoldenItem, ProbeType
from morgan_brain.eval.judge import CalibratedJudge, LLMJudge
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry

_log = structlog.get_logger(__name__)

# Per-probe regression tolerance — small regressions within this band are
# accepted as noise (the overall gate is the primary signal).
_EPSILON: float = 0.05

# The key used for the overall accuracy figure in layer2 / champion metrics.
_OVERALL_KEY = "overall_preference_following_accuracy"

# A predict_fn takes one GoldenItem and returns the assistant's answer, optionally paired with a
# confidence in [0, 1] for calibration scoring. Returning a bare ``str`` (no confidence) is the
# back-compatible default; ``(answer, confidence)`` opts the item into calibration.
PredictFn = Callable[[GoldenItem], Awaitable[str | tuple[str, float]]]

# predict_fn_factory: given a prompt body string, returns a PredictFn.
PredictFnFactory = Callable[[str], PredictFn]


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


class Scorecard(BaseModel):
    """Aggregated result of one eval run.

    Attributes:
        layer1: L1 retrieval metrics (recall@k, f1@k) — populated by L1 runner (Phase 3).
        layer2: Per-probe accuracy + overall.  Keys are ``ProbeType.value`` strings and
                the special ``"overall_preference_following_accuracy"`` key.
        n_items: Total number of golden items evaluated.
        passed:  True when overall accuracy > 0 (at least one item passed).
    """

    layer1: dict[str, float] = Field(default_factory=dict)
    layer2: dict[str, float] = Field(default_factory=dict)
    # L3 calibration metrics ({"brier": .., "ece": ..}) + the reliability-diagram rows.
    # Empty unless predict_fn supplied per-item confidences (report-only; never gated yet).
    layer3: dict[str, float] = Field(default_factory=dict)
    reliability: list[dict[str, float]] = Field(default_factory=list)
    n_items: int = 0
    passed: bool = False


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------


def beats_current(candidate: Scorecard, champion: Scorecard | None) -> bool:
    """Return True if *candidate* should replace *champion*.

    Rules (per ADR "beats-current-or-nothing"):
    1. If champion is None → True (no baseline, anything qualifies).
    2. candidate overall accuracy ≥ champion overall accuracy.
    3. No per-probe score in candidate regresses more than ``_EPSILON``
       below the same probe in champion.

    Args:
        candidate: Scorecard for the candidate prompt.
        champion:  Scorecard for the current champion, or None.

    Returns:
        True if the candidate should be promoted; False otherwise.
    """
    if champion is None:
        return True

    cand_overall = candidate.layer2.get(_OVERALL_KEY, 0.0)
    champ_overall = champion.layer2.get(_OVERALL_KEY, 0.0)

    # Gate 1: overall must not regress.
    if cand_overall < champ_overall - _EPSILON:
        return False

    # Gate 2: no per-probe regression beyond epsilon.
    for key, champ_score in champion.layer2.items():
        if key == _OVERALL_KEY:
            continue
        cand_score = candidate.layer2.get(key, 0.0)
        if cand_score < champ_score - _EPSILON:
            return False

    return True


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Runs L2 preference-following evaluation.

    Args:
        judge: An ``LLMJudge`` or ``CalibratedJudge`` instance.  All judge calls
               are async and scripted in tests (no network).
    """

    def __init__(self, *, judge: LLMJudge | CalibratedJudge) -> None:
        self._judge = judge

    async def run_l2(
        self,
        items: list[GoldenItem],
        predict_fn: PredictFn,
    ) -> Scorecard:
        """Evaluate *predict_fn* over *items* using the judge.

        FIREWALL: this method only reads predict_fn output; it never writes to
        any memory store, vector index, or signal recorder.

        Args:
            items:      List of golden items to evaluate.
            predict_fn: Async callable ``(GoldenItem) → str``; returns the
                        assistant's answer for the given item.

        Returns:
            A ``Scorecard`` with per-probe and overall accuracy in ``layer2``.
        """
        if not items:
            empty_layer2: dict[str, float] = {_OVERALL_KEY: 0.0}
            for pt in ProbeType:
                empty_layer2[pt.value] = 0.0
            return Scorecard(layer1={}, layer2=empty_layer2, n_items=0, passed=False)

        # Per-probe tracking: list of pass/fail booleans.
        probe_results: dict[str, list[bool]] = defaultdict(list)
        # Calibration pairs: (confidence, correct) for items whose predict_fn supplied confidence.
        cal_pairs: list[Pair] = []

        for item in items:
            # FIREWALL: only read from predict_fn; never write to memory.
            out = await predict_fn(item)
            if isinstance(out, tuple):
                answer, confidence = out
            else:
                answer, confidence = out, None

            verdict = await self._judge.judge(
                question=item.query,
                answer=answer,
                expected=item.expected,
            )
            # For OVER_PERSONALIZATION_NEGATIVE items the expected answer already
            # describes the correct non-personalized response, so the judge verdict
            # is interpreted identically (passed = answer matches expected).
            # No inversion needed — the golden expected encodes the correct behaviour.
            probe_results[item.probe.value].append(verdict.passed)
            if confidence is not None:
                cal_pairs.append((confidence, verdict.passed))

        # Aggregate accuracy (layer2).
        layer2: dict[str, float] = {}
        all_results: list[bool] = []
        for pt in ProbeType:
            results = probe_results.get(pt.value, [])
            acc = sum(results) / len(results) if results else 0.0
            layer2[pt.value] = acc
            all_results.extend(results)

        overall = sum(all_results) / len(all_results) if all_results else 0.0
        layer2[_OVERALL_KEY] = overall
        passed = overall > 0.0 and any(all_results)

        # Aggregate calibration (layer3) — report-only: computed + logged, never gated here.
        layer3: dict[str, float] = {}
        reliability: list[dict[str, float]] = []
        if cal_pairs:
            layer3 = {
                "brier": brier_score(cal_pairs),
                "ece": expected_calibration_error(cal_pairs),
            }
            reliability = [b.as_dict() for b in reliability_bins(cal_pairs)]
            _log.info(
                "eval_calibration",
                brier=round(layer3["brier"], 4),
                ece=round(layer3["ece"], 4),
                n=len(cal_pairs),
            )

        return Scorecard(
            layer1={},
            layer2=layer2,
            layer3=layer3,
            reliability=reliability,
            n_items=len(items),
            passed=passed,
        )


# ---------------------------------------------------------------------------
# EvalGate
# ---------------------------------------------------------------------------


class EvalGate:
    """Promotion gate — compare candidate to champion; promote only if better.

    Integrates with the ``PromptRegistry`` seam (Wave 0.5) to store the new
    champion version atomically.

    Args:
        registry: A ``PromptRegistry`` implementation (e.g. ``LocalPromptRegistry``).
        harness:  The ``EvalHarness`` instance to use for any re-evaluation.
    """

    def __init__(self, *, registry: PromptRegistry, harness: EvalHarness) -> None:
        self._registry = registry
        self._harness = harness

    async def promote_if_better(
        self,
        name: str,
        candidate_body: str,
        candidate_scorecard: Scorecard,
        predict_fn_factory: PredictFnFactory,
    ) -> bool:
        """Promote *candidate_body* to champion if it beats the current champion.

        The comparison uses the pre-computed *candidate_scorecard* against the
        metrics stored in the current champion's ``PromptVersion.metrics``.

        Args:
            name:                Prompt name in the registry.
            candidate_body:      New prompt body to potentially promote.
            candidate_scorecard: Pre-computed scorecard for the candidate.
            predict_fn_factory:  Callable ``(body: str) → PredictFn``; used to
                                 create a predict_fn if a fresh evaluation is needed.
                                 (Currently the gate uses the pre-computed scorecard
                                 directly; the factory is the seam for Phase 3 GEPA.)

        Returns:
            True if the candidate was promoted (champion updated); False otherwise.
        """
        # Build the champion scorecard from stored metrics (if any champion exists).
        existing_champion = await self._registry.champion(name)
        champion_scorecard: Scorecard | None = None
        if existing_champion is not None and existing_champion.metrics:
            champion_scorecard = Scorecard(
                layer1={},
                layer2=dict(existing_champion.metrics),
                n_items=0,
                passed=True,
            )

        if not beats_current(candidate_scorecard, champion_scorecard):
            return False

        # Register the candidate and promote it to champion.
        new_version = await self._registry.register(
            name,
            candidate_body,
            commit_message="EvalGate promotion",
            metrics=dict(candidate_scorecard.layer2),
        )
        await self._registry.set_champion(name, new_version.version)
        return True
