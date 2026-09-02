"""Phase 3 Increment D — ChampionTrainer: offline self-improvement gate.

Design refs:
  * Design doc §D (GEPA optimizer loop) — champion flow, promotion gate.
  * self-learning ADR — beats-current-or-nothing; offline only; zero inference-time cost.

ChampionTrainer
---------------
Orchestrates the champion-preprompt improvement loop:

1. Retrieves the current champion body (empty string if none).
2. Calls the ``Optimizer`` to produce a **candidate** ``PromptVersion``.
3. Scores both the candidate and the current champion with the injected scorer.
4. Applies the **strict gate**: promotes ONLY if ``candidate_score > champion_score``.
   - Ties do NOT promote (equal score = no improvement).
   - No champion → the empty-string body is scored as the baseline; a candidate still has
     to strictly beat it. There is no unconditional first-candidate promotion.
5. On promotion: registers the candidate body + sets it as champion.
6. On rejection: the registry is unchanged.

This offline loop only ever runs when ``settings.enable_champion_promotion`` is true (see
``config.py``) — disarmed by default because the current gate is a single scored run over a
small golden set, which is too noisy to trust unattended.

Offline contract
----------------
``ChampionTrainer.train`` runs exclusively in the learning-worker (Cron/idle).
It is NEVER called on the hot request path.  The deployed champion is just a
better prompt string — zero inference-time overhead for the serving path.

Rollback
--------
All historical versions are preserved in the ``PromptRegistry``.  Rolling back to
a previous champion is a single ``registry.set_champion(name, old_version)`` call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from morgan_brain.eval.gate_integrity import (
    GateIntegrityError,
    GateSpec,
    assert_gate_unweakened,
    screen_candidate,
)
from morgan_brain.learning.optimizer import AnyScorer, Example, _call_scorer
from morgan_brain.learning.receipts import ReceiptStore
from morgan_brain.learning_lifecycle.interfaces import Optimizer, PromptRegistry

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces the deprecated ``datetime.utcnow``)."""
    return datetime.now(UTC)


class ChampionTrainer:
    """Orchestrates the offline champion-preprompt improvement loop.

    Args:
        optimizer: An ``Optimizer`` implementation (e.g. ``ReflectiveOptimizer``).
        registry:  A ``PromptRegistry`` implementation (e.g. ``LocalPromptRegistry``).
        clock:     Zero-argument callable returning the current ``datetime``
                   (injected for deterministic testing).
        receipts:  Optional ``ReceiptStore``. When wired, every decision -- promoted or
                   rejected, and for which reason -- is recorded. It is also the only
                   record of which gate certified the standing champion, so the
                   integrity check below depends on it.
        gate:      Optional ``GateSpec`` describing the gate this run is scoring against.
                   When both this and *receipts* are wired, a candidate measured on a
                   different or weaker gate than the champion is refused before its score
                   is even considered.
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        registry: PromptRegistry,
        clock: Callable[[], datetime] = _utcnow,
        receipts: ReceiptStore | None = None,
        gate: GateSpec | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._registry = registry
        self._clock = clock
        self._receipts = receipts
        self._gate = gate

    async def train(
        self,
        name: str,
        *,
        train: list[Example],
        scorer: AnyScorer,
        max_calls: int = 6,
    ) -> bool:
        """Run one champion improvement cycle for prompt *name*.

        Offline contract: this method runs in the learning-worker, never the
        hot request path.  The deployed champion is just a better prompt.

        Algorithm:
        1. Retrieve current champion body (empty string if no champion yet).
        2. Score the current champion body.
        3. Ask the optimizer to produce a candidate.
        4. Score the candidate.
        5. Strict gate: promote only if ``candidate_score > champion_score``
           (ties → no promotion).
        6. On promotion: register + set_champion → return True.
        7. On rejection: registry unchanged → return False.

        Args:
            name:      The prompt name in the registry (e.g. ``"system-prompt"``).
            train:     Training examples to pass to the optimizer.
            scorer:    Sync or async callable ``(body: str) → float``.
            max_calls: Passed through to the optimizer.

        Returns:
            True if the candidate was promoted to champion; False otherwise.
        """
        # Step 1: get the current champion body (empty string if none).
        existing_champion = await self._registry.champion(name)
        current_body: str = existing_champion.body if existing_champion is not None else ""

        # Step 2: score the current champion.
        champion_score = await _call_scorer(scorer, current_body)

        # Step 3: produce a candidate.
        candidate_version = await self._optimizer.optimize(
            name,
            train=train,
            scorer=scorer,
            max_calls=max_calls,
            current_body=current_body,
        )
        candidate_body = candidate_version.body

        # Step 3a: the integrity guards, BEFORE the candidate is scored.
        #
        # Order matters. Scoring first and checking after would let a candidate that
        # addresses the judge produce a number, and a number that exists gets compared --
        # by a later reader if not by this code. Refusing before the measurement means
        # there is no tainted score to reason about.
        try:
            screen_candidate(candidate_body)
            if self._gate is not None:
                assert_gate_unweakened(certified=self._certified_gate(name), current=self._gate)
        except GateIntegrityError as exc:
            logger.warning("ChampionTrainer: refusing candidate for %r: %s", name, exc)
            self._record(
                name,
                verdict="rejected",
                candidate_body=candidate_body,
                reason=str(exc),
                champion_version=(
                    existing_champion.version if existing_champion is not None else None
                ),
                champion_score=champion_score,
            )
            return False

        # Step 4: score the candidate.
        candidate_score = await _call_scorer(scorer, candidate_body)

        # Step 5: strict gate — must STRICTLY improve, always, including when there is no
        # existing champion (scored as the empty-body baseline above). A missing champion is
        # not a free pass: an unconditional first-candidate promotion is how an unvalidated
        # prompt reaches production the moment the judge/reflection roles first become
        # reachable — see the enable_champion_promotion flag in config.py.
        should_promote = candidate_score > champion_score

        if not should_promote:
            logger.debug(
                "ChampionTrainer: candidate for %r (score=%.4f) did not beat champion "
                "(score=%.4f); skipping promotion.",
                name,
                candidate_score,
                champion_score,
            )
            self._record(
                name,
                verdict="rejected",
                candidate_body=candidate_body,
                reason=(
                    f"candidate did not beat the champion "
                    f"({candidate_score:.4f} <= {champion_score:.4f})"
                ),
                champion_version=(
                    existing_champion.version if existing_champion is not None else None
                ),
                champion_score=champion_score,
                candidate_score=candidate_score,
            )
            return False

        # Step 6: register candidate + promote.
        new_version = await self._registry.register(
            name,
            candidate_body,
            commit_message=f"ChampionTrainer: promoted (score {candidate_score:.4f})",
            metrics={"score": candidate_score},
        )
        await self._registry.set_champion(name, new_version.version)

        logger.info(
            "ChampionTrainer: promoted %r v%d (score %.4f → %.4f).",
            name,
            new_version.version,
            champion_score,
            candidate_score,
        )
        self._record(
            name,
            verdict="promoted",
            candidate_body=candidate_body,
            reason=f"beat the champion ({candidate_score:.4f} > {champion_score:.4f})",
            champion_version=(existing_champion.version if existing_champion is not None else None),
            champion_score=champion_score,
            candidate_score=candidate_score,
        )
        return True

    # ------------------------------------------------------------------
    # Integrity + receipts
    # ------------------------------------------------------------------

    def _certified_gate(self, name: str) -> GateSpec | None:
        """The gate the standing champion was certified on.

        Reconstructed from the last promotion receipt, which is the only place that
        survives to be asked -- ``PromptVersion.metrics`` is ``dict[str, float]`` and
        cannot carry a gate description. No receipts wired, or no promotion recorded,
        means there is nothing to compare against, and nothing to weaken.
        """
        if self._receipts is None:
            return None
        last = self._receipts.last_promotion(name)
        if last is None:
            return None
        return GateSpec.from_dict(last.gate_spec)

    def _record(
        self,
        name: str,
        *,
        verdict: str,
        candidate_body: str,
        reason: str,
        champion_version: int | None,
        champion_score: float | None,
        candidate_score: float | None = None,
    ) -> None:
        if self._receipts is None:
            return
        self._receipts.record(
            prompt_name=name,
            verdict=verdict,
            candidate_body=candidate_body,
            now=self._clock(),
            reason=reason,
            champion_version=champion_version,
            champion_score=champion_score,
            candidate_score=candidate_score,
            gate_fingerprint=self._gate.fingerprint() if self._gate else "",
            gate_spec=self._gate.to_dict() if self._gate else {},
            judge_model=self._gate.judge_model if self._gate else "",
        )
