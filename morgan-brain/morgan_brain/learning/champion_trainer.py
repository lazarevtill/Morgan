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

from morgan_brain.learning.optimizer import AnyScorer, Example, _call_scorer
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
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        registry: PromptRegistry,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._optimizer = optimizer
        self._registry = registry
        self._clock = clock

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
        return True
