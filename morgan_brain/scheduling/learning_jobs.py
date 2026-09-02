"""LearningScheduler — automates the self-improvement loop.

Design refs
-----------
* self-learning ADR: nightly consolidation + idle optimizer run.
* Offline contract: jobs run exclusively in the learning-worker; never on the
  hot request path.

Architecture
------------
``LearningScheduler`` wraps a :class:`CronService` and registers two kinds of
recurring jobs for a given user:

1. **consolidate** (nightly, ~86400 s interval): calls
   ``learner.projects_for_user(user_id)`` to discover every project the user has written
   memories under, then ``learner.consolidate(user_id, project=...)`` once per project --
   consolidating only a single (e.g. default) project would silently exclude everything
   written under any other project name.

2. **optimize** (configurable interval, default ~3600 s / idle): if both
   ``champion_trainer`` and ``signal_store`` are provided, mines interaction
   examples from the signal store and runs ``champion_trainer.train(...)``
   with an eval scorer.  This implements the "reflection optimizer offline"
   pattern from the self-learning ADR.

Tests call ``cron.scheduler.tick(now)`` directly — no real sleeping required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from morgan_brain.scheduling.cron import CronService, Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal protocols for learner + trainer (avoids circular imports)
# ---------------------------------------------------------------------------


@runtime_checkable
class _Consolidatable(Protocol):
    """Subset of ConsolidationLearner used by LearningScheduler."""

    async def consolidate(self, user_id: str, *, project: str = ...) -> None: ...

    async def projects_for_user(self, user_id: str) -> list[str]:
        """Distinct projects *user_id* has memories under -- drives the per-project
        consolidation fan-out below (consolidating only one project would silently
        exclude everything written under any other project name)."""
        ...


@runtime_checkable
class _Trainable(Protocol):
    """Subset of ChampionTrainer used by LearningScheduler."""

    async def train(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: Any,
        max_calls: int = 6,
    ) -> bool: ...


@runtime_checkable
class _SignalStore(Protocol):
    """Subset of SignalStore used by LearningScheduler.

    Matches the ``SignalStore.high_value`` interface so that the real
    :class:`~morgan_brain.learning.signals.SignalStore` satisfies the protocol
    without modification.  The optimizer job calls the async free function
    ``mine_examples(store, user_id)`` from ``learning.optimizer`` which in turn
    calls ``store.high_value``.
    """

    async def high_value(
        self, user_id: str, *, min_rank: int = 1, limit: int = 50
    ) -> list[Any]: ...


# ---------------------------------------------------------------------------
# LearningScheduler
# ---------------------------------------------------------------------------

# Default scorer used for optimizer jobs when no custom scorer is provided.
# Returns a constant 0.0 — sufficient for a "no champion yet" first-run scenario.
_DEFAULT_SCORER: Callable[[str], float] = lambda _body: 0.0  # noqa: E731


class LearningScheduler:
    """Registers and manages the nightly learning loop jobs.

    Parameters
    ----------
    cron:
        The :class:`CronService` to register jobs on.
    learner:
        An object with a ``consolidate(user_id)`` async method.
    champion_trainer:
        Optional :class:`ChampionTrainer`; when provided (along with
        *signal_store*) an optimizer job is also registered.
    signal_store:
        Optional :class:`SignalStore`; required for the optimizer job.
    clock:
        Injected callable returning the current :class:`datetime`.
    consolidate_interval_seconds:
        Interval for the consolidation job (default: 86400 s / 24 h).
    optimize_interval_seconds:
        Interval for the optimizer job (default: 3600 s / 1 h).
    prompt_name:
        The prompt registry key passed to ``champion_trainer.train``
        (default: ``"system-prompt"``).
    scorer:
        Scorer passed to ``champion_trainer.train``; defaults to a no-op
        constant-zero scorer (sufficient for a first-run bootstrap).
    max_optimizer_calls:
        ``max_calls`` forwarded to ``champion_trainer.train``.
    """

    def __init__(
        self,
        *,
        cron: CronService,
        learner: _Consolidatable,
        champion_trainer: _Trainable | None = None,
        signal_store: _SignalStore | None = None,
        clock: Callable[[], datetime],
        consolidate_interval_seconds: float = 86400.0,
        optimize_interval_seconds: float = 3600.0,
        prompt_name: str = "system-prompt",
        scorer: Callable[[str], float] | None = None,
        max_optimizer_calls: int = 6,
    ) -> None:
        self._cron = cron
        self._learner = learner
        self._champion_trainer = champion_trainer
        self._signal_store = signal_store
        self._clock = clock
        self._consolidate_interval = consolidate_interval_seconds
        self._optimize_interval = optimize_interval_seconds
        self._prompt_name = prompt_name
        self._scorer: Callable[[str], float] = scorer if scorer is not None else _DEFAULT_SCORER
        self._max_optimizer_calls = max_optimizer_calls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_default_jobs(self, user_id: str) -> None:
        """Register the nightly consolidation job (and optional optimizer job).

        This method is idempotent with respect to the job names: calling it
        multiple times with the same *user_id* will overwrite the previous
        registration in the underlying ``InProcessScheduler``.

        Parameters
        ----------
        user_id:
            The user whose data the jobs will process.
        """
        # ---- 1. Consolidation job (nightly) --------------------------------
        consolidate_name = f"consolidate:{user_id}"

        def _consolidate_fn() -> Any:
            import asyncio

            async def _run() -> None:
                # Fan out across every project the user actually has -- a single
                # (e.g. default-only) call would silently exclude anything written
                # under a real project name.
                projects = await self._learner.projects_for_user(user_id)
                for project in projects:
                    await self._learner.consolidate(user_id, project=project)

            return asyncio.ensure_future(_run())

        self._cron.register(
            Job(name=consolidate_name, interval_seconds=self._consolidate_interval),
            _consolidate_fn,
        )
        logger.debug(
            "LearningScheduler: registered consolidation job %r (interval=%.0fs)",
            consolidate_name,
            self._consolidate_interval,
        )

        # ---- 2. Optimizer job (if trainer + signal_store are wired in) -----
        if self._champion_trainer is not None and self._signal_store is not None:
            self._register_optimize_job(user_id, self._champion_trainer, self._signal_store)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_optimize_job(
        self, user_id: str, trainer: _Trainable, store: _SignalStore
    ) -> None:
        """Both collaborators arrive as parameters, non-optional.

        The caller has already checked them for None; taking them as arguments is what makes
        that check visible to the type checker. Reading them off ``self`` inside the closure
        needed two ``assert``s to narrow -- and ``python -O`` strips asserts, so the narrowing
        would have been gone in exactly the deployment that runs unattended.
        """
        optimize_name = f"optimize:{user_id}"

        def _optimize_fn() -> Any:
            import asyncio

            from morgan_brain.learning.optimizer import mine_examples

            async def _run() -> None:
                # mine_examples is the canonical async free function:
                #   mine_examples(signals: SignalStore, user_id, *, limit=50)
                # It calls store.high_value internally and returns Example objects.
                examples = await mine_examples(store, user_id)  # type: ignore[arg-type]
                if not examples:
                    logger.debug(
                        "LearningScheduler: no high-value examples for %r; skipping optimize",
                        user_id,
                    )
                    return
                promoted = await trainer.train(
                    self._prompt_name,
                    train=examples,
                    scorer=self._scorer,
                    max_calls=self._max_optimizer_calls,
                )
                logger.info("LearningScheduler: optimizer for %r — promoted=%s", user_id, promoted)

            return asyncio.ensure_future(_run())

        self._cron.register(
            Job(name=optimize_name, interval_seconds=self._optimize_interval),
            _optimize_fn,
        )
        logger.debug(
            "LearningScheduler: registered optimizer job %r (interval=%.0fs)",
            optimize_name,
            self._optimize_interval,
        )

    @property
    def cron(self) -> CronService:
        """The underlying CronService (for tests that need tick())."""
        return self._cron
