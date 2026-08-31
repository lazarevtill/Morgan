"""Run the learning-worker: ``python -m morgan_brain.apps.learning_worker``.

Handler path
------------
``RESPONSE_GENERATED`` → reconstruct a :class:`Conversation` (user + assistant
messages, session_id) → ``learner.process_session(conversation)``.

Scheduler path (enable_scheduling=True)
---------------------------------------
Builds a :class:`CronService` + :class:`LearningScheduler` over the REAL
:class:`ConsolidationLearner` from :func:`build_worker_context`, then registers
nightly consolidation always, and the eval-gated optimizer job only when
``settings.enable_champion_promotion`` is true (disarmed by default — see config.py).

Bus wiring
----------
With ``event_bus="redis"`` the worker subscribes to brain-api's Redis stream and
handles turn storage off-path.  With ``event_bus="inproc"`` (single-process dev)
brain-api's bus already handles it — the worker is not needed in that mode.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from morgan_brain.bus import get_event_bus
from morgan_brain.config import Settings, get_settings
from morgan_brain.interfaces.events import Event, EventType, Handler
from morgan_brain.models.memory import DEFAULT_PROJECT
from morgan_brain.models.message import Conversation, Message, Role

if TYPE_CHECKING:
    from typing import Any

    from morgan_brain.interfaces.events import EventBus
    from morgan_brain.learning.learner import ConsolidationLearner
    from morgan_brain.scheduling.cron import CronService
    from morgan_brain.scheduling.learning_jobs import LearningScheduler

log = structlog.get_logger("learning-worker")
_logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Real response handler factory
# ---------------------------------------------------------------------------


def _make_response_handler(
    learner: ConsolidationLearner,
    *,
    clock: Callable[[], datetime] = _utcnow,
) -> Handler:
    """Return an async handler that processes a RESPONSE_GENERATED event.

    Reconstructs a :class:`Conversation` from the event payload and calls
    ``learner.process_session``. Exceptions are caught and logged so the
    handler never crashes the worker event loop.
    """
    _learner = learner
    _clock = clock

    async def _on_response(event: Event) -> None:
        try:
            payload = event.payload
            session_id = payload.get("session_id") or "default"
            project = payload.get("project") or DEFAULT_PROJECT
            query = payload.get("request", "")
            reply = payload.get("response", "")

            convo = Conversation(
                user_id=event.user_id,
                project=project,
                session_id=session_id,
                messages=[
                    Message(user_id=event.user_id, project=project, role=Role.USER, content=query),
                    Message(
                        user_id=event.user_id,
                        project=project,
                        role=Role.ASSISTANT,
                        content=reply,
                    ),
                ],
            )
            await _learner.process_session(convo)
            log.debug(
                "worker.session.processed",
                user_id=event.user_id,
                session_id=session_id,
            )
        except Exception:
            log.exception("worker.session.process-failed", user_id=event.user_id)

    return _on_response


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def _build_cron_service() -> CronService:
    from morgan_brain.scheduling.cron import CronService

    return CronService(clock=_utcnow)


def _build_learning_scheduler(
    cron: CronService,
    learner: ConsolidationLearner,
    champion_trainer: Any | None = None,
    signal_store: Any | None = None,
    eval_scorer: Any | None = None,
) -> LearningScheduler | None:
    """Build a :class:`LearningScheduler` over the real learner from the worker context.

    Registers nightly consolidation always. Registers the eval-gated optimizer job only when
    ``champion_trainer``/``signal_store``/``eval_scorer`` are all provided AND
    ``settings.enable_champion_promotion`` is true — the gate ships disarmed by default (see
    ``enable_champion_promotion`` in config.py for why).
    """
    settings = get_settings()
    optimizer_requested = (
        champion_trainer is not None and signal_store is not None and eval_scorer is not None
    )
    optimizer_enabled = optimizer_requested and settings.enable_champion_promotion
    if optimizer_requested and not settings.enable_champion_promotion:
        champion_trainer = None
        signal_store = None
        eval_scorer = None
    try:
        from morgan_brain.scheduling.learning_jobs import LearningScheduler

        ls = LearningScheduler(
            cron=cron,
            learner=learner,
            champion_trainer=champion_trainer,
            signal_store=signal_store,
            scorer=eval_scorer,
            clock=_utcnow,
            prompt_name="morgan-system",
        )
        ls.register_default_jobs(settings.owner_user_id)
        log.info(
            "learning-scheduler.registered",
            user_id=settings.owner_user_id,
            with_optimizer=optimizer_enabled,
            enable_champion_promotion=settings.enable_champion_promotion,
        )
    except Exception:
        log.exception("learning-scheduler.build-failed")
        return None
    else:
        return ls


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


async def run(settings: Settings | None = None) -> None:
    _settings = settings or get_settings()
    log.info("champion-promotion.flag", enabled=_settings.enable_champion_promotion)

    # Build the real worker context (production stores + configured bus).
    # For inproc dev runs brain-api already handles turn storage, but the
    # worker wiring is harmless (no-op subscriber on an isolated inproc bus).
    from morgan_brain.learning.learner import ConsolidationLearner as _CL

    learner: _CL | None = None
    champion_trainer: Any | None = None
    signal_store_for_sched: Any | None = None
    eval_scorer: Any | None = None
    bus: EventBus | None = None
    try:
        from morgan_brain.composition import build_worker_context

        ctx = build_worker_context(_settings)
        learner = ctx.learner
        champion_trainer = ctx.champion_trainer
        signal_store_for_sched = ctx.signal_store
        eval_scorer = ctx.eval_scorer
        bus = ctx.bus
        log.info("worker-context.built")
    except Exception:
        log.exception("worker-context.build-failed; starting with no-op learner")

    # Use the context's bus -- the SAME instance build_worker_context resolved and shared with
    # _assemble -- so subscribe()/start()/stop() below act on the bus turn-storage registration
    # (and, when Redis, the real stream) actually runs on. get_event_bus() is NOT a singleton
    # (see its docstring): calling it again here would silently create a second, disconnected
    # bus that never receives what _assemble subscribed. Only fall back to a fresh one when the
    # context itself failed to build (graceful degradation path below).
    if bus is None:
        bus = get_event_bus()

    if learner is not None:
        handler = _make_response_handler(learner, clock=_utcnow)
        bus.subscribe(EventType.RESPONSE_GENERATED, handler)
        bus.subscribe(EventType.SESSION_END, handler)
    else:
        # Graceful degradation: log only.
        async def _noop(event: Event) -> None:
            log.info("response.observed.noop", user_id=event.user_id)

        bus.subscribe(EventType.RESPONSE_GENERATED, _noop)
        bus.subscribe(EventType.SESSION_END, _noop)

    await bus.start()
    log.info("learning-worker.started")

    cron = None
    if _settings.enable_scheduling and learner is not None:
        cron = _build_cron_service()
        _build_learning_scheduler(
            cron,
            learner,
            champion_trainer=champion_trainer,
            signal_store=signal_store_for_sched,
            eval_scorer=eval_scorer,
        )
        await cron.start()
        log.info("learning-scheduler.started")
    elif _settings.enable_scheduling:
        log.warning("learning-scheduler.skipped (no learner context)")

    try:
        while True:
            if cron is not None:
                await cron.tick(_utcnow())
            await asyncio.sleep(3600)
    finally:
        if cron is not None:
            await cron.stop()
        await bus.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
