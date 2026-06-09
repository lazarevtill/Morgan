"""Run the learning-worker: ``python -m morgan_brain.apps.learning_worker``.

Handler path
------------
``RESPONSE_GENERATED`` → reconstruct a :class:`Conversation` (user + assistant
messages, session_id) → ``learner.process_session(conversation)``.

Scheduler path (enable_scheduling=True)
---------------------------------------
Builds a :class:`CronService` + :class:`LearningScheduler` over the REAL
:class:`ConsolidationLearner` from :func:`build_worker_context`, then registers
nightly consolidation (and optional optimizer if ChampionTrainer is wired).

Proactivity path (enable_proactivity=True)
------------------------------------------
On every ``HEARTBEAT`` event, loads the current :class:`UserModel` via
``learner.user_model(user_id)`` and derives + publishes consent-gated suggestions.

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
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from morgan_brain.bus import get_event_bus
from morgan_brain.config import Settings, get_settings
from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.models.message import Conversation, Message, Role

if TYPE_CHECKING:
    from morgan_brain.learning.learner import ConsolidationLearner
    from morgan_brain.scheduling.cron import CronService
    from morgan_brain.scheduling.learning_jobs import LearningScheduler
    from morgan_brain.proactivity.engine import ProactivityEngine

log = structlog.get_logger("learning-worker")
_logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Real response handler factory
# ---------------------------------------------------------------------------


def _make_response_handler(
    learner: "ConsolidationLearner",
    *,
    clock: Callable[[], datetime] = _utcnow,
) -> Callable[[Event], "asyncio.coroutines.Coroutine[object, object, None]"]:
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
            query = payload.get("request", "")
            reply = payload.get("response", "")

            convo = Conversation(
                user_id=event.user_id,
                session_id=session_id,
                messages=[
                    Message(user_id=event.user_id, role=Role.USER, content=query),
                    Message(user_id=event.user_id, role=Role.ASSISTANT, content=reply),
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


def _build_cron_service() -> "CronService":
    from morgan_brain.scheduling.cron import CronService

    return CronService(clock=_utcnow)


def _build_learning_scheduler(
    cron: "CronService",
    learner: "ConsolidationLearner",
) -> "LearningScheduler | None":
    """Build a :class:`LearningScheduler` over the real learner from the worker context.

    Registers nightly consolidation. ChampionTrainer is not wired here (requires an
    eval runner — see Wire-D); when it becomes available, pass it as ``champion_trainer``.
    """
    settings = get_settings()
    try:
        from morgan_brain.scheduling.learning_jobs import LearningScheduler

        ls = LearningScheduler(
            cron=cron,
            learner=learner,
            clock=_utcnow,
        )
        ls.register_default_jobs(settings.owner_user_id)
        log.info("learning-scheduler.registered", user_id=settings.owner_user_id)
        return ls
    except Exception:
        log.exception("learning-scheduler.build-failed")
        return None


# ---------------------------------------------------------------------------
# Proactivity helpers
# ---------------------------------------------------------------------------


def _build_proactivity_engine(bus: EventBus) -> "ProactivityEngine | None":
    try:
        from morgan_brain.proactivity.consent import ConsentGate, ConsentRule
        from morgan_brain.proactivity.engine import ProactivityEngine
        from morgan_brain.models.user import RelationshipStage

        gate = ConsentGate(
            rules=[
                ConsentRule(kind="reminder", min_stage=RelationshipStage.FAMILIAR),
                ConsentRule(kind="suggestion", min_stage=RelationshipStage.FAMILIAR),
                ConsentRule(kind="summary", min_stage=RelationshipStage.TRUSTED),
            ]
        )
        engine = ProactivityEngine(gate=gate, bus=bus, clock=_utcnow)
        log.info("proactivity-engine.built")
        return engine
    except Exception:
        log.exception("proactivity-engine.build-failed")
        return None


def _register_proactivity_handler(
    bus: EventBus,
    engine: "ProactivityEngine",
    learner: "ConsolidationLearner | None" = None,
) -> None:
    """Subscribe to HEARTBEAT; derive + consent-gate + publish suggestions.

    When a real *learner* is provided, loads the persisted :class:`UserModel` via
    ``learner.user_model(user_id)``.  Falls back to a default model otherwise.
    """
    _engine = engine
    _learner = learner

    async def _on_heartbeat(event: Event) -> None:
        try:
            if _learner is not None:
                user_model = await _learner.user_model(event.user_id)
            else:
                from morgan_brain.models.user import UserModel

                user_model = UserModel(user_id=event.user_id)
            candidates = _engine.derive_from_patterns(user_model)
            if candidates:
                await _engine.maybe_suggest(
                    user_id=event.user_id,
                    user_model=user_model,
                    candidates=candidates,
                )
        except Exception:
            log.exception("proactivity.heartbeat-handler-failed", user_id=event.user_id)

    bus.subscribe(EventType.HEARTBEAT, _on_heartbeat)
    log.info("proactivity-handler.registered")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


async def run(settings: Settings | None = None) -> None:
    _settings = settings or get_settings()

    # Build the real worker context (production stores + configured bus).
    # For inproc dev runs brain-api already handles turn storage, but the
    # worker wiring is harmless (no-op subscriber on an isolated inproc bus).
    try:
        from morgan_brain.composition import build_worker_context

        ctx = build_worker_context(_settings)
        learner = ctx.learner
        log.info("worker-context.built")
    except Exception:
        log.exception("worker-context.build-failed; starting with no-op learner")
        learner = None  # type: ignore[assignment]

    # Obtain and start the configured event bus.
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
        _build_learning_scheduler(cron, learner)
        await cron.start()
        log.info("learning-scheduler.started")
    elif _settings.enable_scheduling:
        log.warning("learning-scheduler.skipped (no learner context)")

    if _settings.enable_proactivity:
        engine = _build_proactivity_engine(bus)
        if engine is not None:
            _register_proactivity_handler(bus, engine, learner)

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
