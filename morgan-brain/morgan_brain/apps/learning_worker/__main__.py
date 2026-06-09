"""Run the learning-worker: ``python -m morgan_brain.apps.learning_worker``.

Phase 0: subscribes to RESPONSE_GENERATED / SESSION_END and logs.
Phase 2+: replaces the handler body with extraction → UserModel update → consolidation.
Phase 4: when ``enable_scheduling=True``, builds a CronService + LearningScheduler
         and runs the nightly consolidation + optional optimizer jobs.
         When ``enable_proactivity=True``, wires a ProactivityEngine that derives
         candidates from user patterns on every HEARTBEAT and publishes allowed ones.

Default flags are False → current behavior unchanged for existing deployments.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from morgan_brain.bus import get_event_bus
from morgan_brain.config import Settings, get_settings
from morgan_brain.interfaces.events import Event, EventBus, EventType

if TYPE_CHECKING:
    from morgan_brain.scheduling.cron import CronService
    from morgan_brain.scheduling.learning_jobs import LearningScheduler
    from morgan_brain.proactivity.engine import ProactivityEngine

log = structlog.get_logger("learning-worker")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _on_response(event: Event) -> None:
    # Phase 2: queue the session for trait/preference/behavior extraction.
    log.info("response.observed", user_id=event.user_id)


# ---------------------------------------------------------------------------
# Scheduling helpers (Phase 4)
# ---------------------------------------------------------------------------


def _build_cron_service() -> CronService:
    from morgan_brain.scheduling.cron import CronService

    return CronService(clock=_utcnow)


def _build_learning_scheduler(cron: CronService) -> LearningScheduler | None:
    """Build a LearningScheduler over a stub learner.

    In a full production setup the learner + trainer would be built via the
    composition root.  For the learning-worker we keep it lightweight: a
    ConsolidationLearner over a SQLite temporal store.  The scheduler drives the
    same consolidation path as the request-path learner.
    """
    settings = get_settings()
    try:
        from morgan_brain.scheduling.learning_jobs import LearningScheduler

        # Import the composition root's learner factory lazily to avoid circular
        # imports and to keep the worker startup fast when scheduling is disabled.
        from morgan_brain.composition import _assemble
        from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
        from morgan_brain.providers.adapters.fake import FakeChatClient
        from morgan_brain.providers.capability import CapabilityRegistry
        from morgan_brain.providers.router import Binding, RoleRouter

        # Build a minimal learner (no LLM calls from the scheduler itself —
        # consolidation uses the router only when called).
        fake_client = FakeChatClient(reply="")
        reg = CapabilityRegistry.from_seed(
            {"fake/noop": {"supports_tools": False, "json_mode": "none", "context_window": 4096}}
        )
        router = RoleRouter(
            reg=reg,
            bindings={"strong": [Binding("fake", "noop", fake_client)]},
        )
        _orch, _mem, _, _, _, _, learner = _assemble(
            embedder=FakeEmbedder(dim=16),
            router=router,
            settings=Settings(),
            clock=_utcnow,
            temporal_path=":memory:",
        )

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
# Proactivity helpers (Phase 4)
# ---------------------------------------------------------------------------


def _build_proactivity_engine(bus: EventBus) -> ProactivityEngine | None:
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


def _register_proactivity_handler(bus: EventBus, engine: ProactivityEngine) -> None:
    """Subscribe to HEARTBEAT and derive+send suggestions from user patterns."""
    from morgan_brain.models.user import UserModel

    _engine = engine

    async def _on_heartbeat(event: Event) -> None:
        try:
            # In production the user model would be loaded from the store.
            # Here we build a minimal one — the consent gate will gate on stage.
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
    bus = get_event_bus()
    bus.subscribe(EventType.RESPONSE_GENERATED, _on_response)
    bus.subscribe(EventType.SESSION_END, _on_response)
    await bus.start()
    log.info("learning-worker.started")

    cron = None
    if _settings.enable_scheduling:
        cron = _build_cron_service()
        _build_learning_scheduler(cron)
        await cron.start()
        log.info("learning-scheduler.started")

    if _settings.enable_proactivity:
        engine = _build_proactivity_engine(bus)
        if engine is not None:
            _register_proactivity_handler(bus, engine)

    try:
        while True:
            if cron is not None:
                # Drive the in-process scheduler (no-op when using APScheduler).
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
