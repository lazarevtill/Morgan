"""Composition root — assemble the Orchestrator from settings (production) or fakes (tests).

Also registers the cold-path turn-storage subscriber: on RESPONSE_GENERATED, the just-finished
turn is persisted as episodic memory via the Learner. With the in-process bus this runs after the
reply is produced; with the Redis bus (later phases) it runs in the learning-worker, off-path.
"""

from __future__ import annotations

import asyncio
import pathlib
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, cast

import structlog

from morgan_brain.config import Settings, get_settings
from morgan_brain.core.orchestrator import Orchestrator
from morgan_brain.eval.golden import default_golden_path, load_golden_set
from morgan_brain.eval.harness import EvalHarness
from morgan_brain.eval.judge import LLMJudge
from morgan_brain.eval.runner import make_eval_scorer, make_predict_fn
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.interfaces.tools import BaseTool
from morgan_brain.learning.champion_trainer import ChampionTrainer
from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.learning.optimizer import AnyScorer, ReflectiveOptimizer
from morgan_brain.learning.profile import UserProfileBuilder
from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.learning.recorder import SignalRecorder
from morgan_brain.learning.signals import SignalStore
from morgan_brain.learning_lifecycle.factory import build_registry
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.models.memory import DEFAULT_PROJECT, Memory
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.modules.memory.indexing.embedder import Embedder, FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import (
    InMemoryVectorIndex,
    QdrantVectorIndex,
    VectorIndex,
)
from morgan_brain.modules.perception.text.analyzer import TextPerception
from morgan_brain.modules.personalization.adaptive import AdaptivePersonalizer
from morgan_brain.modules.reasoning.reasoner import ReasoningModule
from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
from morgan_brain.modules.tools.builtin.clock_tool import CurrentTimeTool
from morgan_brain.modules.tools.builtin.fetch_url import FetchUrlTool
from morgan_brain.modules.tools.builtin.memory_search import MemorySearchTool
from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry
from morgan_brain.modules.skills.registry import SkillRegistry
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.factory import build_embedder, build_router
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatResult, ToolSpec
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.security.permissions import Grant, PermissionGate, PermissionMode
from morgan_brain.bus import get_event_bus
from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import EventBus

# Name used when storing the system-prompt champion in the registry.
CHAMPION_PROMPT_NAME = "morgan-system"

log = structlog.get_logger("composition")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _run_coro_isolated(coro: Any) -> Any:
    """Run *coro* to completion even when called from inside a running event loop.

    ``build_worker_context`` can be invoked from the learning-worker's async ``run()`` (where
    ``asyncio.run()`` would raise "cannot be called from a running event loop") as well as
    synchronously at brain-api import time (no loop yet). Running the coroutine on a short-lived
    background thread with its own loop works in both cases.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread below
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _probe_embedding_dim(embedder: Embedder, settings: Settings) -> None:
    """Verify the live embedder's output dimension matches ``settings.embedding_dim``.

    Catches the class of bug where ``embedding_model`` and ``embedding_dim`` disagree (the
    vector store is created with a fixed ``dim`` up front — a mismatch fails silently at
    insert time otherwise, deep in the request path). Skipped for the hash backend (no
    provider to ask) and made non-fatal when the endpoint is unreachable — a startup probe
    must not turn a temporarily-down model server into a crash loop.
    """
    if settings.embedding_backend == "hash":
        return
    try:
        vector = _run_coro_isolated(embedder.embed("probe"))
    except Exception as exc:  # noqa: BLE001 — unreachable/misconfigured endpoint, not fatal
        log.warning("embedding-dim-probe.unreachable", error=str(exc))
        return
    if len(vector) != settings.embedding_dim:
        raise RuntimeError(
            f"embedding_model {settings.embedding_model!r} returned a "
            f"{len(vector)}-dimensional vector but settings.embedding_dim="
            f"{settings.embedding_dim}; the two must agree "
            "(set MORGAN_EMBEDDING_DIM to match the model, or vice versa)."
        )


def _build_vector_index(settings: Settings, conn: sqlite3.Connection) -> VectorIndex:
    """Return the configured vector index backend.

    "sqlite" → SqliteVectorIndex (default; persistent, shares *conn* with every other store).
    "memory" → InMemoryVectorIndex (ephemeral; tests/scratch use only).
    "qdrant" → QdrantVectorIndex (persistent; requires Qdrant at settings.qdrant_url).
    """
    if settings.vector_backend == "qdrant":
        return QdrantVectorIndex(
            url=settings.qdrant_url,
            dim=settings.embedding_dim,
        )
    if settings.vector_backend == "memory":
        return InMemoryVectorIndex()
    return SqliteVectorIndex(conn, dim=settings.embedding_dim)


def _register_turn_storage(
    bus: EventBus,
    learner: ConsolidationLearner,
) -> None:
    """Register the in-process **consolidation** subscriber.

    On RESPONSE_GENERATED, store the turn as episodic memory + run consolidation via the
    Learner. Registered only for the in-process bus; with the Redis bus the learning-worker
    subscribes to the same stream and does this off-path. Session history and the base
    interaction signal are NOT written here — the Orchestrator writes those in-process and
    synchronously (``_persist_turn``) regardless of bus backend, so the 2-process Redis
    topology never silently drops them (GAP-2). This subscriber is consolidation only.
    """

    async def _store_turn(event: Event) -> None:
        payload = event.payload
        session_id = payload.get("session_id") or "default"
        project = payload.get("project") or DEFAULT_PROJECT
        query = payload["request"]
        reply = payload["response"]

        convo = Conversation(
            user_id=event.user_id,
            project=project,
            session_id=session_id,
            messages=[
                Message(user_id=event.user_id, project=project, role=Role.USER, content=query),
                Message(user_id=event.user_id, project=project, role=Role.ASSISTANT, content=reply),
            ],
        )
        await learner.process_session(convo)

    bus.subscribe(EventType.RESPONSE_GENERATED, _store_turn)


def _build_tool_executor(
    gate: MemoryGate,
    clock: Callable[[], datetime],
    bus: EventBus,
) -> tuple[ToolExecutorImpl, list[ToolSpec]]:
    """Build the ToolRegistry + executor with builtin tools pre-registered.

    Safe tools (calculator, current_time, memory_search) are granted AUTO so the
    loop can run them without confirmation.  FetchUrlTool is registered but left on
    ASK (the default) so it cannot be called autonomously in the loop.
    """
    registry = ToolRegistry()
    # cast: concrete tools have named params in run() beyond **kwargs; the Protocol
    # uses **kwargs: Any — structurally compatible at runtime but mypy strict disagrees.
    registry.register(cast(BaseTool, CalculatorTool()))
    registry.register(cast(BaseTool, CurrentTimeTool(clock=clock)))
    registry.register(cast(BaseTool, MemorySearchTool(gate=gate)))
    registry.register(cast(BaseTool, FetchUrlTool()))

    perm_gate = PermissionGate(default=PermissionMode.ASK)
    for safe_tool in ("calculator", "current_time", "memory_search"):
        perm_gate.grant(Grant(tool=safe_tool, scope="execute"))

    executor = ToolExecutorImpl(registry=registry, gate=perm_gate, bus=bus)

    specs = [
        ToolSpec(
            name=spec["name"],
            description=spec["description"],
            parameters=spec["schema"],
        )
        for spec in registry.list_specs()
    ]
    return executor, specs


def _assemble(
    *,
    embedder: Embedder,
    router: RoleRouter,
    settings: Settings,
    clock: Callable[[], datetime],
    conn: sqlite3.Connection | None = None,
    temporal_path: str = ":memory:",
    prompt_registry: LocalPromptRegistry | None = None,
    history_store: "SessionHistoryStore | None" = None,
    bus: EventBus | None = None,
    vectors: VectorIndex | None = None,
) -> tuple[
    Orchestrator,
    MemoryModule,
    SignalStore,
    SignalRecorder,
    ToolExecutorImpl,
    SkillRegistry,
    ConsolidationLearner,
]:
    # One connection, shared by every store below (temporal, vectors, FTS, entities, episodics,
    # signals, history) -- required for a single-transaction forget() and for restart survival.
    # Production callers (build_app_context/build_worker_context) open this once over the real
    # data file and pass it in; callers that don't (test helpers) fall back to a private
    # ":memory:" connection so each test stays isolated.
    resolved_conn = conn if conn is not None else open_db(temporal_path)
    temporal = SqliteTemporalStore(conn=resolved_conn)
    # Use the injected vector index -- production callers (build_app_context/build_worker_context)
    # always pass one built from settings via _build_vector_index; test callers that don't care
    # about the vector backend fall back to an ephemeral InMemoryVectorIndex so a FakeEmbedder's
    # low dimensionality never collides with the real embedding_dim a SqliteVectorIndex would
    # enforce.
    resolved_vectors: VectorIndex = vectors if vectors is not None else InMemoryVectorIndex()
    memory_module = MemoryModule(
        embedder=embedder,
        vectors=resolved_vectors,
        temporal=temporal,
        clock=clock,
        fts=FtsIndex(resolved_conn),
        entities=EntityIndex(resolved_conn),
        episodics=EpisodicStore(resolved_conn),
    )
    gate = MemoryGate(memory_module)
    reg = CapabilityRegistry.from_packaged()
    consolidator = MemoryConsolidator(
        gate=gate,
        router=router,
        capability_registry=reg,
        clock=clock,
    )
    signal_store = SignalStore(resolved_conn, clock=clock)
    recorder = SignalRecorder(store=signal_store, clock=clock)
    profile_builder = UserProfileBuilder(
        gate=gate,
        signals=signal_store,
        router=router,
        capability_registry=reg,
        clock=clock,
    )
    learner = ConsolidationLearner(
        consolidator=consolidator,
        gate=gate,
        clock=clock,
        profile_builder=profile_builder,
    )
    # Use the injected bus (tests) or the config-driven one (production).
    # get_event_bus() reads settings.event_bus: "inproc" → InProcessBus, "redis" → RedisStreamsBus.
    resolved_bus: EventBus = bus if bus is not None else get_event_bus()
    # Register the in-process turn-storage subscriber only when the bus is in-process.
    # With the Redis bus the learning-worker process subscribes to the same stream and
    # handles turn storage off-path — registering here would double-process every turn.
    if isinstance(resolved_bus, InProcessBus):
        _register_turn_storage(resolved_bus, learner)
    personalizer = AdaptivePersonalizer(
        profile_builder=profile_builder,
        budget=settings.personalization_budget,
    )
    skills = SkillRegistry(registry=prompt_registry)

    executor, tool_specs = _build_tool_executor(gate=gate, clock=clock, bus=resolved_bus)

    orch = Orchestrator(
        perception=TextPerception(),
        personalizer=personalizer,
        memory=gate,
        skills=skills,
        reasoner=ReasoningModule(router=router, role="strong", executor=executor),
        learner=learner,
        bus=resolved_bus,
        tools=tool_specs,
        recorder=recorder,
        history_store=history_store,
    )
    return orch, memory_module, signal_store, recorder, executor, skills, learner


@dataclass
class AppContext:
    """All handles needed by brain-api create_app + routes."""

    orchestrator: Orchestrator
    signal_store: SignalStore
    signal_recorder: SignalRecorder
    executor: ToolExecutorImpl
    skills: SkillRegistry
    learner: ConsolidationLearner
    prompt_registry: PromptRegistry
    bus: EventBus
    vectors: VectorIndex
    history_store: "SessionHistoryStore | None" = field(default=None)


def _load_champion_override(registry: PromptRegistry) -> str:
    """Best-effort synchronous read of the champion body.

    Returns the champion body for CHAMPION_PROMPT_NAME if one is stored, or
    empty string if no champion exists yet or if any error occurs.  The result
    is read once at startup and passed as system_override on every turn.

    This is intentionally synchronous + blocking (sqlite is fast enough at
    startup) so it fits the existing synchronous factory pattern.
    """
    import asyncio

    try:
        loop = asyncio.new_event_loop()
        try:
            version = loop.run_until_complete(registry.champion(CHAMPION_PROMPT_NAME))
        finally:
            loop.close()
        return version.body if version is not None else ""
    except Exception:
        return ""


class ChampionCache:
    """Live, TTL-cached champion preprompt body.

    The champion is promoted by the learning-worker into the shared PromptRegistry.
    brain-api read it once at startup, so a freshly promoted champion never reached
    live traffic without an operator restart. This refreshes on a short TTL (default
    30s) — at most one cheap SQLite read per window — so a promotion goes live within
    the TTL with negligible per-turn cost and no restart. The last-known body is kept
    if a refresh errors, so a transient registry hiccup never blanks the champion.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        name: str = CHAMPION_PROMPT_NAME,
        *,
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._name = name
        self._ttl = ttl_s
        self._clock = clock
        self._body = ""
        self._expires = 0.0
        self._loaded = False

    async def body(self) -> str:
        now = self._clock()
        if not self._loaded or now >= self._expires:
            try:
                version = await self._registry.champion(self._name)
                self._body = version.body if version is not None else ""
            except Exception:  # noqa: BLE001 — keep last-known body on transient errors
                pass
            self._expires = now + self._ttl
            self._loaded = True
        return self._body


def build_app_context(settings: Settings | None = None) -> AppContext:
    """Production wiring: builds all components and returns an AppContext.

    The AppContext exposes every handle needed by API routes.
    Reads the current champion prompt (if any) and wires it as system_override.
    """
    settings = settings or get_settings()
    temporal_path = _sqlite_path(settings.temporal_db_url)
    if temporal_path != ":memory:":
        pathlib.Path(temporal_path).parent.mkdir(parents=True, exist_ok=True)

    # One connection for the whole process: facts, vectors, FTS, entities, episodics, signals
    # and session history all live in this single database file (required for a
    # single-transaction forget() and for restart survival).
    conn = open_db(temporal_path)
    vectors = _build_vector_index(settings, conn)

    embedder = build_embedder(settings)
    _probe_embedding_dim(embedder, settings)
    router = build_router(settings)
    # Use the persistent registry (same path as the worker) so brain-api reads
    # champions written by the learning-worker.
    prom_registry: PromptRegistry = build_registry(settings)
    # History must be durable: it threads multi-turn context, so an in-memory store would
    # silently collapse every turn to turn 1 across restarts.
    history_store = SessionHistoryStore(conn, clock=_utcnow)
    bus = get_event_bus()

    orch, _, signal_store, recorder, executor, skills, learner = _assemble(
        embedder=embedder,
        router=router,
        settings=settings,
        clock=_utcnow,
        conn=conn,
        prompt_registry=prom_registry if isinstance(prom_registry, LocalPromptRegistry) else None,
        history_store=history_store,
        bus=bus,
        vectors=vectors,
    )

    return AppContext(
        orchestrator=orch,
        signal_store=signal_store,
        signal_recorder=recorder,
        executor=executor,
        skills=skills,
        bus=bus,
        vectors=vectors,
        learner=learner,
        prompt_registry=prom_registry,
        history_store=history_store,
    )


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Production wiring (Ollama + in-memory vectors for Phase 1; Qdrant swaps in later)."""
    return build_app_context(settings).orchestrator


@dataclass
class WorkerContext:
    """Handles needed by the learning-worker process."""

    learner: ConsolidationLearner
    signal_store: SignalStore
    signal_recorder: SignalRecorder
    bus: EventBus
    champion_trainer: ChampionTrainer
    prompt_registry: PromptRegistry
    eval_scorer: AnyScorer


def build_worker_context(settings: Settings | None = None) -> WorkerContext:
    """Build PRODUCTION learning-worker context over configured backends.

    Shares the same database file as brain-api (settings.temporal_db_url / data_dir) so both
    processes read/write the SAME SQLite database. Use a shared Qdrant collection
    (vector_backend="qdrant") for the vector store in multi-process deployments.

    With ``event_bus="redis"``, the worker subscribes to the Redis stream that
    brain-api publishes to. With ``event_bus="inproc"`` (dev/test), both processes
    would share an in-process bus — only useful in single-process mode.

    The worker also builds:
    - A PERSISTENT ``LocalPromptRegistry`` (same data/ path as brain-api).
    - A ``ReflectiveOptimizer`` + ``ChampionTrainer`` for the offline optimize job.
    - An eval-backed scorer (``make_eval_scorer``) so champion promotion is gated
      by the real assistant running the golden set — never on the hot path.
    """
    settings = settings or get_settings()
    temporal_path = _sqlite_path(settings.temporal_db_url)
    if temporal_path != ":memory:":
        pathlib.Path(temporal_path).parent.mkdir(parents=True, exist_ok=True)

    # One connection for the whole process, same as build_app_context.
    conn = open_db(temporal_path)
    vectors = _build_vector_index(settings, conn)

    embedder = build_embedder(settings)
    _probe_embedding_dim(embedder, settings)
    router = build_router(settings)
    bus = get_event_bus()

    orch, _, signal_store, recorder, _, _, learner = _assemble(
        embedder=embedder,
        router=router,
        settings=settings,
        clock=_utcnow,
        conn=conn,
        bus=bus,
        vectors=vectors,
    )

    # Persistent registry — same file as brain-api reads.
    registry: PromptRegistry = build_registry(settings)

    # Optimizer + trainer.
    optimizer = ReflectiveOptimizer(router=router)
    trainer = ChampionTrainer(optimizer=optimizer, registry=registry, clock=_utcnow)

    # Eval-backed scorer: run the real assistant over the golden set.
    golden_path_str = settings.eval_golden_path
    golden_items = load_golden_set(golden_path_str if golden_path_str else default_golden_path())
    judge = LLMJudge(router=router, role="judge")
    harness = EvalHarness(judge=judge)
    # with_confidence=True so the eval harness computes + logs calibration (Brier/ECE) on every
    # worker eval run — report-only (it never gates promotion yet).
    predict_fn = make_predict_fn(orchestrator=orch, clock=_utcnow, with_confidence=True)
    eval_scorer: AnyScorer = make_eval_scorer(
        harness=harness,
        golden_items=golden_items,
        predict_fn=predict_fn,
    )

    return WorkerContext(
        learner=learner,
        signal_store=signal_store,
        signal_recorder=recorder,
        bus=bus,
        champion_trainer=trainer,
        prompt_registry=registry,
        eval_scorer=eval_scorer,
    )


class MemoryTestHandle:
    """Thin handle exposing raw recall for integration-test assertions.

    Also exposes the orchestrator's ``InProcessBus`` (``.bus``): since Task 15 made
    ``publish()`` enqueue rather than run handlers inline, a test that writes a turn and then
    immediately recalls it must ``await handle.bus.start()`` before the turn and
    ``await handle.bus.drain()`` before recalling, or the cold-path storage subscriber won't
    have run yet.
    """

    def __init__(self, memory_module: MemoryModule, bus: InProcessBus) -> None:
        self._memory_module = memory_module
        self.bus = bus

    async def recall_raw(
        self, *, user_id: str, text: str, project: str = "default"
    ) -> list[Memory]:
        from morgan_brain.models.memory import MemoryQuery

        return await self._memory_module.recall(
            MemoryQuery(user_id=user_id, project=project, text=text, top_k=10)
        )


def build_orchestrator_for_test(
    *,
    reply: str,
    clock: Callable[[], datetime],
    chat_results: "list[ChatResult] | None" = None,
) -> tuple[Orchestrator, MemoryTestHandle]:
    """Test wiring: fake embedder + fake LLM + in-memory stores. Returns (orchestrator, memory
    handle) where the memory handle exposes recall_raw() for assertions.

    Args:
        reply:        Default reply text for FakeChatClient (back-compat).
        clock:        Deterministic clock callable.
        chat_results: Optional list of ``ChatResult`` objects to script per-call responses
                      (enables tool-call integration tests without network).
    """
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")

    # Build a RoleRouter backed by FakeChatClient for the test.
    # The fake client is used for both the orchestrator's LLM calls AND consolidation/profile.
    fake_client: FakeChatClient
    if chat_results is not None:
        fake_client = FakeChatClient(results=chat_results)
    else:
        fake_client = FakeChatClient(reply=reply)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    test_router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )

    test_bus = InProcessBus()
    orch, memory_module, _, _, _, _, _ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=test_router,
        settings=settings,
        clock=clock,
        temporal_path=":memory:",
        bus=test_bus,
    )
    return orch, MemoryTestHandle(memory_module, test_bus)


def build_orchestrator_for_test_with_signals(
    *,
    reply: str,
    clock: Callable[[], datetime],
    chat_results: "list[ChatResult] | None" = None,
) -> tuple[Orchestrator, SignalStore, InProcessBus]:
    """Test wiring variant that also returns the SignalStore and InProcessBus.

    Used by tests that assert on signal recording and event payloads.
    """
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")

    if chat_results is not None:
        fake_client: FakeChatClient = FakeChatClient(results=chat_results)
    else:
        fake_client = FakeChatClient(reply=reply)
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    test_router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )

    test_bus = InProcessBus()
    orch, _, signal_store, _, _, _, _ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=test_router,
        settings=settings,
        clock=clock,
        temporal_path=":memory:",
        bus=test_bus,
    )
    # Return the injected bus directly — no cast needed since we created it above.
    return orch, signal_store, test_bus


def _sqlite_path(url: str) -> str:
    """Turn a sqlite:/// URL into a filesystem path; pass through ':memory:'."""
    prefix = "sqlite:///"
    return url[len(prefix) :] if url.startswith(prefix) else url
