"""Composition root — assemble the Orchestrator from settings (production) or fakes (tests).

Also registers the cold-path turn-storage subscriber: on RESPONSE_GENERATED, the just-finished
turn is persisted as episodic memory via the Learner. With the in-process bus this runs after the
reply is produced; with the Redis bus (later phases) it runs in the learning-worker, off-path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, cast

from morgan_brain.config import Settings, get_settings
from morgan_brain.core.orchestrator import Orchestrator
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.interfaces.tools import BaseTool
from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.learning.profile import UserProfileBuilder
from morgan_brain.models.memory import Memory
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.modules.memory.indexing.embedder import Embedder, FakeEmbedder, OllamaEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
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
from morgan_brain.providers.factory import build_router
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatResult, ToolSpec
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.security.permissions import Grant, PermissionGate, PermissionMode
from morgan_brain.bus.inproc import InProcessBus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _register_turn_storage(bus: InProcessBus, learner: ConsolidationLearner) -> None:
    async def _store_turn(event: Event) -> None:
        payload = event.payload
        convo = Conversation(
            user_id=event.user_id,
            session_id=payload.get("session_id") or "default",
            messages=[
                Message(user_id=event.user_id, role=Role.USER, content=payload["request"]),
                Message(user_id=event.user_id, role=Role.ASSISTANT, content=payload["response"]),
            ],
        )
        await learner.process_session(convo)

    bus.subscribe(EventType.RESPONSE_GENERATED, _store_turn)


def _build_tool_executor(
    gate: MemoryGate,
    clock: Callable[[], datetime],
    bus: InProcessBus,
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
    temporal_path: str,
    prompt_registry: LocalPromptRegistry | None = None,
) -> tuple[Orchestrator, MemoryModule]:
    temporal = SqliteTemporalStore(temporal_path)
    memory_module = MemoryModule(
        embedder=embedder,
        vectors=InMemoryVectorIndex(),
        temporal=temporal,
        clock=clock,
    )
    gate = MemoryGate(memory_module)
    reg = CapabilityRegistry.from_packaged()
    consolidator = MemoryConsolidator(
        gate=gate,
        temporal=temporal,
        router=router,
        capability_registry=reg,
        clock=clock,
    )
    profile_builder = UserProfileBuilder(
        gate=gate,
        signals=None,
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
    bus = InProcessBus()
    _register_turn_storage(bus, learner)
    personalizer = AdaptivePersonalizer(
        profile_builder=profile_builder,
        budget=settings.personalization_budget,
    )
    skills = SkillRegistry(registry=prompt_registry)

    executor, tool_specs = _build_tool_executor(gate=gate, clock=clock, bus=bus)

    orch = Orchestrator(
        perception=TextPerception(),
        personalizer=personalizer,
        memory=gate,
        skills=skills,
        reasoner=ReasoningModule(router=router, role="strong", executor=executor),
        learner=learner,
        bus=bus,
        tools=tool_specs,
    )
    return orch, memory_module


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Production wiring (Ollama + in-memory vectors for Phase 1; Qdrant swaps in later)."""
    import pathlib

    settings = settings or get_settings()
    temporal_path = _sqlite_path(settings.temporal_db_url)
    if temporal_path != ":memory:":
        pathlib.Path(temporal_path).parent.mkdir(parents=True, exist_ok=True)
    embedder = OllamaEmbedder(settings.llm_endpoint, settings.embedding_model)
    router = build_router(settings)
    prom_registry = LocalPromptRegistry()
    orch, _ = _assemble(
        embedder=embedder,
        router=router,
        settings=settings,
        clock=_utcnow,
        temporal_path=temporal_path,
        prompt_registry=prom_registry,
    )
    return orch


class MemoryTestHandle:
    """Thin handle exposing raw recall for integration-test assertions."""

    def __init__(self, memory_module: MemoryModule) -> None:
        self._memory_module = memory_module

    async def recall_raw(self, *, user_id: str, text: str) -> list[Memory]:
        from morgan_brain.models.memory import MemoryQuery

        return await self._memory_module.recall(MemoryQuery(user_id=user_id, text=text, top_k=10))


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

    orch, memory_module = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=test_router,
        settings=settings,
        clock=clock,
        temporal_path=":memory:",
    )
    return orch, MemoryTestHandle(memory_module)


def _sqlite_path(url: str) -> str:
    """Turn a sqlite:/// URL into a filesystem path; pass through ':memory:'."""
    prefix = "sqlite:///"
    return url[len(prefix) :] if url.startswith(prefix) else url
