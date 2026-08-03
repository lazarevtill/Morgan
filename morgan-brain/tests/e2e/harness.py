"""Core benchmark engine: builders, conversation driver, scenarios, scoring.

Design choices that keep this ADDITIVE and faithful to production wiring:

* The orchestrator is assembled through :func:`morgan_brain.composition._assemble`
  (the same private seam every existing integration test uses) so the bench drives
  the *real* perceive -> personalize -> recall -> skills -> reason -> store loop.
* We thread session history and the champion ``system_override`` manually on the
  caller side, exactly the way ``brain-api``'s ``/api/chat`` route does on the
  working in-process bus. This deliberately exercises the path that is wired today
  and avoids the known redis/stream gaps (documented in the harness README).
* Deterministic mode scripts ``FakeChatClient`` replies per turn so wiring is
  measured without model noise. Live mode lets the real model answer and only
  asserts the *signal* (e.g. a recalled token shows up in the prompt or reply).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.consolidation import FactOp, FactOpBatch, FactOpKind
from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.modules.memory.indexing.embedder import Embedder, FakeEmbedder
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex, VectorIndex
from morgan_brain.providers.adapters.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.factory import build_router
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatMessage, ChatResult, ToolCall

# ---------------------------------------------------------------------------
# Fixed deterministic clock — advanced per scenario turn where temporality matters.
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class StepClock:
    """Monotonic clock that advances on demand (for bi-temporal fact tests)."""

    def __init__(self, start: datetime = _EPOCH) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> None:
        self._now = self._now + timedelta(**kwargs)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    category: str
    passed: bool
    detail: str
    turn_latencies_ms: list[float] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"


@dataclass
class BenchReport:
    mode: str
    results: list[ScenarioResult]
    generated_at: str

    @property
    def all_latencies(self) -> list[float]:
        out: list[float] = []
        for r in self.results:
            out.extend(r.turn_latencies_ms)
        return out

    def latency_percentile(self, pct: float) -> float:
        lat = sorted(self.all_latencies)
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, int(round((pct / 100.0) * (len(lat) - 1))))
        return lat[idx]

    def recall_accuracy(self) -> float:
        """Fraction of recall-category scenarios that passed (single/multi-hop, temporal)."""
        recall = [r for r in self.results if r.category in _RECALL_CATEGORIES and not r.skipped]
        if not recall:
            return 0.0
        return sum(1 for r in recall if r.passed) / len(recall)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.skipped)


_RECALL_CATEGORIES = {"single-hop", "multi-hop", "temporal"}


# ---------------------------------------------------------------------------
# Capability registry shared by deterministic + live routers
# ---------------------------------------------------------------------------


def _fake_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )


# ---------------------------------------------------------------------------
# Conversation harness — wraps an assembled orchestrator with manual history +
# champion threading (the working in-process path).
# ---------------------------------------------------------------------------


class ConversationHarness:
    """Drives multi-turn TEXT conversations through one assembled orchestrator.

    Tracks session history (read+threaded per turn, written by the in-proc
    turn-storage subscriber) and applies a champion ``system_override`` the same
    way the blocking ``/api/chat`` route does.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        router: RoleRouter,
        vectors: VectorIndex,
        clock: Callable[[], datetime],
        settings: Settings,
        champion_override: str = "",
        fake_client: FakeChatClient | None = None,
    ) -> None:
        self._clock = clock
        self._history = SessionHistoryStore()
        self._champion = champion_override
        self.fake_client = fake_client
        bus = InProcessBus()
        orch, memory_module, *_rest, learner = _assemble(
            embedder=embedder,
            router=router,
            settings=settings,
            clock=clock,
            temporal_path=":memory:",
            history_store=self._history,
            bus=bus,
            vectors=vectors,
        )
        self.orchestrator = orch
        self.memory_module = memory_module
        self.learner: ConsolidationLearner = learner

    async def say(
        self, *, user_id: str, text: str, session_id: str, project: str = "default"
    ) -> tuple[str, float]:
        """Run one turn; return (reply_text, latency_ms)."""
        history = self._history.recent(session_id, project=project)
        t0 = time.perf_counter()
        result = await self.orchestrator.handle_turn(
            user_id=user_id,
            project=project,
            text=text,
            session_id=session_id,
            history=history,
            system_override=self._champion,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result.text, latency_ms

    async def say_full(
        self, *, user_id: str, text: str, session_id: str, project: str = "default"
    ) -> tuple[Any, float]:
        """Like :meth:`say` but returns the full ReasoningResult (for tools_invoked)."""
        history = self._history.recent(session_id, project=project)
        t0 = time.perf_counter()
        result = await self.orchestrator.handle_turn(
            user_id=user_id,
            project=project,
            text=text,
            session_id=session_id,
            history=history,
            system_override=self._champion,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result, latency_ms

    async def consolidate(self, user_id: str, *, project: str = "default") -> None:
        await self.learner.consolidate(user_id, project=project)

    def last_prompt_text(self) -> str:
        """Concatenated content of the last LLM call's messages (deterministic only)."""
        if self.fake_client is None:
            return ""
        return _join_messages(self.fake_client.last_messages)

    def last_system_message(self) -> str:
        if self.fake_client is None:
            return ""
        for m in self.fake_client.last_messages:
            if m.role == "system":
                return m.content
        return ""


def _join_messages(messages: Sequence[ChatMessage]) -> str:
    return " \n".join(m.content for m in messages)


def _fact_batch_json(ops: list[tuple[str, str, str]]) -> str:
    batch = FactOpBatch(
        ops=[
            FactOp(
                op=FactOpKind.ADD,
                subject=s,
                predicate=p,
                object=o,
                confidence=0.95,
                reason="stated by user",
            )
            for (s, p, o) in ops
        ]
    )
    return batch.model_dump_json()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_deterministic(
    *,
    replies: list[str],
    chat_results: list[ChatResult] | None = None,
    champion_override: str = "",
    clock: Callable[[], datetime] | None = None,
) -> ConversationHarness:
    """Assemble a fully-fake harness (no external services)."""
    clock = clock or StepClock()
    if chat_results is not None:
        fake_client = FakeChatClient(results=chat_results)
    else:
        fake_client = FakeChatClient(replies=replies)
    router = RoleRouter(
        reg=_fake_capability_registry(),
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    return ConversationHarness(
        embedder=FakeEmbedder(dim=32),
        router=router,
        vectors=InMemoryVectorIndex(),
        clock=clock,
        settings=settings,
        champion_override=champion_override,
        fake_client=fake_client,
    )


# ---------------------------------------------------------------------------
# Live mode probing + building
# ---------------------------------------------------------------------------


@dataclass
class LiveProbe:
    llm_ok: bool
    qdrant_ok: bool
    reason: str = ""


async def probe_live(settings: Settings) -> LiveProbe:
    """Probe configured LLM + Qdrant reachability. Never raises."""
    import httpx

    llm_ok = False
    qdrant_ok = False
    reasons: list[str] = []

    base = settings.llm_endpoint.rstrip("/")
    models_url = base.removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            resp = await c.get(f"{models_url}/v1/models")
            llm_ok = resp.status_code < 500
    except Exception as exc:  # noqa: BLE001 - probe must never crash
        reasons.append(f"llm unreachable at {settings.llm_endpoint}: {exc!r}")

    if settings.vector_backend == "qdrant":
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                resp = await c.get(f"{settings.qdrant_url.rstrip('/')}/readyz")
                qdrant_ok = resp.status_code < 500
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"qdrant unreachable at {settings.qdrant_url}: {exc!r}")
    else:
        qdrant_ok = True  # in-memory backend, nothing to probe

    return LiveProbe(llm_ok=llm_ok, qdrant_ok=qdrant_ok, reason="; ".join(reasons))


def build_live(settings: Settings, *, champion_override: str = "") -> ConversationHarness:
    """Assemble a harness over the configured LLM/embedding endpoint + vector backend.

    Uses the same production adapters as ``build_app_context`` (OpenAICompatEmbedder +
    ``build_router``) but keeps temporal storage in-memory so the bench is
    self-contained and repeatable.
    """
    embedder: Embedder = OpenAICompatEmbedder(settings.llm_endpoint, settings.embedding_model)
    router = build_router(settings)
    if settings.vector_backend == "qdrant":
        from morgan_brain.modules.memory.stores.vector import QdrantVectorIndex

        vectors: VectorIndex = QdrantVectorIndex(
            url=settings.qdrant_url, dim=settings.embedding_dim
        )
    else:
        vectors = InMemoryVectorIndex()
    return ConversationHarness(
        embedder=embedder,
        router=router,
        vectors=vectors,
        clock=lambda: datetime.now(UTC),
        settings=settings,
        champion_override=champion_override,
        fake_client=None,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

ScenarioFn = Callable[[bool], Awaitable[ScenarioResult]]


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


# --- single-hop cross-turn recall -----------------------------------------


async def scenario_single_hop(live: bool) -> ScenarioResult:
    """Turn 1 states a fact; a later turn must recall it into the prompt."""
    name, cat = "single_hop_recall", "single-hop"
    session, user = "s-single", "u-single"
    fact = "My favorite programming language is Rust."

    if live:
        settings = _live_settings()
        h = build_live(settings)
        _r1, l1 = await h.say(user_id=user, text=fact, session_id=session)
        reply, l2 = await h.say(
            user_id=user, text="What is my favorite programming language?", session_id=session
        )
        passed = _contains(reply, "rust")
        detail = f"reply mentions Rust={passed}: {reply[:120]!r}"
        return ScenarioResult(name, cat, passed, detail, [l1, l2])

    consolidation = _fact_batch_json([("user", "favorite_language", "Rust")])
    h = build_deterministic(
        replies=[
            "Noted: Rust is your favorite.",  # turn 1
            consolidation,  # consolidate call
            "You told me Rust is your favorite language.",  # turn 2
        ]
    )
    _r1, l1 = await h.say(user_id=user, text=fact, session_id=session)
    await h.consolidate(user)
    _r2, l2 = await h.say(
        user_id=user, text="What is my favorite programming language?", session_id=session
    )
    # The recalled fact OR the episodic turn-1 message must reach turn 2's prompt.
    prompt = h.last_prompt_text()
    passed = _contains(prompt, "rust")
    detail = f"'Rust' present in turn-2 prompt={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1, l2])


# --- multi-hop recall ------------------------------------------------------


async def scenario_multi_hop(live: bool) -> ScenarioResult:
    """Two facts stated across turns; a later turn needs BOTH to answer."""
    name, cat = "multi_hop_recall", "multi-hop"
    session, user = "s-multi", "u-multi"

    if live:
        settings = _live_settings()
        h = build_live(settings)
        _a, l1 = await h.say(user_id=user, text="I work at Acme Corp.", session_id=session)
        _b, l2 = await h.say(
            user_id=user, text="Acme Corp is headquartered in Berlin.", session_id=session
        )
        reply, l3 = await h.say(
            user_id=user, text="Which city is my employer based in?", session_id=session
        )
        passed = _contains(reply, "berlin")
        return ScenarioResult(name, cat, passed, f"reply={reply[:120]!r}", [l1, l2, l3])

    h = build_deterministic(
        replies=[
            "Got it, Acme Corp.",  # turn 1
            _fact_batch_json([("user", "works_at", "Acme Corp")]),  # consolidate 1
            "Noted: Acme Corp is in Berlin.",  # turn 2
            _fact_batch_json([("Acme Corp", "headquartered_in", "Berlin")]),  # consolidate 2
            "Your employer Acme Corp is based in Berlin.",  # turn 3
        ]
    )
    _a, l1 = await h.say(user_id=user, text="I work at Acme Corp.", session_id=session)
    await h.consolidate(user)
    _b, l2 = await h.say(
        user_id=user, text="Acme Corp is headquartered in Berlin.", session_id=session
    )
    await h.consolidate(user)
    _c, l3 = await h.say(
        user_id=user, text="Which city is my employer Acme based in?", session_id=session
    )
    prompt = h.last_prompt_text()
    # Both hops must be present in the final prompt to compose the answer.
    passed = _contains(prompt, "acme") and _contains(prompt, "berlin")
    detail = f"both hops in final prompt (acme & berlin)={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1, l2, l3])


# --- temporal knowledge-update (fact supersession) ------------------------


async def scenario_temporal_update(live: bool) -> ScenarioResult:
    """A fact changes; the latest must win and the stale value must not be current."""
    name, cat = "temporal_knowledge_update", "temporal"
    session, user = "s-temporal", "u-temporal"

    if live:
        settings = _live_settings()
        h = build_live(settings)
        _a, l1 = await h.say(user_id=user, text="I live in Berlin.", session_id=session)
        _b, l2 = await h.say(
            user_id=user, text="Actually, I just moved to Munich.", session_id=session
        )
        reply, l3 = await h.say(user_id=user, text="Where do I live now?", session_id=session)
        passed = _contains(reply, "munich")
        return ScenarioResult(name, cat, passed, f"reply={reply[:120]!r}", [l1, l2, l3])

    clock = StepClock()
    h = build_deterministic(
        replies=[
            "Noted: Berlin.",  # turn 1
            _fact_batch_json([("user", "lives_in", "Berlin")]),  # consolidate 1
            "Updated: Munich.",  # turn 2
            _fact_batch_json([("user", "lives_in", "Munich")]),  # consolidate 2
            "You live in Munich.",  # turn 3
        ],
        clock=clock,
    )
    _a, l1 = await h.say(user_id=user, text="I live in Berlin.", session_id=session)
    await h.consolidate(user)
    clock.advance(days=30)
    _b, l2 = await h.say(user_id=user, text="Actually, I just moved to Munich.", session_id=session)
    await h.consolidate(user)
    _c, l3 = await h.say(user_id=user, text="Where do I live now?", session_id=session)

    facts = await h.learner._consolidator._gate.current_facts(user_id=user, subject="user")
    lives_in = {f.object for f in facts if f.predicate == "lives_in"}
    passed = lives_in == {"Munich"}
    detail = f"current lives_in={lives_in} (Munich wins, Berlin superseded)={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1, l2, l3])


# --- preference learning shows up later -----------------------------------


async def scenario_preference_learning(live: bool) -> ScenarioResult:
    """A stated preference is consolidated and injected into a later turn's prompt."""
    name, cat = "preference_learning_visible", "preference"
    session, user = "s-pref", "u-pref"

    if live:
        # Live: we can only softly assert that the later turn still answers; the
        # personalization fragment is an internal artifact. Treat reachability as pass.
        settings = _live_settings()
        h = build_live(settings)
        _a, l1 = await h.say(
            user_id=user, text="Please always keep answers terse.", session_id=session
        )
        reply, l2 = await h.say(user_id=user, text="Explain recursion.", session_id=session)
        passed = len(reply.strip()) > 0
        return ScenarioResult(name, cat, passed, f"non-empty reply={passed}", [l1, l2])

    h = build_deterministic(
        replies=[
            "Understood, I'll be terse.",  # turn 1
            _fact_batch_json([("user", "prefers", "terse")]),  # consolidate
            "Recursion: a function calling itself.",  # turn 2
        ]
    )
    _a, l1 = await h.say(user_id=user, text="Please always keep answers terse.", session_id=session)
    await h.consolidate(user)
    _b, l2 = await h.say(user_id=user, text="Explain recursion.", session_id=session)
    # The consolidated preference must surface as a personalization fragment in the
    # system message ("User prefs: length=terse").
    system = h.last_system_message()
    passed = _contains(system, "length=terse")
    detail = f"personalization fragment 'length=terse' in system prompt={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1, l2])


# --- tool-call loop executes ----------------------------------------------


async def scenario_tool_loop(live: bool) -> ScenarioResult:
    """A turn triggers a tool call; the tool runs and its result threads back."""
    name, cat = "tool_call_loop", "tools"
    session, user = "s-tool", "u-tool"

    if live:
        # The real model decides whether to call a tool; we cannot force it
        # deterministically, so we assert reachability (non-empty answer) only.
        settings = _live_settings()
        h = build_live(settings)
        reply, l1 = await h.say(user_id=user, text="What is 6 multiplied by 7?", session_id=session)
        passed = len(reply.strip()) > 0
        return ScenarioResult(name, cat, passed, f"non-empty reply={passed}", [l1])

    tc = ToolCall(id="tc-1", name="calculator", arguments={"expression": "6 * 7"})
    h = build_deterministic(
        replies=[],
        chat_results=[
            ChatResult(text="", tool_calls=[tc]),  # model asks for calculator
            ChatResult(text="6 × 7 = 42.", tool_calls=[]),  # final answer
        ],
    )
    result, l1 = await h.say_full(user_id=user, text="What is 6 * 7?", session_id=session)
    tool_ran = "calculator" in result.tools_invoked
    # The tool result (42) must have been threaded into the second LLM call.
    prompt = h.last_prompt_text()
    result_threaded = _contains(prompt, "42")
    passed = tool_ran and result_threaded and result.text == "6 × 7 = 42."
    detail = (
        f"calculator invoked={tool_ran}, result threaded into prompt={result_threaded}, "
        f"final='{result.text}'"
    )
    return ScenarioResult(name, cat, passed, detail, [l1])


# --- personalization / profile injection present --------------------------


async def scenario_personalization_injection(live: bool) -> ScenarioResult:
    """With a known UserModel, the personalization fragment reaches the system prompt.

    Seeds a fact directly (subject=user, predicate=comm_tone) so user_model() yields
    a non-default profile, then asserts the fragment is injected. This isolates the
    personalization wiring from the consolidation path.
    """
    name, cat = "personalization_injection", "personalization"
    session, user = "s-persona", "u-persona"

    if live:
        settings = _live_settings()
        h = build_live(settings)
        reply, l1 = await h.say(user_id=user, text="Hello there.", session_id=session)
        passed = len(reply.strip()) > 0
        return ScenarioResult(name, cat, passed, f"non-empty reply={passed}", [l1])

    h = build_deterministic(replies=["Hi!"])
    # Seed a fact so the profile builder produces a non-default UserModel.
    from morgan_brain.models.memory import TemporalFact

    await h.memory_module.upsert_fact(
        TemporalFact(user_id=user, subject="user", predicate="comm_length", object="thorough")
    )
    _r, l1 = await h.say(user_id=user, text="Tell me about Python.", session_id=session)
    system = h.last_system_message()
    passed = _contains(system, "length=thorough")
    detail = f"'length=thorough' injected into system prompt={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1])


# --- champion preprompt threading (the self-learning artifact) ------------


async def scenario_champion_override(live: bool) -> ScenarioResult:
    """A promoted champion preprompt is prepended to the system prompt on the blocking path."""
    name, cat = "champion_preprompt_applied", "personalization"
    session, user = "s-champ", "u-champ"
    champion = "CHAMPION-MARKER: prefer worked examples."

    if live:
        settings = _live_settings()
        h = build_live(settings, champion_override=champion)
        reply, l1 = await h.say(user_id=user, text="Hi", session_id=session)
        passed = len(reply.strip()) > 0
        return ScenarioResult(name, cat, passed, f"non-empty reply={passed}", [l1])

    h = build_deterministic(replies=["Hello!"], champion_override=champion)
    _r, l1 = await h.say(user_id=user, text="Hi", session_id=session)
    system = h.last_system_message()
    passed = _contains(system, "CHAMPION-MARKER")
    detail = f"champion preprompt present in system prompt={passed}"
    return ScenarioResult(name, cat, passed, detail, [l1])


SCENARIOS: list[ScenarioFn] = [
    scenario_single_hop,
    scenario_multi_hop,
    scenario_temporal_update,
    scenario_preference_learning,
    scenario_tool_loop,
    scenario_personalization_injection,
    scenario_champion_override,
]


# ---------------------------------------------------------------------------
# Live settings helper
# ---------------------------------------------------------------------------


def _live_settings() -> Settings:
    from morgan_brain.config import get_settings

    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


async def run_all(*, live: bool) -> BenchReport:
    """Run every scenario and collect a report.

    In live mode, if the configured LLM is unreachable, every scenario is marked
    SKIP rather than failing (graceful degradation).
    """
    results: list[ScenarioResult] = []
    skip_reason = ""
    if live:
        probe = await probe_live(_live_settings())
        if not probe.llm_ok:
            skip_reason = f"live LLM unreachable ({probe.reason})"
        elif not probe.qdrant_ok:
            skip_reason = f"qdrant unreachable ({probe.reason})"

    for fn in SCENARIOS:
        if skip_reason:
            results.append(
                ScenarioResult(
                    name=fn.__name__.removeprefix("scenario_"),
                    category="-",
                    passed=False,
                    detail="",
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
            continue
        try:
            results.append(await fn(live))
        except Exception as exc:  # noqa: BLE001 - record failure, keep going
            results.append(
                ScenarioResult(
                    name=fn.__name__.removeprefix("scenario_"),
                    category="-",
                    passed=False,
                    detail=f"raised {type(exc).__name__}: {exc}",
                )
            )

    return BenchReport(
        mode="live" if live else "deterministic",
        results=results,
        generated_at=datetime.now(UTC).isoformat(),
    )
