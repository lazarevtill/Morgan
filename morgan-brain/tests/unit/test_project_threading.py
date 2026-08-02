"""Task 13B — thread `project` through every caller above MemoryModule.

Before this test existed, every production ``MemoryQuery(...)`` construction above the gate
omitted ``project`` and silently relied on the ``DEFAULT_PROJECT`` field default, and
consolidation hardcoded ``DEFAULT_PROJECT`` at five call sites. That is internally consistent
only as long as nothing ever writes under a real project name -- the moment it does (the CLI,
Task 17), anything it writes becomes invisible to consolidation and to the API.

Fix round 1 found the threading was incomplete in a way neither the implementer nor the first
review caught: ``RESPONSE_GENERATED``'s payload carried no ``project``, so both consumers that
turn it into episodic storage (``composition.py::_store_turn`` and
``apps/learning_worker/__main__.py::_make_response_handler``) reconstructed a ``Conversation``
with no project, and every turn's content collapsed into ``DEFAULT_PROJECT`` regardless of what
project the turn was served under. Not a leak -- a collapse. These tests pin the full,
now-corrected path end to end.

These tests pin:
1. The three orchestrator turn methods accept a ``project`` parameter.
2. Consolidation for one project does not leak into another project's fact base.
3. A full ``handle_turn(project=...)`` call actually lands its episodic content in that
   project (the write-path collapse regression from fix round 1).
4. No production ``MemoryQuery(``, ``Memory(``, or ``TemporalFact(`` construction is left
   unscoped (regression guard, widened in fix round 1 to catch the write-path collapse class
   of bug -- the original guard only scanned ``MemoryQuery(``).
"""

from __future__ import annotations

import inspect
import pathlib
import re
from datetime import UTC, datetime

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.core.orchestrator import Orchestrator
from morgan_brain.learning.consolidation import FactOp, FactOpBatch, FactOpKind, MemoryConsolidator
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. Orchestrator turn methods must accept a `project` parameter.
# ---------------------------------------------------------------------------


def test_orchestrator_turn_methods_require_a_project() -> None:
    for name in ("handle_turn", "handle_turn_with_id", "stream_turn"):
        params = inspect.signature(getattr(Orchestrator, name)).parameters
        assert "project" in params, f"{name} does not accept a project"


# ---------------------------------------------------------------------------
# 2. Consolidation is project-scoped end to end (regression: hardcoded DEFAULT_PROJECT
#    would silently exclude everything the CLI stores under a real project name).
# ---------------------------------------------------------------------------


def _build_stack() -> tuple[MemoryConsolidator, MemoryGate]:
    embedder = FakeEmbedder(dim=16)
    conn = open_db(":memory:")
    module = MemoryModule(
        embedder=embedder,
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: T0,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
    gate = MemoryGate(module)

    reply = FactOpBatch(
        ops=[
            FactOp(
                op=FactOpKind.ADD,
                subject="harbor",
                predicate="is",
                object="mirror",
                reason="stated by user",
            )
        ]
    ).model_dump_json()
    fake_client = FakeChatClient(replies=[reply])
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    consolidator = MemoryConsolidator(
        gate=gate,
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
        role="strong",
    )
    return consolidator, gate


async def test_consolidation_does_not_hardcode_the_default_project() -> None:
    """Regression: consolidating only 'default' would exclude everything the CLI stores."""
    consolidator, gate = _build_stack()

    await gate.store(
        Memory(
            user_id="u",
            project="plata",
            kind=MemoryKind.EPISODIC,
            content="harbor mirror note",
            source=MemorySource.USER_STATED,
        )
    )

    await consolidator.consolidate("u", project="plata")

    assert await gate.current_facts(user_id="u", project="plata") != []
    # And it must NOT have landed in the default project instead.
    assert await gate.current_facts(user_id="u", project="default") == []


# ---------------------------------------------------------------------------
# 3. End-to-end: a full handle_turn(project=...) call must land its episodic content in
#    that project -- the write-path collapse regression from fix round 1.
# ---------------------------------------------------------------------------


async def test_handle_turn_writes_land_in_the_requested_project() -> None:
    """handle_turn(project="acme") must make gate.distinct_projects(user) include "acme".

    Before fix round 1, RESPONSE_GENERATED carried no project, so the in-process turn-storage
    subscriber (_store_turn -> learner.process_session) always wrote episodics to
    DEFAULT_PROJECT regardless of what project the turn was served under -- a silent collapse,
    not a leak (nothing showed up in the wrong project's *recall*, it just never showed up
    anywhere except "default").
    """
    fake_client = FakeChatClient(reply="Got it.")
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _, _, _, _, _, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=lambda: T0,
        temporal_path=":memory:",
        bus=bus,
    )

    await bus.start()
    await orch.handle_turn(user_id="u-acme", project="acme", text="Ship the Q3 report.")
    # publish() now enqueues rather than running the storage subscriber inline (Task 15) —
    # drain the bus before asserting on what it stored.
    await bus.drain()
    await bus.stop()

    projects = await learner._gate.distinct_projects("u-acme")  # type: ignore[attr-defined]
    assert "acme" in projects, f"expected 'acme' in distinct_projects, got {projects}"
    assert projects == ["acme"], (
        f"turn content must NOT collapse into 'default' alongside 'acme', got {projects}"
    )


# ---------------------------------------------------------------------------
# 4. Guard: no production site builds an unscoped MemoryQuery/Memory/TemporalFact.
# ---------------------------------------------------------------------------


def _paren_balanced_snippet(lines: list[str], start: int) -> str:
    """Return the source text of the call starting at 0-based line *start*, up to its
    balanced closing paren (naive but sufficient -- no parens inside string literals in
    this codebase's constructor calls)."""
    depth = 0
    started = False
    snippet: list[str] = []
    i = start
    while i < len(lines):
        snippet.append(lines[i])
        depth += lines[i].count("(") - lines[i].count(")")
        if "(" in lines[i]:
            started = True
        if started and depth <= 0:
            break
        i += 1
    return "\n".join(snippet)


def test_no_production_site_builds_an_unscoped_memory_construction() -> None:
    """Guard against reintroducing the write-path collapse: every `MemoryQuery(`, `Memory(`,
    and `TemporalFact(` CONSTRUCTION (not class definition) must set `project` somewhere in
    its (possibly multi-line) call -- a same-line-only check missed the round-1 bug because
    every real constructor call in this codebase is multi-line.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "morgan_brain"
    patterns = (
        re.compile(r"(?<![A-Za-z0-9_])MemoryQuery\("),
        re.compile(r"(?<![A-Za-z0-9_])Memory\("),
        re.compile(r"(?<![A-Za-z0-9_])TemporalFact\("),
    )
    offenders = []
    for py in root.rglob("*.py"):
        lines = py.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "class " in line:
                continue
            if not any(p.search(line) for p in patterns):
                continue
            snippet = _paren_balanced_snippet(lines, i)
            if "project" not in snippet:
                offenders.append(f"{py.relative_to(root)}:{i + 1}")
    assert offenders == [], offenders
