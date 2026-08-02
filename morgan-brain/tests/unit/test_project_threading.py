"""Task 13B — thread `project` through every caller above MemoryModule.

Before this test existed, every production ``MemoryQuery(...)`` construction above the gate
omitted ``project`` and silently relied on the ``DEFAULT_PROJECT`` field default, and
consolidation hardcoded ``DEFAULT_PROJECT`` at five call sites. That is internally consistent
only as long as nothing ever writes under a real project name -- the moment it does (the CLI,
Task 17), anything it writes becomes invisible to consolidation and to the API.

These tests pin:
1. The three orchestrator turn methods accept a ``project`` parameter.
2. Consolidation for one project does not leak into another project's fact base.
3. No production ``MemoryQuery(`` construction is left unscoped (regression guard).
"""

from __future__ import annotations

import inspect
import pathlib
from datetime import UTC, datetime

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
            project="acme",
            kind=MemoryKind.EPISODIC,
            content="harbor mirror note",
            source=MemorySource.USER_STATED,
        )
    )

    await consolidator.consolidate("u", project="acme")

    assert await gate.current_facts(user_id="u", project="acme") != []
    # And it must NOT have landed in the default project instead.
    assert await gate.current_facts(user_id="u", project="default") == []


# ---------------------------------------------------------------------------
# 3. Guard: no production site builds an unscoped MemoryQuery.
# ---------------------------------------------------------------------------


def test_no_production_site_builds_an_unscoped_memory_query() -> None:
    """Guard against reintroducing the bug: every `MemoryQuery(` line must mention `project`."""
    root = pathlib.Path(__file__).resolve().parents[2] / "morgan_brain"
    offenders = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if "MemoryQuery(" in line and "project" not in line and "class " not in line:
                offenders.append(f"{py.relative_to(root)}:{i}")
    assert offenders == [], offenders
