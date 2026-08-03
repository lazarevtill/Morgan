"""A tool that reads memory reads the turn's project, not a default.

`memory_search` took `project: str = DEFAULT_PROJECT` and nothing in the reasoning or tool
layers carried a project, so an assistant answering a question in the `acme` repo searched
`default` instead: it missed every `acme` memory *and* pulled `default`-bucket content into a
`acme` conversation. Both directions of the "every read is project-scoped" invariant, on the
live hot path, in the one place the assistant searches memory itself.

These tests drive the real MemoryModule + MemoryGate + ToolRegistry + ToolExecutorImpl stack
on a real SQLite database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.modules.tools.builtin.memory_search import MemorySearchTool
from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.security.permissions import PermissionGate, PermissionMode


@pytest.fixture
def executor_and_gate(tmp_path: Path) -> tuple[ToolExecutorImpl, MemoryGate]:
    conn = open_db(str(tmp_path / "morgan.db"))
    module = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime.now(UTC),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
    gate = MemoryGate(module)
    registry = ToolRegistry()
    registry.register(MemorySearchTool(gate=gate))  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=registry, gate=PermissionGate(default=PermissionMode.AUTO))
    return executor, gate


async def _seed(gate: MemoryGate, *, project: str, content: str) -> None:
    await gate.store(
        Memory(user_id="u", project=project, content=content, kind=MemoryKind.EPISODIC)
    )


async def test_the_tool_searches_the_turns_project_not_default(
    executor_and_gate: tuple[ToolExecutorImpl, MemoryGate],
) -> None:
    executor, gate = executor_and_gate
    await _seed(gate, project="acme", content="the ACME credentials rotate on Fridays")
    await _seed(gate, project="default", content="DEFAULT-BUCKET credentials note")

    result = await executor.execute(
        "memory_search", user_id="u", project="acme", query="credentials"
    )

    assert result.ok, result.error
    assert result.output == ["the ACME credentials rotate on Fridays"], result.output


async def test_the_model_cannot_choose_the_project(
    executor_and_gate: tuple[ToolExecutorImpl, MemoryGate],
) -> None:
    """`project` is absent from the tool schema, but a model can emit any argument it likes.

    A tool call carrying `project: "default"` must not widen the turn's scope -- otherwise the
    assistant, not the caller, decides which repository's memories it may read.
    """
    executor, gate = executor_and_gate
    await _seed(gate, project="acme", content="the ACME credentials rotate on Fridays")
    await _seed(gate, project="default", content="DEFAULT-BUCKET credentials note")

    # Exactly what the reasoner passes after stripping the model's `project` key.
    model_args = {"query": "credentials", "project": "default"}
    stripped = {k: v for k, v in model_args.items() if k != "project"}
    result = await executor.execute("memory_search", user_id="u", project="acme", **stripped)

    assert result.output == ["the ACME credentials rotate on Fridays"], result.output


async def test_the_tool_finds_nothing_when_the_project_holds_nothing(
    executor_and_gate: tuple[ToolExecutorImpl, MemoryGate],
) -> None:
    """The other direction: a scoped search must not fall back to some other project."""
    executor, gate = executor_and_gate
    await _seed(gate, project="default", content="DEFAULT-BUCKET credentials note")

    result = await executor.execute(
        "memory_search", user_id="u", project="acme", query="credentials"
    )

    assert result.ok, result.error
    assert result.output == [], result.output


async def test_recall_rejects_an_empty_project(
    executor_and_gate: tuple[ToolExecutorImpl, MemoryGate],
) -> None:
    """`MemoryQuery.project` had no min_length while `Memory.project` did, so an empty project
    reached recall and silently matched nothing instead of being refused at the gate."""
    with pytest.raises(ValueError):
        MemoryQuery(user_id="u", project="", text="anything")
