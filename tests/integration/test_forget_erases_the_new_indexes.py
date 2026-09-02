"""`forget()` must reach the semantic upper index.

It is derived from the memories being erased -- an entity node and a co-occurrence edge
encode what the owner said. Leaving them behind means `morgan forget` reports success while
the derived shape of that project's content is still on disk.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.models import Entity, Memory

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def stack(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=8), dim=8, clock=lambda: T0)
    yield module, module._semantic, conn
    conn.close()


async def _populate(module, project: str) -> None:
    await module.store(
        Memory(
            user_id=U,
            project=project,
            content="Harbor blocked the deploy",
            entities=[Entity(name="harbor"), Entity(name="gitlab")],
        )
    )


async def test_forget_erases_the_semantic_index(stack):
    module, semantic, conn = stack
    await _populate(module, P)
    assert semantic.schema_of(user_id=U, project=P, entity="harbor") == "work"

    await module.forget(user_id=U, project=P)

    assert semantic.schema_of(user_id=U, project=P, entity="harbor") is None
    for table in ("mem_entity_nodes", "mem_entity_edges", "mem_schema_edges", "mem_schemas"):
        sql = {
            "mem_entity_nodes": "SELECT COUNT(*) AS n FROM mem_entity_nodes WHERE project = ?",
            "mem_entity_edges": "SELECT COUNT(*) AS n FROM mem_entity_edges WHERE project = ?",
            "mem_schema_edges": "SELECT COUNT(*) AS n FROM mem_schema_edges WHERE project = ?",
            "mem_schemas": "SELECT COUNT(*) AS n FROM mem_schemas WHERE project = ?",
        }[table]
        assert conn.execute(sql, (P,)).fetchone()["n"] == 0, table


async def test_forget_reports_what_it_erased_from_the_index(stack):
    module, _semantic, _conn = stack
    await _populate(module, P)
    report = await module.forget(user_id=U, project=P)
    assert report.index_entries > 0


async def test_forget_leaves_another_project_alone(stack):
    module, semantic, _conn = stack
    await _populate(module, P)
    await _populate(module, "other")
    await module.forget(user_id=U, project=P)
    assert semantic.schema_of(user_id=U, project="other", entity="harbor") == "work"


async def test_forgetting_a_project_with_no_index_rows_is_not_an_error(stack):
    module, _semantic, _conn = stack
    report = await module.forget(user_id=U, project=P)
    assert report.index_entries == 0
