"""The cold-path job that builds the semantic upper index.

Two properties the index depends on and cannot check for itself: an entity is only ever
filed under a schema that exists, and a model that answers badly must not be able to
corrupt the index -- an unusable answer falls back rather than writing nonsense.
"""

from __future__ import annotations

import pytest

from morgan_brain.learning.semantic_index_builder import (
    FALLBACK_SCHEMA,
    KeywordSchemaClassifier,
    LLMSchemaClassifier,
    SemanticIndexBuilder,
)
from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.semantic_index import PRESET_SCHEMAS, SemanticIndex
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

U = "u1"
P = "plata"


@pytest.fixture
def index():
    conn = open_db(":memory:")
    EntityIndex(conn)
    idx = SemanticIndex(conn)
    yield idx, conn
    conn.close()


def _memory(content: str, names: list[str]) -> Memory:
    return Memory(user_id=U, project=P, content=content, entities=[Entity(name=n) for n in names])


def _router(replies: list[str]) -> tuple[RoleRouter, CapabilityRegistry]:
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    return (
        RoleRouter(
            reg=reg,
            bindings={
                "reflection": [Binding("fake", "test-model", FakeChatClient(replies=replies))]
            },
        ),
        reg,
    )


# ---------------------------------------------------------------------------
# Keyword classifier — the floor that works with no model at all
# ---------------------------------------------------------------------------


async def test_keyword_classifier_assigns_every_entity(index):
    idx, conn = index
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())

    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])

    row = conn.execute("SELECT schema_name FROM mem_entity_nodes WHERE name = 'harbor'").fetchone()
    assert row["schema_name"] in PRESET_SCHEMAS


async def test_presets_are_created_before_anything_is_assigned(index):
    idx, _conn = index
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())
    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])
    assert sorted(idx.schemas(user_id=U, project=P)) == sorted(PRESET_SCHEMAS)


async def test_cooccurrence_is_recorded_for_entities_in_one_memory(index):
    idx, conn = index
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())

    await builder.index(
        user_id=U, project=P, memories=[_memory("Harbor and GitLab", ["Harbor", "GitLab"])]
    )

    rows = conn.execute("SELECT src, dst, weight FROM mem_entity_edges").fetchall()
    assert [(r["src"], r["dst"], r["weight"]) for r in rows] == [("gitlab", "harbor", 1)]


async def test_entities_from_different_memories_are_not_linked(index):
    """Co-occurrence means "in the same memory". Linking across memories would connect
    everything to everything within a session and make one-hop expansion meaningless."""
    idx, conn = index
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())

    await builder.index(
        user_id=U,
        project=P,
        memories=[_memory("Harbor broke", ["Harbor"]), _memory("GitLab is slow", ["GitLab"])],
    )

    assert conn.execute("SELECT COUNT(*) AS n FROM mem_entity_edges").fetchone()["n"] == 0


async def test_a_memory_with_no_entities_is_a_no_op(index):
    idx, conn = index
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())
    await builder.index(user_id=U, project=P, memories=[_memory("nothing named here", [])])
    assert conn.execute("SELECT COUNT(*) AS n FROM mem_entity_nodes").fetchone()["n"] == 0


async def test_an_already_assigned_entity_is_not_reclassified(index):
    """Reclassifying on every pass would let a schema flap between nights, and every flap
    silently rewrites which memories a query can route to."""
    idx, conn = index
    idx.ensure_schemas(user_id=U, project=P)
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="knowledge")
    builder = SemanticIndexBuilder(semantic=idx, classifier=KeywordSchemaClassifier())

    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])

    row = conn.execute("SELECT schema_name FROM mem_entity_nodes WHERE name = 'harbor'").fetchone()
    assert row["schema_name"] == "knowledge"


# ---------------------------------------------------------------------------
# LLM classifier — and what happens when it answers badly
# ---------------------------------------------------------------------------


async def test_llm_classifier_assignment_is_used(index):
    idx, conn = index
    router, reg = _router(['{"assignments": [{"entity": "harbor", "schema_name": "work"}]}'])
    builder = SemanticIndexBuilder(
        semantic=idx,
        classifier=LLMSchemaClassifier(router=router, capability_registry=reg),
    )

    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])

    row = conn.execute("SELECT schema_name FROM mem_entity_nodes WHERE name = 'harbor'").fetchone()
    assert row["schema_name"] == "work"


async def test_an_invented_schema_is_not_written(index):
    """The classifier is a model, so it will eventually invent a slot. The index refuses
    unknown schemas; the builder must land the entity somewhere rather than drop it."""
    idx, conn = index
    router, reg = _router(
        ['{"assignments": [{"entity": "harbor", "schema_name": "infrastructure"}]}']
    )
    builder = SemanticIndexBuilder(
        semantic=idx,
        classifier=LLMSchemaClassifier(router=router, capability_registry=reg),
    )

    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])

    row = conn.execute("SELECT schema_name FROM mem_entity_nodes WHERE name = 'harbor'").fetchone()
    assert row["schema_name"] == FALLBACK_SCHEMA


async def test_an_unreachable_model_falls_back_instead_of_failing_the_job(index):
    """The builder runs in the nightly worker. A model outage must cost index quality,
    not the whole job -- the other work in that run still has to land."""
    idx, conn = index
    reg = CapabilityRegistry.from_seed({})
    router = RoleRouter(reg=reg, bindings={})
    builder = SemanticIndexBuilder(
        semantic=idx,
        classifier=LLMSchemaClassifier(router=router, capability_registry=reg),
    )

    await builder.index(user_id=U, project=P, memories=[_memory("Harbor broke", ["Harbor"])])

    row = conn.execute("SELECT schema_name FROM mem_entity_nodes WHERE name = 'harbor'").fetchone()
    assert row["schema_name"] == FALLBACK_SCHEMA


async def test_entities_the_classifier_omitted_still_get_filed(index):
    idx, conn = index
    router, reg = _router(['{"assignments": [{"entity": "harbor", "schema_name": "work"}]}'])
    builder = SemanticIndexBuilder(
        semantic=idx,
        classifier=LLMSchemaClassifier(router=router, capability_registry=reg),
    )

    await builder.index(
        user_id=U, project=P, memories=[_memory("Harbor and GitLab", ["Harbor", "GitLab"])]
    )

    rows = conn.execute("SELECT name, schema_name FROM mem_entity_nodes ORDER BY name").fetchall()
    assert {r["name"]: r["schema_name"] for r in rows} == {
        "gitlab": FALLBACK_SCHEMA,
        "harbor": "work",
    }
