"""Filing entities into the semantic upper index: an entity is only ever filed under a
schema that exists, is classified once, and is linked only to what shared a memory with it.
"""

from __future__ import annotations

import pytest

from morgan_brain.memory.db import open_db
from morgan_brain.memory.entities import EntityIndex
from morgan_brain.memory.schema_classifier import (
    KeywordSchemaClassifier,
    SemanticIndexBuilder,
)
from morgan_brain.memory.semantic_index import PRESET_SCHEMAS, SemanticIndex
from morgan_brain.models import Entity, Memory

U = "u1"
P = "acme"


@pytest.fixture
def index():
    conn = open_db(":memory:")
    EntityIndex(conn)
    idx = SemanticIndex(conn)
    yield idx, conn
    conn.close()


def _memory(content: str, names: list[str]) -> Memory:
    return Memory(user_id=U, project=P, content=content, entities=[Entity(name=n) for n in names])


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
