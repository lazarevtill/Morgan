"""Persona nodes reaching the system fragment — the hot-path half of the right brain.

Read-only by construction: `build()` may activate nodes, never record them. The write
side is the cold path's job, and a personalizer that wrote would put a persona update
inside the request the invariant exists to keep clean.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.models.base import Entity
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import UserModel
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.personalization.adaptive import AdaptivePersonalizer
from morgan_brain.modules.personalization.persona_graph import PersonaGraph

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def graph():
    conn = open_db(":memory:")
    yield PersonaGraph(conn), SemanticIndex(conn), conn
    conn.close()


def _perception(text: str, names: list[str]) -> FusedPerception:
    return FusedPerception(text=text, entities=[Entity(name=n) for n in names])


async def test_a_cross_entity_node_reaches_the_fragment_when_its_anchor_is_named(graph):
    g, _semantic, _conn = graph
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    p = AdaptivePersonalizer(persona_graph=g)

    ctx = await p.build(
        user_model=UserModel(user_id=U),
        perception=_perception("what about the harbor sync", ["harbor sync"]),
        project=P,
    )
    assert "finds these reviews tedious" in ctx.system_fragment


async def test_the_anchor_is_named_in_the_fragment_not_just_the_attitude(graph):
    """Injecting the attitude without whom it concerns is the collapse the graph exists
    to prevent -- it would read as a statement about the person."""
    g, _semantic, _conn = graph
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    p = AdaptivePersonalizer(persona_graph=g)

    ctx = await p.build(
        user_model=UserModel(user_id=U),
        perception=_perception("what about the harbor sync", ["harbor sync"]),
        project=P,
    )
    assert "harbor sync" in ctx.system_fragment


async def test_an_unrelated_turn_gets_no_persona_text(graph):
    g, _semantic, _conn = graph
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    p = AdaptivePersonalizer(persona_graph=g)

    ctx = await p.build(
        user_model=UserModel(user_id=U),
        perception=_perception("what is the weather", []),
        project=P,
    )
    assert "tedious" not in ctx.system_fragment


async def test_joint_retrieval_reaches_an_anchor_the_turn_did_not_name(graph):
    """Eq. (5): the right brain expands over the entity set the *left* brain activated,
    not only the literal entities in the text. Naming `gitlab` reaches the attitude
    toward `harbor sync` because the two co-occur in the semantic index."""
    g, semantic, _conn = graph
    semantic.ensure_schemas(user_id=U, project=P)
    semantic.assign(user_id=U, project=P, entity="gitlab", schema_name="work")
    semantic.assign(user_id=U, project=P, entity="harbor sync", schema_name="work")
    semantic.observe_cooccurrence(user_id=U, project=P, names=["gitlab", "harbor sync"])
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    p = AdaptivePersonalizer(persona_graph=g, semantic_index=semantic)

    ctx = await p.build(
        user_model=UserModel(user_id=U),
        perception=_perception("what about gitlab", ["gitlab"]),
        project=P,
    )
    assert "finds these reviews tedious" in ctx.system_fragment


async def test_without_the_semantic_index_only_named_anchors_activate(graph):
    g, semantic, _conn = graph
    semantic.ensure_schemas(user_id=U, project=P)
    semantic.assign(user_id=U, project=P, entity="gitlab", schema_name="work")
    semantic.assign(user_id=U, project=P, entity="harbor sync", schema_name="work")
    semantic.observe_cooccurrence(user_id=U, project=P, names=["gitlab", "harbor sync"])
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    p = AdaptivePersonalizer(persona_graph=g)

    ctx = await p.build(
        user_model=UserModel(user_id=U),
        perception=_perception("what about gitlab", ["gitlab"]),
        project=P,
    )
    assert "tedious" not in ctx.system_fragment


async def test_building_a_turn_never_writes_to_the_graph(graph):
    g, _semantic, conn = graph
    g.observe(
        user_id=U,
        project=P,
        description="finds these reviews tedious",
        entity="harbor sync",
        valence=-0.4,
        session_id="s1",
        now=T0,
    )
    before = conn.execute("SELECT observations FROM persona_nodes").fetchone()["observations"]
    p = AdaptivePersonalizer(persona_graph=g)

    for _ in range(3):
        await p.build(
            user_model=UserModel(user_id=U),
            perception=_perception("the harbor sync again", ["harbor sync"]),
            project=P,
        )

    after = conn.execute("SELECT observations FROM persona_nodes").fetchone()["observations"]
    assert after == before


async def test_the_personalizer_still_works_with_no_graph_wired():
    p = AdaptivePersonalizer()
    ctx = await p.build(
        user_model=UserModel(user_id=U), perception=_perception("hello", []), project=P
    )
    assert isinstance(ctx.system_fragment, str)
