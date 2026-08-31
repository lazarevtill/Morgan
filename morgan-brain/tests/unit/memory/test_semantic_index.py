"""The semantic upper index (VoiceMem §3.1) — schema → entity → memory routing.

The property that comes before accuracy is that routing can only ever *narrow* a search
that had somewhere to narrow to. An index that returns a wrong pool silently deletes
memories from recall, which is worse than no index at all, so the no-match and
over-wide cases are pinned first.
"""

from __future__ import annotations

import pytest

from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.semantic_index import PRESET_SCHEMAS, SemanticIndex
from morgan_brain.modules.memory.stores.db import open_db

U = "u1"
P = "plata"


@pytest.fixture
def index():
    conn = open_db(":memory:")
    # EntityIndex owns `memory_entities` -- the leaf level (I_v) this index routes
    # through. Constructing it here is not scaffolding: without it there is no leaf.
    leaves = EntityIndex(conn)
    idx = SemanticIndex(conn)
    idx.ensure_schemas(user_id=U, project=P)
    yield idx, conn, leaves
    conn.close()


def _link(leaves, *, memory_id: str, name: str, user_id: str = U, project: str = P) -> None:
    """Write the leaf level through the index that owns it, not through raw SQL."""
    leaves.add(memory_id, [name], user_id=user_id, project=project)


# ---------------------------------------------------------------------------
# Routing never costs recall
# ---------------------------------------------------------------------------


def test_no_match_returns_none_not_an_empty_pool(index):
    """None means "search everything". An empty list would mean "search nothing", which
    is how a routing layer turns a working recall into silence."""
    idx, _conn, _leaves = index
    assert idx.route(["nothing", "known"], user_id=U, project=P) is None


def test_an_empty_query_returns_none(index):
    idx, _conn, _leaves = index
    assert idx.route([], user_id=U, project=P) is None


def test_a_matched_entity_with_no_memories_returns_none(index):
    """The entity is known but indexes nothing yet -- narrowing to zero ids would erase
    the turn's recall entirely."""
    idx, _conn, _leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    assert idx.route(["harbor"], user_id=U, project=P) is None


def test_an_over_wide_pool_declines_to_narrow(index):
    """A pool that covers most of the store is not routing, it is overhead. Past the cap
    the index steps aside rather than paying for a filter that removes nothing."""
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    for i in range(12):
        _link(leaves, memory_id=f"m{i}", name="harbor")
    assert idx.route(["harbor"], user_id=U, project=P, max_candidates=10) is None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_matched_entity_returns_its_memories(index):
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    _link(leaves, memory_id="m1", name="harbor")
    assert idx.route(["harbor"], user_id=U, project=P) == ["m1"]


def test_an_unrelated_memory_is_excluded_from_the_pool(index):
    """This is the narrowing itself: `m2` exists, is in the same project, and is not in
    the pool because nothing in the query points at it."""
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.assign(user_id=U, project=P, entity="dentist", schema_name="health")
    _link(leaves, memory_id="m1", name="harbor")
    _link(leaves, memory_id="m2", name="dentist")
    assert idx.route(["harbor"], user_id=U, project=P) == ["m1"]


def test_a_matched_schema_pulls_in_every_entity_it_holds(index):
    """V_St in eq. (1): naming the slot reaches the concepts inside it."""
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.assign(user_id=U, project=P, entity="gitlab", schema_name="work")
    _link(leaves, memory_id="m1", name="harbor")
    _link(leaves, memory_id="m2", name="gitlab")
    assert set(idx.route(["work"], user_id=U, project=P)) == {"m1", "m2"}


def test_one_hop_micro_expansion_reaches_a_co_occurring_entity(index):
    """N_1(V_t): harbor and gitlab were seen together, so asking about one reaches the
    other's memories -- but only one hop."""
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.assign(user_id=U, project=P, entity="gitlab", schema_name="work")
    idx.assign(user_id=U, project=P, entity="dentist", schema_name="health")
    idx.observe_cooccurrence(user_id=U, project=P, names=["harbor", "gitlab"])
    idx.observe_cooccurrence(user_id=U, project=P, names=["gitlab", "dentist"])
    _link(leaves, memory_id="m1", name="harbor")
    _link(leaves, memory_id="m2", name="gitlab")
    _link(leaves, memory_id="m3", name="dentist")

    pool = set(idx.route(["harbor"], user_id=U, project=P))
    assert pool == {"m1", "m2"}, "expansion must stop at one hop, not walk to dentist"


def test_routing_is_project_scoped(index):
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    # Schemas are scoped too, so the other project needs its own slots before an entity
    # can be filed there -- assigning into a scope that has none is refused.
    idx.ensure_schemas(user_id=U, project="other")
    idx.assign(user_id=U, project="other", entity="harbor", schema_name="work")
    _link(leaves, memory_id="m1", name="harbor")
    _link(leaves, memory_id="m9", name="harbor", project="other")
    assert idx.route(["harbor"], user_id=U, project=P) == ["m1"]


def test_routing_is_user_scoped(index):
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.ensure_schemas(user_id="u2", project=P)
    idx.assign(user_id="u2", project=P, entity="harbor", schema_name="work")
    _link(leaves, memory_id="m1", name="harbor")
    _link(leaves, memory_id="m9", name="harbor", user_id="u2")
    assert idx.route(["harbor"], user_id=U, project=P) == ["m1"]


def test_matching_is_case_insensitive_and_script_agnostic(index):
    idx, _conn, leaves = index
    idx.assign(user_id=U, project=P, entity="харбор", schema_name="work")
    _link(leaves, memory_id="m1", name="харбор")
    assert idx.route(["Харбор"], user_id=U, project=P) == ["m1"]


def test_pool_order_is_deterministic(index):
    idx, _conn, leaves = index
    for name in ("harbor", "gitlab", "qdrant"):
        idx.assign(user_id=U, project=P, entity=name, schema_name="work")
        _link(leaves, memory_id=f"m-{name}", name=name)
    first = idx.route(["work"], user_id=U, project=P)
    assert first == idx.route(["work"], user_id=U, project=P)
    assert first == sorted(first)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_presets_are_created_once_and_are_idempotent(index):
    idx, conn, _leaves = index
    idx.ensure_schemas(user_id=U, project=P)
    rows = conn.execute(
        "SELECT name FROM mem_schemas WHERE user_id = ? AND project = ?", (U, P)
    ).fetchall()
    assert sorted(r["name"] for r in rows) == sorted(PRESET_SCHEMAS)


def test_an_entity_belongs_to_exactly_one_schema(index):
    """The paper's simplification, and what makes routing non-recursive: reassigning
    moves the entity rather than adding a second membership."""
    idx, conn, _leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="knowledge")
    rows = conn.execute(
        "SELECT schema_name FROM mem_entity_nodes WHERE user_id = ? AND project = ? AND name = ?",
        (U, P, "harbor"),
    ).fetchall()
    assert [r["schema_name"] for r in rows] == ["knowledge"]


def test_cooccurrence_weight_accumulates(index):
    idx, conn, _leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.assign(user_id=U, project=P, entity="gitlab", schema_name="work")
    idx.observe_cooccurrence(user_id=U, project=P, names=["harbor", "gitlab"])
    idx.observe_cooccurrence(user_id=U, project=P, names=["gitlab", "harbor"])
    row = conn.execute(
        "SELECT weight FROM mem_entity_edges WHERE user_id = ? AND project = ? "
        "AND src = ? AND dst = ?",
        (U, P, "gitlab", "harbor"),
    ).fetchone()
    assert row["weight"] == 2, "the pair is undirected: both orderings are the same edge"


def test_cooccurrence_ignores_unknown_entities(index):
    """An edge to an entity with no node would route into a schema-less void."""
    idx, conn, _leaves = index
    idx.assign(user_id=U, project=P, entity="harbor", schema_name="work")
    idx.observe_cooccurrence(user_id=U, project=P, names=["harbor", "never-assigned"])
    rows = conn.execute("SELECT * FROM mem_entity_edges").fetchall()
    assert rows == []


def test_assigning_to_an_unknown_schema_is_refused(index):
    idx, _conn, _leaves = index
    with pytest.raises(ValueError, match="unknown schema"):
        idx.assign(user_id=U, project=P, entity="harbor", schema_name="not-a-slot")
