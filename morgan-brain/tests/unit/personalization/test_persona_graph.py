"""The persona graph — VoiceMem's right brain (§3.2).

The distinction the whole structure exists for: "he is impatient" and "he is impatient
*with the weekly Harbor sync*" are different claims, and a flat trait list records the
first when only the second is true. So the tests that matter most are the ones that
refuse to promote: a disposition seen toward one thing, in one session, is situational,
and turning it into a stable trait is the over-personalization failure Morgan's own
golden eval already probes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.personalization.persona_graph import PersonaGraph, PersonaKind

U = "u1"
P = "plata"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def graph():
    conn = open_db(":memory:")
    yield PersonaGraph(conn), conn
    conn.close()


def _observe(g, description, entity, *, session, day=0, valence=-0.5):
    g.observe(
        user_id=U,
        project=P,
        description=description,
        entity=entity,
        valence=valence,
        session_id=session,
        now=T0 + timedelta(days=day),
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_an_observation_with_an_entity_is_a_cross_entity_node(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    nodes = g.all_nodes(user_id=U, project=P)
    assert [n.kind for n in nodes] == [PersonaKind.CROSS_ENTITY]
    assert nodes[0].entity == "harbor sync"


def test_an_observation_without_an_entity_is_intrinsic(graph):
    g, _conn = graph
    _observe(g, "prefers terse answers", None, session="s1")
    assert [n.kind for n in g.all_nodes(user_id=U, project=P)] == [PersonaKind.INTRINSIC]


def test_repeating_an_observation_strengthens_one_node(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    _observe(g, "impatient", "harbor sync", session="s2", day=1)
    nodes = g.all_nodes(user_id=U, project=P)
    assert len(nodes) == 1
    assert nodes[0].observations == 2
    assert nodes[0].sessions == 2


def test_repeating_within_one_session_does_not_count_as_a_second_session(graph):
    """Recurrence means "came back to it", not "said it twice in one sitting". Counting
    the second one would let a single bad afternoon look like a pattern."""
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    _observe(g, "impatient", "harbor sync", session="s1")
    node = g.all_nodes(user_id=U, project=P)[0]
    assert node.observations == 2
    assert node.sessions == 1


def test_confidence_grows_with_evidence_but_never_reaches_certainty(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    first = g.all_nodes(user_id=U, project=P)[0].confidence
    for i in range(2, 12):
        _observe(g, "impatient", "harbor sync", session=f"s{i}", day=i)
    last = g.all_nodes(user_id=U, project=P)[0].confidence
    assert first < last < 1.0


def test_nodes_are_user_and_project_scoped(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    g.observe(
        user_id="u2",
        project=P,
        description="impatient",
        entity="harbor sync",
        valence=-0.5,
        session_id="s1",
        now=T0,
    )
    assert len(g.all_nodes(user_id=U, project=P)) == 1


# ---------------------------------------------------------------------------
# Activation (hot path, read-only)
# ---------------------------------------------------------------------------


def test_a_cross_entity_node_surfaces_only_when_its_anchor_is_active(graph):
    """The anchor is what gives the attitude its meaning. Surfacing it without the
    anchor is exactly the collapse the paper warns about."""
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")

    assert g.activate(user_id=U, project=P, terms=[], entities={"harbor sync"})
    assert g.activate(user_id=U, project=P, terms=[], entities={"dentist"}) == []


def test_an_intrinsic_node_surfaces_on_a_term_match(graph):
    g, _conn = graph
    _observe(g, "prefers terse answers", None, session="s1")
    assert g.activate(user_id=U, project=P, terms=["terse"], entities=set())


def test_activation_is_case_insensitive_on_the_anchor(graph):
    g, _conn = graph
    _observe(g, "impatient", "Harbor Sync", session="s1")
    assert g.activate(user_id=U, project=P, terms=[], entities={"harbor sync"})


def test_activation_returns_the_strongest_first(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1")
    for i in range(2, 6):
        _observe(g, "resigned", "harbor sync", session=f"s{i}", day=i)
    active = g.activate(user_id=U, project=P, terms=[], entities={"harbor sync"})
    assert [n.description for n in active] == ["resigned", "impatient"]


# ---------------------------------------------------------------------------
# Long-horizon consolidation — the part that must refuse
# ---------------------------------------------------------------------------


def test_one_session_never_becomes_a_trait(graph):
    g, _conn = graph
    for _ in range(20):
        _observe(g, "impatient", "harbor sync", session="s1")
    assert g.consolidate(user_id=U, project=P, now=T0) == []


def test_one_anchor_never_becomes_a_trait_however_often_it_recurs(graph):
    """This is the load-bearing refusal. Impatience toward one recurring meeting is a
    fact about that meeting. Promoting it says something about the person that the
    evidence does not support."""
    g, _conn = graph
    for i in range(1, 15):
        _observe(g, "impatient", "harbor sync", session=f"s{i}", day=i)
    assert g.consolidate(user_id=U, project=P, now=T0) == []


def test_the_same_disposition_across_several_anchors_and_sessions_promotes(graph):
    g, _conn = graph
    _observe(g, "impatient", "harbor sync", session="s1", day=1)
    _observe(g, "impatient", "the release checklist", session="s2", day=2)
    _observe(g, "impatient", "onboarding docs", session="s3", day=3)

    promoted = g.consolidate(user_id=U, project=P, now=T0)
    assert [n.description for n in promoted] == ["impatient"]
    intrinsic = [n for n in g.all_nodes(user_id=U, project=P) if n.kind is PersonaKind.INTRINSIC]
    assert len(intrinsic) == 1


def test_promotion_keeps_the_situational_nodes(graph):
    """The cross-entity nodes are the evidence. Deleting them on promotion would throw
    away whom the disposition concerns, which is the half a flat trait list loses."""
    g, _conn = graph
    for i, anchor in enumerate(("harbor sync", "the release checklist", "onboarding docs"), 1):
        _observe(g, "impatient", anchor, session=f"s{i}", day=i)
    g.consolidate(user_id=U, project=P, now=T0)
    cross = [n for n in g.all_nodes(user_id=U, project=P) if n.kind is PersonaKind.CROSS_ENTITY]
    assert len(cross) == 3


def test_consolidation_is_idempotent(graph):
    g, _conn = graph
    for i, anchor in enumerate(("harbor sync", "the release checklist", "onboarding docs"), 1):
        _observe(g, "impatient", anchor, session=f"s{i}", day=i)
    assert len(g.consolidate(user_id=U, project=P, now=T0)) == 1
    assert g.consolidate(user_id=U, project=P, now=T0) == []


def test_a_promoted_trait_records_what_it_was_generalised_from(graph):
    g, _conn = graph
    for i, anchor in enumerate(("harbor sync", "the release checklist", "onboarding docs"), 1):
        _observe(g, "impatient", anchor, session=f"s{i}", day=i)
    promoted = g.consolidate(user_id=U, project=P, now=T0)
    assert sorted(promoted[0].anchors) == [
        "harbor sync",
        "onboarding docs",
        "the release checklist",
    ]
