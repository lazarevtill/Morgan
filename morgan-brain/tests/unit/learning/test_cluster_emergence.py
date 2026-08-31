"""Cluster emergence (VoiceMem §3.1, Algorithm 1).

Emergence re-partitions the memory index, which changes what every future query can
route to. So the tests that matter are the ones that refuse: too little evidence, a
subgraph that is really the whole slot, and a candidate a judge already turned down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.learning.cluster_emergence import (
    ClusterEmergence,
    RefusingJudge,
    Verdict,
    coherence,
)
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.modules.memory.stores.db import open_db

U = "u1"
P = "plata"
T0 = datetime(2026, 8, 1, tzinfo=UTC)

PETS = ["rex", "leash", "dog park"]
WORK = ["harbor", "gitlab"]


class _Judge:
    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        self.calls: list[list[str]] = []

    async def judge(self, entities, *, schema_name):
        self.calls.append(list(entities))
        return self.verdict


def _build(judge=None, alpha=0.35):
    conn = open_db(":memory:")
    semantic = SemanticIndex(conn)
    semantic.ensure_schemas(user_id=U, project=P)
    emergence = ClusterEmergence(
        semantic=semantic,
        conn=conn,
        judge=judge if judge is not None else RefusingJudge(),
        alpha=alpha,
    )
    return semantic, emergence, conn


def _seed(semantic, emergence, *, queries: int = 8):
    """A slot holding two disconnected groups, one of which is asked about together."""
    for name in [*PETS, *WORK]:
        semantic.assign(user_id=U, project=P, entity=name, schema_name="daily_life")
    semantic.observe_cooccurrence(user_id=U, project=P, names=PETS)
    semantic.observe_cooccurrence(user_id=U, project=P, names=WORK)
    for i in range(queries):
        emergence.log_activation(user_id=U, project=P, entities=PETS, now=T0 + timedelta(days=i))


# ---------------------------------------------------------------------------
# ρ
# ---------------------------------------------------------------------------


def test_coherence_is_one_when_every_query_activates_exactly_the_subset():
    assert coherence({"a", "b"}, [{"a", "b"}, {"a", "b"}]) == 1.0


def test_coherence_is_zero_when_nothing_overlaps():
    assert coherence({"a", "b"}, [{"x"}, {"y"}]) == 0.0


def test_a_rarely_touched_subset_does_not_look_coherent():
    """Queries that activated nothing in the subset still count in the denominator --
    otherwise a subset looks coherent merely because it is rarely asked about."""
    often = coherence({"a", "b"}, [{"a", "b"}] * 4)
    rarely = coherence({"a", "b"}, [{"a", "b"}] + [{"x"}] * 3)
    assert rarely < often


def test_coherence_of_nothing_is_zero():
    assert coherence(set(), [{"a"}]) == 0.0
    assert coherence({"a"}, []) == 0.0


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_too_few_queries_proposes_nothing():
    """One afternoon of asking about the same incident is not a permanent structure."""
    semantic, emergence, conn = _build()
    _seed(semantic, emergence, queries=2)
    assert emergence.candidates(user_id=U, project=P) == []
    conn.close()


def test_a_component_covering_its_whole_slot_is_not_a_candidate():
    """Promoting it would rename the slot, not partition it."""
    semantic, emergence, conn = _build()
    for name in PETS:
        semantic.assign(user_id=U, project=P, entity=name, schema_name="daily_life")
    semantic.observe_cooccurrence(user_id=U, project=P, names=PETS)
    for i in range(8):
        emergence.log_activation(user_id=U, project=P, entities=PETS, now=T0 + timedelta(days=i))
    assert emergence.candidates(user_id=U, project=P) == []
    conn.close()


def test_a_query_touching_one_entity_is_not_logged():
    """It says nothing about co-retrieval, and keeping it dilutes every ρ with a row
    that can never contribute to one."""
    _semantic, emergence, conn = _build()
    assert emergence.log_activation(user_id=U, project=P, entities=["rex"], now=T0) is None
    conn.close()


async def test_no_judge_promotes_nothing():
    """Re-partitioning on a heuristic is not a degraded version of doing it with
    judgement -- it is a different operation whose mistakes are invisible."""
    semantic, emergence, conn = _build()
    _seed(semantic, emergence)
    assert emergence.candidates(user_id=U, project=P)  # a candidate exists
    assert await emergence.run(user_id=U, project=P, now=T0) == []
    assert "rex" not in str(semantic.schemas(user_id=U, project=P))
    conn.close()


async def test_a_judge_that_declines_any_one_check_promotes_nothing():
    semantic, emergence, conn = _build(
        judge=_Judge(Verdict(relevant=True, important=True, complete=False, name="pets"))
    )
    _seed(semantic, emergence)
    assert await emergence.run(user_id=U, project=P, now=T0) == []
    assert semantic.schema_of(user_id=U, project=P, entity="rex") == "daily_life"
    conn.close()


async def test_a_rejected_candidate_is_not_proposed_again():
    """Without this the same subgraph costs a judge call every night until a judge has
    an off day and lets it through."""
    judge = _Judge(Verdict(relevant=False, reason="one holiday, not a topic"))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)

    await emergence.run(user_id=U, project=P, now=T0)
    assert len(judge.calls) == 1
    assert emergence.candidates(user_id=U, project=P) == []
    await emergence.run(user_id=U, project=P, now=T0)
    assert len(judge.calls) == 1, "the refused candidate was proposed a second time"
    conn.close()


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


async def test_a_coherent_subgraph_is_promoted_to_its_own_slot():
    judge = _Judge(Verdict(relevant=True, important=True, complete=True, name="Pets & Outdoor"))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)

    promoted = await emergence.run(user_id=U, project=P, now=T0)

    assert promoted == ["pets_outdoor"]
    for name in PETS:
        assert semantic.schema_of(user_id=U, project=P, entity=name) == "pets_outdoor"
    conn.close()


async def test_promotion_re_partitions_across_the_preset_boundary_only_for_its_members():
    """The entities that were not part of the coherent group stay where they were."""
    judge = _Judge(Verdict(relevant=True, important=True, complete=True, name="pets"))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)

    await emergence.run(user_id=U, project=P, now=T0)

    for name in WORK:
        assert semantic.schema_of(user_id=U, project=P, entity=name) == "daily_life"
    conn.close()


async def test_the_promoted_slot_is_marked_as_emerged():
    judge = _Judge(Verdict(relevant=True, important=True, complete=True, name="pets"))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)
    await emergence.run(user_id=U, project=P, now=T0)

    row = conn.execute(
        "SELECT emerged FROM mem_schemas WHERE user_id = ? AND project = ? AND name = 'pets'",
        (U, P),
    ).fetchone()
    assert row["emerged"] == 1
    conn.close()


async def test_a_judge_verdict_with_no_name_still_produces_a_usable_slot():
    judge = _Judge(Verdict(relevant=True, important=True, complete=True, name=""))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)
    promoted = await emergence.run(user_id=U, project=P, now=T0)
    assert promoted and promoted[0].startswith("emergent_")
    conn.close()


async def test_only_one_candidate_is_promoted_per_run():
    """Re-partitioning changes what every future query routes to; doing several at once
    makes the effect of any one of them unattributable."""
    judge = _Judge(Verdict(relevant=True, important=True, complete=True, name="pets"))
    semantic, emergence, conn = _build(judge=judge)
    _seed(semantic, emergence)
    # A second coherent group in a different slot.
    trip = ["oslo", "flight", "hotel"]
    for name in trip:
        semantic.assign(user_id=U, project=P, entity=name, schema_name="goals")
    semantic.assign(user_id=U, project=P, entity="promotion", schema_name="goals")
    semantic.observe_cooccurrence(user_id=U, project=P, names=trip)
    for i in range(8):
        emergence.log_activation(user_id=U, project=P, entities=trip, now=T0 + timedelta(days=i))

    await emergence.run(user_id=U, project=P, now=T0)
    assert len(judge.calls) == 1
    conn.close()


def test_activations_are_project_scoped():
    semantic, emergence, conn = _build()
    _seed(semantic, emergence)
    assert emergence.candidates(user_id=U, project="other") == []
    conn.close()


@pytest.mark.parametrize("alpha", [0.9, 0.99])
async def test_a_high_threshold_refuses_a_weakly_coherent_group(alpha):
    semantic, emergence, conn = _build(alpha=alpha)
    for name in [*PETS, *WORK]:
        semantic.assign(user_id=U, project=P, entity=name, schema_name="daily_life")
    semantic.observe_cooccurrence(user_id=U, project=P, names=PETS)
    semantic.observe_cooccurrence(user_id=U, project=P, names=WORK)
    for i in range(8):
        # Every query mixes the two groups, so neither is coherent on its own.
        emergence.log_activation(
            user_id=U, project=P, entities=[*PETS, *WORK], now=T0 + timedelta(days=i)
        )
    assert emergence.candidates(user_id=U, project=P) == []
    conn.close()
