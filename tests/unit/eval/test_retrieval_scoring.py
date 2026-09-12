"""Scoring a retrieval run. Pure arithmetic, so it is checkable without a model server."""

from __future__ import annotations

from morgan_brain.eval.retrieval import Probe, ProbeKind, ProbeResult, score_run


def _result(probe_id: str, kind: ProbeKind, retrieved: list[str], **kw) -> ProbeResult:
    probe = Probe(
        id=probe_id,
        kind=kind,
        query="q",
        expected=tuple(kw.get("expected", ["target"])),
        forbidden=tuple(kw.get("forbidden", ())),
    )
    return ProbeResult(probe=probe, retrieved=tuple(retrieved))


def test_recall_counts_a_probe_as_found_when_any_expected_memory_is_in_the_top_k():
    results = [
        _result("a", ProbeKind.SINGLE_HOP, ["noise", "target"]),
        _result("b", ProbeKind.SINGLE_HOP, ["noise", "other"]),
    ]

    card = score_run(results, k=2)

    assert card.recall_at_k == 0.5


def test_recall_ignores_a_hit_that_falls_outside_k():
    """A memory ranked twentieth is not recalled by anyone reading the top few, so the
    scorecard must not count it. Otherwise every unfloored top-k run looks perfect."""
    results = [_result("a", ProbeKind.SINGLE_HOP, ["n1", "n2", "n3", "target"])]

    assert score_run(results, k=3).recall_at_k == 0.0
    assert score_run(results, k=4).recall_at_k == 1.0


def test_reciprocal_rank_rewards_the_position_of_the_first_hit():
    results = [
        _result("a", ProbeKind.SINGLE_HOP, ["target"]),
        _result("b", ProbeKind.SINGLE_HOP, ["noise", "target"]),
    ]

    assert score_run(results, k=5).mrr == 0.75


def test_a_superseded_memory_in_the_results_is_counted_as_a_leak():
    """The knowledge-update case: after a fact changes, the old one must not come back. A
    run that recalls both the new and the old answer scores full recall and is still wrong.
    """
    results = [_result("a", ProbeKind.KNOWLEDGE_UPDATE, ["target", "stale"], forbidden=["stale"])]

    card = score_run(results, k=5)

    assert card.recall_at_k == 1.0
    assert card.leak_rate == 1.0


def test_the_scorecard_breaks_results_down_by_probe_kind():
    """One number hides which kind of recall is broken: temporal questions can fail
    completely while single-hop carries the average."""
    results = [
        _result("a", ProbeKind.SINGLE_HOP, ["target"]),
        _result("b", ProbeKind.TEMPORAL, ["noise"]),
    ]

    card = score_run(results, k=5)

    assert card.by_kind[ProbeKind.SINGLE_HOP].recall_at_k == 1.0
    assert card.by_kind[ProbeKind.TEMPORAL].recall_at_k == 0.0
    assert card.recall_at_k == 0.5


def test_an_empty_run_scores_zero_rather_than_dividing_by_zero():
    card = score_run([], k=5)

    assert card.recall_at_k == 0.0
    assert card.mrr == 0.0
    assert card.leak_rate == 0.0


def test_a_multi_hop_probe_is_only_recalled_when_every_piece_arrives():
    """A multi-hop answer needs two memories together. Counting it as recalled because one
    of them showed up scores half an answer as a whole one, and the composition -- the thing
    multi-hop exists to test -- is never measured at all.
    """
    both = ProbeResult(
        probe=Probe(
            id="a",
            kind=ProbeKind.MULTI_HOP,
            query="q",
            expected=("first", "second"),
            require_all=True,
        ),
        retrieved=("first", "second"),
    )
    half = ProbeResult(probe=both.probe, retrieved=("first", "noise"))

    assert score_run([both], k=5).recall_at_k == 1.0
    assert score_run([half], k=5).recall_at_k == 0.0
