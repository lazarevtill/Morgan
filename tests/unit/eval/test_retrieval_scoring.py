"""Scoring a retrieval run. Pure arithmetic, so it is checkable without a model server."""

from __future__ import annotations

from morgan_brain.eval.retrieval import (
    Probe,
    ProbeKind,
    ProbeResult,
    Split,
    probe_split,
    score_run,
)


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
    assert card.stale_first_rate == 0.0


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


def test_an_unanswerable_probe_is_scored_on_whether_the_run_returned_nothing():
    """The relevance floor's own metric. Recall cannot measure a floor -- a system that
    answers everything scores perfect recall -- so the question "did it correctly say
    nothing" needs a number of its own.
    """
    silent = ProbeResult(
        probe=Probe(id="u1", kind=ProbeKind.UNANSWERABLE, query="q", expected=()),
        retrieved=(),
    )
    talked = ProbeResult(probe=silent.probe, retrieved=("some_unrelated_memory",))

    assert score_run([silent], k=5).abstain_rate == 1.0
    assert score_run([talked], k=5).abstain_rate == 0.0


def test_unanswerable_probes_do_not_drag_down_recall():
    """They have no right answer, so counting them as recall misses would make the headline
    number a blend of two different questions and hide movement in both."""
    results = [
        _result("a", ProbeKind.SINGLE_HOP, ["target"]),
        ProbeResult(
            probe=Probe(id="u1", kind=ProbeKind.UNANSWERABLE, query="q", expected=()),
            retrieved=(),
        ),
    ]

    card = score_run(results, k=5)

    assert card.recall_at_k == 1.0
    assert card.abstain_rate == 1.0


def test_a_superseded_memory_below_the_current_one_is_not_counted_against_the_run():
    """The metric has to measure order, not presence.

    ``city_old`` is forbidden by "where do I live now?" and expected by "where did I live
    before?" -- the same memory, wanted by one question and not the other. A metric that
    counts its mere presence as a failure can only be satisfied by suppressing it, which
    breaks the question that asks for it. What actually goes wrong is the stale answer
    arriving *first*.
    """
    current_first = ProbeResult(
        probe=Probe(
            id="a",
            kind=ProbeKind.KNOWLEDGE_UPDATE,
            query="q",
            expected=("new",),
            forbidden=("old",),
        ),
        retrieved=("new", "old"),
    )
    stale_first = ProbeResult(probe=current_first.probe, retrieved=("old", "new"))

    assert score_run([current_first], k=5).stale_first_rate == 0.0
    assert score_run([stale_first], k=5).stale_first_rate == 1.0


def test_a_run_that_returns_only_the_stale_answer_counts_as_stale_first():
    """Never returning the current answer is the worst case, not an exempt one."""
    only_stale = ProbeResult(
        probe=Probe(
            id="a",
            kind=ProbeKind.KNOWLEDGE_UPDATE,
            query="q",
            expected=("new",),
            forbidden=("old",),
        ),
        retrieved=("old", "unrelated"),
    )

    assert score_run([only_stale], k=5).stale_first_rate == 1.0


def test_which_probes_are_sealed_is_decided_by_hash_not_by_choice():
    """A threshold fitted on the probes that later judge it reports its own training score.
    Deciding the split by hashing the id means whoever authors a probe cannot steer it onto
    the side that flatters them, and the split is the same on every machine and every run.
    """
    ids = [f"probe-{i}" for i in range(300)]
    sealed = [p for p in ids if probe_split(p) is Split.SEALED]

    assert [probe_split(p) for p in ids] == [probe_split(p) for p in ids]
    assert 0.2 < len(sealed) / len(ids) < 0.5
