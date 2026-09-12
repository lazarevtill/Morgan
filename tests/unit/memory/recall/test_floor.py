"""The decision to stay silent.

The gate asks whether the best hit stands out from the background the same query pulled up,
rather than whether its similarity clears a fixed number. A fixed number is a property of one
embedding model; Morgan's model is a configuration value.
"""

from __future__ import annotations

from morgan_brain.memory.recall.floor import (
    MIN_RESULTS_TO_JUDGE,
    answer_margin,
    should_answer,
)


def test_one_clear_winner_leaves_a_wide_margin():
    scores = [0.80, 0.45, 0.42, 0.40, 0.38, 0.37]

    margin = answer_margin(scores)

    assert margin is not None
    assert margin > 0.3


def test_a_flat_list_of_mediocre_neighbours_leaves_almost_none():
    """What an unanswerable question looks like: nothing is close, and nothing stands out.
    The absolute numbers here are unremarkable, which is exactly why an absolute cutoff has
    to be retuned per model and this does not."""
    scores = [0.44, 0.43, 0.42, 0.41, 0.40, 0.40]

    margin = answer_margin(scores)

    assert margin is not None
    assert margin < 0.05


def test_a_question_with_several_good_answers_still_reads_as_answered():
    """A multi-hop question needs two memories by construction, and plenty of ordinary
    questions have three. Averaging those into the background would penalise exactly the
    queries that are answered best, so the background starts below them."""
    scores = [0.78, 0.76, 0.74, 0.40, 0.38, 0.37]

    margin = answer_margin(scores)

    assert margin is not None
    assert margin > 0.3


def test_too_few_results_to_see_a_background_is_not_a_verdict():
    assert answer_margin([0.9] * (MIN_RESULTS_TO_JUDGE - 1)) is None
    assert answer_margin([]) is None


def test_a_margin_above_the_threshold_answers():
    assert should_answer(margin=0.30, threshold=0.13, has_exact_match=False)


def test_a_margin_below_it_stays_silent():
    assert not should_answer(margin=0.02, threshold=0.13, has_exact_match=False)


def test_an_exact_match_answers_whatever_the_margin():
    """A surname or a ticket number carries little of the meaning a sentence embedding
    captures, so a real hit on one can sit flat against its neighbours."""
    assert should_answer(margin=0.0, threshold=0.13, has_exact_match=True)


def test_an_unjudgeable_query_is_answered_rather_than_silenced():
    """On a corpus too small to show a background, answering is the lesser failure."""
    assert should_answer(margin=None, threshold=0.13, has_exact_match=False)
