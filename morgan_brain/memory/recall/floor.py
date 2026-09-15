"""Deciding whether to answer at all.

Recall had no way to say "I don't know". Every signal returned its top-k regardless of score,
so once a project held anything, every question came back with a confident list. On fourteen
questions this corpus cannot answer, it returned eight memories to each.

**Why the gate is here and not inside a signal.** Reciprocal rank fusion is rank-only: the top
result scores ``1/61`` whether it is a perfect match or noise, so nothing downstream of fusion
can tell them apart. Ranking and gating are also different questions -- "which of these is most
relevant" versus "is any of them relevant at all" -- and get different numbers rather than one
formula asked to do both.

**Why a margin and not a similarity.** An absolute cutoff like ``0.5`` is a property of one
embedding model, and Morgan's model is a configuration value: swap it and the number silently
means something else. What is asked instead is whether the best hit stands out from the
background *this same query* pulled up. A query the corpus can answer has one or two memories
far above the rest; a query it cannot has a flat list of equally mediocre neighbours, however
high or low that model happens to score them.

Measured on the labelled probes, that margin separates better than the raw similarity it
replaces -- it keeps more real answers at the same silence -- and it does so without depending
on the model's absolute number range.

An earlier design compared the top score against a baseline built from memory-to-memory
matches. It is recorded here because the failure is instructive: memory-to-memory similarity
averaged 0.576 while a real query's best hit averaged 0.582, so the baseline sat on top of the
answers it was meant to be below, and the gate rejected half of them. A baseline has to be
measured in the same space as the thing it judges.
"""

from __future__ import annotations

import statistics

#: Results from this rank down are "the background this query pulled up". Ranks 1-3 are
#: excluded because a question with two or three good answers is ordinary -- a multi-hop
#: question needs two by construction -- and averaging them into the background would hide
#: exactly the queries that are answered best.
TAIL_FROM_RANK = 3

#: Below this many results there is no background to compare against, and the floor abstains
#: from abstaining rather than judging a query on one or two numbers.
MIN_RESULTS_TO_JUDGE = TAIL_FROM_RANK + 2


def answer_margin(scores: list[float]) -> float | None:
    """How far the best hit stands above the background, or ``None`` if that is unknowable.

    *scores* are similarities, best first, as the vector search returns them.
    """
    if len(scores) < MIN_RESULTS_TO_JUDGE:
        return None
    return scores[0] - statistics.fmean(scores[TAIL_FROM_RANK:])


def should_answer(*, margin: float | None, threshold: float, has_exact_match: bool) -> bool:
    """Whether this query's results are worth returning.

    *has_exact_match* overrides the margin. An exact entity match on a memory the vector search
    also ranked is direct evidence that the corpus contains what was asked about, and it is the
    case an embedding is worst at: an identifier, a surname or a ticket number carries little of
    the meaning a sentence embedding captures, so a real hit on one can sit flat against its
    neighbours. The caller decides what counts; a match on a memory the vector search did not
    rank is only a shared word, and on a real corpus nearly every question shares one.
    """
    if has_exact_match:
        return True
    # Too few results to see a background: the floor has nothing to stand on, so it does not
    # stand in the way. Answering is the lesser failure of the two.
    if margin is None:
        return True
    return margin >= threshold
