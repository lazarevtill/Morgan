"""Which recent episodics the current fact base did not already predict.

Consolidation costs a model call over everything it is given, so episodics the facts
already cover are dropped before the call, except explicit revisions: lexical overlap
does not establish that a correction is redundant. This selection costs no model call.
"""

from __future__ import annotations

from morgan_brain.memory.knowledge.extract import words
from morgan_brain.models import Memory, TemporalFact


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens in any script — the unit of the surprise heuristic."""
    return set(words(text.lower()))


def keep_surprising(
    episodics: list[Memory],
    facts: list[TemporalFact],
    *,
    min_novelty: float = 0.5,
    max_keep: int = 30,
) -> list[Memory]:
    """Prioritize explicit revisions, then novelty-ranked ordinary episodics.

    Inputs must already be scoped, trusted and active; parent IDs here are selection
    metadata, never validation or write authority. Revisions preserve input order and
    bypass lexical novelty. Ordinary records retain their prior threshold and stable
    most-novel-first order. The combined result stays within ``max_keep`` (default 30),
    including cold start. Excess revisions are omitted, and revisions can displace
    novel ordinary records. Retention does not establish semantic truth or entailment.
    """
    known: set[str] = set()
    for f in facts:
        # Underscores become spaces first: a predicate is a snake_case identifier standing
        # for a phrase, and "lives_in" has to meet "lives in" as written in a memory.
        # Recall renders a fact the same way when it surfaces one.
        known |= _tokens(f"{f.subject} {f.predicate} {f.object}".replace("_", " "))

    revisions: list[Memory] = []
    scored: list[tuple[float, Memory]] = []
    for m in episodics:
        if m.revises_event_ids:
            revisions.append(m)
            continue
        toks = _tokens(m.content)
        if not toks:
            continue
        novelty = len(toks - known) / len(toks)
        if novelty >= min_novelty:
            scored.append((novelty, m))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return (revisions + [m for _, m in scored])[:max_keep]
