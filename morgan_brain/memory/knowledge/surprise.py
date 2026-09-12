"""Which recent episodics the current fact base did not already predict.

Consolidation costs a model call over everything it is given, so episodics the facts
already cover are dropped before the call. Deliberately lexical and conservative: it
drops near-duplicates, never borderline-novel content, and costs nothing to run.
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
    """Keep only episodics the current fact base did not already predict (surprise-gating).

    ``novelty`` = fraction of an episodic's tokens absent from the union of current-fact
    tokens. Episodics with ``novelty < min_novelty`` are already-known (low prediction error)
    and dropped; the rest are returned most-surprising-first, capped at ``max_keep``. At cold
    start (no facts) every episodic is fully novel, so nothing is dropped. The heuristic is
    deliberately lexical and conservative — it drops near-duplicates, never borderline-novel
    content — and adds zero LLM cost.
    """
    known: set[str] = set()
    for f in facts:
        # Underscores become spaces first: a predicate is a snake_case identifier standing
        # for a phrase, and "lives_in" has to meet "lives in" as written in a memory.
        # Recall renders a fact the same way when it surfaces one.
        known |= _tokens(f"{f.subject} {f.predicate} {f.object}".replace("_", " "))

    scored: list[tuple[float, Memory]] = []
    for m in episodics:
        toks = _tokens(m.content)
        if not toks:
            continue
        novelty = len(toks - known) / len(toks)
        if novelty >= min_novelty:
            scored.append((novelty, m))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [m for _, m in scored[:max_keep]]
