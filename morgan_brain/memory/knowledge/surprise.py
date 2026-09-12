"""Which recent episodics the current fact base did not already predict.

Consolidation costs a model call over everything it is given, so episodics the facts
already cover are dropped before the call. Deliberately lexical and conservative: it
drops near-duplicates, never borderline-novel content, and costs nothing to run.
"""

from __future__ import annotations

import re

from morgan_brain.models import Memory, TemporalFact

#: A word is a run of letters or digits in any script. ``[^\W_]`` is Unicode-aware for
#: str patterns, so Cyrillic, Greek and CJK all tokenise; the underscore is excluded so a
#: fact's ``lives_in`` predicate splits into the words an episodic would use. The ASCII
#: pattern this replaced produced an empty set for any non-Latin text, and an episodic
#: with no tokens was skipped before it could be scored.
_WORD = re.compile(r"[^\W_]+")


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens in any script — the unit of the surprise heuristic."""
    return set(_WORD.findall(text.lower()))


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
        known |= _tokens(f"{f.subject} {f.predicate} {f.object}")

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
