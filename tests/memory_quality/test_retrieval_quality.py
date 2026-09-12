"""Does recall actually return the right memory?

The project's largest unmeasured claim. The suite that used to sit here ran over a hash
embedder, which has no semantic similarity at all, so it proved the plumbing was connected
and nothing about relevance. These two tests keep those questions apart:

* the plumbing check runs on the hash embedder and asserts only that a run completes and
  scores -- it must never assert a quality number, because any number it produced would be
  an artefact of sha256;
* the quality check needs a real embedding endpoint (``pytest --live``) and asserts a floor
  that a regression would break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.eval.retrieval import ProbeKind, load_probe_set, run_probes, score_run
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.factory import build_embedder

PROBES = Path(__file__).parent / "probes.json"

#: The window a reader of `morgan recall` actually sees.
K = 8


def _gate(tmp_path, embedder, dim):
    conn = open_db(str(tmp_path / "morgan.db"))
    return MemoryGate(build_memory_module(conn=conn, embedder=embedder, dim=dim))


async def test_a_run_completes_and_scores_over_the_labelled_set(tmp_path):
    """Plumbing only. No quality assertion: on a hash embedder there is no quality."""
    probe_set = load_probe_set(PROBES)

    results = await run_probes(
        probe_set, gate=_gate(tmp_path, FakeEmbedder(dim=64), 64), user_id="owner", k=K
    )
    card = score_run(results, k=K)

    assert card.n == len(probe_set.probes)
    assert set(card.by_kind) == {p.kind for p in probe_set.probes}


@pytest.mark.live
async def test_recall_finds_the_right_memory_when_the_query_shares_no_words_with_it(tmp_path):
    """The measurement, against a real embedding endpoint (``pytest --live``).

    The floors below are regression bars, not targets: they sit under what the current
    embedder measured, so a change that breaks semantic retrieval fails here instead of
    being noticed months later. They are deliberately not set to the observed numbers --
    a bar equal to the measurement fails on ordinary model-to-model variation and teaches
    everyone to ignore it.

    Two categories are left unasserted because they are known to be unsolved, and a bar
    they already fail is noise rather than a gate. Both are recorded in the roadmap:
    multi-hop composition, which recall has no mechanism for, and the leak rate, which is
    what happens when a superseded memory has never been consolidated into a fact.
    """
    settings = Settings()
    gate = _gate(tmp_path, build_embedder(settings), settings.embedding_dim)
    probe_set = load_probe_set(PROBES)

    card = score_run(await run_probes(probe_set, gate=gate, user_id="owner", k=K), k=K)
    print("\n" + card.format(K))

    assert card.recall_at_k >= 0.85, card.format(K)
    assert card.by_kind[ProbeKind.SINGLE_HOP].recall_at_k >= 0.9, card.format(K)
    assert card.by_kind[ProbeKind.TEMPORAL].recall_at_k >= 0.9, card.format(K)
    # Ranking, not just retrieval: a corpus this size returns the answer somewhere for
    # almost any query, so the position is what says the ranking means something.
    assert card.mrr >= 0.55, card.format(K)
