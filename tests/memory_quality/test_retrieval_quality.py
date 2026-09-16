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

import subprocess
from pathlib import Path

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.eval.retrieval import (
    ProbeKind,
    ProbeSet,
    Split,
    describe_run,
    load_probe_set,
    probe_split,
    run_probes,
    score_run,
)
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


def _commit() -> str | None:
    """The checkout's commit, so a printed card names the code it measured."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


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
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(
        conn=conn, embedder=build_embedder(settings), dim=settings.embedding_dim
    )
    probe_set = load_probe_set(PROBES)

    card = score_run(
        await run_probes(probe_set, gate=MemoryGate(module), user_id="owner", k=K), k=K
    )
    run = describe_run(
        settings=settings,
        probe_path=PROBES,
        probe_set=probe_set,
        conn=conn,
        k=K,
        floor_margin=None,
        commit=_commit(),
    )
    print("\n" + run.format() + "\n" + card.format(K))

    # Measured on this 60-probe set with Qwen3-Embedding-0.6B: recall@8 0.85 overall, 0.95
    # single-hop, 1.00 temporal, MRR 0.48. Unanswerable and multi-hop probes count toward the
    # overall numbers, which is why they sit below the single-hop ones.
    assert card.recall_at_k >= 0.80, card.format(K)
    assert card.by_kind[ProbeKind.SINGLE_HOP].recall_at_k >= 0.9, card.format(K)
    assert card.by_kind[ProbeKind.TEMPORAL].recall_at_k >= 0.9, card.format(K)
    # Ranking, not just retrieval: a corpus this size returns the answer somewhere for
    # almost any query, so the position is what says the ranking means something.
    assert card.mrr >= 0.45, card.format(K)


@pytest.mark.live
async def test_report_what_each_relevance_floor_would_cost(tmp_path, capsys):
    """Not an assertion -- the fitting procedure, runnable.

    The floor's threshold is the one number that cannot be borrowed: it depends on the corpus
    and the embedding model. This prints what each candidate keeps and what it silences, over
    the fit half only, so a value can be chosen without the sealed probes ever informing it.
    Choose from the middle of a stable stretch rather than the peak: a threshold at the peak
    of a small sweep is fitted to the sweep.
    """
    settings = Settings()
    embedder = build_embedder(settings)
    conn = open_db(str(tmp_path / "morgan.db"))
    probe_set = load_probe_set(PROBES)

    # Embed the corpus once; every threshold below is scored against this same database.
    seed = build_memory_module(conn=conn, embedder=embedder, dim=settings.embedding_dim)
    await run_probes(ProbeSet(probe_set.corpus, ()), gate=MemoryGate(seed), user_id="owner", k=K)

    fit_half = ProbeSet({}, tuple(p for p in probe_set.probes if probe_split(p.id) is Split.FIT))
    lines = [f"{'margin':>7} {'recall@8':>9} {'abstains':>9} {'mrr':>6}"]
    for margin in (None, 0.05, 0.08, 0.10, 0.11, 0.12, 0.13, 0.15, 0.20):
        module = build_memory_module(
            conn=conn, embedder=embedder, dim=settings.embedding_dim, floor_margin=margin
        )
        card = score_run(
            await run_probes(fit_half, gate=MemoryGate(module), user_id="owner", k=K), k=K
        )
        label = "off" if margin is None else f"{margin:.2f}"
        lines.append(
            f"{label:>7} {card.recall_at_k:>9.2f} {card.abstain_rate:>9.2f} {card.mrr:>6.2f}"
        )

    run = describe_run(
        settings=settings,
        probe_path=PROBES,
        probe_set=probe_set,
        conn=conn,
        k=K,
        floor_margin=None,
        commit=_commit(),
    )
    with capsys.disabled():
        print("\n" + run.format())
        print(
            f"relevance floor sweep, each row its own floor, fit half only "
            f"({len(fit_half.probes)} probes):"
        )
        print("\n".join(lines))
