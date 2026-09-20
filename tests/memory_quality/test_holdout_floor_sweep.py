"""The relevance floor, swept against 128 labelled probes over the 2026-09-19 archive.

The floor margin is a property of the embedding model: 0.11 was measured for
Qwen3-Embedding-0.6B on a different snapshot. The live model behind this sweep is an 8B
model at 4,096 dims, so its number is unmeasured until this runs. Scored with
``all_projects=True`` -- the labels were built against the whole snapshot (two projects,
`archive/chatgpt` and `archive/chatgpt-holdout`), and while every expected id happens to sit
in the first, the second is still live as the extra noise a real recall would see.

Works on a copy: a sweep must never write to the file it scores, and it never opens the
owner's live database. The probe text is the owner's private conversation content -- it must
never be printed, stored, or otherwise leave this process; only the aggregate metrics below
do.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.eval.retrieval import (
    Probe,
    ProbeKind,
    ProbeResult,
    Split,
    probe_split,
    score_run,
)
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import MemoryQuery
from morgan_brain.providers.factory import build_embedder

pytestmark = pytest.mark.live

#: The window a reader of `morgan recall` actually sees.
K = 8

#: The snapshot's one owner. Not a secret -- `Settings.owner_user_id` defaults to the same
#: string, and it is the literal value the ChatGPT import wrote into every row.
USER_ID = "owner"


class _CachingEmbedder:
    """Memoizes ``embed`` by text so a multi-threshold sweep embeds each probe once.

    The corpus is already embedded in the snapshot; only the 128 probe queries are embedded
    live, and the same 128 strings are asked again at every threshold -- only the floor
    gating changes downstream of the vector search. Without this a 7-threshold sweep makes
    896 live calls instead of 128.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = await self._inner.embed(text)
        return self._cache[text]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


def _exists(db_path: Path, memory_id: str) -> bool:
    """Whether *memory_id* is a row in *db_path*'s ``memories`` table."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def _load_probes(labels: dict[str, Any]) -> tuple[Probe, ...]:
    """The labels' ``probes`` array as real `Probe`s. No `forbidden`/`require_all`: this set
    holds only single-hop and unanswerable probes, so both stay at their `Probe` defaults."""
    return tuple(
        Probe(
            id=p["id"],
            kind=ProbeKind(p["kind"]),
            query=p["query"],
            expected=tuple(p["expected"]),
        )
        for p in labels["probes"]
    )


async def _sweep(
    conn: sqlite3.Connection,
    embedder: Embedder,
    dim: int,
    probes: tuple[Probe, ...],
    *,
    threshold: float,
    k: int = K,
) -> str:
    """Score every probe once at *threshold*, fit and sealed halves reported apart.

    Recall goes through the gate with ``all_projects=True``, like the fitting run this
    number is meant to reproduce. Only ids, counts and the three metrics ever leave this
    function -- never a probe's query text.
    """
    gate = MemoryGate(
        build_memory_module(conn=conn, embedder=embedder, dim=dim, floor_margin=threshold)
    )
    results = []
    for probe in probes:
        found = await gate.recall(
            MemoryQuery(user_id=USER_ID, all_projects=True, text=probe.query, top_k=k)
        )
        results.append(ProbeResult(probe=probe, retrieved=tuple(m.id for m in found)))

    lines = []
    for split, label in ((Split.FIT, "fit"), (Split.SEALED, "sealed")):
        card = score_run([r for r in results if probe_split(r.probe.id) is split], k=k)
        lines.append(
            f"margin={threshold:.2f} {label:>6}  n={card.n:<4} recall@{k}={card.recall_at_k:.2f}  "
            f"mrr={card.mrr:.2f}  abstain={card.abstain_rate:.2f}"
        )
    return "\n".join(lines)


async def test_floor_sweep_over_the_holdout(snapshot_db, holdout_probes, floor_sweep, tmp_path):
    working = tmp_path / "snapshot.db"
    shutil.copy2(snapshot_db, working)
    labels = json.loads(holdout_probes.read_text(encoding="utf-8"))

    # Every expected id must exist before anything is scored: labels built against a different
    # import score zero and look exactly like a model regression.
    missing = [i for p in labels["probes"] for i in p["expected"] if not _exists(working, i)]
    assert not missing, f"{len(missing)} expected ids are not in the snapshot: {missing[:5]}"

    settings = Settings()
    # composition.build_memory_context probes this automatically; build_memory_module (used
    # below, like the sibling live tests) does not, and a wrong width here does not fail
    # loudly -- it runs a KNN against a table shaped for a different model and returns
    # confident, wrong neighbours that would read as a measured floor.
    assert settings.embedding_dim == 4096, (
        "this snapshot's vectors are 4096-wide (qwen3-embedding:8b); "
        f"settings.embedding_dim={settings.embedding_dim} -- export MORGAN_EMBEDDING_DIM=4096"
    )

    probes = _load_probes(labels)
    embedder: Embedder = _CachingEmbedder(build_embedder(settings))
    conn = open_db(str(working))

    fit_n = sum(1 for p in probes if probe_split(p.id) is Split.FIT)
    print(f"probes: {len(probes)}  fit={fit_n}  sealed={len(probes) - fit_n}")
    for threshold in floor_sweep:
        print(await _sweep(conn, embedder, settings.embedding_dim, probes, threshold=threshold))
