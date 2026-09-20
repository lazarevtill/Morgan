"""run_probes can score a corpus that is already in the database.

The 128 holdout labels are scored against an import of the owner's archive, which is already
stored: storing the corpus again would write 3,010 duplicate memories into the file under
measurement.
"""

from __future__ import annotations

from morgan_brain.composition import build_memory_module
from morgan_brain.eval.retrieval import EVAL_PROJECT, Probe, ProbeKind, ProbeSet, run_probes
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryKind, MemorySource


async def test_store_corpus_false_writes_nothing(tmp_path):
    conn = open_db(str(tmp_path / "m.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=8), dim=8)
    gate = MemoryGate(module)
    await gate.store(
        Memory(
            id="a",
            user_id="u",
            project=EVAL_PROJECT,
            kind=MemoryKind.EPISODIC,
            content="Harbor upgrade plan",
            source=MemorySource.USER_STATED,
        )
    )
    probe_set = ProbeSet(
        corpus={"a": "Harbor upgrade plan", "b": "never stored"},
        probes=(Probe(id="p1", kind=ProbeKind.SINGLE_HOP, query="Harbor", expected=("a",)),),
    )

    results = await run_probes(probe_set, gate=gate, user_id="u", k=5, store_corpus=False)

    assert module._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert results[0].retrieved == ("a",)
