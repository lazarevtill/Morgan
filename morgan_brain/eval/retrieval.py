"""Scoring recall against labelled probes.

A probe is a question with a known answer: the memories that *should* come back, and for a
knowledge update, the superseded memory that must *not*. Running them gives three numbers
that answer different questions, which is why none of them is dropped:

* **recall@k** -- did the right memory arrive at all, within the window a reader sees.
* **MRR** -- and how near the top. A hit at rank eight is not the same product as a hit at
  rank one, and recall alone cannot tell them apart.
* **leak rate** -- how often a superseded memory came back. A run can score perfect recall
  and still be wrong, because returning both the new answer and the old one is wrong.

Broken down per probe kind, because one average hides a category that fails completely:
temporal questions can return nothing while single-hop recall carries the mean.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory, MemoryKind, MemoryQuery, MemorySource


class ProbeKind(str, Enum):
    """What a probe is testing, following the LoCoMo / LongMemEval categories."""

    SINGLE_HOP = "single_hop"  # one memory holds the answer
    MULTI_HOP = "multi_hop"  # the answer needs two memories together
    TEMPORAL = "temporal"  # "where did I *used* to live"
    KNOWLEDGE_UPDATE = "knowledge_update"  # a fact changed; the latest must win
    UNANSWERABLE = "unanswerable"  # the corpus does not hold the answer; say nothing


@dataclass(frozen=True)
class Probe:
    """A question with a known answer."""

    id: str
    kind: ProbeKind
    query: str
    #: Memory keys any one of which counts as recalling the answer.
    expected: tuple[str, ...]
    #: Memory keys that must not come back -- the superseded side of an update.
    forbidden: tuple[str, ...] = ()
    #: When true, every key in ``expected`` must arrive, not just one. A multi-hop answer
    #: needs both its halves; scoring it on either would count half an answer as a whole one
    #: and never measure the composition the probe exists to test.
    require_all: bool = False


@dataclass(frozen=True)
class ProbeResult:
    """What one probe actually returned, most relevant first."""

    probe: Probe
    retrieved: tuple[str, ...]


@dataclass(frozen=True)
class ProbeSet:
    """A corpus to search and the probes to search it with.

    The corpus is keyed rather than ordered so a probe names what it expects in words a
    reader can check -- ``expected=("moved_to_berlin",)`` rather than an index into a list.
    """

    corpus: dict[str, str]
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class Scorecard:
    """One run's numbers. ``by_kind`` is empty on a per-kind card, which has nothing to split."""

    n: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    leak_rate: float = 0.0
    #: Over probes with no right answer, how often the run correctly returned nothing.
    #: Recall cannot measure a relevance floor -- a system that answers everything scores
    #: perfect recall -- so abstaining needs a number of its own.
    abstain_rate: float = 0.0
    by_kind: dict[ProbeKind, Scorecard] = field(default_factory=dict)

    def format(self, k: int) -> str:
        """The card as a reader would want it: the headline, then what it is hiding."""
        lines = [
            f"n={self.n}  recall@{k}={self.recall_at_k:.2f}  "
            f"mrr={self.mrr:.2f}  leak={self.leak_rate:.2f}  "
            f"abstain={self.abstain_rate:.2f}"
        ]
        for kind, card in sorted(self.by_kind.items(), key=lambda kv: kv[0].value):
            lines.append(
                f"  {kind.value:<17} n={card.n:<4} recall@{k}={card.recall_at_k:.2f}  "
                f"mrr={card.mrr:.2f}  leak={card.leak_rate:.2f}  "
                f"abstain={card.abstain_rate:.2f}"
            )
        return "\n".join(lines)


def _reciprocal_rank(result: ProbeResult, k: int) -> float:
    """The reciprocal of the rank at which this probe is answered, or 0 if it never is.

    For a ``require_all`` probe the answer is only complete at the rank of its *last*
    missing piece, so that is the rank scored: the run has not answered the question until
    a reader has seen everything the answer needs.
    """
    window = result.retrieved[:k]
    if result.probe.require_all:
        positions = [window.index(key) + 1 for key in result.probe.expected if key in window]
        if len(positions) < len(result.probe.expected):
            return 0.0
        return 1.0 / max(positions)
    for position, key in enumerate(window, start=1):
        if key in result.probe.expected:
            return 1.0 / position
    return 0.0


def _card(results: Sequence[ProbeResult], k: int) -> Scorecard:
    if not results:
        return Scorecard()
    # A probe with no right answer is asking a different question, so it is scored on its own
    # terms and kept out of recall -- otherwise the headline blends two measurements and hides
    # movement in both.
    answerable = [r for r in results if r.probe.expected]
    unanswerable = [r for r in results if not r.probe.expected]
    ranks = [_reciprocal_rank(r, k) for r in answerable]
    leaks = [
        any(key in r.probe.forbidden for key in r.retrieved[:k])
        for r in results
        if r.probe.forbidden
    ]
    return Scorecard(
        n=len(results),
        recall_at_k=(sum(1 for rank in ranks if rank > 0) / len(ranks)) if ranks else 0.0,
        mrr=(sum(ranks) / len(ranks)) if ranks else 0.0,
        # Probes with nothing forbidden cannot leak, and averaging them in would dilute the
        # rate towards zero and hide the updates that did come back stale.
        leak_rate=(sum(leaks) / len(leaks)) if leaks else 0.0,
        abstain_rate=(
            sum(1 for r in unanswerable if not r.retrieved[:k]) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
    )


def score_run(results: Sequence[ProbeResult], *, k: int) -> Scorecard:
    """Score *results*, counting only the first *k* retrieved memories for each probe."""
    overall = _card(results, k)
    by_kind = {
        kind: _card([r for r in results if r.probe.kind is kind], k)
        for kind in ProbeKind
        if any(r.probe.kind is kind for r in results)
    }
    return Scorecard(
        n=overall.n,
        recall_at_k=overall.recall_at_k,
        mrr=overall.mrr,
        leak_rate=overall.leak_rate,
        abstain_rate=overall.abstain_rate,
        by_kind=by_kind,
    )


#: Where a measurement run stores its corpus. Its own project, because a number that moves
#: when the owner happens to store something unrelated is not a measurement.
EVAL_PROJECT = "eval/retrieval"


async def run_probes(
    probe_set: ProbeSet,
    *,
    gate: MemoryGate,
    user_id: str,
    k: int,
    project: str = EVAL_PROJECT,
) -> list[ProbeResult]:
    """Store *probe_set*'s corpus, run every probe through recall, and report what came back.

    Memory ids are the corpus keys, so a result names what a reader labelled rather than a
    hash they would have to resolve. Recall goes through the gate like any other read: a
    harness that reached past it would measure a path no caller uses.
    """
    for key, text in probe_set.corpus.items():
        await gate.store(
            Memory(
                id=key,
                user_id=user_id,
                project=project,
                kind=MemoryKind.EPISODIC,
                content=text,
                source=MemorySource.USER_STATED,
            )
        )

    results = []
    for probe in probe_set.probes:
        found = await gate.recall(
            MemoryQuery(user_id=user_id, project=project, text=probe.query, top_k=k)
        )
        results.append(ProbeResult(probe=probe, retrieved=tuple(m.id for m in found)))
    return results


def load_probe_set(path: Path) -> ProbeSet:
    """Read a probe set from JSON. Keys beginning with ``_`` are prose for the reader."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ProbeSet(
        corpus=raw["corpus"],
        probes=tuple(
            Probe(
                id=p["id"],
                kind=ProbeKind(p["kind"]),
                query=p["query"],
                expected=tuple(p["expected"]),
                forbidden=tuple(p.get("forbidden", ())),
                require_all=bool(p.get("require_all", False)),
            )
            for p in raw["probes"]
        ),
    )
