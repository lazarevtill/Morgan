"""The gate may not be weakened by the thing it judges.

Ouroboros states this as a constitutional bound: *"Ouroboros may improve the immune
system; it may not weaken it."* (`BIBLE.md`, Principle 3). Morgan needs it for a
concrete reason, not a philosophical one: its optimizer writes a prompt, and its judge
then reads the answer that prompt produced. That is a closed loop in which the thing
being optimised can influence its own evaluation, and "the candidate beat the champion"
is only meaningful if both were measured by the same gate.

Two guards, and neither can be satisfied by trying harder at the task -- which is what
makes them guards rather than difficulty.

**1. The gate is fingerprinted.** A ``GateSpec`` -- item count, a hash of the item ids,
the judge model, the scorer names, the tie epsilon -- is captured when the champion is
certified and again when the candidate is scored. If the two disagree, the candidate was
measured against a different instrument and the comparison is not a comparison, so the
promotion is refused rather than silently accepted. A gate with *fewer* items than the
one that certified the champion is refused as weakening, which is the specific failure
this exists for: shrinking the eval set until the candidate wins.

**2. The candidate is screened for judge-directed text.** A prompt whose body addresses
the evaluator rather than the user is refused before it is ever scored. This is the
reward-hacking path that costs nothing to attempt and, once it works, permanently
poisons every later comparison -- the champion it produces is the baseline everything
after is measured against.

Refusals raise. A promotion path that could log-and-continue past a weakened gate has no
gate, and the caller has to decide what to do with a candidate it cannot trust.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class GateIntegrityError(Exception):
    """Base: the promotion cannot be trusted, for a reason that is not the score."""


class GateWeakened(GateIntegrityError):
    """The gate judging the candidate is weaker than the one that certified the champion."""


class GateChanged(GateIntegrityError):
    """The gate changed between the two measurements, so they are not comparable."""


class JudgeDirectedPrompt(GateIntegrityError):
    """The candidate addresses the evaluator instead of the user."""


@dataclass(frozen=True)
class GateSpec:
    """What the gate consisted of at the moment of a measurement."""

    n_items: int
    items_hash: str
    judge_model: str
    scorers: tuple[str, ...]
    epsilon: float

    @classmethod
    def from_items(
        cls,
        item_ids: Iterable[str],
        *,
        judge_model: str,
        scorers: Sequence[str],
        epsilon: float,
    ) -> GateSpec:
        ids = sorted(str(i) for i in item_ids)
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:32]
        return cls(
            n_items=len(ids),
            items_hash=digest,
            judge_model=judge_model,
            scorers=tuple(sorted(scorers)),
            epsilon=epsilon,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise for a receipt. Paired with ``from_dict`` so the round-trip is typed
        in one place rather than cast at every reader."""
        return {
            "n_items": self.n_items,
            "items_hash": self.items_hash,
            "judge_model": self.judge_model,
            "scorers": list(self.scorers),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> GateSpec | None:
        """Rebuild from a receipt, or ``None`` when the record is absent or malformed.

        A malformed record is treated as "no certified gate" rather than raising: the
        alternative is a nightly job that dies on a row written by an older version, and
        the honest consequence of an unreadable record is that there is nothing to
        compare against.
        """
        if not raw:
            return None
        n_items = raw.get("n_items")
        items_hash = raw.get("items_hash")
        judge_model = raw.get("judge_model")
        scorers = raw.get("scorers")
        epsilon = raw.get("epsilon")
        if not isinstance(n_items, int) or not isinstance(scorers, list):
            return None
        if not isinstance(items_hash, str) or not isinstance(judge_model, str):
            return None
        if not isinstance(epsilon, int | float):
            return None
        return cls(
            n_items=n_items,
            items_hash=items_hash,
            judge_model=judge_model,
            scorers=tuple(str(x) for x in scorers),
            epsilon=float(epsilon),
        )

    def fingerprint(self) -> str:
        """A single value that changes if any part of the gate changed."""
        raw = (
            f"{self.n_items}|{self.items_hash}|{self.judge_model}|"
            f"{','.join(self.scorers)}|{self.epsilon:.6f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def assert_gate_unweakened(*, certified: GateSpec | None, current: GateSpec) -> None:
    """Refuse a promotion measured against a different or weaker gate.

    *certified* is the gate that certified the standing champion, ``None`` when there is
    no champion yet -- in which case there is nothing to compare against and nothing to
    weaken.

    Fewer items is reported as weakening specifically, because it is the failure with a
    motive: every other difference is more likely a configuration change, but a shrunken
    eval set is how a candidate wins by being measured less.
    """
    if certified is None:
        return
    if current.n_items < certified.n_items:
        raise GateWeakened(
            f"the gate scoring this candidate has {current.n_items} items; the gate that "
            f"certified the champion had {certified.n_items}. A candidate measured on "
            f"less than the champion was has not beaten it."
        )
    if current.fingerprint() != certified.fingerprint():
        raise GateChanged(
            "the gate changed between the champion's measurement and the candidate's "
            f"(judge={certified.judge_model!r}→{current.judge_model!r}, "
            f"items={certified.n_items}→{current.n_items}, "
            f"scorers={list(certified.scorers)}→{list(current.scorers)}). "
            "Re-score the champion on the current gate before comparing."
        )


#: Phrases that only make sense if the reader is grading the output. A preprompt telling
#: the assistant how to answer never needs to say any of them, which is why matching here
#: is a refusal rather than a warning. Kept narrow on purpose: a screen that fires on
#: ordinary instructions would train whoever hits it to route around the screen.
_JUDGE_DIRECTED = (
    r"\bignore (all )?(the )?(previous|prior|above|preceding)\b",
    r"\bdisregard (all )?(the )?(previous|prior|above|preceding)\b",
    r"\byou are (the|an|a) (judge|grader|evaluator|examiner)\b",
    r"\bas (the|an|a) (judge|grader|evaluator)\b",
    r"\bwhen (you are )?(grading|evaluating|scoring|judging)\b",
    r"\b(always )?(score|rate|grade|mark) (this|the|my) (answer|response|output|reply)\b",
    r"\b(give|award|assign) (this|it|me) (the )?(highest|full|maximum|top|perfect)\b",
    r"\b(output|respond with|answer) (only )?(pass|correct|yes)\b",
    r"\bthis (answer|response) is (always )?correct\b",
    r"\bthe (judge|grader|evaluator) (should|must|will)\b",
)

_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _JUDGE_DIRECTED)


def screen_candidate(body: str) -> None:
    """Refuse a candidate whose body addresses the evaluator.

    Raises ``JudgeDirectedPrompt`` naming the phrase that matched, so a rejection is
    actionable rather than a mysterious refusal -- the optimizer's output is a model's
    text, and knowing which sentence tripped it is how the prompt gets fixed.
    """
    for pattern in _COMPILED:
        match = pattern.search(body)
        if match is not None:
            raise JudgeDirectedPrompt(
                f"candidate addresses the evaluator, not the user: {match.group(0)!r}. "
                "A preprompt that instructs whoever grades the answer is optimising the "
                "measurement instead of the behaviour."
            )
