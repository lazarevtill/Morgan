"""Calibration metrics for the eval gate (Brier score, ECE, reliability bins).

Calibration answers a question accuracy cannot: *does the system know when it is right?* Given
per-item pairs ``(confidence, correct)`` it scores how well the stated confidence matches the
empirical hit-rate. Pure functions — deterministic, no LLM, no I/O.

* ``brier_score``            — mean squared error of confidence vs outcome (0 best, 1 worst).
* ``expected_calibration_error`` — average |accuracy − confidence| across confidence bins.
* ``reliability_bins``       — the reliability-diagram rows (per-bin confidence/accuracy/count).

Used in **report-only** mode first: the harness computes and surfaces these; promotion is NOT
gated on them until the signal is proven (see the calibration design spec).
"""

from __future__ import annotations

from dataclasses import dataclass

# A single observation: (confidence in [0, 1], was the answer judged correct).
Pair = tuple[float, bool]


@dataclass(frozen=True)
class ReliabilityBin:
    """One row of the reliability diagram for a confidence interval ``[lo, hi)``."""

    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "count": float(self.count),
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }


def _y(correct: bool) -> float:
    return 1.0 if correct else 0.0


def brier_score(pairs: list[Pair]) -> float:
    """Mean squared error between confidence and outcome. 0 = perfect, 1 = confidently wrong.

    Empty input returns 0.0 (no evidence — neutral, never penalises).
    """
    if not pairs:
        return 0.0
    return sum((p - _y(y)) ** 2 for p, y in pairs) / len(pairs)


def _bin_index(p: float, n_bins: int) -> int:
    """Map a confidence to its equal-width bin index in ``[0, n_bins)``."""
    idx = int(p * n_bins)
    return min(n_bins - 1, max(0, idx))  # p == 1.0 lands in the last bin


def reliability_bins(pairs: list[Pair], *, n_bins: int = 10) -> list[ReliabilityBin]:
    """Partition *pairs* into ``n_bins`` equal-width confidence bins over ``[0, 1]``.

    Returns one :class:`ReliabilityBin` per bin (empty bins included, with count 0), so the
    diagram has a stable shape and low-evidence bins are visible.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[Pair]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        buckets[_bin_index(p, n_bins)].append((p, y))

    bins: list[ReliabilityBin] = []
    for i, bucket in enumerate(buckets):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            bins.append(ReliabilityBin(lo=lo, hi=hi, count=0, mean_confidence=0.0, accuracy=0.0))
            continue
        mean_conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(_y(y) for _, y in bucket) / len(bucket)
        bins.append(
            ReliabilityBin(lo=lo, hi=hi, count=len(bucket), mean_confidence=mean_conf, accuracy=acc)
        )
    return bins


def expected_calibration_error(pairs: list[Pair], *, n_bins: int = 10) -> float:
    """Weighted average gap between accuracy and confidence across bins (0 = perfectly calibrated).

    ``ECE = Σ_b (n_b / N) · |acc_b − conf_b|``. Empty input returns 0.0.
    """
    if not pairs:
        return 0.0
    total = len(pairs)
    return sum(
        (b.count / total) * abs(b.accuracy - b.mean_confidence)
        for b in reliability_bins(pairs, n_bins=n_bins)
        if b.count
    )
