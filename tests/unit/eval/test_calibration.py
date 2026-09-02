"""Calibration metrics — verified against hand-computed values."""

from __future__ import annotations

import pytest

from morgan_brain.eval.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
)

# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


def test_brier_perfect_confident_correct() -> None:
    assert brier_score([(1.0, True), (1.0, True)]) == pytest.approx(0.0)


def test_brier_confidently_wrong_is_worst() -> None:
    assert brier_score([(1.0, False), (1.0, False)]) == pytest.approx(1.0)


def test_brier_half_confidence_is_quarter() -> None:
    # (0.5 - y)^2 == 0.25 for y in {0, 1}, so any outcomes give 0.25.
    assert brier_score([(0.5, True), (0.5, False)]) == pytest.approx(0.25)


def test_brier_empty_is_zero() -> None:
    assert brier_score([]) == 0.0


def test_brier_is_bounded() -> None:
    pairs = [(0.2, True), (0.9, False), (0.5, True), (0.0, False), (1.0, True)]
    assert 0.0 <= brier_score(pairs) <= 1.0


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_at_half() -> None:
    # All at conf 0.5 with exactly 50% correct → accuracy == confidence in that bin → ECE 0.
    pairs = [(0.5, True), (0.5, False)]
    assert expected_calibration_error(pairs) == pytest.approx(0.0)


def test_ece_overconfident() -> None:
    # Always 0.5 confidence but always right → bin accuracy 1.0 vs conf 0.5 → ECE 0.5.
    pairs = [(0.5, True), (0.5, True), (0.5, True)]
    assert expected_calibration_error(pairs) == pytest.approx(0.5)


def test_ece_confident_and_correct_is_zero() -> None:
    assert expected_calibration_error([(1.0, True), (1.0, True)]) == pytest.approx(0.0)


def test_ece_confidently_wrong_is_one() -> None:
    assert expected_calibration_error([(1.0, False), (1.0, False)]) == pytest.approx(1.0)


def test_ece_empty_is_zero() -> None:
    assert expected_calibration_error([]) == 0.0


# ---------------------------------------------------------------------------
# reliability_bins
# ---------------------------------------------------------------------------


def test_reliability_bins_partition_and_shape() -> None:
    bins = reliability_bins([(0.05, True), (0.95, False), (0.95, True)], n_bins=10)
    assert len(bins) == 10  # stable shape, empty bins included
    first = bins[0]  # [0.0, 0.1)
    assert first.count == 1 and first.accuracy == pytest.approx(1.0)
    last = bins[9]  # [0.9, 1.0]
    assert last.count == 2
    assert last.mean_confidence == pytest.approx(0.95)
    assert last.accuracy == pytest.approx(0.5)


def test_reliability_confidence_one_lands_in_last_bin() -> None:
    bins = reliability_bins([(1.0, True)], n_bins=10)
    assert bins[9].count == 1


def test_reliability_rejects_zero_bins() -> None:
    with pytest.raises(ValueError):
        reliability_bins([(0.5, True)], n_bins=0)


def test_reliability_as_dict_is_serialisable() -> None:
    row = reliability_bins([(0.5, True)], n_bins=2)[1].as_dict()
    assert set(row) == {"lo", "hi", "count", "mean_confidence", "accuracy"}
    assert all(isinstance(v, float) for v in row.values())
