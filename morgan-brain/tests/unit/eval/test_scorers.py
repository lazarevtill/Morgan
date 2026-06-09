"""Tests for L1 deterministic scorers (recall@k, f1@k) and Cohen's kappa."""

from __future__ import annotations

import pytest

from morgan_brain.eval.scorers import cohen_kappa, f1_at_k, recall_at_k


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall_all_relevant_in_top_k(self) -> None:
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_zero_recall_no_relevant_in_top_k(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}
        # top-2: only "a" found → 1/min(2,3) = 1/2
        result = recall_at_k(retrieved, relevant, k=2)
        assert abs(result - 1 / 2) < 1e-9

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # top-5 but only 2 items; 2 found / min(5,3)=3 → 2/3
        result = recall_at_k(retrieved, relevant, k=5)
        assert abs(result - 2 / 3) < 1e-9

    def test_empty_retrieved(self) -> None:
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        # Guard empty: if relevant is empty, nothing to recall → 0.0
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, k=0) == 0.0

    def test_all_relevant_but_k_limits_window(self) -> None:
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        # top-1 only has "x" → 0/min(1,2)=0
        assert recall_at_k(retrieved, relevant, k=1) == 0.0

    def test_duplicates_in_retrieved_not_double_counted(self) -> None:
        # "a" appears twice but counts as 1 hit
        retrieved = ["a", "a", "b"]
        relevant = {"a", "b"}
        result = recall_at_k(retrieved, relevant, k=3)
        assert abs(result - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# f1_at_k
# ---------------------------------------------------------------------------


class TestF1AtK:
    def test_perfect_f1(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert f1_at_k(retrieved, relevant, k=2) == 1.0

    def test_zero_f1(self) -> None:
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert f1_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_f1(self) -> None:
        retrieved = ["a", "x"]
        relevant = {"a", "b"}
        # precision@2 = 1/2, recall@2 = 1/min(2,2) = 1/2 → f1 = 1/2
        result = f1_at_k(retrieved, relevant, k=2)
        assert abs(result - 0.5) < 1e-9

    def test_empty_retrieved_zero_f1(self) -> None:
        assert f1_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant_zero_f1(self) -> None:
        assert f1_at_k(["a"], set(), k=1) == 0.0

    def test_k_zero_zero_f1(self) -> None:
        assert f1_at_k(["a"], {"a"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_perfect_agreement_all_true(self) -> None:
        a = [True, True, True, True]
        b = [True, True, True, True]
        assert abs(cohen_kappa(a, b) - 1.0) < 1e-9

    def test_perfect_agreement_all_false(self) -> None:
        a = [False, False, False]
        b = [False, False, False]
        assert abs(cohen_kappa(a, b) - 1.0) < 1e-9

    def test_perfect_agreement_mixed(self) -> None:
        a = [True, False, True, False]
        b = [True, False, True, False]
        assert abs(cohen_kappa(a, b) - 1.0) < 1e-9

    def test_complete_disagreement(self) -> None:
        a = [True, True, False, False]
        b = [False, False, True, True]
        kappa = cohen_kappa(a, b)
        # kappa <= 0 for complete flip
        assert kappa <= 0.0

    def test_chance_agreement_approx_zero(self) -> None:
        # 50-50 split, perfectly uncorrelated → kappa ≈ 0
        a = [True, True, False, False]
        b = [True, False, True, False]
        kappa = cohen_kappa(a, b)
        assert abs(kappa) < 0.1

    def test_good_agreement_above_threshold(self) -> None:
        # 8/10 agree, balanced classes → kappa should be > 0.6
        a = [True, True, True, True, True, False, False, False, False, False]
        b = [True, True, True, True, False, False, False, False, False, True]
        kappa = cohen_kappa(a, b)
        # 8 agree / 2 disagree; both balanced → kappa > 0.5
        assert kappa > 0.5

    def test_empty_lists_returns_zero(self) -> None:
        assert cohen_kappa([], []) == 0.0

    def test_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cohen_kappa([True], [True, False])

    def test_kappa_2x2_manual(self) -> None:
        # Manual calculation:
        # a=[T,T,F,F] b=[T,F,F,T]
        # TP=1 FP=1 FN=1 TN=1; n=4
        # po = (TP+TN)/n = 2/4 = 0.5
        # pyes_a=0.5, pyes_b=0.5 → pe=(0.5*0.5)+(0.5*0.5)=0.5
        # kappa=(0.5-0.5)/(1-0.5)=0
        a = [True, True, False, False]
        b = [True, False, False, True]
        kappa = cohen_kappa(a, b)
        assert abs(kappa - 0.0) < 1e-9
