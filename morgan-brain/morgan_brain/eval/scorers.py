"""L1 deterministic scorers — no judge, no network.

All functions are pure: given retrieved ids + a relevant set they return a float
in [0, 1].  The formula for recall@k is:

    |relevant ∩ top_k| / min(k, |relevant|)

which normalises against the smaller of (k, |relevant|) so that asking for k > |relevant|
does not artificially cap recall at a value below 1.

cohen_kappa implements the standard 2×2 binary-classification formula so tests
can verify judge calibration without any external library.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Retrieval scorers (L1)
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-*k* retrieved results.

    Args:
        retrieved: Ordered list of retrieved document ids.
        relevant:  Set of ground-truth relevant ids.
        k:         Cut-off depth.

    Returns:
        |relevant ∩ top_k| / min(k, |relevant|), or 0.0 if k==0 or relevant is empty.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    denominator = min(k, len(relevant))
    return hits / denominator


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k.

    Args:
        retrieved: Ordered list of retrieved document ids.
        relevant:  Set of ground-truth relevant ids.
        k:         Cut-off depth.

    Returns:
        F1 score in [0, 1]; 0.0 when either precision or recall is 0.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant)
    if hits == 0:
        return 0.0
    precision = hits / len(top_k)
    recall = hits / min(k, len(relevant))
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Cohen's kappa (binary) — judge calibration
# ---------------------------------------------------------------------------


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Binary Cohen's kappa between two raters *a* and *b*.

    Implements the standard 2×2 formula:

        kappa = (p_o - p_e) / (1 - p_e)

    where ``p_o`` is observed agreement and ``p_e`` is chance-expected agreement.

    Args:
        a: List of boolean labels from rater A (e.g. the human).
        b: List of boolean labels from rater B (e.g. the judge).

    Returns:
        Kappa in [-1, 1]; 1.0 = perfect agreement, 0 ≈ chance, <0 = worse than chance.
        Returns 0.0 for empty inputs.

    Raises:
        ValueError: If *a* and *b* have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have the same length ({len(a)} vs {len(b)})")
    n = len(a)
    if n == 0:
        return 0.0

    # Build 2×2 contingency table. The four predicates partition every (x, y) pair, and the
    # lengths were checked above, so tp + tn + fp + fn == n by construction -- there is no
    # invariant left for a runtime check to catch.
    # TP: both True, TN: both False, FP: a=False b=True, FN: a=True b=False
    tp = sum(1 for x, y in zip(a, b) if x and y)
    tn = sum(1 for x, y in zip(a, b) if not x and not y)
    fp = sum(1 for x, y in zip(a, b) if not x and y)
    fn = sum(1 for x, y in zip(a, b) if x and not y)

    # Marginals
    a_pos = tp + fn  # a says True
    a_neg = fp + tn  # a says False
    b_pos = tp + fp  # b says True
    b_neg = fn + tn  # b says False

    p_o = (tp + tn) / n
    p_e = (a_pos * b_pos + a_neg * b_neg) / (n * n)

    if abs(1.0 - p_e) < 1e-12:
        # Edge case: p_e == 1 means perfect systematic (dis)agreement with no variance.
        return 1.0 if abs(p_o - p_e) < 1e-12 else 0.0

    return (p_o - p_e) / (1.0 - p_e)
