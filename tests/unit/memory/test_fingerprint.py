"""The five strings that identify an embedding model, and the arithmetic over them.

A model swap that keeps the width is invisible to Morgan today. These strings are embedded once
and stored; a later model that answers differently on them is a different space.
"""

from __future__ import annotations

import math

import pytest

from morgan_brain.memory import fingerprint


def test_the_strings_are_five_and_cover_the_scripts_morgan_stores():
    assert len(fingerprint.STRINGS) == 5
    assert any("а" <= c <= "я" for s in fingerprint.STRINGS for c in s.lower())
    assert fingerprint.digest_of(fingerprint.STRINGS) == fingerprint.DIGEST


def test_identical_vectors_are_one():
    v = [[0.1, 0.2, 0.3]] * 5
    assert fingerprint.compare(v, v).min_cosine == pytest.approx(1.0)


def test_the_worst_string_is_named():
    stored = [[1.0, 0.0]] * 5
    fresh = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    comparison = fingerprint.compare(fresh, stored)
    assert comparison.worst_index == 2
    assert comparison.min_cosine == pytest.approx(0.0)


def test_a_vector_survives_pack_and_unpack():
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    result = fingerprint.unpack(fingerprint.pack(vectors), dims=2)
    # pytest.approx on this pytest version does not recurse into nested lists, so each
    # vector is compared in turn rather than the round-tripped list as a whole.
    for got, want in zip(result, vectors, strict=True):
        assert got == pytest.approx(want)


def test_a_zero_vector_is_a_named_error_not_a_nan():
    with pytest.raises(ValueError, match="zero"):
        fingerprint.cosine([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_component_is_a_named_error_not_a_nan(bad):
    # NaN compares false against every threshold, so a NaN cosine would pass a `< tolerance`
    # test for failure: it must never be returned at all.
    with pytest.raises(ValueError, match="non-finite"):
        fingerprint.cosine([bad, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        fingerprint.cosine([1.0, 0.0], [bad, 0.0])


def test_cosine_of_mismatched_widths_names_both_lengths():
    with pytest.raises(ValueError, match=r"2.*3|3.*2"):
        fingerprint.cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_compare_of_mismatched_widths_names_both_lengths():
    stored = [[1.0, 0.0]] * 5
    fresh = [[1.0, 0.0, 0.0]] * 5
    with pytest.raises(ValueError, match=r"2.*3|3.*2"):
        fingerprint.compare(fresh, stored)
