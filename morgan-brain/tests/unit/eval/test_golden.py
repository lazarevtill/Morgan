"""Tests for golden set loading and probe-type coverage."""

from __future__ import annotations

from morgan_brain.eval.golden import GoldenItem, ProbeType, load_golden_set

# Path to the golden set — relative to the repo root but resolved at runtime.
import pathlib

_GOLDEN_PATH = pathlib.Path(__file__).parent.parent.parent / "eval" / "golden_set.json"


class TestLoadGoldenSet:
    def test_file_exists(self) -> None:
        assert _GOLDEN_PATH.exists(), f"golden_set.json not found at {_GOLDEN_PATH}"

    def test_loads_non_empty(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        assert len(items) >= 12, f"Expected ≥12 items, got {len(items)}"

    def test_all_items_validate(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        for item in items:
            assert isinstance(item, GoldenItem)
            assert item.id
            assert item.query
            assert item.expected

    def test_all_six_probe_types_present(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        found = {item.probe for item in items}
        missing = set(ProbeType) - found
        assert not missing, f"Missing probe types in golden set: {missing}"

    def test_probe_type_distribution(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        counts: dict[ProbeType, int] = {}
        for item in items:
            counts[item.probe] = counts.get(item.probe, 0) + 1
        for pt in ProbeType:
            assert counts.get(pt, 0) >= 1, f"Probe type {pt} has no items"

    def test_over_personalization_items_have_should_inject_false(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        op_items = [i for i in items if i.probe == ProbeType.OVER_PERSONALIZATION_NEGATIVE]
        assert op_items, "No OVER_PERSONALIZATION_NEGATIVE items found"
        for item in op_items:
            assert item.should_inject is False, (
                f"Item {item.id!r}: OVER_PERSONALIZATION_NEGATIVE must have should_inject=False"
            )

    def test_non_negative_items_default_should_inject_true(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        for item in items:
            if item.probe != ProbeType.OVER_PERSONALIZATION_NEGATIVE:
                assert item.should_inject is True, (
                    f"Item {item.id!r} ({item.probe}): expected should_inject=True"
                )

    def test_ids_are_unique(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids)), "Duplicate IDs in golden set"

    def test_setup_is_list_of_strings(self) -> None:
        items = load_golden_set(_GOLDEN_PATH)
        for item in items:
            assert isinstance(item.setup, list)
            for s in item.setup:
                assert isinstance(s, str)


class TestGoldenItemModel:
    def test_default_should_inject(self) -> None:
        item = GoldenItem(
            id="test1",
            probe=ProbeType.EXPLICIT_RECALL,
            setup=["fact A"],
            query="what is A?",
            expected="A",
        )
        assert item.should_inject is True

    def test_negative_probe_explicit_false(self) -> None:
        item = GoldenItem(
            id="neg1",
            probe=ProbeType.OVER_PERSONALIZATION_NEGATIVE,
            setup=["old pref: verbose"],
            query="explain sorting",
            expected="[brief explanation without verbose pref applied]",
            should_inject=False,
        )
        assert item.should_inject is False
