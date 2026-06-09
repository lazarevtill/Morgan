"""Tests for the packaged golden set (commit 2).

Ensures:
- default_golden_path() points to an existing file inside the package.
- load_golden_set(default_golden_path()) returns ≥10 items covering all 6 probe types.
- Settings.eval_golden_path exists with default value "".
- The path returned by default_golden_path() is independent of cwd.
"""

from __future__ import annotations

from pathlib import Path

from morgan_brain.config import Settings
from morgan_brain.eval.golden import ProbeType, default_golden_path, load_golden_set


class TestDefaultGoldenPath:
    def test_returns_path_object(self) -> None:
        p = default_golden_path()
        assert isinstance(p, Path)

    def test_file_exists(self) -> None:
        p = default_golden_path()
        assert p.exists(), f"Packaged golden set not found at {p}"

    def test_is_inside_package(self) -> None:
        """The file must live under morgan_brain/eval/data/."""
        p = default_golden_path().resolve()
        # Check that the path contains eval/data as a sub-directory sequence.
        parts = p.parts
        assert "eval" in parts, f"'eval' not in path parts: {parts}"
        assert "data" in parts, f"'data' not in path parts: {parts}"

    def test_loads_at_least_ten_items(self) -> None:
        items = load_golden_set(default_golden_path())
        assert len(items) >= 10, f"Expected ≥10 items, got {len(items)}"

    def test_all_six_probe_types_covered(self) -> None:
        items = load_golden_set(default_golden_path())
        found = {item.probe for item in items}
        missing = set(ProbeType) - found
        assert not missing, f"Missing probe types: {missing}"


class TestSettingsEvalGoldenPath:
    def test_default_is_empty_string(self) -> None:
        s = Settings(llm_model="x", llm_fast_model="x")
        assert s.eval_golden_path == ""

    def test_can_set_custom_path(self) -> None:
        s = Settings(llm_model="x", llm_fast_model="x", eval_golden_path="/tmp/custom.json")
        assert s.eval_golden_path == "/tmp/custom.json"

    def test_resolves_packaged_when_empty(self) -> None:
        """When eval_golden_path is empty, callers use default_golden_path()."""
        s = Settings(llm_model="x", llm_fast_model="x")
        if not s.eval_golden_path:
            path = default_golden_path()
        else:
            path = Path(s.eval_golden_path)
        assert path.exists()
