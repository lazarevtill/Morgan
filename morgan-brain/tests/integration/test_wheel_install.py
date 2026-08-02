"""A built wheel must carry its packaged data files."""

import subprocess
import sys
import zipfile
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_packaged_data(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path), str(PKG_ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = next(tmp_path.glob("morgan_brain-*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())

    assert "morgan_brain/eval/data/golden_set.json" in names
    assert any(n.startswith("morgan_brain/providers/data/") for n in names), sorted(
        n for n in names if "providers" in n
    )
