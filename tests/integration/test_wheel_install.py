"""A built wheel must carry the package and both console scripts, and nothing else."""

import subprocess
import sys
import zipfile
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_the_package_and_the_entry_points(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path), str(PKG_ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = zipfile.ZipFile(next(tmp_path.glob("morgan_brain-*.whl")))
    names = set(wheel.namelist())

    assert "morgan_brain/cli/__main__.py" in names
    assert "morgan_brain/mcp_server.py" in names
    assert not any(n.startswith("tests/") for n in names)
    entry_points = next(n for n in names if n.endswith("entry_points.txt"))
    scripts = wheel.read(entry_points).decode()
    assert "morgan = morgan_brain.cli.__main__:main" in scripts
    assert "morgan-mcp = morgan_brain.mcp_server:main" in scripts
