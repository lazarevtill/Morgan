"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

import subprocess
from pathlib import Path

from morgan_brain.models.memory import DEFAULT_PROJECT


def detect_project(cwd: Path | None = None) -> str:
    """Return the git repository's directory name, or DEFAULT_PROJECT outside a repo."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DEFAULT_PROJECT
    return Path(root).name or DEFAULT_PROJECT
