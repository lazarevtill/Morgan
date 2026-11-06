"""Render an uninstalled, disabled launchd plan for the existing bounded note worker."""

from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path


def render(python: Path, workspace: Path) -> bytes:
    """No installation, grants, shell, environment credentials or arbitrary command fields."""
    for path in (python, workspace):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("launch agent paths must be absolute without parent traversal")
    return plistlib.dumps(
        {
            "Label": "org.morgan.approved-notes",
            "Disabled": True,
            "ProgramArguments": [
                str(python),
                "-m",
                "morgan_brain.local_notes_cli",
                str(workspace),
                "run",
                "--actions",
                "1",
                "--cycles",
                "1",
                "--seconds",
                "10",
            ],
            "WorkingDirectory": str(workspace),
            "RunAtLoad": True,
            "KeepAlive": False,
            "StartCalendarInterval": [{"Minute": minute} for minute in (0, 15, 30, 45)],
            "ThrottleInterval": 60,
            "ExitTimeOut": 5,
            "ProcessType": "Background",
            "Umask": 0o077,
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "/dev/null",
        },
        sort_keys=True,
    )


def main() -> None:
    """Print a reviewable plist; deliberately never call launchctl or create a service file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    sys.stdout.buffer.write(render(args.python, args.workspace))


if __name__ == "__main__":
    main()
