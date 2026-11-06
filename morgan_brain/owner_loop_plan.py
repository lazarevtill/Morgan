"""Render only a disabled login-agent plan for a finite, explicitly selected local owner loop."""

from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path


def render(python: Path, workspace: Path, server: Path, model: Path) -> bytes:
    """Fixed command and private logs; no installation, credentials, shell or expiry extension."""
    for path in (python, workspace, server, model):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("select absolute paths without parent traversal")
    return plistlib.dumps(
        {
            "Label": "org.morgan.owner-pilot",
            "Disabled": True,
            "ProgramArguments": [
                str(python),
                "-I",
                "-m",
                "morgan_brain.owner_loop",
                str(workspace),
                "--server",
                str(server),
                "--model",
                str(model),
                "--login-agent",
                "--seconds",
                "604800",
            ],
            "WorkingDirectory": str(workspace),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 60,
            "ExitTimeOut": 10,
            "ProcessType": "Background",
            "Umask": 0o077,
            "StandardOutPath": str(workspace / "loop.stdout.log"),
            "StandardErrorPath": str(workspace / "loop.stderr.log"),
        },
        sort_keys=True,
    )


def main() -> None:
    """Print a plan for review; deliberately never write LaunchAgents or invoke launchctl."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "workspace", "server", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.buffer.write(render(args.python, args.workspace, args.server, args.model))


if __name__ == "__main__":
    main()
