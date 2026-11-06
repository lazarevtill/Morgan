"""Internal expiry guardian for one trusted local llama-server, independent of its caller."""

from __future__ import annotations

import os
import signal

# Only an explicitly selected trusted executable receives fixed argv; no shell is used.
import subprocess  # nosec B404
import sys
import time
from contextlib import suppress
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle


def _command(config: ModelCycle, socket: Path) -> list[str]:
    """Use a fixed CPU configuration and existing weights; no download or remote flags."""
    return [
        str(config.server),
        "--model",
        str(config.model),
        "--alias",
        "morgan-cycle",
        "--host",
        str(socket),
        "--gpu-layers",
        "0",
        "--ctx-size",
        "1024",
        "--threads",
        str(config.threads),
        "--batch-size",
        "64",
        "--no-warmup",
    ]


def guard(workspace: NoteWorkspace, config: ModelCycle, socket: Path) -> None:
    """Retain the worker lock, stop on expiry/STOP/signals and reap the owned native child."""
    stopped = False

    def request_stop(signum: int, frame: object) -> None:
        """Translate termination signals into sequential child cleanup."""
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    end = time.monotonic() + config.seconds
    launch_at = time.monotonic() + config.delay
    child = None
    with workspace.locked():
        try:
            while not stopped and time.monotonic() < end:
                workspace.check_stop()
                if child is None and time.monotonic() >= launch_at:
                    child = subprocess.Popen(  # noqa: S603 # nosec B603
                        _command(config, socket),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env={
                            "PATH": "/usr/bin:/bin",
                            "HOME": str(socket.parent),
                            "TMPDIR": str(socket.parent),
                        },
                    )
                if child is not None and child.poll() is not None:
                    break
                time.sleep(0.05)
        except StopRequested:
            pass
        finally:
            if child is not None:
                with suppress(ProcessLookupError):
                    child.terminate()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=1)
            with suppress(FileNotFoundError):
                os.unlink(socket)


def main() -> None:
    """Internal positional entrypoint; configuration validates again across the process boundary."""
    root, server, model, socket, seconds, delay, threads = sys.argv[1:]
    config = ModelCycle(Path(server), Path(model), int(seconds), int(delay), int(threads))
    workspace = NoteWorkspace(Path(root))
    try:
        guard(workspace, config, Path(socket))
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
