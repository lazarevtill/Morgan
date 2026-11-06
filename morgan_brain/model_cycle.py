"""Finite local inference ownership, with a private socket and an independent expiry guardian."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.local_supervisor import _shutdown
from morgan_brain.providers.note_probe import ProbeFailure, private_model_ready, propose_private


@dataclass(frozen=True)
class ModelCycle:
    """Explicit trusted executable/model paths and finite CPU-oriented limits."""

    server: Path
    model: Path
    seconds: int = 90
    delay: int = 0
    threads: int = 4

    def __post_init__(self) -> None:
        """Reject missing files, unbounded schedules and excessive weight/CPU requests."""
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (self.seconds, self.delay, self.threads)
        ):
            raise ValueError("limits require integers")
        if not 5 <= self.seconds <= 120 or not 0 <= self.delay <= 60 or self.delay >= self.seconds:
            raise ValueError("cycle requires 5..120 seconds and delay below its deadline")
        if not 1 <= self.threads <= 4:
            raise ValueError("threads require 1..4")
        if (
            not self.server.is_absolute()
            or not self.server.is_file()
            or not os.access(self.server, os.X_OK)
        ):
            raise ValueError("select an existing trusted server executable")
        if not self.model.is_absolute() or not self.model.is_file():
            raise ValueError("select an existing local model file")
        if not 0 < self.model.stat().st_size <= 8_000_000_000:
            raise ValueError("model must be at most 8 GB")


@dataclass
class ModelSession:
    """A live private socket; results are proposals, never durable permissions."""

    socket: Path
    workspace: NoteWorkspace
    deadline: float

    async def select(self, note: str) -> tuple[str, ...]:
        """Bound each selection by the remaining cycle budget and honor persistent STOP."""
        self.workspace.check_stop()
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ProbeFailure("deadline")
        try:
            result = await propose_private(self.socket, note, seconds=min(20, remaining))
        except ProbeFailure:
            self.workspace.check_stop()
            raise
        self.workspace.check_stop()
        return result


def _guardian_command(config: ModelCycle, workspace: NoteWorkspace, socket: Path) -> list[str]:
    """Construct a fixed isolated Python child; never run a supplied shell command."""
    return [
        sys.executable,
        "-I",
        "-m",
        "morgan_brain.model_guardian",
        str(workspace.path),
        str(config.server),
        str(config.model),
        str(socket),
        str(config.seconds),
        str(config.delay),
        str(config.threads),
    ]


async def _ready(child: asyncio.subprocess.Process, session: ModelSession) -> None:
    """Wait for authenticated-by-private-directory model identity, STOP, expiry or child failure."""
    while asyncio.get_running_loop().time() < session.deadline:
        session.workspace.check_stop()
        if child.returncode is not None:
            raise ProbeFailure("model_start_failed")
        if await private_model_ready(session.socket):
            return
        await asyncio.sleep(0.1)
    raise ProbeFailure("deadline")


@asynccontextmanager
async def managed_model(
    workspace: NoteWorkspace, config: ModelCycle
) -> AsyncIterator[ModelSession]:
    """Own one finite guardian; never adopt a saved PID or restart an inference request."""
    with workspace.locked(supervisor=True):
        workspace.check_stop()
        # A guardian surviving a caller crash retains this lock until its own expiry.
        with workspace.locked():
            pass
        with tempfile.TemporaryDirectory(prefix="morgan-model-") as temporary:
            socket = Path(temporary) / "model.sock"
            deadline = asyncio.get_running_loop().time() + config.seconds
            session = ModelSession(socket, workspace, deadline)
            pending = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *_guardian_command(config, workspace, socket),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
            child = None
            try:
                child = await asyncio.shield(pending)
                await _ready(child, session)
                yield session
            finally:
                if child is None:
                    child = await pending
                await asyncio.shield(_shutdown(child, 5))


def _validate_note(note: str) -> None:
    """Bound caller-supplied text before launching any model process."""
    if not note.strip() or len(note.encode("utf-8")) > 4096 or len(note.splitlines()) > 32:
        raise ValueError("supply at most 4096 UTF-8 bytes and 32 lines")


def _check_existing(existing: object, digest: str, note: str) -> None:
    """Require a complete matching proposal before returning publication recovery."""
    if not isinstance(existing, dict):
        raise TypeError("existing proposal is malformed")
    expected = {
        "version": "morgan.note-proposal.v1",
        "source_sha256": digest,
        "source": "agent_inferred",
        "approved": False,
        "review_required": True,
    }
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValueError("existing proposal conflicts with this note")
    lines = existing.get("selected_lines")
    if not isinstance(lines, list) or len(lines) > 8:
        raise ValueError("existing selections are malformed")
    if any(not isinstance(line, str) or line not in note.splitlines() for line in lines):
        raise ValueError("existing selections are not grounded")


async def run_cycle(workspace: NoteWorkspace, config: ModelCycle, note: str) -> dict[str, object]:
    """Publish one immutable unapproved draft; a matching prior publication needs no inference."""
    _validate_note(note)
    digest = hashlib.sha256(note.encode("utf-8")).hexdigest()
    try:
        existing = json.loads(workspace.read("proposal.json", 16384))
    except FileNotFoundError:
        existing = None
    if existing is not None:
        _check_existing(existing, digest, note)
        return {"reason": "already_written", "approved": False}
    async with managed_model(workspace, config) as session:
        selected = await session.select(note)
        workspace.publish(
            "proposal.json",
            json.dumps(
                {
                    "version": "morgan.note-proposal.v1",
                    "source_sha256": digest,
                    "source": "agent_inferred",
                    "approved": False,
                    "review_required": True,
                    "selected_lines": selected,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    return {"reason": "written", "proposals": len(selected), "approved": False}


def main() -> None:
    """Run one explicitly scheduled local model cycle from bounded stdin, then exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    workspace = None
    try:
        raw = sys.stdin.buffer.read(4097)
        note = raw.decode("utf-8")
        if len(raw) > 4096 or not note.strip() or len(note.splitlines()) > 32:
            parser.error("supply at most 4096 UTF-8 bytes and 32 lines")
        config = ModelCycle(args.server, args.model, args.seconds, args.delay, args.threads)
        workspace = NoteWorkspace(args.workspace)
        result = asyncio.run(run_cycle(workspace, config, note))
    except (OSError, ValueError, TypeError, ProbeFailure, StopRequested) as error:
        reason = str(error) if isinstance(error, ProbeFailure) else "cycle_failed"
        if isinstance(error, StopRequested):
            reason = "stopped"
        print(json.dumps({"reason": reason, "approved": False}))
        raise SystemExit(1) from None
    finally:
        if workspace is not None:
            with suppress(OSError):
                workspace.close()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
