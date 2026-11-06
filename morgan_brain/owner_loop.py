"""Opt-in finite background drafting for a local owner inbox; never confirm or execute tasks."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import time
from contextlib import closing, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle, proposal_filename, run_cycle
from morgan_brain.owner_input import load_pilot
from morgan_brain.owner_model import OwnerModel
from morgan_brain.providers.note_probe import ProbeFailure

_STATE = "owner-loop.json"
_MAX_LAUNCHES = 16


def read_state(workspace: NoteWorkspace) -> dict[str, object]:
    """Read last observed state, not a claim that a process or heartbeat is still alive."""
    try:
        value = json.loads(workspace.read(_STATE, 8192))
    except FileNotFoundError:
        return {"version": 1, "launches": 0, "state": "never_started"}
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("launches"), int)
        or isinstance(value.get("launches"), bool)
        or not 0 <= value["launches"] <= _MAX_LAUNCHES
    ):
        raise ValueError("invalid loop state; preserve it for manual recovery")
    return value


class OwnerLoop:
    """One shared model lane, bounded admission lifetime and durable automatic-attempt markers."""

    def __init__(self, path: Path, config: ModelCycle, *, seconds: int, poll: float = 1) -> None:
        """Require explicit local configuration; do not initialize or extend an owner pilot."""
        if not 5 <= seconds <= 604800 or not 0.05 <= poll <= 60:
            raise ValueError("select 5..604800 seconds and 0.05..60 second polling")
        self.path, self.config, self.seconds, self.poll = path, config, seconds, poll
        self.shutdown = asyncio.Event()
        self.task: asyncio.Task[dict[str, object]] | None = None
        self.active_message: str | None = None
        self.deadline = time.monotonic() + seconds
        self.last_status = ""
        self.last_write = 0.0
        self.launches = 0

    def record(self, workspace: NoteWorkspace, state: str, expires: str) -> None:
        """Publish status changes and a 30-second heartbeat without raw owner text."""
        key = f"{state}:{self.active_message}"
        if key == self.last_status and time.monotonic() - self.last_write < 30:
            return
        payload = {
            "version": 1,
            "launches": self.launches,
            "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
            "expires_at": expires,
            "active_message": self.active_message,
            "automatic_task_confirmation": False,
        }
        workspace.publish(_STATE, json.dumps(payload), replace=True)
        self.last_status, self.last_write = key, time.monotonic()

    async def infer(self, note: str, seconds: int) -> dict[str, object]:
        """Cancel only this owned cycle; the existing guardian cleans up without clearing STOP."""
        with closing(NoteWorkspace(self.path, create=False)) as workspace:
            return await run_cycle(
                workspace,
                replace(self.config, seconds=seconds, delay=0),
                note,
                cache_by_source=True,
            )

    def admit(self, workspace: NoteWorkspace, remaining: float) -> None:
        """Start at most one previously unattempted, unreviewed owner message; never retry."""
        if self.task is not None or remaining < 5:
            return
        _, inbox = load_pilot(workspace)
        view = OwnerModel(self.path, self.config, self.deadline)
        for message in inbox.messages:
            if (
                message.state != "needs_review"
                or view.status(message.text)["state"] != "not_started"
            ):
                continue
            marker = "attempt-" + proposal_filename(message.text)
            try:
                workspace.publish(marker, "Explicit background pilot admission.\n")
            except FileExistsError:
                continue
            self.active_message = message.identity
            self.task = asyncio.create_task(
                self.infer(message.text, min(self.config.seconds, int(remaining)))
            )
            return

    def reap(self) -> None:
        """Observe success or failure once; a failed attempt remains blocked by its marker."""
        if self.task is not None and self.task.done():
            with suppress(OSError, ValueError, TypeError, ProbeFailure, StopRequested):
                self.task.result()
            self.task, self.active_message = None, None

    async def run(self) -> str:
        """Keep the loop lock through cancellation; expiry/STOP/signals do not reset admission."""
        with (
            closing(NoteWorkspace(self.path, create=False)) as workspace,
            os.fdopen(
                workspace.open_file(".owner-loop.lock", os.O_RDWR | os.O_CREAT), "r+"
            ) as lock,
        ):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            launches = read_state(workspace)["launches"]
            if not isinstance(launches, int):
                raise TypeError("invalid launch counter")
            self.launches = launches
            _, inbox = load_pilot(workspace)
            expires = inbox.expires_at.isoformat()
            if self.launches >= _MAX_LAUNCHES:
                self.record(workspace, "launch_limit", expires)
                return "launch_limit"
            self.launches += 1
            self.record(workspace, "starting", expires)
            reason = "shutdown"
            try:
                while not self.shutdown.is_set():
                    workspace.check_stop()
                    _, inbox = load_pilot(workspace)
                    remaining = min(
                        self.deadline - time.monotonic(),
                        (inbox.expires_at - datetime.now(UTC)).total_seconds(),
                    )
                    if remaining <= 0:
                        reason = "expired" if datetime.now(UTC) >= inbox.expires_at else "finished"
                        break
                    self.reap()
                    self.admit(workspace, remaining)
                    self.record(
                        workspace, "processing" if self.task else "waiting_for_owner", expires
                    )
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            self.shutdown.wait(), timeout=min(self.poll, remaining)
                        )
            except StopRequested:
                reason = "stopped"
            except (OSError, ValueError, TypeError, ProbeFailure):
                reason = "error_requires_review"
            finally:
                if self.task is not None:
                    self.task.cancel()
                    with suppress(
                        asyncio.CancelledError,
                        OSError,
                        ValueError,
                        TypeError,
                        ProbeFailure,
                        StopRequested,
                    ):
                        await self.task
                    self.task, self.active_message = None, None
                self.record(workspace, reason, expires)
            return reason


async def _serve(controller: OwnerLoop) -> str:
    """Translate process shutdown into cancellation of the owned operation, not owner resume."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, controller.shutdown.set)
    try:
        return await controller.run()
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main() -> None:
    """Explicitly run or inspect a finite headless loop; never install a launch agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--server", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--poll", type=float, default=1)
    args = parser.parse_args()
    try:
        if args.status:
            with closing(NoteWorkspace(args.workspace, create=False)) as workspace:
                print(
                    json.dumps(
                        {"last_observed": read_state(workspace), "live_process_verified": False}
                    )
                )
            return
        if args.server is None or args.model is None:
            parser.error("select explicit --server and --model")
        controller = OwnerLoop(
            args.workspace,
            ModelCycle(args.server, args.model),
            seconds=args.seconds,
            poll=args.poll,
        )
        print(json.dumps({"reason": asyncio.run(_serve(controller))}))
    except (OSError, ValueError, TypeError) as error:
        # Expected invalid configuration is not an invitation for launchd to retry endlessly.
        print(json.dumps({"reason": "configuration_requires_review", "error": str(error)}))


if __name__ == "__main__":
    main()
