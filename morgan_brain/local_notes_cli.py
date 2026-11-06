"""Explicit account-operated entrypoint for bounded local Markdown delivery."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from morgan_brain.local_notes import (
    ACTION,
    LocalRun,
    NoteWorkspace,
    RunLimits,
    approve,
    enqueue,
    local_status,
    revoke,
    run_local,
)
from morgan_brain.tasks import TaskKey, TaskRequest


async def _run_with_stop_signal(workspace: NoteWorkspace, limits: RunLimits) -> LocalRun:
    """Translate SIGTERM into persistent cooperative STOP for this explicit invocation."""
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, workspace.stop)
    try:
        return await run_local(workspace, limits)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    """Explicitly approve/enqueue notes and operate the finite local worker lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("approve", "replace all grants with one exact note"),
        ("enqueue", "append one exact approved note without losing queued work"),
    ):
        approval = commands.add_parser(command, help=help_text)
        approval.add_argument("--user", required=True)
        approval.add_argument("--project", required=True)
        approval.add_argument("--task", required=True)
        approval.add_argument("--note", required=True, type=Path)
        approval.add_argument("--ttl", type=int, default=3600)
        approval.add_argument(
            "--yes", action="store_true", help="explicit trusted automation approval"
        )
    run = commands.add_parser("run")
    run.add_argument("--actions", type=int, default=1)
    run.add_argument("--cycles", type=int, default=1)
    run.add_argument("--seconds", type=float, default=10)
    run.add_argument("--poll", type=float, default=1)
    commands.add_parser("stop")
    commands.add_parser("resume")
    commands.add_parser("status")
    commands.add_parser("revoke")
    args = parser.parse_args()
    workspace = NoteWorkspace(args.workspace)
    try:
        if args.command in ("approve", "enqueue"):
            with args.note.open("rb") as source:
                text = source.read(16385).decode("utf-8")
            request = TaskRequest(TaskKey(args.user, args.project, args.task), ACTION, text)
            print(json.dumps(asdict(request), ensure_ascii=False), file=sys.stderr)
            if not args.yes:
                print("Approve this exact note? Type APPROVE: ", end="", file=sys.stderr)
                if input() != "APPROVE":
                    parser.exit(1, "Not approved.\n")
            if args.command == "enqueue":
                enqueue(workspace, request, ttl_seconds=args.ttl)
            else:
                approve(workspace, [request], ttl_seconds=args.ttl)
            print(json.dumps({"approved": True}))
        elif args.command == "run":
            limits = RunLimits(args.actions, args.cycles, args.seconds, args.poll)
            result = asyncio.run(_run_with_stop_signal(workspace, limits))
            print(json.dumps(asdict(result)))
            if result.reason not in ("cycle_limit", "action_limit"):
                raise SystemExit(2)
        elif args.command == "stop":
            workspace.stop()
            print(json.dumps({"stopped": True}))
        elif args.command == "status":
            print(json.dumps(local_status(workspace)))
        elif args.command == "revoke":
            revoke(workspace)
            print(json.dumps({"revoked": True}))
        else:
            workspace.resume()
            print(json.dumps({"resumed": True}))
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
