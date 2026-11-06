"""Explicit account-operated entrypoint for bounded local Markdown delivery."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.local_notes import (
    ACTION,
    LocalRun,
    NoteWorkspace,
    RunLimits,
    approve,
    approve_occurrences,
    enqueue,
    local_status,
    revoke,
    run_local,
)
from morgan_brain.local_recurrence import Occurrence, RepeatLimits, repeat_plan
from morgan_brain.local_supervisor import (
    Supervision,
    SupervisorLimits,
    supervise,
    supervisor_status,
)
from morgan_brain.local_worker_state import read_observation, scoped_counts
from morgan_brain.tasks import TaskKey, TaskRequest


async def _run_with_stop_signal(
    workspace: NoteWorkspace, limits: RunLimits, supervisor: SupervisorLimits | None = None
) -> LocalRun | Supervision:
    """Translate SIGTERM into persistent cooperative STOP for this explicit invocation."""
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, workspace.stop)
    try:
        if supervisor is not None:
            return await supervise(workspace, worker_limits=limits, limits=supervisor)
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
        ("repeat", "replace grants with a finite interval plan for one exact note"),
        ("enqueue", "append one exact approved note without losing queued work"),
    ):
        approval = commands.add_parser(command, help=help_text)
        approval.add_argument("--user", required=True)
        approval.add_argument("--project", required=True)
        approval.add_argument("--task", required=True)
        approval.add_argument("--note", required=True, type=Path)
        if command == "repeat":
            approval.add_argument("--start", type=datetime.fromisoformat, required=True)
            approval.add_argument("--interval", type=int, required=True)
            approval.add_argument("--count", type=int, required=True)
            approval.add_argument("--window", type=int, required=True)
        else:
            approval.add_argument("--ttl", type=int, default=3600)
        if command == "approve":
            approval.add_argument(
                "--not-before",
                type=datetime.fromisoformat,
                help="earliest delivery ISO timestamp with UTC offset; TTL starts now",
            )
        approval.add_argument(
            "--yes", action="store_true", help="explicit trusted automation approval"
        )
    for name in ("run", "supervise"):
        run = commands.add_parser(name)
        run.add_argument(
            "--actions",
            type=int,
            default=1,
            help="per-worker dispatch limit; supervised maximum is starts times actions",
        )
        run.add_argument("--cycles", type=int, default=1)
        run.add_argument("--seconds", type=float, default=10)
        run.add_argument("--poll", type=float, default=1)
        if name == "supervise":
            run.add_argument("--starts", type=int, default=3)
            run.add_argument("--supervisor-seconds", type=float, default=60)
            run.add_argument("--stop-grace", type=float, default=2)
            run.add_argument("--restart-delay", type=float, default=0.1)
    commands.add_parser("stop")
    commands.add_parser("resume")
    commands.add_parser("status")
    commands.add_parser("revoke")
    diagnosis = commands.add_parser("diagnose")
    diagnosis.add_argument("--user", required=True)
    diagnosis.add_argument("--project", required=True)
    args = parser.parse_args()
    workspace = NoteWorkspace(args.workspace)
    try:
        if args.command in ("approve", "enqueue", "repeat"):
            with args.note.open("rb") as source:
                text = source.read(16385).decode("utf-8")
            request = TaskRequest(TaskKey(args.user, args.project, args.task), ACTION, text)
            occurrences: tuple[Occurrence, ...] = ()
            preview: dict[str, object] = {"request": asdict(request)}
            if args.command == "repeat":
                occurrences = repeat_plan(
                    request,
                    start=args.start,
                    limits=RepeatLimits(args.interval, args.count, args.window),
                    now=datetime.now(UTC),
                )
                preview["occurrences"] = [item.document() for item in occurrences]
            else:
                preview["ttl_seconds"] = args.ttl
            if args.command == "approve" and args.not_before is not None:
                preview["not_before"] = args.not_before.isoformat()
            print(json.dumps(preview, ensure_ascii=False), file=sys.stderr)
            if not args.yes:
                print("Approve this exact note? Type APPROVE: ", end="", file=sys.stderr)
                if input() != "APPROVE":
                    parser.exit(1, "Not approved.\n")
            if args.command == "repeat":
                approve_occurrences(workspace, occurrences)
            elif args.command == "enqueue":
                enqueue(workspace, request, ttl_seconds=args.ttl)
            else:
                approve(workspace, [request], ttl_seconds=args.ttl, not_before=args.not_before)
            print(json.dumps({"approved": True}))
        elif args.command in ("run", "supervise"):
            limits = RunLimits(args.actions, args.cycles, args.seconds, args.poll)
            supervisor = (
                SupervisorLimits(
                    args.starts, args.supervisor_seconds, args.stop_grace, args.restart_delay
                )
                if args.command == "supervise"
                else None
            )
            result = asyncio.run(_run_with_stop_signal(workspace, limits, supervisor))
            print(json.dumps(asdict(result)))
            if result.reason not in ("cycle_limit", "action_limit", "complete"):
                raise SystemExit(2)
        elif args.command == "stop":
            workspace.stop()
            print(json.dumps({"stopped": True}))
        elif args.command == "diagnose":
            state = local_status(workspace)
            print(
                json.dumps(
                    {
                        "controls": state,
                        "supervisor": supervisor_status(workspace),
                        "last_run": read_observation(
                            workspace, active=bool(state["worker_active"])
                        ),
                        "task_counts": scoped_counts(
                            workspace, user_id=args.user, project=args.project
                        ),
                    }
                )
            )
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
