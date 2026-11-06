"""Explicit local-account alert invocation; stdin is data, never an execution grant."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import closing
from pathlib import Path

from morgan_brain.local_alert import ACTION, Alert, LocalAlert
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.tasks import Task, TaskKey, run_task


async def send(workspace: NoteWorkspace, key: TaskKey, alert: Alert) -> Task:
    """Authorize one exact request for this invocation and never replay a started attempt."""
    payload = alert.model_dump_json()

    def authorize(task: Task) -> None:
        """Require the exact account-selected specification and a clear STOP marker."""
        workspace.check_stop()
        if (task.key, task.action, task.payload) != (key, ACTION, payload):
            raise ValueError("alert request differs from explicit invocation")

    with workspace.locked(), closing(workspace.journal()) as journal:
        workspace.check_stop()
        task = journal.create(key, action=ACTION, payload=payload)
        if task.status != "pending":
            return task
        return await run_task(
            journal,
            key,
            handlers={ACTION: LocalAlert(workspace)},
            timeout_seconds=alert.seconds + 10,
            authorize=authorize,
        )


def main() -> int:
    """Send approved stdin text or manage STOP, without storing a standing permission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("operation", choices=("send", "stop", "resume"))
    parser.add_argument("--user")
    parser.add_argument("--project")
    parser.add_argument("--task-id")
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()
    try:
        with closing(NoteWorkspace(args.workspace)) as workspace:
            if args.operation == "stop":
                workspace.stop()
                print('{"control":"stopped"}')
            elif args.operation == "resume":
                workspace.resume()
                print('{"control":"resumed","retried":false}')
            else:
                key = TaskKey(args.user, args.project, args.task_id)
                alert = Alert(text=sys.stdin.read(2001), seconds=args.seconds)
                task = asyncio.run(send(workspace, key, alert))
                print(
                    json.dumps(
                        {
                            "status": task.status,
                            "result": json.loads(task.result) if task.result else None,
                        }
                    )
                )
                return 0 if task.status == "succeeded" else 2
    except (ValueError, OSError, RuntimeError, StopRequested, TimeoutError) as error:
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "guidance": "Inspect the scoped journal; do not retry with a new ID.",
                }
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
