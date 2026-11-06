"""One explicit owner push; credentials are prompted, not installed or saved."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import warnings
from contextlib import closing
from pathlib import Path

from pydantic import SecretStr

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.owner_push import OwnerPush, Push, send
from morgan_brain.tasks import TaskKey


def main() -> int:
    """Require an explicit destination, scope and task identity for one send invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--server", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    try:
        push = Push(server=args.server, topic=args.topic, text=sys.stdin.read(2001))
        push.endpoint()
        key = TaskKey(args.user, args.project, args.task_id)
        with closing(NoteWorkspace(args.workspace)) as workspace:
            workspace.check_stop()
            with open("/dev/tty", "r+") as terminal, warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = SecretStr(getpass.getpass("Transient ntfy token: ", stream=terminal))
            task = asyncio.run(send(workspace, key, push, OwnerPush(workspace, token)))
            print(
                json.dumps(
                    {
                        "status": task.status,
                        "result": json.loads(task.result) if task.result else None,
                    }
                )
            )
            return 0 if task.status == "succeeded" else 2
    except (ValueError, OSError, RuntimeError, StopRequested, TimeoutError, getpass.GetPassWarning):
        print('{"error":"push refused or outcome unknown; inspect journal before any new send"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
