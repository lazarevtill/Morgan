"""Subprocess-only synthetic pilot: never connects to a provider or reads a real token."""

import asyncio
import json
import sys
from contextlib import closing
from pathlib import Path

import httpx
from pydantic import SecretStr

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.owner_push import OwnerPush, Push
from morgan_brain.owner_push_cli import send_with_stop_signal
from morgan_brain.tasks import TaskKey


async def run(path: Path) -> int:
    """Expose one deterministic effect boundary and let the test terminate this process."""
    calls = []

    async def provider(request: httpx.Request) -> httpx.Response:
        """Stand in for an accepted request whose response has not arrived."""
        calls.append(request)
        print("request_started", flush=True)
        await asyncio.sleep(30)
        raise AssertionError("test must stop the in-flight request")

    with closing(NoteWorkspace(path)) as workspace:
        adapter = OwnerPush(
            workspace, SecretStr("synthetic-only"), transport=httpx.MockTransport(provider)
        )
        try:
            task = await send_with_stop_signal(
                workspace,
                TaskKey("demo", "personal", "shutdown"),
                Push(server="https://push.example", topic="demo", text="Synthetic shutdown pilot"),
                adapter,
            )
        except StopRequested:
            print(json.dumps({"stopped": True, "calls": len(calls)}))
            return 2
        print(json.dumps({"status": task.status, "calls": len(calls)}))
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(Path(sys.argv[1]))))
