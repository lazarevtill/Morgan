"""Real process termination and recovery, with a synthetic network effect boundary."""

import asyncio
import json
import signal
import sys
from contextlib import closing
from pathlib import Path

import pytest

from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.tasks import TaskKey

PILOT = Path(__file__).parents[1] / "fixtures" / "owner_push_shutdown.py"


@pytest.mark.parametrize("graceful", [True, False])
@pytest.mark.asyncio
async def test_signal_and_fresh_process_recovery(tmp_path: Path, graceful: bool) -> None:
    """SIGTERM persists STOP; SIGKILL leaves running fenced, and neither permits replay."""
    workspace_path = tmp_path / "outbox"
    command = [sys.executable, str(PILOT), str(workspace_path)]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        assert process.stdout is not None
        async with asyncio.timeout(5):
            assert await process.stdout.readline() == b"request_started\n"
        process.send_signal(signal.SIGTERM if graceful else signal.SIGKILL)
        async with asyncio.timeout(5):
            output, error = await process.communicate()
        assert not error
        assert process.returncode == (2 if graceful else -signal.SIGKILL)
        if graceful:
            assert json.loads(output) == {"stopped": True, "calls": 1}
        with closing(NoteWorkspace(workspace_path, create=False)) as workspace:
            assert (workspace.path / "STOP").exists() == graceful
            with closing(workspace.journal()) as journal:
                task = journal.get(TaskKey("demo", "personal", "shutdown"))
                assert task.status == ("outcome_unknown" if graceful else "running")
                assert task.result is None
            if graceful:
                blocked = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                async with asyncio.timeout(5):
                    output, error = await blocked.communicate()
                assert blocked.returncode == 2
                assert json.loads(output) == {"stopped": True, "calls": 0}
                assert not error
                workspace.resume()
            replay = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            async with asyncio.timeout(5):
                output, error = await replay.communicate()
            assert replay.returncode == 0
            assert not error
            assert json.loads(output) == {"status": task.status, "calls": 0}
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
