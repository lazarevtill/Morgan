"""Real producer races, interrupted intake and the finite worker lifecycle."""

import json
import os
import signal
import subprocess
import sys
import time

from morgan_brain.local_notes import (
    ACTION,
    FileApproval,
    NoteWorkspace,
    enqueue,
    local_status,
)
from morgan_brain.tasks import TaskKey, TaskRequest


def test_concurrent_producers_preserve_every_request(tmp_path):
    workspace = NoteWorkspace(tmp_path / "agent")
    script = """
import sys
from pathlib import Path
from morgan_brain.local_notes import ACTION, NoteWorkspace, enqueue
from morgan_brain.tasks import TaskRequest, TaskKey
workspace = NoteWorkspace(Path(sys.argv[1]))
try:
    request = TaskRequest(TaskKey('owner','personal',sys.argv[2]), ACTION, sys.argv[2])
    enqueue(workspace, request, ttl_seconds=60)
finally:
    workspace.close()
"""
    processes = []
    try:
        for identity in range(8):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(workspace.path), str(identity)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        for process in processes:
            _, error = process.communicate(timeout=15)
            assert process.returncode == 0, error
        requests, _ = FileApproval(workspace).snapshot()
        assert {r.key.task_id for r in requests} == {str(i) for i in range(8)}
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        workspace.close()


def test_intake_ack_loss_keeps_one_request_and_releases_lock(tmp_path):
    workspace = NoteWorkspace(tmp_path / "agent")
    script = """
import os, sys
from pathlib import Path
from morgan_brain.local_notes import ACTION, NoteWorkspace, enqueue
from morgan_brain.tasks import TaskRequest, TaskKey
original = NoteWorkspace.publish

def die(self, *args, **kwargs):
    original(self, *args, **kwargs)
    os._exit(23)
NoteWorkspace.publish = die
request = TaskRequest(TaskKey('owner','personal','one'), ACTION, '# One')
enqueue(NoteWorkspace(Path(sys.argv[1])), request, ttl_seconds=60)
"""
    try:
        process = subprocess.run(
            [sys.executable, "-c", script, str(workspace.path)],
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert process.returncode == 23, process.stderr
        original, expiry = FileApproval(workspace).snapshot()
        enqueue(
            workspace,
            TaskRequest(TaskKey("owner", "personal", "one"), ACTION, "# One"),
            ttl_seconds=600,
        )
        assert FileApproval(workspace).snapshot() == (original, expiry)
        assert len(original) == 1
    finally:
        workspace.close()


def test_cli_live_intake_sigterm_stop_and_restart(tmp_path):
    workspace = NoteWorkspace(tmp_path / "agent")
    command = [sys.executable, "-m", "morgan_brain.local_notes_cli", str(workspace.path)]
    worker = subprocess.Popen(
        [*command, "run", "--cycles", "60", "--poll", "0.1", "--seconds", "10", "--actions", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not local_status(workspace)["worker_active"]:
            assert worker.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for identity in ("one", "two"):
            source = tmp_path / f"{identity}.md"
            source.write_text(f"# {identity}\n")
            accepted = subprocess.run(
                [
                    *command,
                    "enqueue",
                    "--user",
                    "owner",
                    "--project",
                    "personal",
                    "--task",
                    identity,
                    "--note",
                    str(source),
                    "--yes",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            assert accepted.returncode == 0, accepted.stderr
        while len(list(workspace.path.glob("note-*.md"))) != 2:
            assert worker.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        os.kill(worker.pid, signal.SIGTERM)
        output, error = worker.communicate(timeout=5)
        assert worker.returncode == 2, error
        assert json.loads(output) == {"reason": "stopped", "actions": 2, "recovered": 0}
        state = local_status(workspace)
        assert state["worker_active"] is False and state["stop_requested"] is True
        workspace.resume()
        restarted = subprocess.run(
            [*command, "run"], capture_output=True, text=True, timeout=5, check=False
        )
        assert restarted.returncode == 0, restarted.stderr
        assert json.loads(restarted.stdout)["actions"] == 0
        assert len(list(workspace.path.glob("note-*.md"))) == 2
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        workspace.close()
