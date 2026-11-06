"""A real CLI worker may die while waiting and resume one approved delivery."""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

from morgan_brain.local_notes import NoteWorkspace, local_status


def test_cli_schedule_kill_restart_and_repeat(tmp_path):
    """Persist the window before killing the waiter; restart delivers once when due."""
    workspace = NoteWorkspace(tmp_path / "agent")
    note = tmp_path / "plan.md"
    note.write_text("# Утренний план\n- Полить растения\n")
    command = [sys.executable, "-m", "morgan_brain.local_notes_cli", str(workspace.path)]
    due = datetime.now(UTC) + timedelta(seconds=8)
    try:
        approved = subprocess.run(
            [
                *command,
                "approve",
                "--user",
                "owner",
                "--project",
                "personal",
                "--task",
                "morning",
                "--note",
                str(note),
                "--ttl",
                "60",
                "--not-before",
                due.isoformat(),
                "--yes",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert approved.returncode == 0, approved.stderr
        assert json.loads(approved.stderr)["not_before"] == due.isoformat()
        run = [*command, "run", "--cycles", "60", "--poll", "0.25", "--seconds", "20"]
        with subprocess.Popen(run, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as worker:
            try:
                deadline = time.monotonic() + 5
                while not local_status(workspace)["worker_active"]:
                    assert worker.poll() is None and time.monotonic() < deadline
                    time.sleep(0.01)
                assert datetime.now(UTC) < due
                assert not list(workspace.path.glob("note-*.md"))
                worker.kill()
                worker.wait(timeout=5)
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait(timeout=5)
        resumed = subprocess.run(run, capture_output=True, text=True, timeout=25, check=False)
        assert resumed.returncode == 0, resumed.stderr
        assert json.loads(resumed.stdout)["actions"] == 1
        again = subprocess.run(
            [*command, "run"], capture_output=True, text=True, timeout=5, check=False
        )
        assert again.returncode == 0, again.stderr
        assert json.loads(again.stdout)["actions"] == 0
        assert [p.read_text() for p in workspace.path.glob("note-*.md")] == [note.read_text()]
    finally:
        workspace.close()
