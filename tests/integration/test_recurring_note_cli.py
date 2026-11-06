"""Real CLI invocations with controlled wall time for a finite personal reminder series."""

import json
import subprocess
import sys

CLOCKED_CLI = """
import sys
from datetime import datetime, UTC
from morgan_brain import local_notes, task_batches, local_notes_cli
stamp = datetime.fromisoformat(sys.argv.pop(1))
class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return stamp.astimezone(tz)
for module in (local_notes, task_batches, local_notes_cli):
    module.datetime = Clock
local_notes_cli.main()
"""


def call(workspace, stamp, *args, input_text=None):
    """Launch a new process for each owner operation, without real waiting or personal data."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            CLOCKED_CLI,
            f"2026-11-06T08:{stamp}:00+00:00",
            str(workspace),
            *args,
        ],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_repeat_preview_stop_restart_and_missed_slot(tmp_path):
    """Approve three breaks; STOP misses one, resume delivers the next only once."""
    workspace = tmp_path / "agent"
    note = tmp_path / "break.md"
    note.write_text("# Разминка\nСделать перерыв от экрана.\n")
    accepted = call(
        workspace,
        "00",
        "repeat",
        "--user",
        "owner",
        "--project",
        "personal",
        "--task",
        "breaks",
        "--note",
        str(note),
        "--start",
        "2026-11-06T08:00:00Z",
        "--interval",
        "60",
        "--count",
        "3",
        "--window",
        "30",
        "--yes",
    )
    assert accepted.returncode == 0, accepted.stderr
    preview = json.loads(accepted.stderr)
    assert len(preview["occurrences"]) == 3
    assert len({x["request"]["key"]["task_id"] for x in preview["occurrences"]}) == 3
    first = call(workspace, "00", "run")
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["actions"] == 1
    assert call(workspace, "00", "stop").returncode == 0
    assert json.loads(call(workspace, "01", "run").stdout)["reason"] == "stopped"
    assert call(workspace, "02", "resume").returncode == 0
    assert json.loads(call(workspace, "02", "run").stdout)["actions"] == 1
    assert json.loads(call(workspace, "02", "run").stdout)["actions"] == 0
    assert len(list(workspace.glob("note-*.md"))) == 2
    assert json.loads(call(workspace, "03", "run").stdout)["reason"] == "permission_denied"
