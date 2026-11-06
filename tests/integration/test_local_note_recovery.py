"""Real process death around a local effect, plus the account-operated CLI path."""

import asyncio
import json
import subprocess
import sys

import pytest

from morgan_brain.local_notes import ACTION, NoteWorkspace, approve, run_local
from morgan_brain.tasks import TaskKey, TaskRequest


@pytest.mark.parametrize("after_effect", [True, False])
def test_process_death_never_repeats_uncertain_delivery(tmp_path, after_effect):
    workspace = NoteWorkspace(tmp_path / "private")
    try:
        request = TaskRequest(TaskKey("owner", "personal", "daily"), ACTION, "# Daily note\n")
        approve(workspace, [request], ttl_seconds=60)
        script = """
import asyncio, os, sys
from pathlib import Path
from morgan_brain.local_notes import LocalNoteAdapter, NoteWorkspace, run_local
original = LocalNoteAdapter.execute
async def die(self, task, control):
    if sys.argv[2] == 'after':
        await original(self, task, control)
    os._exit(23)
LocalNoteAdapter.execute = die
asyncio.run(run_local(NoteWorkspace(Path(sys.argv[1]))))
"""
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(workspace.path),
                "after" if after_effect else "before",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert process.returncode == 23, process.stderr
        before = list(workspace.path.glob("note-*.md"))
        inode = before[0].stat().st_ino if before else None
        result = asyncio.run(run_local(workspace))
        assert result.actions == 0
        if after_effect:
            assert result.recovered == 1
            after = list(workspace.path.glob("note-*.md"))
            assert len(after) == 1 and after[0].stat().st_ino == inode
            assert after[0].read_text() == request.payload
        else:
            assert result.reason == "needs_reconciliation"
            assert not list(workspace.path.glob("note-*.md"))
        journal = workspace.journal()
        try:
            task = journal.get(request.key)
            assert task.status == ("succeeded" if after_effect else "outcome_unknown")
            assert bool(task.receipt_json) == after_effect
        finally:
            journal.close()
    finally:
        workspace.close()


def test_cli_end_to_end(tmp_path):
    root = tmp_path / "agent"
    source = tmp_path / "plan.md"
    source.write_text("# Morning plan\n\nReview priorities.\n")

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "morgan_brain.local_notes_cli", str(root), *args],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    approved = cli(
        "approve",
        "--user",
        "me",
        "--project",
        "personal",
        "--task",
        "today",
        "--note",
        str(source),
        "--yes",
    )
    assert approved.returncode == 0, approved.stderr
    first = cli("run")
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["actions"] == 1
    assert json.loads(cli("run").stdout)["actions"] == 0
    assert cli("stop").returncode == 0
    stopped = cli("run")
    assert stopped.returncode == 2
    assert json.loads(stopped.stdout)["reason"] == "stopped"
    assert cli("resume").returncode == 0
    assert cli("run").returncode == 0
    assert [p.read_text() for p in root.glob("note-*.md")] == [source.read_text()]
