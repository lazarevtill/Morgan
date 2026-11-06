"""A complete synthetic text-to-plan loop using real CLI processes and isolated memory."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace

NOTE = "Нужно полить растения.\nЯ уже купил хлеб.\nНадо позвонить в библиотеку."
LINES = ["Нужно полить растения.", "Надо позвонить в библиотеку."]


def invoke(root, module, args, *, success=True):
    """Run a clean-environment CLI, excluding account settings and external embeddings."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root),
        "MORGAN_DATA_DIR": str(root / "memory"),
        "MORGAN_OWNER_USER_ID": "pilot",
        "MORGAN_EMBEDDING_BACKEND": "hash",
    }
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return json.loads(result.stdout) if result.stdout.strip().startswith("{") else result.stdout


def accept_fixture(root, index, identity):
    """Review only the exact known synthetic fixture, never arbitrary model suggestions."""
    args = [
        "review-proposal",
        str(root / "proposal.json"),
        "--note",
        str(root / "note.txt"),
        "--selection",
        str(index),
        "--due",
        "2026-11-06",
        "--minutes",
        "10",
        "--priority",
        "normal",
        "--author-id",
        "pilot",
        "--event-id",
        identity,
        "--effective-at",
        "2026-11-01T08:00:00Z",
        "--project",
        "personal",
        "--json",
    ]
    preview = invoke(root, "morgan_brain.surfaces.cli", args)
    assert preview["review"]["source_note"] == NOTE
    assert preview["review"]["original_line"] == LINES[index - 1]
    assert not preview["stored"] and not preview["execution_authorized"]
    accepted = invoke(
        root, "morgan_brain.surfaces.cli", [*args, "--accept", preview["review_sha256"]]
    )
    assert accepted["id"] == identity and not accepted["execution_authorized"]


def make_plan(root, filename, *, follow=False, success=True):
    """Keep the original selected task IDs unchanged across completion and replanning."""
    args = [
        "--memory-db",
        str(root / "memory/morgan.db"),
        "--user",
        "pilot",
        "--project",
        "personal",
        "--task",
        filename,
        "--evidence",
        "plants",
        "--evidence",
        "library",
        "--day",
        "2026-11-06",
        "--budget",
        "30",
        "--output",
        str(root / filename),
    ]
    if follow:
        args.append("--follow-revisions")
    return invoke(root, "morgan_brain.daily_brief_cli", args, success=success)


def run_pilot(root: Path, proposal: dict) -> dict:
    """Check capture, completion, current planning, explicit delivery and STOP/restart."""
    workspace = NoteWorkspace(root)
    try:
        workspace.publish("note.txt", NOTE)
        workspace.publish("proposal.json", json.dumps(proposal, ensure_ascii=False))
    finally:
        workspace.close()
    assert proposal["selected_lines"] == LINES
    accept_fixture(root, 1, "plants")
    accept_fixture(root, 2, "library")
    before = make_plan(root, "before.md")
    assert before["selected"] == 2 and before["planned_minutes"] == 20
    completed = invoke(
        root,
        "morgan_brain.surfaces.cli",
        [
            "commitment",
            LINES[0],
            "--due",
            "2026-11-06",
            "--minutes",
            "10",
            "--priority",
            "normal",
            "--status",
            "done",
            "--source",
            "user_stated",
            "--author-id",
            "pilot",
            "--event-id",
            "plants-done",
            "--revises-event-id",
            "plants",
            "--effective-at",
            "2026-11-02T08:00:00Z",
            "--project",
            "personal",
            "--json",
        ],
    )
    assert completed["id"] == "plants-done"
    make_plan(root, "stale.md", success=False)
    assert not (root / "stale.md").exists()
    after = make_plan(root, "after.md", follow=True)
    assert after["selected"] == 1 and after["excluded"] == 1 and after["planned_minutes"] == 10
    assert after["resolved_sources"] == {"plants": "plants-done", "library": "library"}
    delivery = str(root / "delivery")
    module = "morgan_brain.local_notes_cli"
    invoke(
        root,
        module,
        [
            delivery,
            "approve",
            "--user",
            "pilot",
            "--project",
            "personal",
            "--task",
            "reviewed-plan",
            "--note",
            str(root / "after.md"),
            "--yes",
        ],
    )
    invoke(root, module, [delivery, "stop"])
    stopped = invoke(root, module, [delivery, "run"], success=False)
    assert stopped["reason"] == "stopped" and stopped["actions"] == 0
    invoke(root, module, [delivery, "resume"])
    resumed = invoke(root, module, [delivery, "run"])
    repeated = invoke(root, module, [delivery, "run"])
    assert resumed["actions"] == 1 and repeated["actions"] == 0
    assert len(list((root / "delivery").glob("note-*.md"))) == 1
    result = {
        "synthetic": True,
        "before": before,
        "after": after,
        "stop_actions": stopped["actions"],
        "resume_actions": resumed["actions"],
        "replay_actions": repeated["actions"],
        "original_ids_reused": True,
        "automatic_approval_of_arbitrary_model_output": False,
    }
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def test_synthetic_personal_plan_loop(tmp_path):
    """A controlled fixture represents model output; real-model measurements are separate."""
    proposal = {
        "version": "morgan.note-proposal.v1",
        "source_sha256": hashlib.sha256(NOTE.encode()).hexdigest(),
        "source": "agent_inferred",
        "approved": False,
        "review_required": True,
        "selected_lines": LINES,
    }
    run_pilot(tmp_path / "pilot", proposal)
