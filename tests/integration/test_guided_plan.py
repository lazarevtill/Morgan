"""Scripted human decisions for a single guided session; no real user data or model calls."""

import hashlib
import json
import subprocess
import sys
import time

import pytest

from morgan_brain import guided_plan
from morgan_brain.local_notes import NoteWorkspace

CASES = [
    (
        "Нужно полить растения.\nХлеб уже куплен.\nНадо позвонить в библиотеку.",
        [0, 2],
        "готово 1",
        "выход",
    ),
    (
        "I need to water the plants.\nI already bought bread.\nI need to call the library.",
        [0, 2],
        "done 1",
        "quit",
    ),
]


def inputs(root, note, selected):
    """Write one controlled private proposal; selected may deliberately model an ownership error."""
    workspace = NoteWorkspace(root / "input")
    try:
        workspace.publish("note.txt", note)
        workspace.publish(
            "proposal.json",
            json.dumps(
                {
                    "version": "morgan.note-proposal.v1",
                    "source_sha256": hashlib.sha256(note.encode()).hexdigest(),
                    "source": "agent_inferred",
                    "approved": False,
                    "review_required": True,
                    "selected_lines": selected,
                }
            ),
        )
    finally:
        workspace.close()
    return [
        "--note",
        str(root / "input/note.txt"),
        "--proposal",
        str(root / "input/proposal.json"),
        "--user",
        "pilot",
        "--project",
        "personal",
        "--day",
        "2026-11-06",
        "--budget",
        "30",
    ]


def run(root, answers, args=(), *, script=None, overrides=None):
    """Run the actual foreground CLI with explicitly scripted fixture decisions."""
    command = (
        [sys.executable, "-m", "morgan_brain.guided_plan"]
        if script is None
        else [sys.executable, "-c", script]
    )
    return subprocess.run(
        [*command, str(root / "session"), *args],
        input=answers,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(root), **(overrides or {})},
    )


@pytest.mark.parametrize("note,indices,done,quit_command", CASES)
def test_one_launch_six_answers_captures_plans_and_completes(
    tmp_path, note, indices, done, quit_command
):
    """Two languages complete the same useful loop with one launch and six answer submissions."""
    selected = [note.splitlines()[i] for i in indices]
    args = inputs(tmp_path, note, selected)
    answers = ["1 3", "10 normal 2026-11-06", "10 normal 2026-11-06", "CONFIRM", done, quit_command]
    result = run(tmp_path, "\n".join(answers) + "\n", args)
    assert result.returncode == 0, result.stderr
    assert "planned: 20 min" in result.stdout and "planned: 10 min" in result.stdout
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["captured"] and len(state["captures"]) == 2 and state["pending_done"] is None
    assert len(list((tmp_path / "session").glob("plan-*.md"))) == 2
    assert not (tmp_path / "session/.approval.json").exists()
    resumed = run(tmp_path, "quit\n")
    assert resumed.returncode == 0 and "planned: 10 min" in resumed.stdout


@pytest.mark.parametrize(
    "note",
    [
        "Антон планирует почистить велосипед.\nМне нужно забрать посылку.",
        "Anton plans to clean his bicycle.\nI need to collect my parcel.",
    ],
)
def test_owner_can_reject_foreign_model_choice_and_select_omitted_own_task(tmp_path, note):
    """A wrong model suggestion never grants authority and does not hide omitted source lines."""
    args = inputs(tmp_path, note, [note.splitlines()[0]])
    result = run(tmp_path, "2\n10 normal 2026-11-06\nCONFIRM\nquit\n", args)
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["captures"][0]["commitment"]["title"] == note.splitlines()[1]
    assert "1 [model]" in result.stdout and "2:" in result.stdout


def test_cancel_does_not_store_and_allows_reselection(tmp_path):
    """Rejecting the review preserves source artifacts but does not create memory."""
    note = "I need to water the plants."
    args = inputs(tmp_path, note, [note])
    result = run(tmp_path, "1\n10 normal 2026-11-06\nNO\n", args)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "session/memory").exists()
    assert json.loads((tmp_path / "session/session.json").read_text())["captures"] == []
    assert run(tmp_path, "1\n10 normal 2026-11-06\nCONFIRM\nquit\n").returncode == 0


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("phase", ["capture", "done"])
def test_crash_after_memory_write_resumes_without_duplicate(tmp_path, phase, count):
    """Kill the session after a successful memory write but before the journal acknowledgement."""
    lines = ["I need to water the plants.", "I need to call the library."][:count]
    note = "\n".join(lines)
    args = inputs(tmp_path, note, lines)
    script = f"""
import os
from morgan_brain import guided_plan
original = guided_plan._capture
async def crash(workspace, session, capture):
    await original(workspace, session, capture)
    if bool(capture.parents) == {phase == "done"}:
        os._exit(23)
guided_plan._capture = crash
guided_plan.main()
"""
    answers = " ".join(str(i + 1) for i in range(count)) + "\n"
    answers += "10 normal 2026-11-06\n" * count + "CONFIRM\ndone 1\nquit\n"
    result = run(tmp_path, answers, args, script=script)
    assert result.returncode == 23, result.stderr
    resumed = run(tmp_path, "CONFIRM\nquit\n")
    assert resumed.returncode == 0, resumed.stderr
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["captured"] and state["pending_done"] is None
    expected_minutes = (count - (phase == "done")) * 10
    assert f"planned: {expected_minutes} min" in resumed.stdout
    assert len(state["captures"]) == count


def test_changed_source_on_resume_is_refused(tmp_path):
    """A saved review cannot silently adopt modified source text."""
    note = "I need to water the plants."
    args = inputs(tmp_path, note, [note])
    assert run(tmp_path, "\n", args).returncode == 0
    (tmp_path / "session/source.txt").write_text("Something else")
    assert run(tmp_path, "quit\n").returncode == 2


def test_notes_escape_terminal_controls():
    """Untrusted source text cannot inject terminal or bidi controls into the prompt."""
    assert "\x1b" not in guided_plan._quote("bad\x1b[31m")
    assert "\u202e" not in guided_plan._quote("bad\u202e")


def test_second_session_fails_instead_of_waiting_for_another_users_prompt(tmp_path):
    """A nonblocking private session lock excludes overlapping foreground conversations."""
    note = "I need to water the plants."
    args = inputs(tmp_path, note, [note])
    command = [sys.executable, "-m", "morgan_brain.guided_plan", str(tmp_path / "session"), *args]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as first:
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "session/session.json").exists():
                assert first.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            second = run(tmp_path, "quit\n")
            assert second.returncode == 2
            assert not (tmp_path / "session/memory").exists()
        finally:
            first.communicate("\n", timeout=5)


def test_embedding_environment_change_cannot_break_resumed_completion(tmp_path):
    """Pin local hash-vector width rather than inheriting a different width on resume."""
    note = "I need to water the plants."
    args = inputs(tmp_path, note, [note])
    first = run(
        tmp_path,
        "1\n10 normal 2026-11-06\nCONFIRM\nquit\n",
        args,
        overrides={"MORGAN_EMBEDDING_DIM": "8"},
    )
    assert first.returncode == 0, first.stderr
    resumed = run(tmp_path, "done 1\nquit\n", overrides={"MORGAN_EMBEDDING_DIM": "4096"})
    assert resumed.returncode == 0, resumed.stderr
    assert "planned: 0 min" in resumed.stdout
