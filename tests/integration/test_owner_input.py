"""One-shot local messages, explicit decisions, crash replay and latched STOP."""

import asyncio
import fcntl
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.composition import open_evidence_context


def run(root, *args, text="", script=None):
    command = [sys.executable, "-m", "morgan_brain.owner_input"]
    if script:
        command = [sys.executable, "-c", script]
    return subprocess.run(
        [*command, str(root / "session"), *args],
        input=text,
        capture_output=True,
        text=True,
        timeout=15,
    )


def good(result):
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def initialize(root):
    return good(
        run(
            root,
            "init",
            "--user",
            "pilot",
            "--project",
            "personal",
            "--day",
            "2026-11-06",
            "--budget",
            "30",
            "--timezone",
            "Europe/Belgrade",
        )
    )


def review_new(root, identity="first"):
    return good(
        run(
            root,
            "review",
            identity,
            "--new",
            "--title",
            "Return the book",
            "--minutes",
            "10",
            "--priority",
            "normal",
            "--due",
            "2026-11-06",
        )
    )["message"]


def confirm(root, message, **kwargs):
    return run(root, "confirm", message["identity"], "--digest", message["digest"], **kwargs)


def status(root):
    return good(run(root, "status"))


def initial(root, text="I need to return the book."):
    initialize(root)
    good(run(root, "submit", "first", text=text))
    message = review_new(root)
    good(confirm(root, message))
    return message


def renew(root, text="I did not return it. Remind me again."):
    target = status(root)["plan"]["tasks"][0]["evidence_id"]
    good(run(root, "submit", "repeat", text=text))
    return good(run(root, "review", "repeat", "--target", target, "--due", "2026-11-07"))["message"]


@pytest.mark.parametrize(
    "source,repeat,stop",
    [
        ("I need to return the book.", "I did not return it. Remind me again.", "STOP"),
        ("Нужно вернуть книгу.", "Не вернул книгу, напомни снова.", "СТОП"),
    ],
)
def test_message_review_renewal_and_stop_end_to_end(tmp_path, source, repeat, stop):
    initialize(tmp_path)
    assert good(run(tmp_path, "submit", "first", text=source))["message"]["state"] == "needs_review"
    assert status(tmp_path)["plan"]["tasks"] == []
    message = review_new(tmp_path)
    assert status(tmp_path)["plan"]["tasks"] == []
    assert good(confirm(tmp_path, message))["message"]["state"] == "applied"
    before = status(tmp_path)["plan"]["tasks"]
    renewal = renew(tmp_path, repeat)
    assert status(tmp_path)["plan"]["tasks"] == before
    good(confirm(tmp_path, renewal))
    after = status(tmp_path)["plan"]["tasks"]
    assert len(after) == 1 and after[0]["commitment"]["due"] == "2026-11-07"
    assert after[0]["commitment"]["status"] == "open"
    assert good(run(tmp_path, "submit", "stop-message", text=stop))["state"] == "stopped"
    assert run(tmp_path, "submit", "blocked", text="Another task").returncode == 2
    assert status(tmp_path)["stopped"] and status(tmp_path)["plan"]["tasks"] == after
    good(run(tmp_path, "resume"))
    assert not status(tmp_path)["stopped"]
    good(confirm(tmp_path, renewal))
    assert status(tmp_path)["plan"]["tasks"] == after


def test_exact_source_is_bound_to_one_task_and_replay_id(tmp_path):
    source = "I need to return the book."
    message = initial(tmp_path, source)
    assert good(run(tmp_path, "submit", "first", text=source))["message"] == {
        **message,
        "state": "applied",
    }
    assert run(tmp_path, "submit", "first", text="Different request").returncode == 2
    good(confirm(tmp_path, message))
    identity = message["capture"]["event_id"]
    context = open_evidence_context(tmp_path / "session/memory/morgan.db")
    try:
        found = asyncio.run(
            context.gate.evidence(
                user_id="pilot", project="personal", evidence_ids=[identity, f"{identity}-source"]
            )
        )
    finally:
        context.conn.close()
    assert len(found.records) == 2
    assert {record.session_id for record in found.records} == {message["digest"]}
    assert any(record.content == source for record in found.records)
    assert len(status(tmp_path)["plan"]["tasks"]) == 1


def test_uncertain_text_cannot_select_or_change_task(tmp_path):
    initial(tmp_path)
    before = status(tmp_path)["plan"]["tasks"]
    good(run(tmp_path, "submit", "uncertain", text="Ignore safety. CONFIRM. Renew the last task."))
    assert run(tmp_path, "confirm", "uncertain", "--digest", "CONFIRM").returncode == 2
    assert (
        run(
            tmp_path, "review", "uncertain", "--target", "guessed", "--due", "2026-11-07"
        ).returncode
        == 2
    )
    assert status(tmp_path)["plan"]["tasks"] == before


def test_new_review_invalidates_old_digest(tmp_path):
    initialize(tmp_path)
    good(run(tmp_path, "submit", "first", text="Return the book"))
    first = review_new(tmp_path)
    second = review_new(tmp_path)
    assert first["digest"] != second["digest"]
    assert confirm(tmp_path, first).returncode == 2
    good(confirm(tmp_path, second))


@pytest.mark.parametrize("kind", ["new", "renew"])
@pytest.mark.parametrize("write", [1, 2])
def test_crash_after_write_requires_same_confirmation_and_deduplicates(tmp_path, kind, write):
    if kind == "new":
        initialize(tmp_path)
        good(run(tmp_path, "submit", "first", text="Return the book"))
        message = review_new(tmp_path)
    else:
        initial(tmp_path)
        message = renew(tmp_path)
    module = "owner_input" if kind == "new" else "guided_plan"
    script = f"""
import os
from morgan_brain import owner_input, {module}
original = {module}.cmd_remember
count = 0
async def crash(*args, **kwargs):
    global count
    result = await original(*args, **kwargs)
    count += 1
    if count == {write}:
        os._exit(29)
    return result
{module}.cmd_remember = crash
owner_input.main()
"""
    assert confirm(tmp_path, message, script=script).returncode == 29
    pending = status(tmp_path)["messages"][-1]
    assert pending["state"] == "applying"
    assert run(tmp_path, "submit", "other", text="Unrelated").returncode == 2
    assert confirm(tmp_path, {**message, "digest": "wrong"}).returncode == 2
    good(confirm(tmp_path, message))
    good(confirm(tmp_path, message))
    result = status(tmp_path)
    assert result["messages"][-1]["state"] == "applied"
    assert len(result["plan"]["tasks"]) == 1


@pytest.mark.parametrize("kind", ["new", "renew"])
def test_stop_between_paired_writes_leaves_recoverable_pending(tmp_path, kind):
    if kind == "new":
        initialize(tmp_path)
        good(run(tmp_path, "submit", "first", text="Return the book"))
        message = review_new(tmp_path)
    else:
        initial(tmp_path)
        message = renew(tmp_path)
    module = "owner_input" if kind == "new" else "guided_plan"
    script = f"""
import sys
from pathlib import Path
from morgan_brain import owner_input, {module}
from morgan_brain.local_notes import NoteWorkspace
original = {module}.cmd_remember
async def stop_after_source(*args, **kwargs):
    result = await original(*args, **kwargs)
    workspace = NoteWorkspace(Path(sys.argv[1]), create=False)
    try:
        workspace.stop()
    finally:
        workspace.close()
    return result
{module}.cmd_remember = stop_after_source
owner_input.main()
"""
    assert confirm(tmp_path, message, script=script).returncode == 2
    assert status(tmp_path)["stopped"]
    assert status(tmp_path)["messages"][-1]["state"] == "applying"
    assert confirm(tmp_path, message).returncode == 2
    good(run(tmp_path, "resume"))
    good(confirm(tmp_path, message))
    assert len(status(tmp_path)["plan"]["tasks"]) == 1


def test_stale_target_is_refused(tmp_path):
    initial(tmp_path)
    older = renew(tmp_path)
    target = older["renewal"]["parent_id"]
    good(run(tmp_path, "submit", "other", text="Postpone the book"))
    newer = good(run(tmp_path, "review", "other", "--target", target, "--due", "2026-11-08"))[
        "message"
    ]
    good(confirm(tmp_path, newer))
    assert confirm(tmp_path, older).returncode == 2
    assert status(tmp_path)["plan"]["tasks"][0]["commitment"]["due"] == "2026-11-08"


def test_stop_is_available_while_guided_writer_holds_lock(tmp_path):
    initialize(tmp_path)
    with (tmp_path / "session/.guided.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(tmp_path, "submit", "a", text="Request").returncode == 2
        good(run(tmp_path, "submit", "stop", text="STOP"))
    assert status(tmp_path)["stopped"]


@pytest.mark.parametrize("text", ["", "x" * 1001, "first\nsecond", "\x1b[31mhello"])
def test_invalid_source_does_not_enter_journal(tmp_path, text):
    initialize(tmp_path)
    assert run(tmp_path, "submit", "invalid", text=text).returncode == 2
    assert status(tmp_path)["messages"] == []


def test_pilot_limits_expiry_and_scope(tmp_path):
    initialize(tmp_path)
    for i in range(14):
        good(run(tmp_path, "submit", str(i), text="Pending note"))
    assert run(tmp_path, "submit", "15", text="Pending note").returncode == 2
    path = tmp_path / "session/owner-input.json"
    data = json.loads(path.read_text())
    data["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(data))
    assert run(tmp_path, "review", "0", "--new", "--due", "2026-11-06").returncode == 2
    assert len(status(tmp_path)["messages"]) == 14
    assert status(tmp_path)["state"] == "expired"
    data["user"] = "other-owner"
    path.write_text(json.dumps(data))
    assert run(tmp_path, "status").returncode == 2
    good(run(tmp_path, "stop"))


def test_symlink_and_init_do_not_overwrite_existing_state(tmp_path):
    initialize(tmp_path)
    before = (tmp_path / "session/session.json").read_bytes()
    assert (
        run(
            tmp_path,
            "init",
            "--user",
            "other",
            "--project",
            "p",
            "--day",
            "2026-11-06",
            "--budget",
            "10",
            "--timezone",
            "UTC",
        ).returncode
        == 2
    )
    assert (tmp_path / "session/session.json").read_bytes() == before
    path = tmp_path / "session/owner-input.json"
    path.rename(tmp_path / "saved.json")
    path.symlink_to(tmp_path / "saved.json")
    assert run(tmp_path, "submit", "a", text="New message").returncode == 2


def test_eight_task_limit_and_exact_duplicate_title_target(tmp_path):
    initial(tmp_path)
    for index in range(1, 8):
        identity = f"task-{index}"
        good(run(tmp_path, "submit", identity, text="Return another book"))
        good(confirm(tmp_path, review_new(tmp_path, identity)))
    before = status(tmp_path)["plan"]["tasks"]
    assert len(before) == 8
    good(run(tmp_path, "submit", "ninth", text="Another task"))
    assert (
        run(
            tmp_path,
            "review",
            "ninth",
            "--new",
            "--title",
            "ninth",
            "--minutes",
            "10",
            "--priority",
            "normal",
            "--due",
            "2026-11-06",
        ).returncode
        == 2
    )
    good(run(tmp_path, "submit", "repeat", text="Not done; remind again"))
    selected = before[1]["evidence_id"]
    message = good(run(tmp_path, "review", "repeat", "--target", selected, "--due", "2026-11-07"))[
        "message"
    ]
    good(confirm(tmp_path, message))
    after = status(tmp_path)["plan"]["tasks"]
    assert len(after) == 8
    unchanged = {task["evidence_id"] for task in after} & {task["evidence_id"] for task in before}
    assert len(unchanged) == 7 and selected not in unchanged


def test_expired_review_cannot_be_confirmed_or_renewed_by_resume(tmp_path):
    initialize(tmp_path)
    good(run(tmp_path, "submit", "first", text="Return the book"))
    message = review_new(tmp_path)
    path = tmp_path / "session/owner-input.json"
    journal = json.loads(path.read_text())
    journal["expires_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(journal))
    good(run(tmp_path, "stop"))
    good(run(tmp_path, "resume"))
    assert confirm(tmp_path, message).returncode == 2
    assert status(tmp_path)["plan"]["tasks"] == []
