"""Explicit reopening keeps one family and recovers exact writes without granting authority."""

import json
import subprocess
import sys

import pytest

from morgan_brain import owner_input
from morgan_brain.guided_renewal import RenewalUpdate
from tests.integration.test_owner_search import capture
from tests.integration.test_owner_ui import drive, initialize


def closed(root, *, cancel=False):
    initialize(root)
    parent = capture(root)
    owner_input.request(root, ["submit", "close"], text="I mistakenly reported this closed.")
    review = owner_input.request(
        root, ["review", "close", "--cancel" if cancel else "--complete", parent]
    )["message"]
    owner_input.request(root, ["confirm", "close", "--digest", review["digest"]])
    return review["renewal"]["event_id"]


def reopen_review(root, parent, identity="reopen"):
    owner_input.request(
        root, ["submit", identity], text="Not actually completed; reopen this task."
    )
    return owner_input.request(
        root, ["review", identity, "--reopen", parent, "--due", "2026-11-09"]
    )["message"]


def confirm(root, review):
    return owner_input.request(root, ["confirm", review["identity"], "--digest", review["digest"]])


@pytest.mark.parametrize(
    "text", ["Я ошибся: ещё не вернул книгу.", "I was mistaken: I have not returned the book."]
)
def test_owner_reopens_same_task_through_dialogue(tmp_path, text):
    parent = closed(tmp_path)
    dialogs = drive(
        tmp_path,
        [
            ("choose", "Новое сообщение"),
            ("text", text),
            ("choose", "Вернуть выполненную задачу в работу"),
            ("choose", 0),
            ("text", "2026-11-09"),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    result = owner_input.request(tmp_path, ["task-state", "книг"])
    assert result["answer"] == "open" and result["current_task"]["due"] == "2026-11-09"
    assert len(result["sources"]) == 3
    status = owner_input.request(tmp_path, ["status"])
    assert len(status["plan"]["tasks"]) == 1
    update = status["messages"][-1]["renewal"]
    assert update["parent_id"] == parent and update["before"]["status"] == "done"
    assert update["source_text"] == text
    prompts = [d["prompt"] for d in dialogs.seen if d["kind"] == "confirm"]
    assert len(prompts) == 1 and "история сохраняется" in prompts[0] and parent in prompts[0]
    selections = [d["options"] for d in dialogs.seen if d["prompt"] == "Выберите конкретную задачу"]
    assert "выполнено" in selections[0][0]


@pytest.mark.parametrize("state", ["open", "cancelled"])
def test_reopen_rejects_noncompleted_tasks(tmp_path, state):
    if state == "open":
        initialize(tmp_path)
        parent = capture(tmp_path)
    else:
        parent = closed(tmp_path, cancel=True)
    with pytest.raises(ValueError, match="reopening"):
        reopen_review(tmp_path, parent)
    assert owner_input.request(tmp_path, ["task-state", "книг"])["answer"] == state


def test_reopen_requires_date_digest_current_parent_and_stop_clear(tmp_path):
    parent = closed(tmp_path)
    review = reopen_review(tmp_path, parent)
    with pytest.raises(ValueError):
        owner_input.request(tmp_path, ["review", "reopen", "--reopen", parent])
    with pytest.raises(ValueError, match="digest"):
        confirm(tmp_path, {**review, "digest": "wrong"})
    owner_input.request(tmp_path, ["stop"])
    from morgan_brain.local_notes import StopRequested

    with pytest.raises(StopRequested):
        confirm(tmp_path, review)
    owner_input.request(tmp_path, ["resume"])
    competing = reopen_review(tmp_path, parent, "second-review")
    confirm(tmp_path, competing)
    with pytest.raises(ValueError, match="changed"):
        confirm(tmp_path, review)
    assert len(owner_input.request(tmp_path, ["status"])["plan"]["tasks"]) == 1


@pytest.mark.parametrize(
    "phase", ["inbox-1", "session-1", "write-1", "write-2", "session-2", "inbox-2"]
)
def test_reopen_recovers_after_process_exit_without_duplicate_family(tmp_path, phase):
    parent = closed(tmp_path)
    review = reopen_review(tmp_path, parent)
    kind, count = phase.split("-")
    target = {
        "inbox": "owner_input._save",
        "session": "owner_input.guided_save" if count == "1" else "guided_plan._save",
        "write": "guided_plan.cmd_remember",
    }[kind]
    if kind == "session":
        count = "1"
    script = f"""
import os
from morgan_brain import owner_input, guided_plan
original = {target}
count = 0
{"async " if kind == "write" else ""}def crash(*args, **kwargs):
    global count
    result = {"await " if kind == "write" else ""}original(*args, **kwargs)
    count += 1
    if count == {count}:
        os._exit(29)
    return result
{target} = crash
owner_input.main()
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            "confirm",
            "reopen",
            "--digest",
            review["digest"],
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 29, proc.stderr
    confirm(tmp_path, review)
    confirm(tmp_path, review)
    result = owner_input.request(tmp_path, ["task-state", "книг"])
    assert (
        result["answer"] == "open"
        and result["current_evidence_id"] == review["renewal"]["event_id"]
    )
    assert len(result["sources"]) == 3
    assert len(owner_input.request(tmp_path, ["status"])["plan"]["tasks"]) == 1


def test_expired_reopening_preserves_review(tmp_path):
    parent = closed(tmp_path)
    review = reopen_review(tmp_path, parent)
    path = tmp_path / "owner-input.json"
    inbox = json.loads(path.read_text())
    inbox["expires_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(inbox))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="expired"):
        confirm(tmp_path, review)
    assert before == path.read_bytes()
    assert owner_input.request(tmp_path, ["task-state", "книг"])["answer"] == "done"


@pytest.mark.parametrize(
    "field,value",
    [("title", "Different task"), ("minutes", 99), ("priority", "high"), ("status", "cancelled")],
)
def test_reopening_cannot_smuggle_other_field_changes(tmp_path, field, value):
    review = reopen_review(tmp_path, closed(tmp_path))
    update = review["renewal"]
    update["after"][field] = value
    with pytest.raises(ValueError, match="reopening"):
        RenewalUpdate.model_validate_json(json.dumps(update))
