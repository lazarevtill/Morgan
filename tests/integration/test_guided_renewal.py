"""Explicit EN/RU renewal, exact target binding and replay under real process failures."""

import asyncio
import json

import pytest

from morgan_brain.composition import open_evidence_context
from tests.integration.test_guided_availability import state
from tests.integration.test_guided_plan import inputs, run
from tests.integration.test_guided_status import status


def initial(root, note="I need to return the book."):
    """Use a deliberately empty model proposal; the owner can still select the source line."""
    return run(root, "1\n10 normal 2026-11-06\nCONFIRM\nquit\n", inputs(root, note, []))


def renewal(
    command="renew 1", statement="I did not return it. Remind me again.", confirm="CONFIRM"
):
    """Script explicit fixture decisions; these are not inferred from the statement."""
    return f"{command}\n{statement}\n2026-11-07\n{confirm}\nquit\n"


def evidence(root, identities):
    """Inspect only scoped evidence through the gate, never raw storage tables."""
    context = open_evidence_context(root / "session/memory/morgan.db")
    try:
        return asyncio.run(
            context.gate.evidence(user_id="pilot", project="personal", evidence_ids=identities)
        )
    finally:
        context.conn.close()


@pytest.mark.parametrize(
    "command,statement",
    [
        ("renew 1", "I did not return the book. Remind me again."),
        ("повторить 1", "Книгу не вернул, напомни снова."),
    ],
)
def test_explicit_renewal_preserves_one_family_and_exact_source(tmp_path, command, statement):
    """A missed task stays open with a reviewed new date, not a new duplicate obligation."""
    assert initial(tmp_path).returncode == 0
    original = state(tmp_path)["captures"][0]["event_id"]
    result = run(tmp_path, renewal(command, statement))
    assert result.returncode == 0, result.stderr
    assert "not notification delivery" in result.stdout
    payload = json.loads(status(tmp_path).stdout)
    assert len(payload["tasks"]) == 1 and payload["planned_minutes"] == 0
    task = payload["tasks"][0]
    assert task["commitment"]["due"] == "2026-11-07" and task["commitment"]["status"] == "open"
    identity = task["evidence_id"]
    assert payload["resolved_sources"] == {original: identity}
    found = evidence(tmp_path, [identity, f"{identity}-source"])
    by_id = {record.id: record for record in found.records}
    assert not found.missing_ids and by_id[f"{identity}-source"].content == statement
    assert by_id[identity].session_id == by_id[f"{identity}-source"].session_id
    assert by_id[identity].revises_event_ids == [original]
    assert all(record.source.value == "user_stated" for record in found.records)
    resumed = run(tmp_path, "quit\n")
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(status(tmp_path).stdout)["tasks"] == payload["tasks"]
    assert len(state(tmp_path)["captures"]) == 1
    assert not (tmp_path / "session/.approval.json").exists()


def test_identical_titles_require_explicit_index_and_do_not_update_both(tmp_path):
    """Use exact displayed identity, not string matching or recency, for two similar tasks."""
    note = "Return the book.\nReturn the book."
    args = inputs(tmp_path, note, [])
    answers = "1 2\n10 normal 2026-11-06\n10 normal 2026-11-06\nCONFIRM\nquit\n"
    assert run(tmp_path, answers, args).returncode == 0
    before = json.loads(status(tmp_path).stdout)["tasks"]
    result = run(tmp_path, renewal("renew 2"))
    assert result.returncode == 0, result.stderr
    after = json.loads(status(tmp_path).stdout)["tasks"]
    assert after[0] == before[0] and after[1]["evidence_id"] != before[1]["evidence_id"]
    assert after[1]["commitment"]["due"] == "2026-11-07"


@pytest.mark.parametrize(
    "answers",
    [
        "I did not do it. Remind me again.\nquit\n",
        "Не сделал, напомни снова.\nquit\n",
        "renew 99\nquit\n",
        "renew 1\nI did not do it.\n",
        renewal(confirm="NO"),
    ],
)
def test_no_target_date_or_confirmation_does_not_change_task(tmp_path, answers):
    """Natural text alone, an unknown index, EOF and refusal must not imply a target or grant."""
    assert initial(tmp_path).returncode == 0
    before = json.loads(status(tmp_path).stdout)["tasks"]
    result = run(tmp_path, answers)
    assert result.returncode == 0, result.stderr
    assert json.loads(status(tmp_path).stdout)["tasks"] == before
    assert state(tmp_path)["pending_renewal"] is None


def test_completed_task_is_not_reopened(tmp_path):
    """Renewal is not an implicit reversal of a previous explicit completion."""
    assert initial(tmp_path).returncode == 0
    assert run(tmp_path, "done 1\nquit\n").returncode == 0
    before = json.loads(status(tmp_path).stdout)["tasks"]
    result = run(tmp_path, "renew 1\nquit\n")
    assert result.returncode == 0 and "Cannot renew a completed task" in result.stdout
    assert json.loads(status(tmp_path).stdout)["tasks"] == before


def crash(root, number):
    """Lose acknowledgement after either the exact source or task revision was committed."""
    script = f"""
import os
from morgan_brain import guided_plan
original = guided_plan.cmd_remember
count = 0
async def fail(*args, **kwargs):
    global count
    result = await original(*args, **kwargs)
    if kwargs.get('client') == 'guided-renewal-v1':
        count += 1
        if count == {number}:
            os._exit(29)
    return result
guided_plan.cmd_remember = fail
guided_plan.main()
"""
    return run(root, renewal(), script=script)


@pytest.mark.parametrize("number", [1, 2])
def test_crash_reconfirms_and_replays_exact_pair_without_duplicates(tmp_path, number):
    """Both partial-write boundaries recover with the original IDs and an explicit review."""
    assert initial(tmp_path).returncode == 0
    killed = crash(tmp_path, number)
    assert killed.returncode == 29, killed.stderr
    pending = state(tmp_path)["pending_renewal"]
    payload = json.loads(status(tmp_path).stdout)
    assert payload["pending_reviews"] == ["renewal_confirmation"] and "tasks" not in payload
    assert run(tmp_path, "NO\n").returncode == 0
    assert state(tmp_path)["pending_renewal"] == pending
    resumed = run(tmp_path, "CONFIRM\nquit\n")
    assert resumed.returncode == 0, resumed.stderr
    assert state(tmp_path)["pending_renewal"] is None
    task = json.loads(status(tmp_path).stdout)["tasks"][0]
    assert task["evidence_id"] == pending["event_id"]
    found = evidence(tmp_path, [pending["event_id"], pending["event_id"] + "-source"])
    assert len(found.records) == 2 and not found.missing_ids
    assert run(tmp_path, "quit\n").returncode == 0
    assert json.loads(status(tmp_path).stdout)["tasks"] == [task]


def external_change(root):
    """Simulate another trusted local writer changing the displayed parent after review."""
    from morgan_brain import guided_plan
    from morgan_brain.local_notes import NoteWorkspace

    workspace = NoteWorkspace(root / "session")
    try:
        session = guided_plan.Session.model_validate_json(workspace.read("session.json", 32768))
        pending = session.pending_renewal
        changed = guided_plan.Capture(
            event_id="external-change",
            effective_at=pending.effective_at,
            commitment=pending.before.model_copy(update={"status": "done"}),
            parents=[pending.parent_id],
        )
        asyncio.run(guided_plan._capture(workspace, session, changed))
    finally:
        workspace.close()


@pytest.mark.parametrize("number", [1, 2])
def test_stale_or_conflicted_parent_cannot_be_overwritten_on_resume(tmp_path, number):
    """A later change or competing branch blocks recovery rather than reopening the task."""
    assert initial(tmp_path).returncode == 0 and crash(tmp_path, number).returncode == 29
    pending = state(tmp_path)["pending_renewal"]
    external_change(tmp_path)
    result = run(tmp_path, "CONFIRM\n")
    assert result.returncode == 2 and "ambiguous" in result.stderr
    assert state(tmp_path)["pending_renewal"] == pending
    record = evidence(tmp_path, ["external-change"]).records[0]
    assert json.loads(record.content)["status"] == "done"


@pytest.mark.parametrize("field", ["user", "project"])
def test_pending_parent_in_another_scope_is_refused(tmp_path, field):
    """Account-controlled journal fields cannot broaden the exact scoped evidence query."""
    assert initial(tmp_path).returncode == 0 and crash(tmp_path, 1).returncode == 29
    saved = state(tmp_path)
    saved[field] = "different"
    (tmp_path / "session/session.json").write_text(json.dumps(saved))
    result = run(tmp_path, "CONFIRM\n")
    assert result.returncode == 2 and "unavailable in this scope" in result.stderr


def test_competing_writer_between_source_and_revision_is_reported_as_conflict(tmp_path):
    """A concurrent revision creates an explicit unresolved branch, never last-writer-wins."""
    assert initial(tmp_path).returncode == 0
    script = """
import argparse
import sys
from pathlib import Path
from morgan_brain import guided_plan
original = guided_plan.cmd_remember
async def race(*args, **kwargs):
    capture = args[0]
    if kwargs.get('client') == 'guided-renewal-v1' and not capture.event_id.endswith('-source'):
        session = guided_plan.Session.model_validate_json(
            (Path(sys.argv[1]) / 'session.json').read_text()
        )
        pending = session.pending_renewal
        other = argparse.Namespace(
            text=pending.before.model_copy(update={'status':'done'}).model_dump_json(),
            source='user_stated', author_id=session.user, event_id='racing-change',
            effective_at=pending.effective_at, revises_event_ids=[pending.parent_id],
            assert_session_id=True,
        )
        await original(other, *args[1:], client='other-local-writer', session_id='other-review')
    return await original(*args, **kwargs)
guided_plan.cmd_remember = race
guided_plan.main()
"""
    result = run(tmp_path, renewal(), script=script)
    assert result.returncode == 2 and "ambiguous" in result.stderr
    pending = state(tmp_path)["pending_renewal"]
    parent = evidence(tmp_path, [pending["parent_id"]]).records[0]
    assert parent.eligible_leaf_count == 2
    assert set(parent.eligible_leaf_ids) == {pending["event_id"], "racing-change"}
    assert json.loads(status(tmp_path).stdout)["pending_reviews"] == ["renewal_confirmation"]
    assert run(tmp_path, "CONFIRM\n").returncode == 2


def test_second_renewal_follows_current_version_without_growing_task_list(tmp_path):
    """A later reviewed renewal revises the current leaf, retaining the original task family."""
    assert initial(tmp_path).returncode == 0
    assert run(tmp_path, renewal()).returncode == 0
    first = json.loads(status(tmp_path).stdout)["tasks"][0]["evidence_id"]
    second_answers = renewal(statement="Still not done, please move the date again.")
    second_answers = second_answers.replace("2026-11-07", "2026-11-08")
    assert run(tmp_path, second_answers).returncode == 0
    tasks = json.loads(status(tmp_path).stdout)["tasks"]
    assert len(tasks) == 1 and tasks[0]["commitment"]["due"] == "2026-11-08"
    current = evidence(tmp_path, [tasks[0]["evidence_id"]]).records[0]
    assert current.revises_event_ids == [first]
