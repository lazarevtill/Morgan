"""Real foreground dialogue, synthetic statements, explicit confirmation and crash recovery."""

import json

import pytest

from tests.integration.test_guided_plan import inputs, run

START = "2026-11-06T09:00:00+01:00 2026-11-06T10:00:00+01:00"
FREE = "2026-11-07T09:00:00+01:00 2026-11-07T10:00:00+01:00"


def setup(tmp_path):
    """Prepare a synthetic task proposal without invoking a model."""
    note = "I need to read a chapter."
    return inputs(tmp_path, note, [note])


def task_answers():
    """Provide explicit fixture review decisions for the ordinary task intake."""
    return "1\n10 normal 2026-11-06\nCONFIRM\n"


def busy_answers(command="busy", statement="I am unavailable during this interval."):
    """Provide a reviewed first interval with an explicit planning zone."""
    return f"{command}\n{statement}\nEurope/Paris\n{START}\nCONFIRM\n"


def state(tmp_path):
    """Inspect the private dialogue journal without bypassing memory access."""
    return json.loads((tmp_path / "session/session.json").read_text())


@pytest.mark.parametrize("command", ["busy", "недоступность"])
def test_reviewed_interval_blocks_plan_and_survives_resume(tmp_path, command):
    """Both menu languages persist confirmed constraints and defer the next resumed plan."""
    result = run(tmp_path, task_answers() + busy_answers(command) + "quit\n", setup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "planned: 10 min" in result.stdout and "planned: 0 min" in result.stdout
    assert "Unavailability does not imply location" in result.stdout
    assert "I am unavailable during this interval." in result.stdout
    saved = state(tmp_path)
    assert saved["timezone"] == "Europe/Paris" and len(saved["unavailability_ids"]) == 1
    assert saved["pending_availability"] is None
    resumed = run(tmp_path, "quit\n")
    assert resumed.returncode == 0 and "planned: 0 min" in resumed.stdout
    assert "sources" in resumed.stdout and "Europe/Paris" in resumed.stdout
    assert not (tmp_path / "session/.approval.json").exists()


def test_correction_cancellation_and_completion_keep_stable_constraint_number(tmp_path):
    """Follow revisions under one stable menu index through cancellation and task completion."""
    answers = task_answers() + busy_answers()
    answers += f"изменить 1\nMy unavailable interval moved to the next day.\n{FREE}\nCONFIRM\n"
    answers += "отменить 1\nI cancel that interval.\nПОДТВЕРЖДАЮ\nготово 1\nвыход\n"
    result = run(tmp_path, answers, setup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "U1: cancelled" in result.stdout and "U2:" not in result.stdout
    assert result.stdout.count("planned: 0 min") == 2
    assert len(state(tmp_path)["unavailability_ids"]) == 1
    resumed = run(tmp_path, "quit\n")
    assert "planned: 0 min" in resumed.stdout and "done" in resumed.stdout
    assert "U1: cancelled" in resumed.stdout


def test_declined_confirmation_does_not_save_source_or_constraint(tmp_path):
    """Rejecting the review must not publish a source, interval or planning zone."""
    answers = task_answers() + busy_answers().removesuffix("CONFIRM\n") + "NO\nquit\n"
    result = run(tmp_path, answers, setup(tmp_path))
    assert result.returncode == 0, result.stderr
    saved = state(tmp_path)
    assert not saved["unavailability_ids"] and saved["timezone"] is None
    assert saved["pending_availability"] is None
    assert "planned: 0 min" not in result.stdout


@pytest.mark.parametrize("write_number", [1, 2])
@pytest.mark.parametrize("operation", ["add", "change", "cancel"])
def test_crash_after_source_or_assertion_replays_exact_ids(tmp_path, write_number, operation):
    """Reconfirm and replay each interrupted operation without duplicate constraint families."""
    args = setup(tmp_path)
    initial = task_answers()
    if operation != "add":
        initial += busy_answers()
    assert run(tmp_path, initial + "quit\n", args).returncode == 0
    script = f"""
import os
from morgan_brain import guided_plan
original = guided_plan.cmd_remember
count = 0
async def crash(*args, **kwargs):
    global count
    result = await original(*args, **kwargs)
    if kwargs.get('client') == 'guided-availability-v1':
        count += 1
        if count == {write_number}:
            os._exit(29)
    return result
guided_plan.cmd_remember = crash
guided_plan.main()
"""
    if operation == "add":
        answers = busy_answers()
    elif operation == "change":
        answers = f"change 1\nMy interval moved.\n{FREE}\nCONFIRM\n"
    else:
        answers = "cancel 1\nI cancel this interval.\nCONFIRM\n"
    killed = run(tmp_path, answers, script=script)
    assert killed.returncode == 29, killed.stderr
    pending = state(tmp_path)["pending_availability"]
    assert pending is not None
    declined = run(tmp_path, "NO\n")
    assert declined.returncode == 0 and state(tmp_path)["pending_availability"] == pending
    resumed = run(tmp_path, "CONFIRM\nquit\n")
    assert resumed.returncode == 0, resumed.stderr
    saved = state(tmp_path)
    assert saved["pending_availability"] is None and len(saved["unavailability_ids"]) == 1
    expected = 0 if operation == "add" else 10
    assert f"planned: {expected} min" in resumed.stdout
    assert pending["source_id"] in resumed.stdout and pending["event_id"] in resumed.stdout
    again = run(tmp_path, "quit\n")
    assert again.returncode == 0 and f"planned: {expected} min" in again.stdout


def test_invalid_zone_nonexistent_wall_time_and_end_boundary(tmp_path):
    """Retry invalid zones and DST gaps, while respecting an exclusive midnight end."""
    args = setup(tmp_path)
    answers = task_answers() + "busy\nI am unavailable.\nUnknown/Zone\nEurope/Paris\n"
    answers += "2027-03-28T02:30+01:00 2027-03-28T04:00+02:00\n"
    answers += "2026-11-05T23:00+01:00 2026-11-06T00:00+01:00\nCONFIRM\nquit\n"
    result = run(tmp_path, answers, args)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Invalid input") == 2
    assert "planned: 0 min" not in result.stdout


def test_cancelled_interval_can_be_reactivated_only_after_new_confirmation(tmp_path):
    """A rejected reactivation must retain cancellation until a later confirmed update."""
    answers = task_answers() + busy_answers()
    answers += "cancel 1\nI cancel it.\nCONFIRM\n"
    answers += f"change 1\nI am unavailable again.\n{START}\nNO\nquit\n"
    result = run(tmp_path, answers, setup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("U1: cancelled") == 2
    resumed = run(tmp_path, f"change 1\nI am unavailable again.\n{START}\nCONFIRM\nquit\n")
    assert resumed.returncode == 0 and "planned: 0 min" in resumed.stdout


def test_old_session_without_new_fields_still_resumes(tmp_path):
    """Upgrade an actual legacy journal only when the new reviewed interval is saved."""
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    path = tmp_path / "session/session.json"
    saved = state(tmp_path)
    for key in ("timezone", "unavailability_ids", "pending_availability"):
        saved.pop(key)
    saved["version"] = "morgan.guided-plan.v1"
    path.write_text(json.dumps(saved))
    resumed = run(tmp_path, busy_answers() + "quit\n")
    assert resumed.returncode == 0 and "planned: 0 min" in resumed.stdout
    assert state(tmp_path)["version"] == "morgan.guided-plan.v2"


def test_control_characters_in_source_require_reentry(tmp_path):
    """Reject source text that could manipulate the terminal before capturing it."""
    answers = task_answers() + "busy\nbad\x1b[31m\nI am unavailable.\nEurope/Paris\n"
    result = run(tmp_path, answers + START + "\nCONFIRM\nquit\n", setup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Invalid input" in result.stdout and "\x1b" not in result.stdout


def test_two_constraints_require_cancelling_both_to_release_day(tmp_path):
    """Cancelling one family cannot release a day still covered by another."""
    answers = task_answers() + busy_answers()
    answers += f"busy\nI have another unavailable interval.\n{START}\nCONFIRM\n"
    answers += "cancel 1\nI cancel the first interval.\nCONFIRM\nquit\n"
    result = run(tmp_path, answers, setup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert len(state(tmp_path)["unavailability_ids"]) == 2
    assert "U1: cancelled" in result.stdout and "U2: active" in result.stdout
    resumed = run(tmp_path, "cancel 2\nI cancel the second interval.\nCONFIRM\nquit\n")
    assert resumed.returncode == 0, resumed.stderr
    assert "planned: 0 min" in resumed.stdout and "planned: 10 min" in resumed.stdout


def test_saved_source_is_exact_and_bound_to_confirmed_assertion(tmp_path):
    """Read source provenance through MemoryGate and verify the shared immutable binding."""
    import asyncio

    from morgan_brain.composition import open_evidence_context

    statement = "Недоступен в это время; местоположение не сообщаю."
    assert (
        run(
            tmp_path, task_answers() + busy_answers(statement=statement) + "quit\n", setup(tmp_path)
        ).returncode
        == 0
    )
    saved = state(tmp_path)
    context = open_evidence_context(tmp_path / "session/memory/morgan.db")

    async def inspect():
        """Check the exact captured quote and provenance without raw database access."""
        constraint = await context.gate.evidence(
            user_id="pilot",
            project="personal",
            evidence_ids=saved["unavailability_ids"],
        )
        payload = json.loads(constraint.records[0].content)
        source = await context.gate.evidence(
            user_id="pilot",
            project="personal",
            evidence_ids=payload["source_ids"],
        )
        assert len(source.records) == 1 and source.records[0].content == statement
        assert source.records[0].source.value == "user_stated"
        assert source.records[0].session_id == constraint.records[0].session_id
        assert source.records[0].created_at == constraint.records[0].created_at
        assert "location" not in payload

    try:
        asyncio.run(inspect())
    finally:
        context.conn.close()


def test_invalid_pending_source_binding_refuses_resume(tmp_path):
    """Refuse a journal whose pending assertion names a different source."""
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    saved = state(tmp_path)
    saved["pending_availability"] = {
        "source_id": "source",
        "source_text": "I am unavailable.",
        "event_id": "assertion",
        "effective_at": "2026-11-06T08:00:00+00:00",
        "parent_id": None,
        "interval": {
            "version": "morgan.unavailability.v1",
            "starts_at": "2026-11-06T09:00:00+01:00",
            "ends_at": "2026-11-06T10:00:00+01:00",
            "timezone": "Europe/Paris",
            "status": "active",
            "source_ids": ["different-source"],
        },
    }
    (tmp_path / "session/session.json").write_text(json.dumps(saved))
    result = run(tmp_path, "CONFIRM\nquit\n")
    assert result.returncode == 2 and "bind its exact source" in result.stderr
