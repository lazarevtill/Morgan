"""Date suggestions follow a status snapshot's explicit zone, never the host timezone."""

import json
import time
from datetime import datetime

import pytest

from morgan_brain import daily_brief, guided_plan, owner_input
from morgan_brain.owner_ui import OwnerUI
from tests.integration.test_owner_plan_preview import hashes
from tests.integration.test_owner_ui import DialogFixture, initialize, state


@pytest.mark.parametrize(
    "zone,instant,stored,expected",
    [
        ("Europe/Belgrade", "2026-11-06T23:30:00+00:00", "2026-11-06", "2026-11-07"),
        ("Pacific/Kiritimati", "2026-11-06T12:00:00+00:00", "2026-11-06", "2026-11-07"),
        ("America/Los_Angeles", "2026-11-06T12:00:00+00:00", "2026-11-06", "2026-11-06"),
        ("Europe/Belgrade", "2026-11-06T23:30:00+00:00", "2026-11-08", "2026-11-08"),
        (None, "2026-11-06T23:30:00+00:00", "2026-11-06", "2026-11-06"),
        ("Europe/Belgrade", "2026-11-30T23:30:00+00:00", "2026-11-06", "2026-12-01"),
    ],
)
def test_new_task_and_preview_suggestions_do_not_rewrite_saved_day(
    tmp_path, monkeypatch, zone, instant, stored, expected
):
    initialize(tmp_path)
    session_path = tmp_path / "session.json"
    session = json.loads(session_path.read_text())
    session.update(timezone=zone, day=stored)
    session_path.write_text(json.dumps(session))
    owner_input.request(tmp_path, ["submit", "new"], text="My own task: return the atlas.")
    fixed = datetime.fromisoformat(instant)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz)

    monkeypatch.setattr(guided_plan, "datetime", Clock)
    monkeypatch.setattr(daily_brief, "datetime", Clock)
    snapshot = state(tmp_path)
    before = hashes(tmp_path)
    preview = DialogFixture([("text", None)])
    OwnerUI(tmp_path, preview, time.monotonic() + 30).preview_plan(snapshot)
    assert preview.seen[0]["initial"] == expected
    assert hashes(tmp_path) == before
    dialogs = DialogFixture(
        [
            ("choose", "Новая задача"),
            ("text", "Return the atlas"),
            ("text", "10"),
            ("choose", "Обычный"),
            ("text", "2026-12-20"),
            ("confirm", None),
        ]
    )
    OwnerUI(tmp_path, dialogs, time.monotonic() + 30).review(snapshot["messages"][0], snapshot)
    due_prompt = next(
        x for x in dialogs.seen if x["kind"] == "text" and "ГГГГ-ММ-ДД" in x["prompt"]
    )
    assert due_prompt["initial"] == expected
    assert not state(tmp_path)["plan"]["tasks"]
    assert json.loads(session_path.read_text())["day"] == stored
    assert state(tmp_path)["messages"][0]["capture"]["commitment"]["due"] == "2026-12-20"
    # Confirmation uses real event time; the frozen snapshot tests only date suggestions.
    monkeypatch.undo()
    reviewed = state(tmp_path)["messages"][0]
    owner_input.request(tmp_path, ["confirm", "new", "--digest", reviewed["digest"]])
    assert state(tmp_path)["plan"]["tasks"][0]["commitment"]["due"] == "2026-12-20"
    assert json.loads(session_path.read_text())["day"] == stored


def test_reschedule_suggestion_does_not_change_existing_task_without_confirmation(tmp_path):
    from tests.integration.test_owner_search import capture

    initialize(tmp_path)
    capture(tmp_path)
    owner_input.request(tmp_path, ["submit", "repeat"], text="I have not returned the book.")
    snapshot = state(tmp_path)
    snapshot["plan"].update(as_of="2026-11-07T00:30:00+00:00", day_relation="past")
    dialogs = DialogFixture(
        [
            ("choose", "Перенести задачу"),
            ("choose", 0),
            ("text", "2026-12-20"),
            ("confirm", None),
        ]
    )
    OwnerUI(tmp_path, dialogs, time.monotonic() + 30).review(snapshot["messages"][-1], snapshot)
    date_prompt = next(x for x in dialogs.seen if x["kind"] == "text")
    assert date_prompt["initial"] == "2026-11-07"
    result = state(tmp_path)
    assert result["plan"]["tasks"][0]["commitment"]["due"] == "2026-11-06"
    assert result["messages"][-1]["renewal"]["after"]["due"] == "2026-12-20"
