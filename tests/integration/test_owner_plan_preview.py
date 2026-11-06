"""Native date previews surface existing planning evidence without changing the pilot."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain import owner_input
from tests.integration.test_owner_ui import create, drive, initialize, state


def hashes(root):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("session.json", "owner-input.json", "memory/morgan.db")
    }


def test_preview_next_day_and_completion_are_read_only(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    answers = create()
    answers[6] = ("text", "2026-11-07")
    drive(root, [*answers, ("choose", "Выход")])
    original = state(root)
    assert not original["plan"]["selected_ids"]
    before = hashes(root)
    dialogs = drive(
        root,
        [("choose", "План на дату"), ("text", "2026-11-07"), ("info", "OK"), ("choose", "Выход")],
    )
    preview = owner_input.request(root, ["status", "--preview-day", "2026-11-07"])
    assert preview["plan"]["selected_ids"]
    assert (
        preview["plan"]["preview_only"] and preview["plan"]["session_planning_day"] == "2026-11-06"
    )
    assert any(
        "Предпросмотр" in call["prompt"]
        and "2026-11-07" in call["prompt"]
        and "в плане" in call["prompt"]
        for call in dialogs.seen
    )
    assert hashes(root) == before
    assert state(root)["plan"]["planning_day"] == "2026-11-06"
    drive(
        root,
        [
            ("choose", "Новое сообщение"),
            ("text", "Вернул книгу."),
            ("choose", "Завершить задачу"),
            ("choose", 0),
            ("confirm", "Подтвердить"),
            ("info", "OK"),
            ("choose", "Выход"),
        ],
    )
    preview = owner_input.request(root, ["status", "--preview-day", "2026-11-07"])
    assert not preview["plan"]["selected_ids"]
    assert preview["plan"]["tasks"][0]["commitment"]["status"] == "done"


def test_plan_shows_budget_exclusion_and_past_date(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    answers = create()
    answers[4] = ("text", "40")
    drive(root, [*answers, ("choose", "Выход")])
    dialogs = drive(root, [("choose", "План"), ("info", "OK"), ("choose", "Выход")])
    assert any(
        "не помещается в бюджет" in item["prompt"] and "бюджет: 30" in item["prompt"]
        for item in dialogs.seen
    )
    old = owner_input.request(root, ["status", "--preview-day", "2000-01-01"])
    assert old["plan"]["day_relation"] == "past"


@pytest.mark.parametrize("expired", [False, True])
def test_preview_after_stop_and_expiry_never_resumes(tmp_path, expired):
    root = tmp_path / "pilot"
    initialize(root)
    if expired:
        path = root / "owner-input.json"
        data = json.loads(path.read_text())
        data["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        path.write_text(json.dumps(data))
    owner_input.request(root, ["stop"])
    before = hashes(root)
    dialogs = drive(
        root,
        [("choose", "План на дату"), ("text", "2026-11-07"), ("info", "OK"), ("choose", "Выход")],
    )
    assert all(item["allow_stopped"] for item in dialogs.seen)
    assert state(root)["stopped"] and state(root)["expired"] == expired
    assert hashes(root) == before


@pytest.mark.parametrize("answer", [None, "not-a-date"])
def test_cancel_or_invalid_preview_preserves_state(tmp_path, answer):
    root = tmp_path / "pilot"
    initialize(root)
    before = hashes(root)
    responses = [("choose", "План на дату"), ("text", answer)]
    if answer is not None:
        responses.append(("info", "OK"))
    dialogs = drive(root, [*responses, ("choose", "Выход")])
    assert hashes(root) == before
    if answer is not None:
        assert any("Предпросмотр не завершён" in item["prompt"] for item in dialogs.seen)


def test_pending_completion_is_not_presented_as_empty_plan(tmp_path, monkeypatch):
    from tests.integration.test_owner_completion import confirm, setup

    root = tmp_path / "pilot"
    message = setup(root)

    async def fail(*args):
        raise OSError("synthetic interrupted apply")

    monkeypatch.setattr(owner_input, "guided_apply_renewal", fail)
    with pytest.raises(OSError):
        confirm(root, message)
    before = hashes(root)
    dialogs = drive(root, [("choose", "План"), ("info", "OK"), ("choose", "Выход")])
    assert any(
        "План не сформирован" in item["prompt"] and "изменение задачи" in item["prompt"]
        for item in dialogs.seen
    )
    assert not any("Нет доступных задач" in item["prompt"] for item in dialogs.seen)
    assert hashes(root) == before
