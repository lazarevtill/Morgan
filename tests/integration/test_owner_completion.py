"""Explicit owner completion reuses reviewed transitions and crash-safe paired writes."""

import json

import pytest

from morgan_brain import owner_input
from morgan_brain.guided_renewal import RenewalUpdate
from morgan_brain.local_notes import StopRequested
from tests.integration.test_owner_ui import create, drive, initialize, state


def setup(root):
    initialize(root)
    drive(root, [*create(), ("choose", "Выход")])
    target = state(root)["plan"]["tasks"][0]["evidence_id"]
    owner_input.request(root, ["submit", "completion"], text="Книгу вернул.")
    return owner_input.request(root, ["review", "completion", "--complete", target])["message"]


def confirm(root, message):
    return owner_input.request(root, ["confirm", "completion", "--digest", message["digest"]])


@pytest.mark.parametrize("text", ["Книгу вернул.", "I returned the book."])
def test_native_completion_keeps_source_and_updates_search(tmp_path, text):
    root = tmp_path / "pilot"
    initialize(root)
    dialogs = drive(
        root,
        [
            *create(),
            ("choose", "Новое сообщение"),
            ("text", text),
            ("choose", "Завершить задачу"),
            ("choose", 0),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    result = state(root)
    assert len(result["plan"]["tasks"]) == 1
    assert result["plan"]["tasks"][0]["commitment"]["status"] == "done"
    assert result["plan"]["tasks"][0]["commitment"]["due"] == "2026-11-06"
    assert any(
        "Отметить выполненной" in item["prompt"] and text in item["prompt"] for item in dialogs.seen
    )
    found = owner_input.request(root, ["find", text])
    assert found["results"][0]["current_task"]["status"] == "done"
    assert found["results"][0]["quote"] == text


def test_cancel_completion_and_resume_exact_review(tmp_path):
    root = tmp_path / "pilot"
    setup(root)
    drive(
        root,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("choose", "Показать подтверждение"),
            ("confirm", None),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["plan"]["tasks"][0]["commitment"]["status"] == "open"
    drive(
        root,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("choose", "Показать подтверждение"),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["plan"]["tasks"][0]["commitment"]["status"] == "done"


def test_completion_recovers_after_source_write_without_duplicate(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    message = setup(root)
    original = owner_input.guided_apply_renewal.__globals__["cmd_remember"]
    calls = 0

    async def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interrupted task write")
        return await original(*args, **kwargs)

    monkeypatch.setitem(owner_input.guided_apply_renewal.__globals__, "cmd_remember", interrupt)
    with pytest.raises(OSError):
        confirm(root, message)
    assert state(root)["messages"][-1]["state"] == "applying"
    monkeypatch.setitem(owner_input.guided_apply_renewal.__globals__, "cmd_remember", original)
    confirm(root, message)
    before = (root / "memory/morgan.db").read_bytes()
    confirm(root, message)
    assert (root / "memory/morgan.db").read_bytes() == before
    assert state(root)["plan"]["tasks"][0]["commitment"]["status"] == "done"


def test_completion_rejects_stale_parent_stop_and_field_changes(tmp_path):
    root = tmp_path / "pilot"
    message = setup(root)
    update = message["renewal"]
    bad = json.loads(json.dumps(update))
    bad["after"]["due"] = "2026-11-07"
    with pytest.raises(ValueError, match="completion"):
        RenewalUpdate.model_validate_json(json.dumps(bad))
    owner_input.request(root, ["stop"])
    with pytest.raises(StopRequested):
        confirm(root, message)
    owner_input.request(root, ["resume"])
    target = update["parent_id"]
    owner_input.request(root, ["submit", "later"], text="Перенеси на завтра.")
    later = owner_input.request(
        root, ["review", "later", "--target", target, "--due", "2026-11-07"]
    )["message"]
    owner_input.request(root, ["confirm", "later", "--digest", later["digest"]])
    with pytest.raises(ValueError, match="changed"):
        confirm(root, message)
    assert state(root)["plan"]["tasks"][0]["commitment"]["status"] == "open"


def test_expiry_and_wrong_digest_do_not_complete(tmp_path):
    from datetime import UTC, datetime, timedelta

    root = tmp_path / "pilot"
    message = setup(root)
    with pytest.raises(ValueError, match="digest"):
        owner_input.request(root, ["confirm", "completion", "--digest", "wrong"])
    journal = root / "owner-input.json"
    payload = json.loads(journal.read_text())
    payload["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    journal.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expired"):
        confirm(root, message)
    assert state(root)["plan"]["tasks"][0]["commitment"]["status"] == "open"


def test_completion_cannot_change_date_or_complete_done_task(tmp_path):
    root = tmp_path / "pilot"
    message = setup(root)
    with pytest.raises(ValueError, match="preserves"):
        owner_input.request(
            root,
            [
                "review",
                "completion",
                "--complete",
                message["renewal"]["parent_id"],
                "--due",
                "2026-11-07",
            ],
        )
    confirm(root, message)
    target = state(root)["plan"]["tasks"][0]["evidence_id"]
    owner_input.request(root, ["submit", "again"], text="Already done.")
    with pytest.raises(ValueError, match="completion"):
        owner_input.request(root, ["review", "again", "--complete", target])
