"""Cancelling an obligation is distinct from claiming it was completed."""

import pytest

from morgan_brain import owner_input
from tests.integration.test_owner_ui import create, drive, initialize, state


@pytest.mark.parametrize(
    "text", ["Больше не нужно возвращать книгу.", "I no longer need to return the book."]
)
def test_cancel_preserves_source_and_excludes_task_without_claiming_done(tmp_path, text):
    initialize(tmp_path)
    dialogs = drive(
        tmp_path,
        [
            *create(),
            ("choose", "Новое сообщение"),
            ("text", text),
            ("choose", "Отменить задачу"),
            ("choose", 0),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    plan = state(tmp_path)["plan"]
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["commitment"]["status"] == "cancelled"
    assert not plan["selected_ids"]
    found = owner_input.request(tmp_path, ["find", text])["results"][0]
    assert found["quote"] == text and found["current_task"]["status"] == "cancelled"
    assert any("Отменить обязательство" in d["prompt"] for d in dialogs.seen)


def reviewed(root):
    initialize(root)
    drive(root, [*create(), ("choose", "Выход")])
    target = state(root)["plan"]["tasks"][0]["evidence_id"]
    owner_input.request(
        root, ["submit", "cancel"], text="Synthetic cancellation, no longer needed."
    )
    return owner_input.request(root, ["review", "cancel", "--cancel", target])["message"]


def confirm(root, message):
    return owner_input.request(root, ["confirm", "cancel", "--digest", message["digest"]])


def test_cancel_review_can_be_declined_then_resumed(tmp_path):
    message = reviewed(tmp_path)
    drive(
        tmp_path,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("choose", "Показать подтверждение"),
            ("confirm", None),
            ("choose", "Выход"),
        ],
    )
    assert state(tmp_path)["plan"]["tasks"][0]["commitment"]["status"] == "open"
    confirm(tmp_path, message)
    result = state(tmp_path)
    assert result["plan"]["tasks"][0]["commitment"]["status"] == "cancelled"
    assert not result["plan"]["selected_ids"]
    dialogs = drive(
        tmp_path,
        [
            ("choose", "Найти источник"),
            ("text", "Synthetic cancellation"),
            ("choose", 0),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert any("Текущее состояние: отменена" in d["prompt"] for d in dialogs.seen)


@pytest.mark.parametrize("write", [1, 2])
def test_cancel_replay_after_partial_write_is_idempotent(tmp_path, monkeypatch, write):
    message = reviewed(tmp_path)
    namespace = owner_input.guided_apply_renewal.__globals__
    original = namespace["cmd_remember"]
    count = 0

    async def interrupted(*args, **kwargs):
        nonlocal count
        result = await original(*args, **kwargs)
        count += 1
        if count == write:
            raise OSError("synthetic interruption")
        return result

    monkeypatch.setitem(namespace, "cmd_remember", interrupted)
    with pytest.raises(OSError):
        confirm(tmp_path, message)
    assert state(tmp_path)["messages"][-1]["state"] == "applying"
    monkeypatch.setitem(namespace, "cmd_remember", original)
    confirm(tmp_path, message)
    database = tmp_path / "memory/morgan.db"
    before = database.read_bytes()
    confirm(tmp_path, message)
    assert database.read_bytes() == before
    assert len(state(tmp_path)["plan"]["tasks"]) == 1


def test_cancel_respects_stop_stale_target_and_terminal_state(tmp_path):
    from morgan_brain.local_notes import StopRequested

    message = reviewed(tmp_path)
    owner_input.request(tmp_path, ["stop"])
    with pytest.raises(StopRequested):
        confirm(tmp_path, message)
    owner_input.request(tmp_path, ["resume"])
    target = message["renewal"]["parent_id"]
    owner_input.request(tmp_path, ["submit", "change"], text="Move the synthetic task.")
    later = owner_input.request(
        tmp_path, ["review", "change", "--target", target, "--due", "2026-11-07"]
    )["message"]
    owner_input.request(tmp_path, ["confirm", "change", "--digest", later["digest"]])
    with pytest.raises(ValueError, match="target changed"):
        confirm(tmp_path, message)
    current = state(tmp_path)["plan"]["tasks"][0]["evidence_id"]
    message = owner_input.request(tmp_path, ["review", "cancel", "--cancel", current])["message"]
    confirm(tmp_path, message)
    current = state(tmp_path)["plan"]["tasks"][0]["evidence_id"]
    owner_input.request(tmp_path, ["submit", "again"], text="No implicit reopening.")
    for option in ("--complete", "--cancel"):
        with pytest.raises(ValueError, match="open task"):
            owner_input.request(tmp_path, ["review", "again", option, current])
    with pytest.raises(ValueError, match="open task"):
        owner_input.request(
            tmp_path, ["review", "again", "--target", current, "--due", "2026-11-08"]
        )
