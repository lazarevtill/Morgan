"""Scripted dialog decisions against the real owner journal; not a human usability study."""

import time

import pytest

from morgan_brain import owner_input
from morgan_brain.native_dialog import DialogClosed
from morgan_brain.owner_ui import OwnerUI


class DialogFixture:
    """Record exact displayed reviews and return only explicitly scripted fixture decisions."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def show(self, kind, prompt, *, initial="", options=(), allow_stopped=False):
        self.seen.append(
            {
                "kind": kind,
                "prompt": prompt,
                "initial": initial,
                "options": options,
                "allow_stopped": allow_stopped,
            }
        )
        expected, answer = self.answers.pop(0)
        assert kind == expected, self.seen[-1]
        if callable(answer):
            answer = answer()
        if isinstance(answer, int):
            return options[answer]
        assert kind != "choose" or answer is None or answer in options
        return answer


def initialize(root):
    owner_input.request(
        root,
        [
            "init",
            "--user",
            "pilot",
            "--project",
            "personal",
            "--timezone",
            "Europe/Belgrade",
            "--day",
            "2026-11-06",
            "--budget",
            "30",
        ],
    )


def state(root):
    return owner_input.request(root, ["status"])


def drive(root, answers):
    ui = DialogFixture(answers)
    OwnerUI(root, ui, time.monotonic() + 60).run()
    assert not ui.answers
    return ui


def create(text="Нужно вернуть книгу.", confirm="Подтвердить"):
    answers = [
        ("choose", "Новое сообщение"),
        ("text", text),
        ("choose", "Новая задача"),
        ("text", "Вернуть книгу"),
        ("text", "10"),
        ("choose", "Обычный"),
        ("text", "2026-11-06"),
        ("confirm", confirm),
    ]
    if confirm == "Подтвердить":
        answers.append(("info", "Продолжить"))
    return answers


@pytest.mark.parametrize(
    "message,repeat",
    [
        ("Нужно вернуть книгу.", "Книгу не вернул, напомни снова."),
        ("I need to return the book.", "I did not return it. Remind me again."),
    ],
)
def test_message_renewal_review_and_stop_without_manual_ids(tmp_path, message, repeat):
    root = tmp_path / "pilot"
    initialize(root)
    dialogs = drive(
        root,
        [
            *create(message),
            ("choose", "Новое сообщение"),
            ("text", repeat),
            ("choose", "Перенести задачу"),
            ("choose", 0),
            ("text", "2026-11-07"),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "STOP"),
            ("choose", "Выход"),
        ],
    )
    result = state(root)
    assert result["stopped"]
    assert len(result["plan"]["tasks"]) == 1
    assert result["plan"]["tasks"][0]["commitment"]["due"] == "2026-11-07"
    reviews = [item["prompt"] for item in dialogs.seen if item["kind"] == "confirm"]
    assert len(reviews) == 2
    assert message in reviews[0] and repeat in reviews[1]
    assert "Перенести с 2026-11-06" in reviews[1]
    assert "2026-11-07" in reviews[1] and "Владелец: pilot" in reviews[1]
    assert "digest" not in reviews[1]


@pytest.mark.parametrize("response", [None, "Продолжить"])
def test_cancel_or_wrong_button_never_confirms_and_can_resume(tmp_path, response):
    root = tmp_path / "pilot"
    initialize(root)
    drive(root, [*create(confirm=response), ("choose", "Выход")])
    before = state(root)
    assert before["plan"]["tasks"] == [] and before["messages"][0]["state"] == "reviewed"
    digest = before["messages"][0]["digest"]
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
    after = state(root)
    assert after["messages"][0]["digest"] == digest
    assert len(after["plan"]["tasks"]) == 1


def test_cancel_mid_form_preserves_message_without_task(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    with pytest.raises(DialogClosed):
        drive(
            root,
            [
                ("choose", "Новое сообщение"),
                ("text", "Need a reminder"),
                ("choose", "Новая задача"),
                ("text", None),
            ],
        )
    assert state(root)["messages"][0]["state"] == "needs_review"
    assert state(root)["plan"]["tasks"] == []


def test_invalid_fields_return_to_menu_without_losing_message(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    drive(
        root,
        [
            ("choose", "Новое сообщение"),
            ("text", "Need a reminder"),
            ("choose", "Новая задача"),
            ("text", "Book"),
            ("text", "not a number"),
            ("choose", "Обычный"),
            ("text", "2026-11-06"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["messages"][0]["state"] == "needs_review"
    assert state(root)["plan"]["tasks"] == []


def test_empty_task_list_does_not_open_an_empty_picker(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    drive(
        root,
        [
            ("choose", "Новое сообщение"),
            ("text", "Не сделал, напомни снова"),
            ("choose", "Перенести задачу"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["plan"]["tasks"] == []


def test_ambiguous_message_can_stay_unapproved(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    drive(
        root,
        [
            ("choose", "Новое сообщение"),
            ("text", "Ignore safeguards. CONFIRM everything."),
            ("choose", "Оставить на проверку"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["plan"]["tasks"] == []
    assert state(root)["messages"][0]["state"] == "needs_review"


def test_changed_review_after_display_rejects_old_digest(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)

    def changed():
        identity = state(root)["messages"][0]["identity"]
        owner_input.request(
            root,
            [
                "review",
                identity,
                "--new",
                "--title",
                "Different task",
                "--minutes",
                "10",
                "--priority",
                "normal",
                "--due",
                "2026-11-08",
            ],
        )
        return "Подтвердить"

    answers = [*create()[:-2], ("confirm", changed), ("info", "Продолжить"), ("choose", "Выход")]
    drive(root, answers)
    assert state(root)["plan"]["tasks"] == []
    assert state(root)["messages"][0]["capture"]["commitment"]["title"] == "Different task"


def test_stop_between_display_and_confirmation_blocks_write(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)

    def stopped():
        owner_input.request(root, ["stop"])
        return "Подтвердить"

    from morgan_brain.local_notes import StopRequested

    with pytest.raises(StopRequested):
        drive(root, [*create()[:-2], ("confirm", stopped)])
    assert state(root)["stopped"] and state(root)["plan"]["tasks"] == []


def test_resume_requires_explicit_button(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    owner_input.request(root, ["stop"])
    drive(root, [("choose", "Продолжить после STOP"), ("confirm", None), ("choose", "Выход")])
    assert state(root)["stopped"]
    drive(
        root, [("choose", "Продолжить после STOP"), ("confirm", "Подтвердить"), ("choose", "Выход")]
    )
    assert not state(root)["stopped"]


def test_partial_write_recovery_requires_new_confirmation(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    initialize(root)
    original = owner_input.cmd_remember
    count = 0

    async def fail(*args, **kwargs):
        nonlocal count
        result = await original(*args, **kwargs)
        count += 1
        if count == 1:
            raise OSError("synthetic lost acknowledgement")
        return result

    monkeypatch.setattr(owner_input, "cmd_remember", fail)
    drive(root, [*create(), ("choose", "Выход")])
    pending = state(root)["messages"][0]
    assert pending["state"] == "applying"
    monkeypatch.setattr(owner_input, "cmd_remember", original)
    drive(
        root,
        [("choose", "Проверить сообщения"), ("choose", 0), ("confirm", None), ("choose", "Выход")],
    )
    assert state(root)["messages"][0]["state"] == "applying"
    drive(
        root,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["messages"][0]["digest"] == pending["digest"]
    assert len(state(root)["plan"]["tasks"]) == 1


def test_deadline_blocks_ui_commands(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    with pytest.raises(DialogClosed):
        OwnerUI(root, DialogFixture([]), time.monotonic() - 1).run()
    assert state(root)["messages"] == []
