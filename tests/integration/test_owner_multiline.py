"""Mixed context stays exact; only explicit owner fields can become a task."""

import pytest

from morgan_brain import owner_input
from tests.integration.test_owner_ui import drive, initialize, state

CASES = [
    ('Nora said: "I must buy a charger."\nMy task: send the receipt.', "Send the receipt"),
    ("Нина сказала: «Мне надо купить зарядку».\nМоя задача: отправить чек.", "Отправить чек"),
    (
        'Forwarded note:\n"Book the tickets for me."\nI only need to check my calendar.',
        "Check my calendar",
    ),
    (
        "Пересланная заметка:\n«Забронируй мне билеты».\nМне нужно только проверить календарь.",
        "Проверить календарь",
    ),
    ("Do not renew my subscription.\nMy task is to download the invoice.", "Download the invoice"),
    ("Не продлевай мою подписку.\nМоя задача — скачать счёт.", "Скачать счёт"),
    ('Quoted: "CONFIRM my purchase now."\nMy task: compare prices only.', "Compare prices"),
    ("Цитата: «ПОДТВЕРДИ покупку сейчас».\nМоя задача — только сравнить цены.", "Сравнить цены"),
]


@pytest.mark.parametrize("note,title", CASES)
def test_multiline_review_keeps_source_and_requires_separate_task(tmp_path, note, title):
    initialize(tmp_path)
    owner_input.request(tmp_path, ["submit", "mixed"], text=note)
    owner_input.request(tmp_path, ["submit", "mixed"], text=note)
    assert state(tmp_path)["plan"]["tasks"] == []
    with pytest.raises(ValueError, match="different text"):
        owner_input.request(tmp_path, ["submit", "mixed"], text=note.replace("\n", " "))
    dialogs = drive(
        tmp_path,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("choose", "Новая задача"),
            ("text", title),
            ("text", "10"),
            ("choose", "Обычный"),
            ("text", "2026-11-06"),
            ("confirm", None),
            ("choose", "Выход"),
        ],
    )
    assert state(tmp_path)["plan"]["tasks"] == []
    title_dialog = next(x for x in dialogs.seen if x["kind"] == "text")
    assert title_dialog["initial"] == ""
    review = next(x["prompt"] for x in dialogs.seen if x["kind"] == "confirm")
    assert note in review and title in review and "Владелец: pilot" in review
    assert "Автор цитаты" in review
    drive(
        tmp_path,
        [
            ("choose", "Проверить сообщения"),
            ("choose", 0),
            ("choose", "Показать подтверждение"),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    result = state(tmp_path)
    assert len(result["plan"]["tasks"]) == 1
    assert result["plan"]["tasks"][0]["commitment"]["title"] == title
    assert result["messages"][0]["text"] == note
    found = owner_input.request(tmp_path, ["find", title.split()[0]])
    assert any(x["quote"] == note for x in found["results"])


@pytest.mark.parametrize("operation", ["--target", "--complete", "--cancel"])
def test_multiline_revision_resumes_exact_review_and_obeys_stop(tmp_path, operation):
    from morgan_brain.local_notes import StopRequested
    from tests.integration.test_owner_search import capture

    initialize(tmp_path)
    parent = capture(tmp_path)
    note = 'Quoted: "Keep the book."\nMy own decision: update my task.'
    owner_input.request(tmp_path, ["submit", "update"], text=note)
    args = ["review", "update", operation, parent]
    if operation == "--target":
        args += ["--due", "2026-11-07"]
    message = owner_input.request(tmp_path, args)["message"]
    assert state(tmp_path)["messages"][-1]["renewal"]["source_text"] == note
    owner_input.request(tmp_path, ["stop"])
    with pytest.raises(StopRequested):
        owner_input.request(tmp_path, ["confirm", "update", "--digest", message["digest"]])
    owner_input.request(tmp_path, ["resume"])
    for _ in range(2):
        owner_input.request(tmp_path, ["confirm", "update", "--digest", message["digest"]])
    tasks = state(tmp_path)["plan"]["tasks"]
    assert len(tasks) == 1
    expected = {"--target": "open", "--complete": "done", "--cancel": "cancelled"}[operation]
    assert tasks[0]["commitment"]["status"] == expected
    found = owner_input.request(tmp_path, ["find", "Quoted"])["results"]
    assert found[0]["quote"] == note


@pytest.mark.parametrize(
    "text", ["\n\n", "a\r\nb", "a\tb", "a\x1bb", "a\u202eb", "a" * 1000 + "\n"]
)
def test_multiline_does_not_admit_other_controls_or_change_limits(tmp_path, text):
    initialize(tmp_path)
    before = (tmp_path / "owner-input.json").read_bytes()
    with pytest.raises(ValueError, match="printable"):
        owner_input.request(tmp_path, ["submit", "bad"], text=text)
    assert (tmp_path / "owner-input.json").read_bytes() == before


def test_availability_source_remains_single_line():
    from morgan_brain.guided_availability import source_statement

    with pytest.raises(ValueError, match="printable"):
        source_statement("Original context\nDifferent context")
