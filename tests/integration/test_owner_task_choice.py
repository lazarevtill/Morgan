"""Explicit task choice distinguishes equal titles using verified source quotes."""

import pytest

from morgan_brain import owner_input
from tests.integration.test_owner_ui import drive, initialize, state


@pytest.mark.parametrize(
    "sources,title,completion",
    [
        (
            ("Return the cobalt atlas.", "Return the amber atlas."),
            "Return atlas",
            "Atlas returned.",
        ),
        (("Вернуть синий атлас.", "Вернуть янтарный атлас."), "Вернуть атлас", "Атлас возвращён."),
    ],
)
def test_equal_titles_show_distinct_sources_and_only_selected_task_changes(
    tmp_path, sources, title, completion
):
    initialize(tmp_path)
    for index, source in enumerate(sources):
        identity = f"task-{index}"
        owner_input.request(tmp_path, ["submit", identity], text=source)
        message = owner_input.request(
            tmp_path,
            [
                "review",
                identity,
                "--new",
                "--title",
                title,
                "--minutes",
                "10",
                "--priority",
                "normal",
                "--due",
                "2026-11-06",
            ],
        )["message"]
        owner_input.request(tmp_path, ["confirm", identity, "--digest", message["digest"]])
    dialogs = drive(
        tmp_path,
        [
            ("choose", "Новое сообщение"),
            ("text", completion),
            ("choose", "Завершить задачу"),
            ("choose", 1),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    picker = next(d for d in dialogs.seen if d["prompt"] == "Выберите конкретную задачу")
    assert sources[0] in picker["options"][0]
    assert sources[1] in picker["options"][1]
    confirmation = next(d["prompt"] for d in dialogs.seen if d["kind"] == "confirm")
    assert sources[1] in confirmation
    assert "ID задачи:" in confirmation
    tasks = state(tmp_path)["plan"]["tasks"]
    assert [t["commitment"]["status"] for t in tasks] == ["open", "done"]


def test_source_preview_does_not_use_unconfirmed_receipts(tmp_path):
    import time

    from morgan_brain.owner_ui import OwnerUI
    from tests.integration.test_owner_ui import DialogFixture

    initialize(tmp_path)
    owner_input.request(tmp_path, ["submit", "pending"], text="Return atlas: unverified quote.")
    ui = OwnerUI(tmp_path, DialogFixture([]), time.monotonic() + 60)
    assert ui.task_source("missing-task", "Return atlas") == "Проверенный источник недоступен"


def test_truncated_choice_keeps_full_verified_quote_in_confirmation(tmp_path):
    from tests.integration.test_owner_ui import create

    initialize(tmp_path)
    quote = "Синтетический источник " + "подробность " * 20 + "нужное окончание."
    drive(tmp_path, [*create(text=quote), ("choose", "Выход")])
    before = state(tmp_path)["plan"]["tasks"]
    dialogs = drive(
        tmp_path,
        [
            ("choose", "Новое сообщение"),
            ("text", "Закончил."),
            ("choose", "Завершить задачу"),
            ("choose", 0),
            ("confirm", None),
            ("choose", "Выход"),
        ],
    )
    picker = next(d for d in dialogs.seen if d["prompt"] == "Выберите конкретную задачу")
    assert picker["options"][0].endswith("…")
    confirmation = next(d["prompt"] for d in dialogs.seen if d["kind"] == "confirm")
    assert quote in confirmation
    assert state(tmp_path)["plan"]["tasks"] == before
