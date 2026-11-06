"""A bounded native owner review session; no listener, background service or model calls."""

from __future__ import annotations

import argparse
import fcntl
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from morgan_brain import owner_input
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.native_dialog import DialogClosed, Dialogs, NativeDialogs
from morgan_brain.tasks import TaskConflict


class OwnerUI:
    """Present existing receipts and exact reviews without a second persistence mechanism."""

    def __init__(self, workspace: Path, dialogs: Dialogs, deadline: float) -> None:
        """Bind one foreground session to one private owner-input workspace."""
        self.workspace, self.dialogs, self.deadline = workspace, dialogs, deadline

    def call(self, *arguments: str, text: str = "") -> dict[str, Any]:
        """Keep command scope fixed and reject mutations after the foreground deadline."""
        if time.monotonic() >= self.deadline:
            raise DialogClosed("UI session deadline reached")
        return owner_input.request(self.workspace, list(arguments), text=text)

    def text(self, prompt: str, initial: str = "") -> str:
        """Treat cancellation at any form field as incomplete input, never a default decision."""
        result = self.dialogs.show("text", prompt, initial=initial)
        if result is None:
            raise DialogClosed("form cancelled; saved messages remain pending")
        return result

    def confirm(self, envelope: dict[str, Any]) -> None:
        """Display the exact persisted proposal and bind only an explicit button to its digest."""
        message = envelope["message"]
        renewal = message["renewal"]
        item = renewal["after"] if renewal else message["capture"]["commitment"]
        action = "Создать задачу" if not renewal else f"Перенести с {renewal['before']['due']}"
        prompt = (
            f"Владелец: {envelope['owner']}\nПроект: {envelope['project']}\n"
            f"Источник: {message['text']}\n\n{action}\n"
            f"Задача: {item['title']}\nСрок: {item['due']}\n"
            f"Минуты: {item['minutes']}\nПриоритет: {item['priority']}\n"
            f"Состояние: {item['status']}\n\n"
            "Подтверждение изменит только план. Уведомление не отправляется."
        )
        if self.dialogs.show("confirm", prompt) != "Подтвердить":
            return
        self.call("confirm", message["identity"], "--digest", message["digest"])
        self.dialogs.show("info", "Изменение сохранено в плане.")

    def review(self, message: dict[str, Any], state: dict[str, Any]) -> None:
        """Resume exact attempted reviews or collect explicitly chosen new planning fields."""
        envelope = {
            "owner": state["plan"]["owner"],
            "project": state["plan"]["project"],
            "message": message,
        }
        if message["state"] == "applying":
            self.confirm(envelope)
            return
        if message["state"] == "reviewed":
            choice = self.dialogs.show(
                "choose",
                "Сохранённая проверка сообщения",
                options=("Показать подтверждение", "Изменить поля", "Позже"),
            )
            if choice == "Показать подтверждение":
                self.confirm(envelope)
                return
            if choice != "Изменить поля":
                return
        action = self.dialogs.show(
            "choose",
            f"Ваше сообщение:\n{message['text']}\n\nЧто сделать?",
            options=("Новая задача", "Перенести задачу", "Оставить на проверку"),
        )
        day = state["plan"]["planning_day"]
        if action == "Новая задача":
            title = self.text("Название вашей задачи", message["text"][:240])
            minutes = self.text("Сколько минут выделить задаче?", "10")
            priority = self.dialogs.show(
                "choose", "Приоритет задачи", options=("Обычный", "Высокий", "Низкий")
            )
            if priority is None:
                return
            due = self.text("Срок задачи: ГГГГ-ММ-ДД", day)
            envelope = self.call(
                "review",
                message["identity"],
                "--new",
                "--title",
                title,
                "--minutes",
                minutes,
                "--priority",
                {"Обычный": "normal", "Высокий": "high", "Низкий": "low"}[priority],
                "--due",
                due,
            )
        elif action == "Перенести задачу":
            tasks = [
                item
                for item in state["plan"].get("tasks", [])
                if item["commitment"]["status"] == "open"
            ]
            if not tasks:
                self.dialogs.show("info", "Нет открытых задач для переноса.")
                return
            labels = tuple(
                f"{i + 1}. {item['commitment']['title']} — {item['commitment']['due']}"
                for i, item in enumerate(tasks)
            )
            selected = self.dialogs.show("choose", "Выберите конкретную задачу", options=labels)
            if selected is None:
                return
            target = tasks[labels.index(selected)]["evidence_id"]
            due = self.text("Новый срок: ГГГГ-ММ-ДД", day)
            envelope = self.call("review", message["identity"], "--target", target, "--due", due)
        else:
            return
        self.confirm(envelope)

    def pending(self, state: dict[str, Any]) -> None:
        """Select by a stable receipt identity even when two messages have identical text."""
        messages = [item for item in state["messages"] if item["state"] != "applied"]
        if not messages:
            self.dialogs.show("info", "Нет сообщений, ожидающих проверки.")
            return
        labels = tuple(
            f"{i + 1}. {item['text'][:120]} [{item['state']}]" for i, item in enumerate(messages)
        )
        selected = self.dialogs.show("choose", "Сообщения для проверки", options=labels)
        if selected is not None:
            self.review(messages[labels.index(selected)], state)

    def run(self) -> None:
        """Run a finite foreground menu; cancel/exit never sends or confirms queued work."""
        for _ in range(30):
            state = self.call("status")
            options = (
                ("Продолжить после STOP", "План", "Выход")
                if state["stopped"]
                else ("Новое сообщение", "Проверить сообщения", "План", "STOP", "Выход")
            )
            choice = self.dialogs.show(
                "choose",
                "Morgan — локальный план\n"
                f"Срок пилота: {state['expires_at']}\n"
                f"Дата плана: {state['plan']['planning_day']}",
                options=options,
                allow_stopped=True,
            )
            if choice in (None, "Выход"):
                return
            try:
                if choice == "STOP":
                    self.call("stop")
                elif choice == "Продолжить после STOP":
                    if (
                        self.dialogs.show(
                            "confirm",
                            "Снять STOP? Задачи автоматически не выполняются.",
                            allow_stopped=True,
                        )
                        == "Подтвердить"
                    ):
                        self.call("resume")
                elif choice == "План":
                    tasks = state["plan"].get("tasks", [])
                    lines = [
                        f"{item['commitment']['title']} — {item['commitment']['due']} "
                        f"({item['commitment']['status']})"
                        for item in tasks
                    ]
                    self.dialogs.show(
                        "info",
                        "\n".join(lines) or "Нет доступных задач; проверьте сообщения.",
                        allow_stopped=True,
                    )
                elif choice == "Новое сообщение":
                    text = self.text("Ваше сообщение (одна строка, до 1000 символов)")
                    receipt = self.call("submit", f"ui-{uuid4().hex}", text=text)
                    if "message" in receipt:
                        self.review(receipt["message"], self.call("status"))
                elif choice == "Проверить сообщения":
                    self.pending(state)
            except (ValueError, OSError, TaskConflict) as error:
                self.dialogs.show(
                    "info",
                    f"Изменение не завершено: {error}\nПроверьте сохранённое состояние.",
                    allow_stopped=True,
                )


def main() -> None:
    """Open an existing private pilot in a bounded GUI session; never install a service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--init-user")
    parser.add_argument("--init-project")
    parser.add_argument("--init-timezone")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 1800:
        parser.error("select a foreground lifetime of 5..1800 seconds")
    if any((args.init_user, args.init_project, args.init_timezone)):
        if not all((args.init_user, args.init_project, args.init_timezone)):
            parser.error("initialization requires user, project and timezone")
        day = datetime.now(ZoneInfo(args.init_timezone)).date().isoformat()
        owner_input.request(
            args.workspace,
            [
                "init",
                "--user",
                args.init_user,
                "--project",
                args.init_project,
                "--timezone",
                args.init_timezone,
                "--day",
                day,
                "--budget",
                "30",
            ],
        )
    workspace = NoteWorkspace(args.workspace, create=False)
    try:
        with os.fdopen(workspace.open_file(".owner-ui.lock", os.O_RDWR | os.O_CREAT), "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + args.seconds
            OwnerUI(args.workspace, NativeDialogs(args.workspace, deadline), deadline).run()
    except (DialogClosed, StopRequested, KeyboardInterrupt):
        print("UI closed; pending messages remain unconfirmed.")
    except (ValueError, OSError, TaskConflict) as error:
        parser.exit(2, f"Owner UI failed: {error}\n")
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
