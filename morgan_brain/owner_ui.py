"""A bounded native owner review session with optional explicit local-model drafts."""

from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from morgan_brain import owner_input
from morgan_brain.guided_plan import _human_status as render_plan_status
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle
from morgan_brain.native_dialog import DialogClosed, Dialogs, NativeDialogs
from morgan_brain.owner_model import OwnerModel
from morgan_brain.tasks import TaskConflict


def _suggested_day(plan: dict[str, Any]) -> str:
    """Suggest local today only for a past session day; keep the saved plan unchanged."""
    if plan["day_relation"] == "past" and plan["timezone"] is not None:
        instant = datetime.fromisoformat(plan["as_of"])
        return instant.astimezone(ZoneInfo(plan["timezone"])).date().isoformat()
    return str(plan["planning_day"])


class OwnerUI:
    """Present existing receipts and exact reviews without a second persistence mechanism."""

    def __init__(
        self, workspace: Path, dialogs: Dialogs, deadline: float, model: OwnerModel | None = None
    ) -> None:
        """Bind one foreground session to one private owner-input workspace."""
        self.workspace, self.dialogs, self.deadline = workspace, dialogs, deadline
        self.model = model

    def call(self, *arguments: str, text: str = "") -> dict[str, Any]:
        """Keep command scope fixed and reject mutations after the foreground deadline."""
        if time.monotonic() >= self.deadline:
            raise DialogClosed("UI session deadline reached")
        if arguments == ("stop",) and self.model is not None:
            self.model.stop()
        return owner_input.request(self.workspace, list(arguments), text=text)

    def text(self, prompt: str, initial: str = "") -> str:
        """Treat cancellation at any form field as incomplete input, never a default decision."""
        result = self.dialogs.show("text", prompt, initial=initial)
        if result is None:
            raise DialogClosed("form cancelled; saved messages remain pending")
        return result

    def task_source(self, identity: str, title: str) -> str:
        """Read a verified quote for this exact task, never substitute an inbox receipt."""
        query = title.strip().splitlines()[0][:200]
        found = self.call("find", "--", query)
        for item in found["results"]:
            if item["source_kind"] == "verified_memory" and (
                item["current_evidence_id"] == identity or item["source_id"] == f"{identity}-source"
            ):
                return str(item["quote"])
        return "Проверенный источник недоступен"

    def task_label(self, index: int, task: dict[str, Any]) -> str:
        """Keep every candidate visible; shorten only the clearly marked source excerpt."""
        item = task["commitment"]
        quote = " ".join(self.task_source(task["evidence_id"], item["title"]).split())
        excerpt = quote[:120] + ("…" if len(quote) > 120 else "")
        return f"{index + 1}. {item['title']} — {item['due']} — открыта — Источник: {excerpt}"

    def confirm(self, envelope: dict[str, Any]) -> None:
        """Display the exact persisted proposal and bind only an explicit button to its digest."""
        message = envelope["message"]
        renewal = message["renewal"]
        item = renewal["after"] if renewal else message["capture"]["commitment"]
        action = "Создать задачу" if not renewal else f"Перенести с {renewal['before']['due']}"
        if renewal and item["status"] == "done":
            action = "Отметить выполненной"
        if renewal and item["status"] == "cancelled":
            action = "Отменить обязательство (не выполнено)"
        context = ""
        if renewal:
            target = renewal["parent_id"]
            quote = self.task_source(target, renewal["before"]["title"])
            quoted = "\n".join(f"│ {line}" for line in quote.split("\n"))
            context = f"ID задачи: {target}\nСохранённый источник задачи:\n{quoted}\n\n"
        quoted_source = "\n".join(f"│ {line}" for line in message["text"].split("\n"))
        prompt = (
            f"Владелец: {envelope['owner']}\nПроект: {envelope['project']}\n"
            f"Предоставленный текст:\n{quoted_source}\n"
            "Автор цитаты не определяется. Вы подтверждаете выбранную задачу владельца.\n\n"
            f"{context}{action}\n"
            f"Задача: {item['title']}\nСрок: {item['due']}\n"
            f"Минуты: {item['minutes']}\nПриоритет: {item['priority']}\n"
            f"Состояние: {item['status']}\n\n"
            "Подтверждение изменит только план. Уведомление не отправляется."
        )
        if self.dialogs.show("confirm", prompt) != "Подтвердить":
            return
        self.call("confirm", message["identity"], "--digest", message["digest"])
        self.dialogs.show("info", "Изменение сохранено в плане.")

    def model_review(self, note: str) -> bool:
        """Show model progress/results without turning suggestions into approval."""
        if self.model is None:
            return False
        while time.monotonic() < self.deadline:
            status = self.model.status(note)
            if status["state"] == "ready":
                summary = (
                    f"Сохранённое предложение модели (не подтверждено): {status['lines']}"
                    if status["lines"]
                    else "Модель не нашла обязательств. Это может быть ошибкой."
                )
                return (
                    self.dialogs.show(
                        "choose",
                        f"{summary}\n"
                        "Проверьте владельца, отрицания и выполненные задачи. "
                        "Модель может ошибаться. Поля и запись подтверждаются отдельно.",
                        options=("Заполнить задачу вручную", "Оставить на проверку"),
                    )
                    == "Заполнить задачу вручную"
                )
            if status["state"] != "running":
                label = {
                    "not_started": "ещё не запускалась",
                    "stopped": "остановлена",
                    "failed": "ошибка; результат не получен",
                    "interrupted": "предыдущий запуск прерван; результата нет",
                }[str(status["state"])]
                if (
                    self.dialogs.show(
                        "confirm",
                        f"Состояние модели: {label}. Код: {status.get('reason', '—')}.\n"
                        "Запустить или повторить локальное предложение для этого сообщения? "
                        "До 90 секунд, 4 CPU threads, без внешних API. "
                        "Задача не будет подтверждена автоматически.",
                    )
                    != "Подтвердить"
                ):
                    return False
                self.model.start(note)
            choice = self.dialogs.show(
                "choose",
                "Модель выполняется. Можно проверить результат или остановить её. "
                "Предыдущее прерванное обращение не повторяется автоматически.",
                options=("Обновить состояние", "STOP: остановить модель и ввод"),
            )
            if choice != "Обновить состояние":
                self.model.stop()
                return False
        self.model.stop()
        raise DialogClosed("foreground session ended during model review")

    def review_action(self, message: dict[str, Any]) -> str | None:
        """Show source context and only the actions allowed for its saved state."""
        return self.dialogs.show(
            "choose",
            f"Ваше сообщение:\n{message['text']}\n\nЧто сделать?",
            options=(
                "Новая задача",
                "Перенести задачу",
                "Завершить задачу",
                "Отменить задачу",
                "Оставить на проверку",
            )
            + (("Не задача",) if message["state"] == "needs_review" else ())
            + (("Предложение локальной модели",) if self.model is not None else ()),
        )

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
        action = self.review_action(message)
        if action == "Не задача":
            self.classify(message, restore=False)
            return
        if action == "Предложение локальной модели":
            if not self.model_review(message["text"]):
                return
            action = "Новая задача"
        day = _suggested_day(state["plan"])
        if action == "Новая задача":
            initial_title = "" if "\n" in message["text"] else message["text"][:240]
            title = self.text("Название вашей задачи (не чужая цитата)", initial_title)
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
                f"--title={title}",
                "--minutes",
                minutes,
                "--priority",
                {"Обычный": "normal", "Высокий": "high", "Низкий": "low"}[priority],
                "--due",
                due,
            )
        elif action in ("Перенести задачу", "Завершить задачу", "Отменить задачу"):
            tasks = [
                item
                for item in state["plan"].get("tasks", [])
                if item["commitment"]["status"] == "open"
            ]
            if not tasks:
                self.dialogs.show("info", "Нет открытых задач для изменения.")
                return
            labels = tuple(self.task_label(i, item) for i, item in enumerate(tasks))
            selected = self.dialogs.show("choose", "Выберите конкретную задачу", options=labels)
            if selected is None:
                return
            target = tasks[labels.index(selected)]["evidence_id"]
            if action in ("Завершить задачу", "Отменить задачу"):
                option = "--complete" if action == "Завершить задачу" else "--cancel"
                envelope = self.call("review", message["identity"], option, target)
            else:
                due = self.text("Новый срок: ГГГГ-ММ-ДД", day)
                envelope = self.call(
                    "review", message["identity"], "--target", target, "--due", due
                )
        else:
            return
        self.confirm(envelope)

    def classify(self, message: dict[str, Any], *, restore: bool) -> None:
        """Show the exact retained source before changing only its inbox classification."""
        explanation = (
            "Вернуть на проверку? Фоновый пилот может подготовить черновик, "
            "если попытки ещё не было."
            if restore
            else "Отметить «не задача»? Источник сохранится; можно вернуть его на проверку. "
            "Текущий запуск модели не отменяется; для остановки используйте STOP."
        )
        if (
            self.dialogs.show("confirm", f"Источник: {message['text']}\n\n{explanation}")
            == "Подтвердить"
        ):
            self.call("restore" if restore else "dismiss", message["identity"])
            self.dialogs.show("info", "Состояние сообщения сохранено. План не изменён.")

    def read_drafts(self, state: dict[str, Any]) -> None:
        """Read validated saved suggestions under STOP without offering execution or approval."""
        if self.model is None:
            return
        drafts = []
        unavailable = 0
        for message in state["messages"]:
            if message["state"] != "needs_review":
                continue
            try:
                status = self.model.status(message["text"])
            except (OSError, ValueError, TypeError):
                unavailable += 1
                continue
            if status["state"] == "ready":
                drafts.append((message, status))
        summary = f"Недоступных черновиков: {unavailable}."
        if not drafts:
            self.dialogs.show(
                "info",
                f"Готовых черновиков ожидающих проверки сообщений нет. {summary}",
                allow_stopped=True,
            )
            return
        labels = tuple(f"{i + 1}. {message['text'][:120]}" for i, (message, _) in enumerate(drafts))
        selected = self.dialogs.show(
            "choose",
            f"Сохранённые черновики — только чтение. {summary}",
            options=labels,
            allow_stopped=True,
        )
        if selected is None:
            return
        message, status = drafts[labels.index(selected)]
        result = (
            f"Предложение модели (не подтверждено): {status['lines']}"
            if status["lines"]
            else "Модель не нашла обязательств. Это может быть ошибкой."
        )
        self.dialogs.show(
            "info",
            f"Источник: {message['text']}\n\n{result}\n"
            "Это сохранённый черновик, не текущее состояние задачи. "
            "Просмотр не запускает модель, не снимает STOP и не подтверждает изменения.",
            allow_stopped=True,
        )

    def draft_state(self, message: dict[str, Any]) -> str:
        """Inspect saved draft state without inference; one bad cache must not hide the inbox."""
        if self.model is None or message["state"] != "needs_review":
            return "not_applicable"
        try:
            return str(self.model.status(message["text"])["state"])
        except (OSError, ValueError, TypeError):
            return "unavailable"

    def inbox_summary(self, state: dict[str, Any]) -> str:
        """Expose pending work on entry, without claiming drafts are approved tasks."""
        pending = [m for m in state["messages"] if m["state"] not in ("applied", "dismissed")]
        summary = f"Ожидают проверки: {len(pending)}."
        if self.model is not None:
            drafts = [self.draft_state(message) for message in pending]
            problems = sum(value in ("failed", "interrupted", "unavailable") for value in drafts)
            summary += (
                f"\nГотовых черновиков: {drafts.count('ready')} (не подтверждены). "
                f"Сбоев или прерванных попыток: {problems}."
            )
        return summary

    def draft_label(self, message: dict[str, Any]) -> str:
        """A busy shared model lane is not proof that this particular message is running."""
        labels = {
            "ready": "черновик готов, не подтверждён",
            "failed": "ошибка модели",
            "interrupted": "попытка прервана",
            "not_started": "модель ещё не запускалась",
            "stopped": "модель остановлена",
            "running": "общий канал модели занят",
            "unavailable": "состояние черновика недоступно",
        }
        label = labels.get(self.draft_state(message))
        return f" — {label}" if label else ""

    def pending(self, state: dict[str, Any], *, dismissed: bool = False) -> None:
        """Select by a stable receipt identity even when two messages have identical text."""
        messages = [
            item
            for item in state["messages"]
            if (
                item["state"] == "dismissed"
                if dismissed
                else item["state"] not in ("applied", "dismissed")
            )
        ]
        if not messages:
            self.dialogs.show(
                "info",
                "Нет сообщений «не задача»." if dismissed else "Нет сообщений, ожидающих проверки.",
            )
            return
        labels = tuple(
            f"{i + 1}. {item['text'][:120]} [{item['state']}]{self.draft_label(item)}"
            for i, item in enumerate(messages)
        )
        selected = self.dialogs.show(
            "choose",
            "Отмеченные «не задача»" if dismissed else "Сообщения для проверки",
            options=labels,
        )
        if selected is not None:
            message = messages[labels.index(selected)]
            if dismissed:
                self.classify(message, restore=True)
            else:
                self.review(message, state)

    def search_sources(self) -> None:
        """Show exact source quotations and verified task state without calling a model."""
        query = self.dialogs.show(
            "text",
            "Поиск по фрагменту текста или названию задачи (не по смыслу)",
            allow_stopped=True,
        )
        if query is None:
            return
        try:
            found = self.call("find", "--", query)
        except (ValueError, OSError, sqlite3.Error) as error:
            self.dialogs.show(
                "info", f"Поиск не завершён: {error}\nДанные не изменены.", allow_stopped=True
            )
            return
        results = found["results"]
        summary = (
            "Поиск только в сообщениях этого пилота. Не вся личная память.\n"
            f"Недоступных подтверждённых источников: {found['unavailable_confirmed_sources']}."
        )
        if not results:
            self.dialogs.show(
                "info",
                summary + "\nСовпадений нет; это не доказывает отсутствие задачи.",
                allow_stopped=True,
            )
            return
        labels = tuple(f"{i + 1}. {item['quote'][:100]}" for i, item in enumerate(results))
        selected = self.dialogs.show("choose", summary, options=labels, allow_stopped=True)
        if selected is None:
            return
        item = results[labels.index(selected)]
        task = item["current_task"]
        statuses = {"open": "открыта", "done": "выполнено", "cancelled": "отменена"}
        current = (
            f"Текущее состояние: {statuses[task['status']]}; "
            f"срок: {task['due']}\nЗадача: {task['title']}"
            if task
            else "Подтверждённое текущее состояние задачи не установлено."
        )
        review = {
            "needs_review": "ожидает проверки",
            "reviewed": "поля проверены, запись не подтверждена",
            "applying": "запись не завершена",
            "applied": "изменение ранее подтверждено",
            "dismissed": "отмечено как не задача",
        }[item["review_state"]]
        source_label = (
            "память" if item["source_kind"] == "verified_memory" else "входящее сообщение"
        )
        self.dialogs.show(
            "info",
            f"Источник: {source_label} / {item['source_id']}\n"
            f"Цитата: {item['quote']}\nПроверка сообщения: {review}\n\n"
            f"{current}\nЗапись состояния: {item['current_evidence_id']}\n"
            f"Проверено на: {found['as_of']}\nЭто сведения, не разрешение на действие.",
            allow_stopped=True,
        )

    def preview_plan(self, state: dict[str, Any]) -> None:
        """Read another day's existing evidence without changing dates, approval or STOP."""
        day = self.dialogs.show(
            "text",
            "Дата предпросмотра: ГГГГ-ММ-ДД. Сохранённый план не изменится.",
            initial=_suggested_day(state["plan"]),
            allow_stopped=True,
        )
        if day is None:
            return
        try:
            snapshot = self.call("status", "--preview-day", day)
        except (ValueError, OSError, sqlite3.Error) as error:
            self.dialogs.show(
                "info",
                f"Предпросмотр не завершён: {error}\nДанные не изменены.",
                allow_stopped=True,
            )
            return
        self.dialogs.show("info", render_plan_status(snapshot["plan"]), allow_stopped=True)

    def run(self) -> None:
        """Run a finite foreground menu; cancel/exit never sends or confirms queued work."""
        for _ in range(30):
            state = self.call("status")
            options: tuple[str, ...] = (
                ("Продолжить после STOP", "План", "План на дату", "Найти источник", "Выход")
                if state["stopped"]
                else (
                    "Новое сообщение",
                    "Проверить сообщения",
                    "Не задачи",
                    "Найти источник",
                    "План",
                    "План на дату",
                    "STOP",
                    "Выход",
                )
            )
            if self.model is not None:
                options = (*options[:-1], "Читать черновики", *options[-1:])
            choice = self.dialogs.show(
                "choose",
                "Morgan — локальный план\n"
                f"Срок пилота: {state['expires_at']}\n"
                f"Дата плана: {state['plan']['planning_day']}\n{self.inbox_summary(state)}",
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
                    self.dialogs.show("info", render_plan_status(state["plan"]), allow_stopped=True)
                elif choice == "План на дату":
                    self.preview_plan(state)
                elif choice == "Найти источник":
                    self.search_sources()
                elif choice == "Читать черновики":
                    self.read_drafts(state)
                elif choice == "Новое сообщение":
                    text = self.text("Ваше сообщение (до 1000 символов, переносы строк разрешены)")
                    receipt = self.call("submit", f"ui-{uuid4().hex}", text=text)
                    if "message" in receipt:
                        self.review(receipt["message"], self.call("status"))
                elif choice in ("Проверить сообщения", "Не задачи"):
                    self.pending(state, dismissed=choice == "Не задачи")
            except sqlite3.Error:
                self.dialogs.show(
                    "info",
                    "Хранилище недоступно. Сохранённую проверку можно возобновить "
                    "после восстановления. "
                    "Сеанс закрывается без автоматического повтора.",
                    allow_stopped=True,
                )
                return
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
    parser.add_argument("--server", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--init-user")
    parser.add_argument("--init-project")
    parser.add_argument("--init-timezone")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 1800:
        parser.error("select a foreground lifetime of 5..1800 seconds")
    if (args.server is None) != (args.model is None):
        parser.error("local model requires both --server and --model")
    workspace = None
    model = None
    try:
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
        with os.fdopen(workspace.open_file(".owner-ui.lock", os.O_RDWR | os.O_CREAT), "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + args.seconds
            if args.server is not None:
                model = OwnerModel(args.workspace, ModelCycle(args.server, args.model), deadline)
            try:
                OwnerUI(
                    args.workspace, NativeDialogs(args.workspace, deadline), deadline, model
                ).run()
            finally:
                if model is not None:
                    model.close()
    except (DialogClosed, StopRequested, KeyboardInterrupt):
        print("UI closed; pending messages remain unconfirmed.")
    except (ValueError, OSError, TaskConflict, ZoneInfoNotFoundError, sqlite3.Error) as error:
        parser.exit(2, f"Owner UI failed: {error}\n")
    finally:
        if workspace is not None:
            workspace.close()


if __name__ == "__main__":
    main()
