"""One foreground, local-only conversation for reviewed personal tasks and completion."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from morgan_brain.availability import Unavailability
from morgan_brain.composition import open_evidence_context
from morgan_brain.config import Settings
from morgan_brain.daily_brief import BriefDraft, BriefRequest, Commitment, prepare_brief
from morgan_brain.guided_availability import (
    AvailabilityUpdate,
    explicit_zone,
    interval_entry,
    source_statement,
)
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.memory.continuation_context import ineligible
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.model_cycle import ModelCycle, managed_model
from morgan_brain.providers.note_probe import ProbeFailure
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.cli.proposal_review import Proposal, _read_note, _read_proposal
from morgan_brain.tasks import TaskConflict, TaskKey


class Capture(BaseModel):
    """Persist exact reviewed fields before capture so interrupted writes can be retried."""

    model_config = ConfigDict(extra="forbid")
    event_id: str
    effective_at: str
    commitment: Commitment
    parents: list[str] = Field(default_factory=list, max_length=1)


class Session(BaseModel):
    """A bounded workspace-local review journal, not authorization against the OS account."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["morgan.guided-plan.v1", "morgan.guided-plan.v2"] = "morgan.guided-plan.v2"
    user: str = Field(min_length=1, max_length=128)
    project: str = Field(min_length=1, max_length=128)
    day: date
    budget: int = Field(ge=1, le=480)
    note_sha256: str
    proposal_sha256: str
    captures: list[Capture] = Field(default_factory=list, max_length=8)
    captured: bool = False
    pending_done: Capture | None = None
    timezone: str | None = None
    unavailability_ids: list[str] = Field(default_factory=list, max_length=8)
    pending_availability: AvailabilityUpdate | None = None


def _quote(text: str) -> str:
    """Render untrusted notes with escaped terminal controls while retaining readable Unicode."""
    return "".join(
        char if char.isprintable() else f"\\u{ord(char):04x}"
        for char in json.dumps(text, ensure_ascii=False)
    )


def _save(workspace: NoteWorkspace, session: Session) -> None:
    """Persist only the small local workflow journal; memory remains the source of task state."""
    workspace.publish("session.json", session.model_dump_json(indent=2), replace=True)


def _settings(workspace: NoteWorkspace, session: Session) -> Settings:
    """Use explicit private memory and hash embeddings; never inherit a provider destination."""
    return Settings(
        _env_file=None,
        owner_user_id=session.user,
        data_dir=str(workspace.path / "memory"),
        temporal_db_url=f"sqlite:///{workspace.path / 'memory/morgan.db'}",
        embedding_backend="hash",
        embedding_dim=1024,
    )


def _read_inputs(workspace: NoteWorkspace, session: Session) -> tuple[str, Proposal]:
    """Refuse changed source/proposal artifacts on every resumed session."""
    note = workspace.read("source.txt", 4096)
    proposal, digest = _read_proposal(workspace.path / "proposal.json")
    if hashlib.sha256(note.encode()).hexdigest() != session.note_sha256:
        raise ValueError("source changed; use a new workspace")
    if digest != session.proposal_sha256 or proposal.source_sha256 != session.note_sha256:
        raise ValueError("proposal changed; use a new workspace")
    return note, proposal


async def _new_session(workspace: NoteWorkspace, args: argparse.Namespace) -> Session:
    """Run an explicitly selected local model once, or import its existing unapproved draft."""
    if any(value is None for value in (args.note, args.user, args.project, args.day, args.budget)):
        raise ValueError("new session requires note, user, project, day and budget")
    note, note_hash = _read_note(args.note)
    if len(note.splitlines()) > 32:
        raise ValueError("note must have at most 32 lines")
    session = Session(
        user=args.user,
        project=args.project,
        day=args.day,
        budget=args.budget,
        note_sha256=note_hash,
        proposal_sha256="",
    )
    if args.proposal and (args.server is not None or args.model is not None):
        raise ValueError("select either proposal or local server/model")
    if args.proposal:
        proposal, _ = _read_proposal(args.proposal)
    else:
        if args.server is None or args.model is None:
            raise ValueError("select local server/model or an existing proposal")
        print("Локальная модель / Local model: up to 90 seconds; Ctrl-C stops.", flush=True)
        async with managed_model(
            workspace, ModelCycle(args.server, args.model, seconds=90)
        ) as model:
            selected = await model.select(note)
        proposal = Proposal(
            version="morgan.note-proposal.v1",
            source_sha256=note_hash,
            source="agent_inferred",
            approved=False,
            review_required=True,
            selected_lines=list(selected),
        )
    if proposal.source_sha256 != note_hash:
        raise ValueError("proposal does not match note")
    payload = proposal.model_dump_json(indent=2)
    session.proposal_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    workspace.publish("source.txt", note)
    workspace.publish("proposal.json", payload)
    _save(workspace, session)
    return session


def _ask[Value](prompt: str, parse: Callable[[str], Value]) -> Value:
    """Retry invalid entries locally; EOF/Ctrl-C still stop without hidden confirmation."""
    for _ in range(5):
        try:
            return parse(input(prompt))
        except ValueError:
            print("Неверный ввод, попробуйте ещё раз / Invalid input, please try again.")
    raise ValueError("too many invalid entries; resume the session to try again")


def _indices(lines: list[str], raw: str) -> list[int]:
    """Allow a bounded set of usable source lines, or an explicit empty selection to exit."""
    if not raw.strip():
        return []
    indices = [int(value) for value in raw.split()]
    if not 1 <= len(indices) <= 8 or len(set(indices)) != len(indices):
        raise ValueError("choose 1..8 distinct lines")
    for index in indices:
        if not 1 <= index <= len(lines):
            raise ValueError("line number is outside source")
        title = lines[index - 1]
        if not title.strip() or len(title) > 240 or not title.isprintable():
            raise ValueError("select a nonblank printable task line of at most 240 characters")
    return indices


def _commitment(title: str, raw: str) -> Commitment:
    """Validate one human-entered duration, priority and date without guessing corrections."""
    fields = raw.split()
    if len(fields) != 3:
        raise ValueError("provide minutes, priority and date")
    return Commitment(
        version="morgan.commitment.v1",
        title=title,
        minutes=int(fields[0]),
        priority=cast(
            Literal["high", "normal", "low"],
            {"высокий": "high", "обычный": "normal", "низкий": "low"}.get(fields[1], fields[1]),
        ),
        due=date.fromisoformat(fields[2]),
        status="open",
    )


def _done_index(items: list[tuple[str, Commitment]], raw: str) -> int | None:
    """Accept an explicit completion index or exit, retaining the displayed task numbering."""
    command = raw.split()
    if command in (["quit"], ["выход"]):
        return None
    if len(command) != 2 or command[0] not in ("done", "готово"):
        raise ValueError("use done N or quit")
    index = int(command[1]) - 1
    if not 0 <= index < len(items):
        raise ValueError("unknown task number")
    return index


def _select(note: str, proposal: Proposal, session: Session) -> list[Capture]:
    """Ask the owner to choose their own tasks, including lines omitted by the model."""
    lines = note.splitlines()
    print("Исходная заметка / Source note (model suggestions are untrusted):")
    for index, line in enumerate(lines, 1):
        mark = " [model]" if line in proposal.selected_lines else ""
        print(f"{index}{mark}: {_quote(line)}")
    indices = _ask(
        "Номера только ВАШИХ задач / Your task numbers (space-separated; empty = exit): ",
        lambda raw: _indices(lines, raw),
    )
    captures = []
    for index in indices:
        print(f"Задача / Task: {_quote(lines[index - 1])}")
        commitment = _ask(
            f"Минуты priority due / minutes high|normal|low YYYY-MM-DD "
            f"(example: 10 normal {session.day}): ",
            partial(_commitment, lines[index - 1]),
        )
        captures.append(
            Capture(
                event_id=f"guided-{uuid4().hex}",
                effective_at=datetime.now(UTC).isoformat(),
                commitment=commitment,
            )
        )
    return captures


def _confirm(session: Session, captures: list[Capture]) -> bool:
    """Confirm ownership and complete planning fields; no default or automatic acceptance."""
    print(f"Владелец / Owner: {_quote(session.user)}; project: {_quote(session.project)}")
    for item in captures:
        task = item.commitment
        print(
            f"- {_quote(task.title)} — {task.minutes} min; {task.priority}; "
            f"due {task.due}; {task.status}"
        )
    answer = input("Это мои задачи / These are my tasks. CONFIRM / ПОДТВЕРЖДАЮ (else cancel): ")
    return answer.strip() in ("CONFIRM", "ПОДТВЕРЖДАЮ")


async def _capture(workspace: NoteWorkspace, session: Session, capture: Capture) -> None:
    """Reuse the normal scoped write path with immutable IDs and source-bound provenance."""
    digest = hashlib.sha256(
        (session.note_sha256 + session.proposal_sha256 + capture.model_dump_json()).encode()
    ).hexdigest()
    args = argparse.Namespace(
        text=capture.commitment.model_dump_json(),
        source="user_stated",
        author_id=session.user,
        event_id=capture.event_id,
        effective_at=capture.effective_at,
        revises_event_ids=capture.parents,
        assert_session_id=True,
    )
    await cmd_remember(
        args,
        _settings(workspace, session),
        session.project,
        client="guided-plan-v1",
        session_id=digest,
    )


@dataclass(frozen=True)
class PlanView:
    """One displayed snapshot: task and constraint indices remain stable within the menu."""

    tasks: list[tuple[str, Commitment]]
    constraints: list[tuple[str, Unavailability]]


async def _plan(workspace: NoteWorkspace, session: Session) -> PlanView:
    """Show and save a derived plan, reading current commitments through MemoryGate only."""
    context = open_evidence_context(workspace.path / "memory/morgan.db")
    try:
        request = BriefRequest(
            TaskKey(session.user, session.project, "guided-plan"),
            tuple(item.event_id for item in session.captures),
            session.day,
            session.budget,
            follow_revisions=True,
            unavailability_ids=tuple(session.unavailability_ids),
            timezone=session.timezone,
        )
        draft = await prepare_brief(context.gate, request)
        evidence = await context.gate.evidence(
            user_id=session.user,
            project=session.project,
            evidence_ids=[target for _, target in draft.resolved_ids],
        )
        if evidence.missing_ids or any(
            ineligible(item, draft.as_of) is not None for item in evidence.records
        ):
            raise ValueError("tasks changed during planning; resume to refresh")
        items = [
            (item.id, Commitment.model_validate_json(item.content)) for item in evidence.records
        ]
        constraints = await _constraint_view(context.gate, session, draft)
    finally:
        context.conn.close()
    filename = f"plan-{uuid4().hex}.md"
    workspace.publish(filename, draft.task.payload)
    print(f"План / Plan — budget: {session.budget} min; planned: {draft.planned_minutes} min")
    print(f"Файл / File: {workspace.path / filename}")
    for index, (identity, item) in enumerate(items, 1):
        label = "focus" if identity in draft.selected_ids else "outside this plan"
        if identity in draft.blocked_ids:
            label = "отложено: недоступность / deferred: unavailability"
        print(
            f"{index}: {item.status} — {_quote(item.title)}; {item.minutes} min; "
            f"due {item.due}; {label}"
        )
    _show_constraints(session, constraints)
    return PlanView(items, constraints)


def _interact(workspace: NoteWorkspace, session: Session) -> None:
    """Keep explicit review separate from completion; resume interrupted captures idempotently."""
    note, proposal = _read_inputs(workspace, session)
    if not session.captures:
        session.captures = _select(note, proposal, session)
        if not session.captures:
            return
        _save(workspace, session)
    if not session.captured:
        print("Проверьте контекст / Review context:", _quote(note))
        if not _confirm(session, session.captures):
            if not (workspace.path / "memory/morgan.db").exists():
                session.captures = []
                _save(workspace, session)
            return
        for capture in session.captures:
            asyncio.run(_capture(workspace, session, capture))
        session.captured = True
        _save(workspace, session)
    if session.pending_done is not None:
        print("Незавершённая запись выполнения / Pending completion:")
        if not _confirm(session, [session.pending_done]):
            return
        asyncio.run(_capture(workspace, session, session.pending_done))
        session.pending_done = None
        _save(workspace, session)
    if session.pending_availability is not None:
        print("Незавершённая запись недоступности / Pending availability update:")
        if not _confirm_availability(session, session.pending_availability):
            return
        asyncio.run(_apply_availability(workspace, session))
    for _ in range(20):
        view = asyncio.run(_plan(workspace, session))
        command, index = _ask(
            "done N / готово N; busy / недоступность; change N / изменить N; "
            "cancel N / отменить N; quit / выход: ",
            partial(_menu, view),
        )
        if command == "quit":
            return
        if command != "done":
            _edit_availability(workspace, session, view, command, index)
            continue
        if index is None:
            raise ValueError("completion requires a task index")
        identity, item = view.tasks[index]
        if item.status == "done":
            print("Уже выполнено / Already done")
            continue
        session.pending_done = Capture(
            event_id=f"guided-done-{uuid4().hex}",
            effective_at=datetime.now(UTC).isoformat(),
            commitment=item.model_copy(update={"status": "done"}),
            parents=[identity],
        )
        _save(workspace, session)
        asyncio.run(_capture(workspace, session, session.pending_done))
        session.pending_done = None
        _save(workspace, session)


async def _constraint_view(
    gate: MemoryGate, session: Session, draft: BriefDraft
) -> list[tuple[str, Unavailability]]:
    """Read the exact constraint versions just used by the planner before offering edits."""
    if not draft.unavailability_resolved_ids:
        return []
    targets = dict(draft.unavailability_resolved_ids)
    result = await gate.evidence(
        user_id=session.user,
        project=session.project,
        evidence_ids=list(dict.fromkeys(targets.values())),
    )
    if result.missing_ids or any(
        ineligible(item, draft.as_of) is not None for item in result.records
    ):
        raise ValueError("constraints changed during planning; resume to refresh")
    records = {item.id: item for item in result.records}
    return [
        (targets[root], Unavailability.model_validate_json(records[targets[root]].content))
        for root in session.unavailability_ids
    ]


def _show_constraints(session: Session, constraints: list[tuple[str, Unavailability]]) -> None:
    """State coverage honestly and show the provenance and conservative day-level policy."""
    if not constraints:
        print("Недоступность не задана / No unavailability recorded; availability is not verified.")
        return
    print(f"Зона плана / Planning zone: {_quote(str(session.timezone))}")
    print("Пересечение откладывает весь дневной план / Any overlap defers the whole daily plan.")
    print("Недоступность не означает местоположение / Unavailability does not imply location.")
    for index, (identity, interval) in enumerate(constraints, 1):
        print(
            f"U{index}: {interval.status}; {interval.starts_at.isoformat()} → "
            f"{interval.ends_at.isoformat()} (end excluded); {_quote(interval.timezone)}; "
            f"record {_quote(identity)}; sources {_quote(', '.join(interval.source_ids))}"
        )


def _menu(view: PlanView, raw: str) -> tuple[str, int | None]:
    """Separate task indices from displayed constraint indices; no ambiguous bare numbers."""
    command = raw.split()
    if command in (["busy"], ["недоступность"]):
        if len(view.constraints) >= 8:
            raise ValueError("this session already has eight constraint families")
        return "busy", None
    if len(command) == 2 and command[0] in ("change", "изменить", "cancel", "отменить"):
        index = int(command[1]) - 1
        if not 0 <= index < len(view.constraints):
            raise ValueError("unknown constraint number")
        return ("change" if command[0] in ("change", "изменить") else "cancel"), index
    task_index = _done_index(view.tasks, raw)
    return ("quit" if task_index is None else "done"), task_index


def _availability_input(
    session: Session, current: tuple[str, Unavailability] | None, cancel: bool
) -> AvailabilityUpdate:
    """Collect an actual user assertion, explicit local zone and offset-bearing bounds."""
    source = _ask(
        "Ваше утверждение о недоступности/изменении/отмене / Your availability statement: ",
        source_statement,
    )
    zone = session.timezone or _ask(
        "Зона плана IANA (пример Europe/Paris) / Planning timezone: ",
        explicit_zone,
    )
    source_id = f"guided-source-{uuid4().hex}"
    if cancel:
        if current is None:
            raise ValueError("cancellation requires an existing interval")
        interval = current[1].model_copy(update={"status": "cancelled", "source_ids": [source_id]})
    else:
        interval = _ask(
            "Начало и исключительный конец с UTC-смещением / Start and exclusive end "
            "(example: 2027-04-10T09:00+02:00 2027-04-10T10:00+02:00): ",
            lambda raw: interval_entry(raw, zone, source_id),
        )
    return AvailabilityUpdate(
        source_id=source_id,
        source_text=source,
        event_id=f"guided-unavailable-{uuid4().hex}",
        effective_at=datetime.now(UTC).isoformat(),
        interval=interval,
        parent_id=current[0] if current else None,
    )


def _confirm_availability(session: Session, update: AvailabilityUpdate) -> bool:
    """Review exact words, owner, dates, timezone, provenance and correction before any write."""
    interval = update.interval
    print(f"Владелец / Owner: {_quote(session.user)}; project: {_quote(session.project)}")
    print(f"Источник / Your statement: {_quote(update.source_text)}; ID {_quote(update.source_id)}")
    print(
        f"{interval.status}: {interval.starts_at.isoformat()} → {interval.ends_at.isoformat()}; "
        f"zone {_quote(interval.timezone)}; end excluded; "
        f"revises {_quote(update.parent_id or 'none')}"
    )
    print("Это недоступность, не местоположение / This is unavailability, not location.")
    print("Любое пересечение откладывает весь план дня / Any overlap defers the whole daily plan.")
    return input("Подтвердить / Confirm: CONFIRM / ПОДТВЕРЖДАЮ (else cancel): ").strip() in (
        "CONFIRM",
        "ПОДТВЕРЖДАЮ",
    )


async def _apply_availability(workspace: NoteWorkspace, session: Session) -> None:
    """Replay exact source/assertion writes; clear pending only after both acknowledgements."""
    update = session.pending_availability
    if update is None:
        raise ValueError("no pending availability update")
    if session.timezone is not None and session.timezone != update.interval.timezone:
        raise ValueError("pending interval changed the session timezone")
    digest = hashlib.sha256(update.model_dump_json().encode()).hexdigest()
    for identity, content, parents in (
        (update.source_id, update.source_text, []),
        (
            update.event_id,
            update.interval.model_dump_json(),
            [update.parent_id] if update.parent_id else [],
        ),
    ):
        args = argparse.Namespace(
            text=content,
            source="user_stated",
            author_id=session.user,
            event_id=identity,
            effective_at=update.effective_at,
            revises_event_ids=parents,
            assert_session_id=True,
        )
        await cmd_remember(
            args,
            _settings(workspace, session),
            session.project,
            client="guided-availability-v1",
            session_id=digest,
        )
    if update.parent_id is None and update.event_id not in session.unavailability_ids:
        session.unavailability_ids.append(update.event_id)
    session.timezone = update.interval.timezone
    session.version = "morgan.guided-plan.v2"
    session.pending_availability = None
    _save(workspace, session)


def _edit_availability(
    workspace: NoteWorkspace, session: Session, view: PlanView, command: str, index: int | None
) -> None:
    """Do not persist unconfirmed input; journal confirmed values before starting capture."""
    current = None if index is None else view.constraints[index]
    update = _availability_input(session, current, command == "cancel")
    if not _confirm_availability(session, update):
        print("Отменено без записи / Cancelled without writing")
        return
    session.pending_availability = update
    _save(workspace, session)
    asyncio.run(_apply_availability(workspace, session))


def main() -> None:
    """Start/resume one foreground conversation; no service, credentials or notifications."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--note", type=Path)
    parser.add_argument("--proposal", type=Path, help="Existing unapproved local-model draft")
    parser.add_argument("--server", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--user")
    parser.add_argument("--project")
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    workspace = NoteWorkspace(args.workspace)
    try:
        with os.fdopen(workspace.open_file(".guided.lock", os.O_RDWR | os.O_CREAT), "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                session = Session.model_validate_json(workspace.read("session.json", 32768))
            except FileNotFoundError:
                session = asyncio.run(_new_session(workspace, args))
            else:
                if any(
                    value is not None
                    for value in (
                        args.note,
                        args.proposal,
                        args.server,
                        args.model,
                        args.user,
                        args.project,
                        args.day,
                        args.budget,
                    )
                ):
                    raise ValueError(
                        "resume uses only workspace; new inputs require a new workspace"
                    )
            print("Локальный пилот / Local pilot: no external actions or notifications.")
            _interact(workspace, session)
    except (EOFError, KeyboardInterrupt):
        print("Остановлено; можно продолжить / Stopped; resume with workspace only.")
    except (ValueError, OSError, ProbeFailure, StopRequested, TaskConflict) as error:
        parser.exit(2, f"Ошибка / Error: {error}\n")
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
