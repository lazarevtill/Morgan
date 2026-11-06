"""One-shot local owner messages with explicit review; no listener, model or external effects."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from morgan_brain import guided_plan as guided
from morgan_brain.availability import planning_zone
from morgan_brain.composition import build_memory_context
from morgan_brain.daily_brief import Commitment
from morgan_brain.guided_availability import source_statement
from morgan_brain.guided_renewal import RenewalUpdate
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.cli.proposal_review import Proposal
from morgan_brain.tasks import TaskConflict


class Message(BaseModel):
    """A local receipt; the OS account, not the supplied message ID, establishes trust."""

    model_config = ConfigDict(extra="forbid")
    identity: str = Field(min_length=1, max_length=128)
    text: str
    state: Literal["needs_review", "reviewed", "applying", "applied"] = "needs_review"
    capture: guided.Capture | None = None
    renewal: RenewalUpdate | None = None
    digest: str | None = None


class Inbox(BaseModel):
    """Finite pilot journal; no eviction, hidden retries or unbounded queue."""

    model_config = ConfigDict(extra="forbid")
    user: str
    project: str
    expires_at: datetime
    messages: list[Message] = Field(default_factory=list, max_length=14)


def _save(workspace: NoteWorkspace, inbox: Inbox) -> None:
    workspace.publish("owner-input.json", inbox.model_dump_json(indent=2), replace=True)


def _digest(inbox: Inbox, message: Message) -> str:
    payload = [
        inbox.user,
        inbox.project,
        message.identity,
        message.text,
        message.capture.model_dump(mode="json") if message.capture else None,
        message.renewal.model_dump(mode="json") if message.renewal else None,
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _init(workspace: NoteWorkspace, args: argparse.Namespace) -> dict[str, object]:
    planning_zone(args.timezone)
    # Do not overwrite or adopt a partially initialized/existing guided workspace.
    for name in ("session.json", "source.txt", "proposal.json", "owner-input.json"):
        try:
            workspace.read(name, 65536)
        except FileNotFoundError:
            continue
        raise ValueError("init requires a fresh workspace; preserve existing data")
    note = "Local owner inbox; every message requires explicit review."
    note_hash = hashlib.sha256(note.encode()).hexdigest()
    proposal = Proposal(
        version="morgan.note-proposal.v1",
        source_sha256=note_hash,
        source="agent_inferred",
        approved=False,
        review_required=True,
        selected_lines=[],
    ).model_dump_json(indent=2)
    session = guided.Session(
        user=args.user,
        project=args.project,
        day=args.day,
        budget=args.budget,
        timezone=args.timezone,
        captured=True,
        note_sha256=note_hash,
        proposal_sha256=hashlib.sha256(proposal.encode()).hexdigest(),
    )
    workspace.publish("source.txt", note)
    workspace.publish("proposal.json", proposal)
    context = build_memory_context(guided._settings(workspace, session))
    context.conn.close()
    guided._save(workspace, session)
    _save(
        workspace,
        Inbox(
            user=session.user,
            project=session.project,
            expires_at=datetime.now(UTC) + timedelta(days=7),
        ),
    )
    return {"state": "ready", "message_limit": 14, "task_limit": 8}


def _find(inbox: Inbox, identity: str) -> Message:
    for message in inbox.messages:
        if message.identity == identity:
            return message
    raise ValueError("unknown message ID")


def _submit(inbox: Inbox, identity: str, text: str) -> Message:
    if not identity.strip() or not identity.isprintable() or len(identity) > 128:
        raise ValueError("message ID requires 1..128 printable characters")
    for existing in inbox.messages:
        if existing.identity == identity:
            if existing.text != text:
                raise ValueError("message ID already contains different text")
            return existing
    if len(inbox.messages) >= 14:
        raise ValueError("pilot message limit reached; preserve journal")
    message = Message(identity=identity, text=source_statement(text))
    inbox.messages.append(message)
    return message


async def _review(
    workspace: NoteWorkspace,
    session: guided.Session,
    inbox: Inbox,
    message: Message,
    args: argparse.Namespace,
) -> None:
    if message.state in ("applying", "applied"):
        raise ValueError("review cannot change an attempted operation")
    if guided._pending_reviews(session):
        raise ValueError("resolve pending guided review first")
    capture = None
    renewal = None
    effective = datetime.now(UTC).isoformat()
    if args.target:
        if args.title is not None or args.minutes is not None or args.priority is not None:
            raise ValueError("renewal accepts target and due only")
        _, view = await guided._read_plan(workspace, session)
        matches = [task for identity, task in view.tasks if identity == args.target]
        if len(matches) != 1:
            raise ValueError("select one exact current task ID from status")
        before = matches[0]
        renewal = RenewalUpdate(
            event_id=f"guided-renew-{uuid4().hex}",
            parent_id=args.target,
            source_text=message.text,
            effective_at=effective,
            before=before,
            after=before.model_copy(update={"due": args.due}),
        )
    else:
        if len(session.captures) >= 8:
            raise ValueError("pilot task limit reached")
        if args.title is None or args.minutes is None or args.priority is None:
            raise ValueError("new task requires explicit title, minutes, priority and due")
        capture = guided.Capture(
            event_id=f"owner-input-{uuid4().hex}",
            effective_at=effective,
            commitment=Commitment(
                version="morgan.commitment.v1",
                title=args.title,
                minutes=args.minutes,
                priority=args.priority,
                due=args.due,
                status="open",
            ),
        )
    message.capture, message.renewal = capture, renewal
    message.digest = _digest(inbox, message)
    message.state = "reviewed"


async def _add(workspace: NoteWorkspace, session: guided.Session, message: Message) -> None:
    capture = message.capture
    if message.digest is None:
        raise ValueError("missing review digest")
    if capture is None:
        raise ValueError("missing reviewed capture")
    known = [item for item in session.captures if item.event_id == capture.event_id]
    if known and known != [capture]:
        raise ValueError("capture identity conflict")
    if not known and len(session.captures) >= 8:
        raise ValueError("pilot task limit reached")
    for identity, content in (
        (f"{capture.event_id}-source", message.text),
        (capture.event_id, capture.commitment.model_dump_json()),
    ):
        workspace.check_stop()
        await cmd_remember(
            argparse.Namespace(
                text=content,
                source="user_stated",
                author_id=session.user,
                event_id=identity,
                effective_at=capture.effective_at,
                revises_event_ids=[],
                assert_session_id=True,
            ),
            guided._settings(workspace, session),
            session.project,
            client="owner-input-v1",
            session_id=message.digest,
        )
    if not known:
        session.captures.append(capture)
        guided._save(workspace, session)


async def _confirm(
    workspace: NoteWorkspace, session: guided.Session, inbox: Inbox, message: Message, digest: str
) -> None:
    if message.digest is None or digest != message.digest or digest != _digest(inbox, message):
        raise ValueError("confirmation must match the exact displayed review digest")
    if message.state == "applied":
        return
    if message.state not in ("reviewed", "applying"):
        raise ValueError("review before confirmation")
    pending = guided._pending_reviews(session)
    own_pending = (
        message.renewal is not None
        and session.pending_renewal == message.renewal
        and pending == ["renewal_confirmation"]
    )
    if pending and not own_pending:
        raise ValueError("another guided review is pending")
    workspace.check_stop()
    if message.renewal is not None:
        await guided._check_renewal(workspace, session, message.renewal)
    message.state = "applying"
    _save(workspace, inbox)
    if message.renewal is not None:
        session.pending_renewal = message.renewal
        guided._save(workspace, session)
        await guided._apply_renewal(workspace, session)
    else:
        await _add(workspace, session, message)
    message.state = "applied"
    _save(workspace, inbox)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--user", required=True)
    init.add_argument("--project", required=True)
    init.add_argument("--day", required=True, type=date.fromisoformat)
    init.add_argument("--budget", required=True, type=int)
    init.add_argument("--timezone", required=True)
    submit = commands.add_parser("submit", help="Read one bounded owner message from stdin")
    submit.add_argument("identity")
    review = commands.add_parser("review")
    review.add_argument("identity")
    action = review.add_mutually_exclusive_group(required=True)
    action.add_argument("--new", action="store_true")
    action.add_argument("--target")
    review.add_argument("--title")
    review.add_argument("--minutes", type=int)
    review.add_argument("--priority", choices=("high", "normal", "low"))
    review.add_argument("--due", required=True, type=date.fromisoformat)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("identity")
    confirm.add_argument("--digest", required=True)
    for name in ("status", "stop", "resume"):
        commands.add_parser(name)
    return parser


def _status(workspace: NoteWorkspace, session: guided.Session, inbox: Inbox) -> dict[str, object]:
    """Show admission expiry and pending receipts alongside the explicitly scoped plan."""
    try:
        workspace.check_stop()
        stopped = False
    except StopRequested:
        stopped = True
    expired = datetime.now(UTC) >= inbox.expires_at
    state = "ready"
    if any(message.state != "applied" for message in inbox.messages):
        state = "needs_review"
    if expired:
        state = "expired"
    if stopped:
        state = "stopped"
    return {
        "state": state,
        "stopped": stopped,
        "expired": expired,
        "expires_at": inbox.expires_at.isoformat(),
        "messages": [message.model_dump(mode="json") for message in inbox.messages],
        "plan": asyncio.run(guided._status_payload(workspace, session)),
    }


def _dispatch(workspace: NoteWorkspace, args: argparse.Namespace) -> dict[str, object]:
    if args.command == "init":
        workspace.check_stop()
        return _init(workspace, args)
    session = guided.Session.model_validate_json(workspace.read("session.json", 32768))
    guided._read_inputs(workspace, session)
    inbox = Inbox.model_validate_json(workspace.read("owner-input.json", 262144))
    if (inbox.user, inbox.project) != (session.user, session.project):
        raise ValueError("inbox owner/project does not match session")
    if inbox.expires_at.utcoffset() is None:
        raise ValueError("pilot expiry requires a timezone")
    if args.command not in ("status", "resume") and datetime.now(UTC) >= inbox.expires_at:
        raise ValueError("seven-day pilot expired; preserve journal")
    if args.command == "resume":
        workspace.resume()
        return {"state": "resumed", "automatic_processing": False}
    if args.command == "status":
        return _status(workspace, session, inbox)
    workspace.check_stop()
    if any(
        m.state == "applying" and m.identity != getattr(args, "identity", None)
        for m in inbox.messages
    ):
        raise ValueError("reconfirm the interrupted message before another mutation")
    if args.command == "submit":
        message = _submit(inbox, args.identity, args.text)
    else:
        message = _find(inbox, args.identity)
        if args.command == "review":
            asyncio.run(_review(workspace, session, inbox, message, args))
        else:
            asyncio.run(_confirm(workspace, session, inbox, message, args.digest))
    _save(workspace, inbox)
    return {
        "owner": inbox.user,
        "project": inbox.project,
        "message": message.model_dump(mode="json"),
        "execution_authorized": False,
    }


def main() -> None:
    """Execute one local account command and exit; never prompt or consume model authority."""
    parser = _parser()
    args = parser.parse_args()
    if args.command == "submit":
        args.text = sys.stdin.read(1002).removesuffix("\n")
        if args.text.strip().casefold() in ("stop", "стоп"):
            args.command = "stop"
    workspace = NoteWorkspace(args.workspace, create=args.command == "init")
    try:
        # STOP must remain available while a cooperating writer owns the dialogue lock.
        if args.command == "stop":
            workspace.stop()
            print(json.dumps({"state": "stopped"}))
            return
        with os.fdopen(workspace.open_file(".guided.lock", os.O_RDWR | os.O_CREAT), "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(_dispatch(workspace, args), ensure_ascii=True))
    except (ValueError, OSError, StopRequested, TaskConflict) as error:
        parser.exit(2, f"Error: {error}\n")
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
