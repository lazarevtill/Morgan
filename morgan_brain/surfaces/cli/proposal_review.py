"""Preview-bound, explicit capture of one reviewed model proposal through normal memory writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import UTC, date
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from morgan_brain.composition import sqlite_path
from morgan_brain.config import Settings
from morgan_brain.daily_brief import Commitment
from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.models import PERSONAL_PROJECT
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.cli.project import Repository
from morgan_brain.surfaces.cli.validation import effective_time, event_identity


class Proposal(BaseModel):
    """Strict existing draft format; model text remains unapproved until explicit capture."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["morgan.note-proposal.v1"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["agent_inferred"]
    approved: Literal[False]
    review_required: Literal[True]
    selected_lines: list[str] = Field(max_length=8)

    @field_validator("approved", "review_required", mode="before")
    @classmethod
    def strict_flags(cls, value: object) -> object:
        """Do not accept numeric stand-ins for boolean review metadata."""
        if not isinstance(value, bool):
            raise TypeError("proposal flags must be booleans")
        return value


def _read_proposal(path: Path) -> tuple[Proposal, str]:
    """Read one bounded private file once; never follow a final-component symlink."""
    workspace = NoteWorkspace(path.parent, create=False)
    try:
        text = workspace.read(path.name, 16384)
    finally:
        workspace.close()
    return Proposal.model_validate_json(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_note(path: Path) -> tuple[str, str]:
    """Read the explicitly selected regular source file without following symlinks or FIFOs."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("source note must be a regular file")
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("source note exceeds 4096 bytes")
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _destination(settings: Settings) -> str:
    """Bind review to the resolved local database, not merely its project label."""
    path = sqlite_path(settings.temporal_db_url)
    return path if path == ":memory:" else str(Path(path).resolve())


def _review(args: argparse.Namespace, settings: Settings, project: str | None) -> dict[str, Any]:
    """Validate and shape the exact claim before opening memory or calling any model."""
    proposal, proposal_hash = _read_proposal(args.proposal)
    if not 1 <= args.selection <= len(proposal.selected_lines):
        raise ValueError("select an existing proposal line")
    original = proposal.selected_lines[args.selection - 1]
    note, note_hash = _read_note(args.note)
    if note_hash != proposal.source_sha256 or original not in note.splitlines():
        raise ValueError("proposal does not match the selected source note")
    commitment = Commitment(
        version="morgan.commitment.v1",
        title=args.title if args.title is not None else original,
        due=date.fromisoformat(args.due),
        minutes=args.minutes,
        priority=args.priority,
        status="open",
    )
    identity = event_identity(args.event_id)
    timestamp = effective_time(args.effective_at)
    if identity is None or timestamp is None or not args.author_id.strip():
        raise ValueError("review requires event ID, explicit event time and reported author")
    return {
        "version": "morgan.commitment-review.v1",
        "owner": settings.owner_user_id,
        "project": project if project is not None else PERSONAL_PROJECT,
        "database": _destination(settings),
        "author_id": args.author_id,
        "event_id": identity,
        "effective_at": timestamp.astimezone(UTC).isoformat(),
        "proposal_sha256": proposal_hash,
        "source_sha256": proposal.source_sha256,
        "selection": args.selection,
        "original_line": original,
        "source_note": note,
        "commitment": commitment.model_dump(mode="json"),
        "source_after_confirmation": "user_stated",
        "execution_authorized": False,
    }


def _digest(review: dict[str, Any]) -> str:
    """Hash the entire displayed review, including scope, provenance and planning fields."""
    encoded = json.dumps(review, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def cmd_review_proposal(
    args: argparse.Namespace,
    settings: Settings,
    project: str | None,
    *,
    repository: Repository | None = None,
) -> dict[str, Any]:
    """Preview by default; only an exact supplied review digest permits one scoped capture."""
    review = _review(args, settings, project)
    digest = _digest(review)
    if args.accept is None:
        return {
            "stored": False,
            "review": review,
            "review_sha256": digest,
            "execution_authorized": False,
        }
    if args.accept != digest:
        raise ValueError("review changed: preview again before accepting")
    capture = argparse.Namespace(
        text=json.dumps(review["commitment"], ensure_ascii=False, separators=(",", ":")),
        source="user_stated",
        author_id=review["author_id"],
        event_id=review["event_id"],
        effective_at=review["effective_at"],
        revises_event_ids=None,
    )
    result = await cmd_remember(
        capture,
        settings,
        project,
        client="reviewed-proposal-v1",
        session_id=digest,
        repository=repository,
    )
    result.update(
        review_sha256=digest, proposal_sha256=review["proposal_sha256"], execution_authorized=False
    )
    return result


def add_review_parser(sub: Any) -> argparse.ArgumentParser:
    """Add explicit review fields; common project/output options are supplied by the main CLI."""
    parser = cast(
        argparse.ArgumentParser,
        sub.add_parser("review-proposal", help="Preview and explicitly accept one proposal."),
    )
    parser.add_argument("proposal", type=Path)
    parser.add_argument(
        "--note", type=Path, required=True, help="Original source note for context."
    )
    parser.add_argument("--selection", type=int, required=True)
    parser.add_argument("--title", help="Optional reviewed correction to the selected line.")
    parser.add_argument("--due", required=True)
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--priority", choices=("high", "normal", "low"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--effective-at", required=True)
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--accept", help="Exact review_sha256 after reviewing the preview.")
    return parser
