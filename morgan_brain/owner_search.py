"""Read-only literal source lookup within one finite owner inbox; never generated answers."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from morgan_brain.composition import open_evidence_context
from morgan_brain.daily_brief import BriefRequest, Commitment, _sources
from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.memory.continuation_context import ineligible
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import MemorySource
from morgan_brain.tasks import TaskKey

if TYPE_CHECKING:
    from morgan_brain.guided_plan import Session
    from morgan_brain.owner_input import Inbox, Message


async def _confirmed(
    gate: MemoryGate, session: Session, message: Message, cutoff: datetime
) -> dict[str, Any]:
    """Resolve recorded companions and current revisions through scoped evidence reads only."""
    identity = (
        message.capture.event_id
        if message.capture
        else (message.renewal.event_id if message.renewal else None)
    )
    if identity is None:
        raise ValueError("confirmed receipt has no task reference")
    source_id = f"{identity}-source"
    evidence = await gate.evidence(
        user_id=session.user,
        project=session.project,
        evidence_ids=[source_id],
        effective_at=cutoff,
    )
    if evidence.missing_ids or len(evidence.records) != 1:
        raise ValueError("source unavailable")
    source = evidence.records[0]
    if (
        ineligible(source, cutoff) is not None
        or source.source != MemorySource.USER_STATED
        or source.author_id != session.user
        or source.content != message.text
    ):
        raise ValueError("source cannot be verified")
    request = BriefRequest(
        TaskKey(session.user, session.project, "owner-source-search"),
        (identity,),
        session.day,
        session.budget,
        follow_revisions=True,
    )
    records, _ = await _sources(gate, request, cutoff)
    current = records[0]
    if current.author_id != session.user or ineligible(current, cutoff) is not None:
        raise ValueError("current task cannot be verified")
    return {
        "quote": source.content,
        "source_id": source.id,
        "source_kind": "verified_memory",
        "current_evidence_id": current.id,
        "current_task": Commitment.model_validate_json(current.content).model_dump(mode="json"),
    }


async def search(
    workspace: NoteWorkspace, session: Session, inbox: Inbox, query: str
) -> dict[str, Any]:
    """Search at most fourteen retained sources, including explicit unconfirmed receipts."""
    needle = query.strip().casefold()
    if not needle or len(query) > 200 or not query.isprintable():
        raise ValueError("search requires 1..200 printable characters")
    cutoff = datetime.now(UTC)
    results = []
    unavailable = 0
    context = None
    if any(message.state == "applied" for message in inbox.messages):
        # Receipt-only results remain usable without an evidence database.
        with suppress(OSError, sqlite3.Error, ValueError):
            context = open_evidence_context(workspace.path / "memory/morgan.db")
    try:
        for message in inbox.messages:
            if message.state == "applied":
                if context is None:
                    unavailable += 1
                    continue
                try:
                    item = await _confirmed(context.gate, session, message, cutoff)
                except (OSError, sqlite3.Error, ValueError):
                    unavailable += 1
                    continue  # Never resurrect missing memory from the workflow journal.
            else:
                item = {
                    "quote": message.text,
                    "source_id": message.identity,
                    "source_kind": "inbox_receipt",
                    "current_evidence_id": None,
                    "current_task": None,
                }
            title = item["current_task"]["title"] if item["current_task"] else ""
            if needle in item["quote"].casefold() or needle in title.casefold():
                results.append(
                    {"message_id": message.identity, "review_state": message.state, **item}
                )
    finally:
        if context is not None:
            context.conn.close()
    return {
        "query": query,
        "match": "literal_casefold_fragment",
        "scope": "owner_inbox",
        "scope_complete": False,
        "as_of": cutoff.isoformat(),
        "results": results,
        "unavailable_confirmed_sources": unavailable,
        "execution_authorized": False,
    }
