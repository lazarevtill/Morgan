"""Seed the brain from a ChatGPT export.

An empty brain cannot be measured and cannot be useful on first run. This reads the export's
own JSON shape -- a list of conversations, each a ``mapping`` of message nodes -- and writes
every usable turn through the gate, so imported memories take the same one write path as
anything typed at the terminal and land in every index together.

**The holdout is a project, not a flag.** Conversations whose id hashes into the holdout go
to a separate project. Golden items for the promotion gate are drawn from there, and
consolidation is project-scoped, so nothing the optimizer can mine and nothing it is scored
against ever share a conversation. Reusing the scoping invariant means no future caller has
to remember a rule; the one that already guards every read and write guards this too.

Membership follows from the conversation id alone, so a re-import selects the same holdout
without a seed having to survive between runs, and re-importing updates in place: the memory
id is derived from the message id, and the write path replaces by id.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory, MemoryKind, MemorySource, OriginKind

#: Where imported conversations live. Not a working project: a corpus to recall across and
#: consolidate from, kept out of the way of the projects real work happens in.
ARCHIVE_PROJECT = "archive/chatgpt"

#: The held-out slice. Never consolidated, never optimized against -- read explicitly when
#: authoring golden items, and by nothing else.
HOLDOUT_PROJECT = "archive/chatgpt-holdout"

#: One conversation in this many is held out. A fifth of ~300 conversations is enough to
#: draw 100+ labelled items from while leaving the bulk of the corpus usable.
HOLDOUT_EVERY = 5

#: The most text one memory may carry, so its embedding fits the model server's context.
#: Budgeted in characters rather than tokens because the import must not depend on a
#: tokenizer it does not own, and sized for the worst case measured on this corpus: Russian
#: costs about three characters per token, so 6,000 characters is about 2,000 tokens -- well
#: inside the 8,192 a default llama-server offers, with room for a smaller one.
MAX_EMBED_CHARS = 6_000

#: Where a split is allowed to fall, best first. A paragraph break is a real boundary in the
#: text; a line break is usually one; a space rarely is but beats cutting a word in half.
_BOUNDARIES = ("\n\n", "\n", " ")

#: Which roles become memories, and what attribution each one carries. A tool result is the
#: environment talking, not the owner and not the assistant's reasoning, and carries no
#: preference worth recalling -- so it is skipped rather than mis-attributed.
_SOURCE_BY_ROLE = {
    "user": MemorySource.USER_STATED,
    "assistant": MemorySource.AGENT_INFERRED,
}


@dataclass(frozen=True)
class ImportReport:
    """What the import did, in the terms the owner asked it in."""

    conversations: int = 0
    held_out: int = 0
    memories: int = 0
    #: Turns not stored: tool output, empty content, and pieces an earlier run already
    #: imported unchanged.
    skipped_turns: int = 0


def is_held_out(conversation_id: str) -> bool:
    """True when this conversation belongs to the eval holdout.

    Decided by hashing the id rather than by sampling, so the split is reproducible from the
    export alone. sha256 rather than ``hash()``: the built-in is salted per process and would
    put the same conversation on different sides of the firewall on consecutive runs.
    """
    digest = hashlib.sha256(conversation_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % HOLDOUT_EVERY == 0


def split_for_embedding(text: str, budget: int = MAX_EMBED_CHARS) -> list[str]:
    """Split *text* into pieces of at most *budget* characters, losing nothing.

    A turn longer than the embedding server's context is refused outright, and truncating it
    instead would index the whole text for keyword search while the vector saw only its
    opening -- a memory visible to one signal and not another, which is the failure the one
    write path exists to prevent. Splitting keeps every character reachable by both.

    Cuts fall on the latest boundary in range, preferring a paragraph break to a line break
    to a space. An unbroken run longer than the budget -- a log dump, a base64 blob -- has no
    boundary to find, so it is cut at the budget rather than handed over intact and rejected.
    """
    remaining = text.strip()
    if len(remaining) <= budget:
        return [remaining] if remaining else []

    chunks: list[str] = []
    while len(remaining) > budget:
        window = remaining[:budget]
        cut = next((c for c in (window.rfind(b) for b in _BOUNDARIES) if c > 0), -1)
        if cut <= 0:
            cut = budget
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def _memory_id(message_id: str, part: int) -> str:
    """A stable memory id for one piece of a message.

    Derived from the message id so a second import replaces rather than duplicates, and from
    the piece's index so a split turn keeps one id per piece across runs.
    """
    return hashlib.sha256(f"chatgpt:{message_id}#{part}".encode()).hexdigest()[:32]


def _turn_text(message: dict[str, Any]) -> str:
    """The message's text, or empty when it carries none.

    Non-text parts (images, tool payloads) appear as objects in the same list; they are not
    something to recall as prose, so only the string parts are joined.
    """
    parts = (message.get("content") or {}).get("parts") or []
    return "\n".join(p for p in parts if isinstance(p, str)).strip()


def _created_at(message: dict[str, Any]) -> datetime | None:
    ts = message.get("create_time")
    return datetime.fromtimestamp(ts, tz=UTC) if isinstance(ts, int | float) else None


def _messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation's messages in the order they were sent.

    The export stores a node tree, not a list, because a conversation can branch; ordering by
    ``create_time`` gives the transcript a reader would recognise without walking the tree.
    """
    nodes = (conversation.get("mapping") or {}).values()
    messages = [n["message"] for n in nodes if isinstance(n, dict) and n.get("message")]
    return sorted(messages, key=lambda m: m.get("create_time") or 0)


async def import_chatgpt(
    path: Path, *, gate: MemoryGate, user_id: str, progress: Any = None
) -> ImportReport:
    """Import every usable turn of *path* through *gate*, returning what was written.

    *progress*, when given, is called with ``(conversations_done, total)`` -- an import of a
    few thousand turns takes minutes of embedding calls and a silent one looks hung.
    """
    # to_thread: a real export is tens of megabytes, and reading it inline would block the
    # loop the embedding calls below run on.
    raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
    conversations = json.loads(raw)
    kept = held = stored = skipped = 0

    for done, conversation in enumerate(conversations, start=1):
        conversation_id = str(conversation.get("conversation_id") or conversation.get("id") or done)
        project = HOLDOUT_PROJECT if is_held_out(conversation_id) else ARCHIVE_PROJECT

        wrote_any = False
        for message in _messages(conversation):
            source = _SOURCE_BY_ROLE.get((message.get("author") or {}).get("role", ""))
            text = _turn_text(message)
            if source is None or not text:
                skipped += 1
                continue
            message_id = str(message.get("id") or f"{conversation_id}-{stored}")
            for part, piece in enumerate(split_for_embedding(text)):
                memory_id = _memory_id(message_id, part)
                # Already imported, unchanged: skip it. Every piece costs an embedding call
                # and a real export is thousands of them, so an import that redoes finished
                # work is one that never finishes on a machine that gets interrupted. The
                # content check keeps a corrected turn from being frozen out by its own id.
                existing = await gate.get(memory_id, user_id=user_id)
                if existing is not None and existing.content == piece:
                    skipped += 1
                    wrote_any = True
                    continue
                await gate.store(
                    Memory(
                        id=memory_id,
                        user_id=user_id,
                        project=project,
                        kind=MemoryKind.EPISODIC,
                        content=piece,
                        source=source,
                        created_at=_created_at(message),
                        origin_kind=OriginKind.IMPORT,
                        cwd=str(Path.cwd()),
                        author_id=user_id,
                    )
                )
                stored += 1
                wrote_any = True

        if wrote_any:
            held += project == HOLDOUT_PROJECT
            kept += project == ARCHIVE_PROJECT
        if progress is not None:
            progress(done, len(conversations))

    return ImportReport(conversations=kept, held_out=held, memories=stored, skipped_turns=skipped)
