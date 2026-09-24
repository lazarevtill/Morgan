"""Seed the brain from a ChatGPT export.

An empty brain cannot be measured and cannot be useful on first run. This reads the export's
own JSON shape -- a list of conversations, each a ``mapping`` of message nodes -- and writes
every usable turn through the gate, so imported memories take the same one write path as
anything typed at the terminal and land in every index together.

**An export is history nobody can rephrase.** Each turn is scanned whole under ``redact``
before it is split, so a secret a cut would halve is still seen whole: a provider token is
redacted and counted, never refused, and every piece records the hits that fall inside it.

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
import bisect
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.config import DEFAULT_IMPORT_CANARY_EVERY
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets import Hit, placeholder_spans
from morgan_brain.memory.secrets.rules import PROVIDER_RULE_NAMES
from morgan_brain.models import Memory, MemoryKind, MemorySource, OriginKind
from morgan_brain.providers.wire import EmbeddingSpaceMismatch

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
    #: Pieces the gate redacted on the way in, and how many of the hits were provider tokens:
    #: an export is history nobody can rephrase, so nothing in it is refused.
    redacted_pieces: int = 0
    provider_hits: int = 0


class ImportStopped(Exception):
    """The import canary caught the model answering outside tolerance, and stopped before
    trusting anything stored since the last good check.

    *first* and *last* are the 1-based ordinals of the suspect stretch, counting only what
    *this run* actually stored -- never a piece skipped because it was already there
    unchanged, and never the export's own numbering, which a resumed run does not track.
    *suspect_ids* carries the memory ids themselves, in order, so a caller can act on them
    directly rather than parse the message; a resumed run's ids are its own, not an earlier
    stopped run's, because a suspect skipped as already-stored is never re-named. *setting*
    and *detail* are the failing ``EmbeddingSpaceMismatch``'s own, carried through unchanged
    rather than paraphrased into a single guessed cause.

    **The suspects are not repaired.** They are already stored, in every index, with
    whatever vectors they were given -- the canary runs after the stretch is stored, not
    before. Storage is idempotent by id, so once ``morgan doctor --vectors`` confirms the
    model is sound again, re-running the same import skips every id already stored, suspects
    included, rather than re-embedding them. Real remediation -- re-embedding the suspects,
    or holding a stretch back until its own canary passes -- is not built.
    """

    def __init__(
        self, *, first: int, last: int, suspect_ids: list[str], setting: str, detail: str
    ) -> None:
        if not suspect_ids:
            # `_run_canary`, the only caller today, never passes an empty list -- both call
            # sites append to it before any interval check can fire. Still a public exception
            # with a public list attribute the type does not forbid being empty, so a future
            # caller that violates the precondition gets a named refusal, not a bare
            # IndexError two lines down.
            raise ValueError("ImportStopped needs at least one suspect id, got suspect_ids=[]")
        self.first = first
        self.last = last
        self.suspect_ids = suspect_ids
        self.setting = setting
        self.detail = detail
        first_id, last_id = suspect_ids[0], suspect_ids[-1]
        super().__init__(
            f"import stopped: memory {first} through memory {last} of this run's own store "
            f"order are suspect -- ids {first_id} through {last_id} ({setting}: {detail}). "
            "They stay stored exactly as given; a re-run skips them unchanged, it does not "
            "repair them. Run `morgan doctor --vectors`, then re-run the import to store "
            "the rest."
        )


def is_held_out(conversation_id: str) -> bool:
    """True when this conversation belongs to the eval holdout.

    Decided by hashing the id rather than by sampling, so the split is reproducible from the
    export alone. sha256 rather than ``hash()``: the built-in is salted per process and would
    put the same conversation on different sides of the firewall on consecutive runs.
    """
    digest = hashlib.sha256(conversation_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % HOLDOUT_EVERY == 0


def split_for_embedding(text: str, budget: int = MAX_EMBED_CHARS) -> list[tuple[int, str]]:
    """Split *text* into pieces of at most *budget* characters, losing nothing: each piece
    with its offset, where it starts in *text* -- the whitespace a cut strips counted -- so
    the hits of a scan of *text* shift onto the piece by it.

    A turn longer than the embedding server's context is refused outright, and truncating it
    instead would index the whole text for keyword search while the vector saw only its
    opening -- a memory visible to one signal and not another, which is the failure the one
    write path exists to prevent. Splitting keeps every character reachable by both.

    Cuts fall on the latest boundary in range, preferring a paragraph break to a line break
    to a space. An unbroken run longer than the budget -- a log dump, a base64 blob -- has no
    boundary to find, so it is cut at the budget rather than handed over intact and rejected.
    A cut never falls inside a placeholder the gate wrote: cut in two, neither half is one,
    and the piece records nothing where the gate redacted something. A placeholder holds no
    whitespace, so only a cut at the budget can land in one; it moves to the placeholder's
    start, or to its end when the placeholder opens the piece, which then runs past the
    budget by less than a placeholder's length.

    A flag-only hit (``passport_rf``, ``phone_rf``) is not guarded that way: a cut can fall
    inside its kept text, most often on a space the text holds. The hit then lies wholly
    inside neither piece, and no piece records it. Nothing leaks, because a flag-only hit's
    text is kept in clear by design.
    """
    guards = placeholder_spans(text)
    guard_starts = [start for start, _ in guards]
    start, end = _stripped(text, 0, len(text))
    pieces: list[tuple[int, str]] = []
    while end - start > budget:
        window = text[start : start + budget]
        cut = next((c for c in (window.rfind(b) for b in _BOUNDARIES) if c > 0), -1)
        if cut <= 0:
            cut = budget
            inside = bisect.bisect_left(guard_starts, start + cut) - 1
            if inside >= 0 and guards[inside][1] > start + cut:
                guard_start, guard_end = guards[inside]
                cut = (guard_start if guard_start > start else guard_end) - start
        piece_start, piece_end = _stripped(text, start, start + cut)
        if piece_end > piece_start:
            pieces.append((piece_start, text[piece_start:piece_end]))
        start, end = _stripped(text, start + cut, end)
    if end > start:
        pieces.append((start, text[start:end]))
    return pieces


def _stripped(text: str, start: int, end: int) -> tuple[int, int]:
    """The bounds of ``text[start:end].strip()`` in *text*."""
    span = text[start:end]
    lead = len(span) - len(span.lstrip())
    return start + lead, start + lead + len(span.strip())


def _hits_within(hits: Sequence[Hit], offset: int, length: int) -> list[Hit]:
    """The hits that lie wholly inside the piece at *offset*, *length* characters long,
    positioned on the piece."""
    return [
        replace(hit, start=hit.start - offset)
        for hit in hits
        if offset <= hit.start and hit.start + hit.length <= offset + length
    ]


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
    path: Path,
    *,
    gate: MemoryGate,
    user_id: str,
    progress: Any = None,
    canary_every: int = DEFAULT_IMPORT_CANARY_EVERY,
) -> ImportReport:
    """Import every usable turn of *path* through *gate*, returning what was written.

    *progress*, when given, is called with ``(conversations_done, total)`` -- an import of a
    few thousand turns takes minutes of embedding calls and a silent one looks hung.

    Every *canary_every* memories this actually stores -- never a piece skipped because it
    was already there unchanged, which costs no embedding -- it re-checks the active embedding
    space against its recorded fingerprint (``MemoryGate.check_embedding_space``), and once
    more at the end for whatever was stored since the last good check. This bounds how many
    memories a model that is *still* answering wrong when a check runs can reach before it is
    named; it does not catch a vector that was wrong only in between two checks. A mismatch
    raises ``ImportStopped`` naming the suspect range; the suspects stay stored with whatever
    vectors they were given, and a re-run skips them like anything else already stored, rather
    than repairing them. ``surfaces/cli/commands.py::cmd_import`` passes
    ``settings.import_canary_every``, whose default is the same
    ``config.DEFAULT_IMPORT_CANARY_EVERY`` this parameter defaults to; this function reads no
    settings of its own, so a caller with no ``Settings`` still gets a sensible interval.

    Raises ``ValueError`` by name if *canary_every* is below 1 -- a caller bypassing
    ``Settings``'s own ``ge=1`` validation would otherwise divide by zero.
    """
    if canary_every < 1:
        raise ValueError(f"canary_every must be at least 1, got {canary_every}")
    # to_thread: a real export is tens of megabytes, and reading it inline would block the
    # loop the embedding calls below run on.
    raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
    conversations = json.loads(raw)
    kept = held = stored = skipped = 0
    redacted = provider = 0
    #: Memory ids stored since the last canary that still matched -- the suspects a failure
    #: names, and what a caller acts on directly without parsing the message.
    since_last_canary: list[str] = []

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
            # The turn is scanned whole, once, before any cut: a secret a cut would halve is
            # seen whole, and each piece is handed the hits that fall inside it.
            scanned = gate.scan_result(text, verdict="redact")
            hits = (*scanned.redactions, *scanned.flags)
            for part, (offset, piece) in enumerate(split_for_embedding(scanned.text)):
                memory_id = _memory_id(message_id, part)
                # Already imported, unchanged: skip it. Every piece costs an embedding call
                # and a real export is thousands of them, so an import that redoes finished
                # work is one that never finishes on a machine that gets interrupted. The
                # content check keeps a corrected turn from being frozen out by its own id. It
                # compares with the piece as `store` writes it: `store` scans the piece again,
                # and a shorter text can redact what the whole turn did not.
                existing = await gate.get(memory_id, user_id=user_id)
                if (
                    existing is not None
                    and existing.content == gate.scan_result(piece, verdict="redact").text
                ):
                    skipped += 1
                    wrote_any = True
                    continue
                memory = Memory(
                    id=memory_id,
                    user_id=user_id,
                    project=project,
                    kind=MemoryKind.EPISODIC,
                    content=piece,
                    source=source,
                    created_at=_created_at(message),
                    origin_kind=OriginKind.IMPORT,
                    client="cli",
                    cwd=str(Path.cwd()),
                    author_id=user_id,
                )
                await gate.store(
                    memory, verdict="redact", hits=_hits_within(hits, offset, len(piece))
                )
                recorded = json.loads(memory.redactions)
                if recorded:
                    redacted += 1
                    provider += sum(1 for hit in recorded if hit["rule"] in PROVIDER_RULE_NAMES)
                stored += 1
                since_last_canary.append(memory_id)
                wrote_any = True
                if stored % canary_every == 0:
                    await _run_canary(gate, stored, since_last_canary)
                    since_last_canary = []

        if wrote_any:
            held += project == HOLDOUT_PROJECT
            kept += project == ARCHIVE_PROJECT
        if progress is not None:
            progress(done, len(conversations))

    if since_last_canary:
        await _run_canary(gate, stored, since_last_canary)

    return ImportReport(
        conversations=kept,
        held_out=held,
        memories=stored,
        skipped_turns=skipped,
        redacted_pieces=redacted,
        provider_hits=provider,
    )


async def _run_canary(gate: MemoryGate, stored: int, since_last_canary: list[str]) -> None:
    """Re-check the active embedding space; a mismatch stops the import rather than trusting
    anything stored since the last good check.

    *stored* is this run's own count of memories stored so far (``import_chatgpt``'s ``stored``
    at the moment of the call); *since_last_canary* the ids stored since the previous good
    canary (or the start of the run), in order, so the suspect range is
    ``stored - len(since_last_canary) + 1`` through *stored*.
    """
    try:
        await gate.check_embedding_space()
    except EmbeddingSpaceMismatch as exc:
        raise ImportStopped(
            first=stored - len(since_last_canary) + 1,
            last=stored,
            suspect_ids=list(since_last_canary),
            setting=exc.setting,
            detail=exc.detail,
        ) from exc
