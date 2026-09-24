"""One chat turn: recall, answer, remember.

This is the whole cognitive loop of the core. Recall what this project knows, put it in
front of the model with the recent history, answer, and store both halves of the exchange
as episodic memory so the next turn -- and the next consolidation -- can find them.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets import Verdict
from morgan_brain.memory.store.history import SessionHistoryStore, session_key
from morgan_brain.models import Memory, MemoryQuery, MemorySource, Message, OriginKind, Role
from morgan_brain.providers.wire import ChatClient, ChatMessage

_SYSTEM = (
    "You are Morgan, a personal assistant that knows the user well. "
    "Use the provided memories when relevant. If a memory conflicts with general knowledge, "
    "prefer the memory. Be helpful and concise."
)


def build_messages(
    *, memories: list[Memory], history: list[Message], text: str
) -> list[ChatMessage]:
    """The prompt: system + recalled memories, the prior history, then the user's turn.
    Pure and deterministic."""
    system = _SYSTEM
    if memories:
        system += "\n\nRelevant memories:\n" + "\n".join(f"- {m.content}" for m in memories)
    messages = [ChatMessage(role="system", content=system)]
    messages.extend(ChatMessage(role=m.role.value, content=m.content) for m in history)
    messages.append(ChatMessage(role="user", content=text))
    return messages


class Chat:
    def __init__(
        self,
        *,
        gate: MemoryGate,
        history: SessionHistoryStore,
        client: ChatClient,
        model: str,
        clock: Callable[[], datetime],
    ) -> None:
        self._gate = gate
        self._history = history
        self._client = client
        self._model = model
        self._clock = clock

    async def ask(
        self,
        *,
        user_id: str,
        project: str,
        text: str,
        session_id: str | None = None,
        caller_client: str = "",
        caller_session_id: str = "",
    ) -> str:
        """Answer *text* for *user_id* in *project*, and remember the exchange.

        On a database waiting for ``morgan migrate`` it refuses before anything else: the
        model call would otherwise happen, and only the rows after it would be refused.

        A question holding a provider token is refused here, before the recall or any model
        call, and one with a generic hit goes to the recall and the prompt redacted. The reply
        was produced and paid for, so nobody can rephrase it: it is redacted, never refused,
        and what this returns is the redacted reply. Each history row and each memory is
        scanned once, by the gate method that writes it, from its own raw text -- the reply's
        under ``redact`` -- so each memory records what the gate redacted in it.

        *caller_client*/*caller_session_id* are provenance for the two memories this turn
        writes -- named apart from *session_id* (history bucketing) and ``Chat``'s own
        ``client`` (the ``ChatClient`` this instance calls the model through) on purpose, so
        neither collides with an existing parameter of a different kind.
        """
        self._gate.require_writable()
        # The question is scanned before it is embedded, sent or stored: a provider token
        # refuses the turn here, and a generic hit goes on as its redacted text.
        _, redacted = self._gate.scan_text(text)
        hkey = session_key(user_id, session_id)
        history = self._history.recent(hkey, project=project)
        recalled = await self._gate.recall(
            MemoryQuery(user_id=user_id, project=project, text=redacted)
        )
        result = await self._client.agenerate(
            build_messages(memories=recalled.memories, history=history, text=redacted),
            model=self._model,
        )
        # The reply was produced and paid for; nobody can rephrase it, so what is returned is
        # redacted, never refused.
        _, reply = self._gate.scan_text(result.text, verdict="redact")

        # Each writer scans the raw text once itself, so a memory records what the gate
        # redacted in it; a text scanned twice keeps its redaction and loses the record.
        for role, content in ((Role.USER, text), (Role.ASSISTANT, result.text)):
            await self._gate.append_history(
                user_id=user_id,
                project=project,
                session_key=hkey,
                message=Message(user_id=user_id, role=role, content=content),
            )
        # Both halves are remembered, attributed to who said them: the user's words are a
        # statement, the reply is an inference and must never be mistaken for one. Both rows
        # came from the same ask turn, so both carry origin_kind=ASK -- source is what tells
        # them apart, not origin. The question is stored under ``refuse``, which it passed
        # above; the reply under ``redact``, the verdict its history row gets.
        halves: tuple[tuple[str, MemorySource, Verdict], ...] = (
            (text, MemorySource.USER_STATED, "refuse"),
            (result.text, MemorySource.AGENT_INFERRED, "redact"),
        )
        for content, source, verdict in halves:
            await self._gate.store(
                Memory(
                    user_id=user_id,
                    project=project,
                    content=content,
                    source=source,
                    created_at=self._clock(),
                    origin_kind=OriginKind.ASK,
                    author_id=user_id,
                    cwd=str(Path.cwd()),
                    client=caller_client,
                    session_id=caller_session_id,
                ),
                verdict=verdict,
            )
        return reply
