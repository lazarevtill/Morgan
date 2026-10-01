"""One chat turn: recall, answer, remember.

This is the whole cognitive loop of the core. Recall what this project knows, put it in
front of the model with the recent history, answer, and store both halves of the exchange
as episodic memory so the next turn -- and the next consolidation -- can find them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from morgan_brain.app.strict_context import (
    AnswerBudget,
    AnswerResult,
    StrictContextConfig,
    StrictContextError,
    count_request,
    fits,
    pack_context,
    render_messages,
    resolve_closure,
    serialized_request_bytes,
    strict_request,
    validate_answer,
)
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.history import SessionHistoryStore, session_key
from morgan_brain.models import Memory, MemoryQuery, MemorySource, Message, OriginKind, Role
from morgan_brain.providers.context import StrictChatBackend, request_fingerprint
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
        strict_backend: StrictChatBackend | None = None,
        strict_config: StrictContextConfig | None = None,
    ) -> None:
        self._gate = gate
        self._history = history
        self._client = client
        self._model = model
        self._clock = clock
        self._strict_backend = strict_backend
        self._strict_config = strict_config or StrictContextConfig()

    async def aclose(self) -> None:
        """Release the optional counted backend's per-context HTTP client."""
        close = getattr(self._strict_backend, "aclose", None)
        if callable(close):
            await close()

    async def ask(
        self,
        *,
        user_id: str,
        project: str,
        text: str,
        session_id: str | None = None,
        caller_client: str = "",
        caller_session_id: str = "",
        source: MemorySource = MemorySource.UNKNOWN,
        author_id: str = "",
        strict_context: bool = False,
    ) -> str:
        """Answer and atomically remember a turn; retains the legacy string result."""
        reply, _ = await self._turn(
            user_id=user_id,
            project=project,
            text=text,
            session_id=session_id,
            caller_client=caller_client,
            caller_session_id=caller_session_id,
            source=source,
            author_id=author_id,
            strict_context=strict_context,
        )
        return reply

    async def ask_evidence(
        self,
        *,
        user_id: str,
        project: str,
        text: str,
        session_id: str | None = None,
        caller_client: str = "",
        caller_session_id: str = "",
        source: MemorySource = MemorySource.UNKNOWN,
        author_id: str = "",
    ) -> AnswerResult:
        """Return a strict cited answer and measured budget after atomic persistence."""
        _, detailed = await self._turn(
            user_id=user_id,
            project=project,
            text=text,
            session_id=session_id,
            caller_client=caller_client,
            caller_session_id=caller_session_id,
            source=source,
            author_id=author_id,
            strict_context=True,
        )
        if detailed is None:
            raise RuntimeError("Strict turn did not produce an answer contract")
        return detailed

    async def _turn(
        self,
        *,
        user_id: str,
        project: str,
        text: str,
        session_id: str | None = None,
        caller_client: str = "",
        caller_session_id: str = "",
        source: MemorySource = MemorySource.UNKNOWN,
        author_id: str = "",
        strict_context: bool = False,
    ) -> tuple[str, AnswerResult | None]:
        """Answer *text* for *user_id* in *project*, and remember the exchange.

        On a database waiting for ``morgan migrate`` it refuses before anything else: the
        model call and the two history rows below would otherwise happen, and only the
        memories after them would be refused.

        *caller_client*/*caller_session_id* are provenance for the two memories this turn
        writes -- named apart from *session_id* (history bucketing) and ``Chat``'s own
        ``client`` (the ``ChatClient`` this instance calls the model through) on purpose, so
        neither collides with an existing parameter of a different kind.
        """
        source = MemorySource(source)
        self._gate.require_writable()
        generation = self._gate.capture_erasure_generation()
        input_at = self._clock()
        backend = self._strict_backend
        request = None
        base_count = None
        if strict_context:
            if (
                backend is None
                or not callable(getattr(backend, "count_request", None))
                or not callable(getattr(backend, "generate_counted", None))
            ):
                raise StrictContextError("token_counter_unavailable")
            if len(text.encode("utf-8")) > self._strict_config.max_input_bytes:
                raise StrictContextError("input_budget_exceeded")
            request = strict_request(self._model, self._strict_config)
            base_messages = render_messages([], [], text)
            if (
                serialized_request_bytes(base_messages, request)
                > self._strict_config.max_input_bytes
            ):
                raise StrictContextError("input_budget_exceeded")
            base_count = await count_request(backend, base_messages, request)
            if not fits(base_count, self._strict_config):
                raise StrictContextError("input_budget_exceeded")
        hkey = session_key(user_id, session_id)
        history = self._history.recent(hkey, project=project, user_id=user_id)
        recalled = await self._gate.recall(MemoryQuery(user_id=user_id, project=project, text=text))
        evidence_basis = None
        detailed = None
        if strict_context:
            if backend is None or request is None or base_count is None:
                raise StrictContextError("token_counter_unavailable")
            closure = await resolve_closure(
                self._gate,
                user_id=user_id,
                project=project,
                candidates=recalled.memories,
                config=self._strict_config,
                effective_at=self._clock(),
            )
            if closure.abstention_reason:
                raise StrictContextError(
                    closure.abstention_reason,
                    evidence_ids=list(
                        dict.fromkeys(
                            identity
                            for record in closure.records
                            for identity in record.eligible_leaf_ids
                        )
                    ),
                )
            packed = await pack_context(
                closure,
                history=history,
                text=text,
                backend=backend,
                request=request,
                base_count=base_count,
                config=self._strict_config,
            )
            if not packed.records:
                raise StrictContextError("evidence_incomplete")
            try:
                result = await asyncio.wait_for(
                    backend.generate_counted(packed.messages, request=request, count=packed.count),
                    timeout=60,
                )
            except TimeoutError as exc:
                raise StrictContextError("generation_timeout") from exc
            if request_fingerprint(packed.messages, request) != packed.count.request_fingerprint:
                raise StrictContextError("token_counter_unavailable")
            if result.finish_reason != "stop" or result.tool_calls:
                raise StrictContextError("generation_incomplete")
            if result.usage.output_tokens > request.output_tokens:
                raise StrictContextError("output_budget_exceeded")
            if (
                result.model != self._model
                or result.usage.input_tokens <= 0
                or result.usage.input_tokens != packed.count.input_tokens
            ):
                raise StrictContextError("token_counter_unavailable")
            validated = validate_answer(result.text, packed.records)
            reply = validated.answer
            detailed = AnswerResult(
                user_id=user_id,
                project=project,
                model=result.model,
                answer=reply,
                evidence_ids=list(validated.evidence_ids),
                abstained=validated.abstained,
                budget=AnswerBudget(
                    input_tokens=packed.count.input_tokens,
                    reported_output_tokens=result.usage.output_tokens,
                    total_tokens=self._strict_config.total_tokens,
                    output_reserve_tokens=request.output_tokens,
                    safety_tokens=self._strict_config.safety_tokens,
                    template_id=packed.count.template_id,
                    counter_calls=packed.counter_calls,
                ),
            )
            evidence_basis = packed.records
        else:
            result = await self._client.agenerate(
                build_messages(memories=recalled.memories, history=history, text=text),
                model=self._model,
            )
            reply = result.text
        reply_at = self._clock()

        # Source is reported evidence provenance, independent of the chat wire role.
        # Client-authored input never becomes a user statement merely by using ask.
        memories = []
        for content, evidence_source, reported_author, effective_at in (
            (text, source, author_id, input_at),
            (reply, MemorySource.AGENT_INFERRED, f"model:{self._model}", reply_at),
        ):
            memories.append(
                Memory(
                    user_id=user_id,
                    project=project,
                    content=content,
                    source=evidence_source,
                    created_at=effective_at,
                    origin_kind=OriginKind.ASK,
                    author_id=reported_author,
                    cwd=str(Path.cwd()),
                    client=caller_client,
                    session_id=caller_session_id,
                )
            )
        await self._gate.store_turn(
            memories,
            history=self._history,
            expected_generation=generation,
            evidence_basis=evidence_basis,
            history_entries=[
                (
                    hkey,
                    project,
                    Message(user_id=user_id, project=project, role=Role.USER, content=text),
                ),
                (
                    hkey,
                    project,
                    Message(user_id=user_id, project=project, role=Role.ASSISTANT, content=reply),
                ),
            ],
        )
        return reply, detailed
