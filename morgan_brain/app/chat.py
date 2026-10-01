"""One chat turn: recall, answer, remember.

This is the whole cognitive loop of the core. Recall what this project knows, put it in
front of the model with the recent history, answer, and store both halves of the exchange
as episodic memory so the next turn -- and the next consolidation -- can find them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from morgan_brain.app.strict_context import (
    AnswerBudget,
    AnswerResult,
    CountedRequest,
    EvidenceScope,
    PackedContext,
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
from morgan_brain.memory.errors import EvidenceChanged
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.recall.language import of as query_language
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
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


@dataclass(frozen=True)
class TurnRequest:
    """Per-call input; source and author are reported attribution, not credentials."""

    user_id: str
    project: str
    text: str
    session_id: str | None = None
    caller_client: str = ""
    caller_session_id: str = ""
    source: MemorySource = MemorySource.UNKNOWN
    author_id: str = ""


@dataclass(frozen=True)
class DefaultAnswerResult:
    answer: str
    model_used: str | None
    response_author_id: str


@dataclass(frozen=True)
class TurnBasis:
    input_at: datetime
    generation: int
    evidence: list[Memory] | None
    reply_author_id: str | None = None


async def _no_backend_to_close() -> None:
    """A counted capability may have no transport to release."""


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
        close = getattr(self._strict_backend, "aclose", _no_backend_to_close)
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
        reply, _, _ = await self._turn(
            TurnRequest(
                user_id=user_id,
                project=project,
                text=text,
                session_id=session_id,
                caller_client=caller_client,
                caller_session_id=caller_session_id,
                source=source,
                author_id=author_id,
            ),
            strict_context=strict_context,
        )
        return reply

    async def ask_with_provenance(self, request: TurnRequest) -> DefaultAnswerResult:
        """Default answer attribution returned per call, without mutable shared state."""
        reply, _, author = await self._turn(request)
        return DefaultAnswerResult(
            answer=reply,
            model_used=None if author == "morgan:conflict-guard" else self._model,
            response_author_id=author,
        )

    async def ask_evidence(self, request: TurnRequest) -> AnswerResult:
        """Return a strict cited answer and measured budget after atomic persistence."""
        try:
            _, detailed, _ = await self._turn(request, strict_context=True)
        except EvidenceChanged as exc:
            raise StrictContextError(exc.reason) from exc
        except StoreInterruptedByForget as exc:
            raise StrictContextError("store_interrupted_by_forget") from exc
        if detailed is None:
            raise RuntimeError("Strict turn did not produce an answer contract")
        return detailed

    async def _turn(
        self, turn: TurnRequest, *, strict_context: bool = False
    ) -> tuple[str, AnswerResult | None, str]:
        # Admission and erasure capture precede every asynchronous operation.
        MemorySource(turn.source)
        self._gate.require_writable()
        generation = self._gate.capture_erasure_generation()
        input_at = self._clock()
        counter = await self._prepare_strict(turn) if strict_context else None
        hkey = session_key(turn.user_id, turn.session_id)
        history = self._history.recent(hkey, project=turn.project, user_id=turn.user_id)
        recalled = await self._gate.recall(
            MemoryQuery(user_id=turn.user_id, project=turn.project, text=turn.text)
        )
        detailed = None
        evidence_basis = [record.model_copy(deep=True) for record in recalled.memories]
        reply_author_id = None
        if counter is not None:
            packed = await self._pack_strict(turn, recalled.memories, history, counter)
            detailed = await self._answer_strict(turn, packed, counter)
            reply = detailed.answer
            evidence_basis = packed.records
        elif any(memory.revision_state == "conflicted" for memory in recalled.memories):
            # Default prose omits revision metadata. Do not let a model select a fork.
            reply = (
                "В найденной памяти есть неразрешённые версии. Используйте recall и evidence; "
                "проверьте eligible_leaf_count и revision_truncated. Уточните версию "
                "через remember "
                "с revises_event_ids: не более 8 родителей за шаг; большие группы объединяйте "
                "поэтапно, не пропуская оставшиеся ветви."
                if query_language(turn.text) == "ru"
                else "Recalled memories have unresolved versions. Use recall and evidence; "
                "check eligible_leaf_count and revision_truncated. Clarify via remember with "
                "revises_event_ids: at most 8 parents per step; join larger groups in stages "
                "without omitting remaining branches."
            )
            reply_author_id = "morgan:conflict-guard"
        else:
            result = await self._client.agenerate(
                build_messages(memories=recalled.memories, history=history, text=turn.text),
                model=self._model,
            )
            reply = result.text
        await self._persist_turn(
            turn, reply, TurnBasis(input_at, generation, evidence_basis, reply_author_id)
        )
        return reply, detailed, reply_author_id or f"model:{self._model}"

    async def _prepare_strict(self, turn: TurnRequest) -> CountedRequest:
        backend = self._strict_backend
        if (
            backend is None
            or not callable(getattr(backend, "count_request", None))
            or not callable(getattr(backend, "generate_counted", None))
        ):
            raise StrictContextError("token_counter_unavailable")
        if len(turn.text.encode("utf-8")) > self._strict_config.max_input_bytes:
            raise StrictContextError("input_budget_exceeded")
        request = strict_request(self._model, self._strict_config)
        base_messages = render_messages([], [], turn.text)
        if serialized_request_bytes(base_messages, request) > self._strict_config.max_input_bytes:
            raise StrictContextError("input_budget_exceeded")
        base_count = await count_request(backend, base_messages, request)
        if not fits(base_count, self._strict_config):
            raise StrictContextError("input_budget_exceeded")
        return CountedRequest(backend, request, base_count)

    async def _pack_strict(
        self,
        turn: TurnRequest,
        memories: list[Memory],
        history: list[Message],
        counter: CountedRequest,
    ) -> PackedContext:
        closure = await resolve_closure(
            self._gate,
            scope=EvidenceScope(turn.user_id, turn.project),
            candidates=memories,
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
        return await pack_context(
            closure,
            history=history,
            text=turn.text,
            counter=counter,
            config=self._strict_config,
        )

    async def _answer_strict(
        self, turn: TurnRequest, packed: PackedContext, counter: CountedRequest
    ) -> AnswerResult:
        if not packed.records:
            raise StrictContextError("evidence_incomplete")
        try:
            result = await asyncio.wait_for(
                counter.backend.generate_counted(
                    packed.messages, request=counter.request, count=packed.count
                ),
                # The calibrated adapter classifies its own 60-second deadline first.
                timeout=61,
            )
        except TimeoutError as exc:
            raise StrictContextError("generation_timeout") from exc
        if (
            request_fingerprint(packed.messages, counter.request)
            != packed.count.request_fingerprint
        ):
            raise StrictContextError("token_counter_unavailable")
        if result.finish_reason != "stop" or result.tool_calls:
            raise StrictContextError("generation_incomplete")
        if result.usage.output_tokens > counter.request.output_tokens:
            raise StrictContextError("output_budget_exceeded")
        if (
            result.model != self._model
            or result.usage.input_tokens <= 0
            or result.usage.input_tokens != packed.count.input_tokens
        ):
            raise StrictContextError("token_counter_unavailable")
        validated = validate_answer(result.text, packed.records)
        return AnswerResult(
            user_id=turn.user_id,
            project=turn.project,
            model=result.model,
            answer=validated.answer,
            evidence_ids=list(validated.evidence_ids),
            abstained=validated.abstained,
            budget=AnswerBudget(
                input_tokens=packed.count.input_tokens,
                reported_output_tokens=result.usage.output_tokens,
                total_tokens=self._strict_config.total_tokens,
                output_reserve_tokens=counter.request.output_tokens,
                safety_tokens=self._strict_config.safety_tokens,
                template_id=packed.count.template_id,
                counter_calls=packed.counter_calls,
            ),
        )

    async def _persist_turn(
        self,
        turn: TurnRequest,
        reply: str,
        basis: TurnBasis,
    ) -> None:
        hkey = session_key(turn.user_id, turn.session_id)
        reply_at = self._clock()

        # Source is reported evidence provenance, independent of the chat wire role.
        # Client-authored input never becomes a user statement merely by using ask.
        memories = []
        for content, evidence_source, reported_author, effective_at in (
            (turn.text, MemorySource(turn.source), turn.author_id, basis.input_at),
            (
                reply,
                MemorySource.AGENT_INFERRED,
                basis.reply_author_id or f"model:{self._model}",
                reply_at,
            ),
        ):
            memories.append(
                Memory(
                    user_id=turn.user_id,
                    project=turn.project,
                    content=content,
                    source=evidence_source,
                    created_at=effective_at,
                    origin_kind=OriginKind.ASK,
                    author_id=reported_author,
                    cwd=str(Path.cwd()),
                    client=turn.caller_client,
                    session_id=turn.caller_session_id,
                )
            )
        await self._gate.store_turn(
            memories,
            history=self._history,
            expected_generation=basis.generation,
            evidence_basis=basis.evidence,
            history_entries=[
                (
                    hkey,
                    turn.project,
                    Message(
                        user_id=turn.user_id,
                        project=turn.project,
                        role=Role.USER,
                        content=turn.text,
                    ),
                ),
                (
                    hkey,
                    turn.project,
                    Message(
                        user_id=turn.user_id,
                        project=turn.project,
                        role=Role.ASSISTANT,
                        content=reply,
                    ),
                ),
            ],
        )
