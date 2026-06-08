"""EgressGate — privacy chokepoint wrapping a ChatClient.

For **local** providers (Ollama, vLLM, llama.cpp, LM Studio) the gate is a
transparent pass-through: full context is sent without modification.

For **remote** providers the gate:
  1. Classifies each outbound message.  If any message is SECRET and
     ``block_secret=True``, raises ``PermissionError`` before sending anything.
  2. Redacts PII from all outbound message *content* via the supplied
     ``EgressRedactor`` (or skips if ``redactor`` is ``None``).
  3. Calls the inner ``ChatClient``.
  4. Rehydrates placeholders in the response text so the owner sees real values.

Streaming (``astream``): outbound messages are redacted (same as non-streaming);
the response deltas are rehydrated best-effort per chunk.

The gate satisfies the ``ChatClient`` Protocol unconditionally so it can be
dropped in anywhere a ``ChatClient`` is expected.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from morgan_brain.interfaces.llm import ChatClient
from morgan_brain.privacy.classification import DataClass, classify
from morgan_brain.privacy.redaction import EgressRedactor, RedactionMap
from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


class EgressGate:
    """Privacy gate wrapping a ``ChatClient``.

    Parameters
    ----------
    inner:
        The wrapped ``ChatClient`` (real adapter or fake, for tests).
    is_remote:
        ``True`` → apply redaction / secret-blocking.
        ``False`` → pass through unchanged (local provider).
    redactor:
        The ``EgressRedactor`` instance to use for outbound redaction.  If
        ``None``, no redaction is performed (even for remote).
    block_secret:
        When ``True`` (default), any message classified as SECRET raises
        ``PermissionError`` before the remote call is attempted.
    """

    def __init__(
        self,
        inner: ChatClient,
        *,
        is_remote: bool,
        redactor: EgressRedactor | None,
        block_secret: bool = True,
    ) -> None:
        self._inner = inner
        self._is_remote = is_remote
        self._redactor = redactor
        self._block_secret = block_secret

    # ------------------------------------------------------------------
    # ChatClient interface
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if not self._is_remote:
            return await self._inner.agenerate(
                messages, model=model, tools=tools, response_format=response_format
            )

        # --- Remote path ---
        self._check_secret(messages)
        redacted_messages, rmap = self._redact_messages(messages)
        result = await self._inner.agenerate(
            redacted_messages, model=model, tools=tools, response_format=response_format
        )
        return self._rehydrate_result(result, rmap)

    def astream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        return self._astream_impl(messages, model=model, tools=tools)

    async def _astream_impl(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        if not self._is_remote:
            async for delta in self._inner.astream(messages, model=model, tools=tools):
                yield delta
            return

        # --- Remote path ---
        self._check_secret(messages)
        redacted_messages, rmap = self._redact_messages(messages)

        # Build a rehydration step-function for streaming
        from morgan_brain.privacy.redaction import rehydrate_stream

        step = rehydrate_stream(rmap)

        async for delta in self._inner.astream(redacted_messages, model=model, tools=tools):
            if delta.kind == "text_delta" and delta.text:
                rehydrated = step(delta.text)
                yield StreamDelta(kind="text_delta", text=rehydrated)
            elif delta.kind == "finish":
                # Flush the stream buffer
                remainder = step(None)
                if remainder:
                    yield StreamDelta(kind="text_delta", text=remainder)
                yield delta
            else:
                yield delta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_secret(self, messages: list[ChatMessage]) -> None:
        if not self._block_secret:
            return
        for msg in messages:
            if classify(msg.content) >= DataClass.SECRET:
                raise PermissionError(
                    "secret-tier blocked from remote provider: "
                    "message content classified as SECRET.  "
                    "Use a local provider for secret-tier data."
                )

    def _redact_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[ChatMessage], RedactionMap]:
        if self._redactor is None:
            return messages, {}

        merged_map: RedactionMap = {}
        redacted_list: list[ChatMessage] = []
        for msg in messages:
            redacted_content, rmap = self._redactor.redact(msg.content)
            merged_map.update(rmap)
            redacted_list.append(msg.model_copy(update={"content": redacted_content}))
        return redacted_list, merged_map

    def _rehydrate_result(self, result: ChatResult, rmap: RedactionMap) -> ChatResult:
        if not rmap or self._redactor is None:
            return result
        rehydrated_text = self._redactor.rehydrate(result.text, rmap)
        return result.model_copy(update={"text": rehydrated_text})
