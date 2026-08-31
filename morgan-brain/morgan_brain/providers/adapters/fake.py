"""Deterministic fake adapters for tests.

``FakeChatClient`` — scripted ChatResult responses, no network.
``FakeEmbedder``   — SHA-256-based L2-normalized vectors, no network.

Both satisfy their respective Protocols (``ChatClient``, ``Embedder``) so they can be
substituted anywhere a real adapter is expected.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import AsyncIterator

from morgan_brain.providers.wire import (
    ChatMessage,
    ChatResult,
    StreamDelta,
    ToolCall,
    ToolSpec,
)


class FakeChatClient:
    """Scripted chat client — returns pre-set replies without any network call.

    Args:
        reply:      Single reply text repeated for every call (back-compat).
        replies:    Queue of reply texts consumed one per call; when exhausted, the last
                    item is repeated (so tests that do one extra call don't crash).
        tool_calls: Optional tool calls to include in every ChatResult.
        results:    Queue of full ``ChatResult`` objects consumed one per call (highest
                    priority; overrides *reply* / *replies* / *tool_calls* when set).
                    When the queue is exhausted the last item is repeated.

    Attributes:
        calls:         Total number of ``agenerate`` invocations.
        last_messages: The message list from the most recent ``agenerate`` call.
        last_model:    The model string from the most recent ``agenerate`` call.
    """

    def __init__(
        self,
        reply: str = "",
        replies: list[str] | None = None,
        tool_calls: list[ToolCall] | None = None,
        results: list[ChatResult] | None = None,
    ) -> None:
        # Full-result queue: each entry is returned verbatim for one agenerate call.
        # When exhausted, the last entry is repeated indefinitely.
        # Mutually exclusive with reply/replies/tool_calls (results takes priority).
        if results is not None:
            self._result_queue: deque[ChatResult] = deque(results)
            self._last_result: ChatResult | None = results[-1] if results else None
        else:
            self._result_queue = deque()
            self._last_result = None

        if replies is not None:
            self._queue: deque[str] = deque(replies)
            self._last_reply: str = replies[-1] if replies else reply
        else:
            self._queue = deque([reply])
            self._last_reply = reply
        self._tool_calls = tool_calls or []

        self.calls: int = 0
        self.last_messages: list[ChatMessage] = []
        self.last_model: str = ""
        self.last_response_format: dict[str, object] | None = None

    def _next_reply(self) -> str:
        if self._queue:
            text = self._queue.popleft()
            self._last_reply = text
            return text
        return self._last_reply

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, object] | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.last_messages = messages
        self.last_model = model
        self.last_response_format = response_format
        # Full-result queue takes priority (enables per-call tool_calls scripting).
        if self._result_queue:
            r = self._result_queue.popleft()
            self._last_result = r
            return ChatResult(text=r.text, model=model, tool_calls=list(r.tool_calls))
        if self._last_result is not None:
            # results= was provided and queue is exhausted — repeat last entry.
            return ChatResult(
                text=self._last_result.text,
                model=model,
                tool_calls=list(self._last_result.tool_calls),
            )
        reply = self._next_reply()
        return ChatResult(text=reply, model=model, tool_calls=list(self._tool_calls))

    async def _astream_impl(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        # Record the prompt so tests can assert what reached the model on the
        # streaming path too (e.g. the champion system_override).
        self.last_messages = messages
        # Peek at the next reply without consuming the call counter.
        # astream does NOT increment calls — only agenerate does.
        reply = self._last_reply if not self._queue else self._queue[0]
        yield StreamDelta(kind="text_delta", text=reply)
        yield StreamDelta(kind="finish", finish_reason="stop")

    def astream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        return self._astream_impl(messages, model=model, tools=tools)


class FakeEmbedder:
    """Deterministic embedder using SHA-256 → L2-normalised vector.

    Content-sensitive (different texts produce different vectors) and stable across
    runs without any model dependency. Matches the approach used by the Phase-1
    ``morgan_brain.modules.memory.indexing.embedder.FakeEmbedder``.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
