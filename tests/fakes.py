"""Deterministic test doubles. Nothing here ships."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


class FakeChatClient:
    """A scripted chat client. ``replies`` are consumed one per call; when exhausted the last
    one repeats. Records the last prompt so a test can assert what reached the model."""

    def __init__(self, reply: str = "", replies: list[str] | None = None) -> None:
        self._queue: deque[str] = deque(replies if replies is not None else [reply])
        self._last_reply = self._queue[-1] if self._queue else reply
        self.calls = 0
        self.last_messages: list[ChatMessage] = []
        self.last_model = ""
        self.last_response_format: dict[str, Any] | None = None

    def _next_reply(self) -> str:
        if self._queue:
            self._last_reply = self._queue.popleft()
        return self._last_reply

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.last_messages = messages
        self.last_model = model
        self.last_response_format = response_format
        return ChatResult(text=self._next_reply(), model=model)

    async def _astream_impl(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]:
        self.last_messages = messages
        yield StreamDelta(kind="text_delta", text=self._next_reply())
        yield StreamDelta(kind="finish", finish_reason="stop")

    def astream(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]:
        return self._astream_impl(messages, model=model, tools=tools)
