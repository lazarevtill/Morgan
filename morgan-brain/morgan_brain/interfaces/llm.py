"""ChatClient seam — provider-agnostic chat. Callers normally use RoleRouter, not a client directly."""

from __future__ import annotations
from typing import Any, AsyncIterator, Protocol, runtime_checkable
from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


@runtime_checkable
class ChatClient(Protocol):
    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def astream(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]: ...
