"""ChatClient seam — provider-agnostic chat.

Callers normally use RoleRouter, not a client directly.

Every adapter behind this seam (chat and embeddings alike) raises ``ProviderUnreachable``
when the endpoint cannot be reached or does not answer — the one failure a personal brain on
a remote model server meets every day, and the one every surface has to report by name:
brain-api as a 502, the CLI and the MCP server as a message that says which endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


class ProviderUnreachable(ConnectionError):
    """The model endpoint could not be reached, or gave no answer in time.

    Carries the endpoint so the message can name it: "Connection error." tells the owner
    nothing about *which* of their configured servers is down.
    """

    def __init__(self, endpoint: str, detail: str) -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(
            f"model endpoint {endpoint} is unreachable ({detail}); check MORGAN_LLM_ENDPOINT "
            "and run `morgan doctor`"
        )


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
