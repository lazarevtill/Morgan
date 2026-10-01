"""Optional whole-request counting/generation capability; ChatClient stays unchanged."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from morgan_brain.providers.wire import ChatMessage, ChatResult


@dataclass(frozen=True)
class StrictRequest:
    model: str
    output_tokens: int
    response_format: dict[str, Any]
    temperature: float = 0.0
    seed: int = 42
    enable_thinking: bool = False


def request_fingerprint(messages: list[ChatMessage], request: StrictRequest) -> str:
    raw = {
        "messages": [message.to_openai() for message in messages],
        "model": request.model,
        "output_tokens": request.output_tokens,
        "response_format": request.response_format,
        "temperature": request.temperature,
        "seed": request.seed,
        "enable_thinking": request.enable_thinking,
    }
    return hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RequestCount:
    input_tokens: int
    exact: bool
    model: str
    template_id: str
    request_fingerprint: str


class StrictChatBackend(Protocol):
    async def count_request(
        self, messages: list[ChatMessage], *, request: StrictRequest
    ) -> RequestCount:
        """Count the complete calibrated request including the generation prefix."""

    async def generate_counted(
        self, messages: list[ChatMessage], *, request: StrictRequest, count: RequestCount
    ) -> ChatResult:
        """Generate with exactly the counted model/template/options and output cap."""
