"""Caller-declared byte bounds for canonical OpenAI-compatible request JSON.

This is not a tokenizer or HTTP/header/template budget. The payload includes all
ChatClient input arguments; custom backends must not treat it as their wire size.
"""

from __future__ import annotations

import json
from typing import Any

from morgan_brain.providers.wire import ChatMessage, ToolSpec


class StructuredRequestTooLarge(ValueError):
    """A complete structured request exceeded the caller's explicit byte ceiling."""

    def __init__(self, *, measured_bytes: int, limit_bytes: int, attempt: int) -> None:
        self.measured_bytes = measured_bytes
        self.limit_bytes = limit_bytes
        self.attempt = attempt
        super().__init__(
            f"Structured request attempt {attempt} is {measured_bytes} UTF-8 JSON bytes; "
            f"caller limit is {limit_bytes} bytes. No generation attempted for this request."
        )


def validate_request_byte_limit(limit: int | None) -> None:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("request_byte_limit must be a positive integer or None (disabled)")


def chat_request_payload(
    messages: list[ChatMessage],
    *,
    model: str,
    tools: list[ToolSpec] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The same complete kwargs used by OpenAICompatAdapter.agenerate."""
    payload: dict[str, Any] = {"model": model, "messages": [m.to_openai() for m in messages]}
    if tools:
        payload["tools"] = [tool.to_openai() for tool in tools]
    if response_format:
        payload["response_format"] = response_format
    return payload


def serialized_request_bytes(payload: dict[str, Any]) -> int:
    """Compact UTF-8 JSON, including field names, escaping, model and schema overhead."""
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
