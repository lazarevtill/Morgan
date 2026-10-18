"""Caller-declared byte bounds for canonical OpenAI-compatible request JSON.

This is not a tokenizer or HTTP/header/template budget. The payload includes all
ChatClient input arguments; custom backends must not treat it as their wire size.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from morgan_brain.providers.wire import ChatMessage, ToolSpec


@dataclass(frozen=True)
class ChatRequestOptions:
    """Explicit SDK request controls; None leaves the provider's default unchanged.

    enable_thinking requires a compatible server/template. No model detection or
    arbitrary extra payload is accepted. max_output_tokens includes reasoning.
    """

    max_output_tokens: int | None = None
    enable_thinking: bool | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        limit = self.max_output_tokens
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None")
        temperature = self.temperature
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
            or not math.isfinite(temperature)
        ):
            raise ValueError("temperature must be a finite number between 0 and 2 or None")


def request_options_for(client: object) -> ChatRequestOptions | None:
    """Optional client-owned controls, shared by the adapter and structured accounting."""
    options = getattr(client, "request_options", None)
    if options is not None and not isinstance(options, ChatRequestOptions):
        raise TypeError("request_options must be ChatRequestOptions or None")
    return options


def openai_request_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the one supported server extension to the SDK's extra_body argument."""
    kwargs = dict(payload)
    if "chat_template_kwargs" in kwargs:
        kwargs["extra_body"] = {"chat_template_kwargs": kwargs.pop("chat_template_kwargs")}
    return kwargs


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
    request_options: ChatRequestOptions | None = None,
) -> dict[str, Any]:
    """Canonical wire payload used by the adapter and structured accounting."""
    payload: dict[str, Any] = {"model": model, "messages": [m.to_openai() for m in messages]}
    if tools:
        payload["tools"] = [tool.to_openai() for tool in tools]
    if response_format:
        payload["response_format"] = response_format
    if request_options is not None:
        if not isinstance(request_options, ChatRequestOptions):
            raise TypeError("request_options must be ChatRequestOptions or None")
        if request_options.max_output_tokens is not None:
            payload["max_tokens"] = request_options.max_output_tokens
        if request_options.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": request_options.enable_thinking}
        if request_options.temperature is not None:
            payload["temperature"] = request_options.temperature
    return payload


def serialized_request_bytes(payload: dict[str, Any]) -> int:
    """Compact UTF-8 JSON, including field names, escaping, model and schema overhead."""
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
