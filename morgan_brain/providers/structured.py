"""Structured output: ask the model for JSON, validate it against a Pydantic schema, re-ask
with the validation error when it is wrong.

How the JSON is requested is a setting (``MORGAN_LLM_JSON_MODE``), not a per-model capability
table: the core has one chat model, and the owner knows what their server supports.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from morgan_brain.providers.wire import ChatClient, ChatMessage

JsonMode = Literal["json_schema", "json_object", "prompted"]


class StructuredError(Exception):
    """Raised when every attempt (the first plus the re-asks) failed to produce a valid parse."""


def structured_request(
    messages: list[ChatMessage], *, schema: type[BaseModel], json_mode: JsonMode
) -> tuple[list[ChatMessage], dict[str, Any] | None]:
    """Prepare the exact mode-specific messages and response format once."""
    working = list(messages)
    response_format: dict[str, Any] | None = None
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    if json_mode == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    elif json_mode == "json_object":
        response_format = {"type": "json_object"}
        working = [
            ChatMessage(
                role="system",
                content=f"Respond ONLY with a JSON object matching this schema:\n{schema_json}",
            ),
            *working,
        ]
    else:
        working = [
            *working,
            ChatMessage(
                role="user",
                content=(
                    "Respond ONLY with a valid JSON object matching this schema "
                    f"(no markdown, no explanation):\n{schema_json}"
                ),
            ),
        ]

    return working, response_format


def structured_input_size(
    messages: list[ChatMessage], response_format: dict[str, Any] | None, *, model: str
) -> int:
    payload: dict[str, Any] = {"messages": [m.to_openai() for m in messages], "model": model}
    if response_format is not None:
        payload["response_format"] = response_format
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


async def generate_structured[M: BaseModel](
    client: ChatClient,
    messages: list[ChatMessage],
    *,
    model: str,
    schema: type[M],
    json_mode: JsonMode = "json_schema",
    max_reask: int = 2,
    max_input_bytes: int | None = None,
) -> M:
    working, response_format = structured_request(messages, schema=schema, json_mode=json_mode)
    if max_input_bytes is not None and (
        isinstance(max_input_bytes, bool)
        or not isinstance(max_input_bytes, int)
        or max_input_bytes <= 0
    ):
        raise ValueError("max_input_bytes must be a positive integer")

    last_error = ""
    attempts = 1 + max_reask
    for attempt in range(attempts):
        if attempt > 0:
            working = [
                *working,
                ChatMessage(
                    role="user",
                    content=(
                        f"The previous response was invalid. Error: {last_error}\n"
                        "Please respond ONLY with valid JSON matching the schema."
                    ),
                ),
            ]
        if (
            max_input_bytes is not None
            and structured_input_size(working, response_format, model=model) > max_input_bytes
        ):
            raise ValueError("structured model input exceeds byte limit")
        result = await client.agenerate(working, model=model, response_format=response_format)
        if result.finish_reason != "stop" or result.tool_calls:
            raise StructuredError(
                "Structured generation incomplete or tool-bearing; no proposal accepted"
            )
        try:
            return schema.model_validate_json(result.text)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)

    raise StructuredError(
        f"Failed to parse a valid {schema.__name__} after {attempts} attempt(s). "
        f"Last error: {last_error}"
    )
