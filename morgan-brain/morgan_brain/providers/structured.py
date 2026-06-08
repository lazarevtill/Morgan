"""Structured-output ladder — tier-aware JSON generation + pydantic validation.

Tier selection is driven by ``CapabilityDescriptor.json_mode``:

* ``JSON_SCHEMA``  — pass ``response_format={"type":"json_schema",...}`` directly to the model;
                    model returns schema-constrained JSON natively.
* ``JSON_OBJECT``  — pass ``response_format={"type":"json_object"}`` + inject the JSON schema
                    into a system instruction so the model knows the target shape.
* ``NONE``         — append a plain-text instruction telling the model to respond *only*
                    with JSON matching the Pydantic model's schema.

In all tiers the response is always parsed and validated with ``schema.model_validate_json``.
On ``ValidationError`` or ``json.JSONDecodeError`` the error text is appended as a user
message and the call is retried up to *max_reask* times.  When retries are exhausted a
``StructuredError`` is raised.

Grammar/tool-as-schema tiers are intentionally left as TODO for live adapters — the
validate-and-re-ask spine already works without them.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from morgan_brain.providers.capability import CapabilityDescriptor, JsonMode
from morgan_brain.providers.wire import ChatMessage, ChatResult

if TYPE_CHECKING:
    from morgan_brain.interfaces.llm import ChatClient

M = TypeVar("M", bound=BaseModel)


class StructuredError(Exception):
    """Raised when *generate_structured* exhausts all re-ask retries without a valid parse."""


async def generate_structured(
    client: "ChatClient",
    messages: list[ChatMessage],
    *,
    model: str,
    schema: Type[M],
    descriptor: CapabilityDescriptor,
    max_reask: int = 2,
) -> M:
    """Generate a response and parse it into *schema*.

    Args:
        client:     A ``ChatClient`` (e.g. ``FakeChatClient`` or ``OpenAICompatAdapter``).
        messages:   Conversation history to pass to the model.
        model:      Model identifier string forwarded to ``client.agenerate``.
        schema:     Pydantic ``BaseModel`` subclass to validate against.
        descriptor: Capability descriptor controlling the tier.
        max_reask:  Maximum additional attempts after the first parse failure (so
                    total attempts = ``1 + max_reask``).

    Returns:
        A validated instance of *schema*.

    Raises:
        StructuredError: When all attempts (1 + max_reask) fail to produce valid JSON.
    """
    # Build the initial message list (copy so we don't mutate the caller's list).
    working_messages: list[ChatMessage] = list(messages)
    response_format: dict[str, Any] | None = None

    json_mode = descriptor.json_mode

    if json_mode == JsonMode.JSON_SCHEMA:
        # Tier 1: native schema-constrained generation.
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        }

    elif json_mode == JsonMode.JSON_OBJECT:
        # Tier 2: native JSON object mode + schema injected into the prompt.
        response_format = {"type": "json_object"}
        schema_instruction = (
            f"Respond ONLY with a JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        working_messages = [
            ChatMessage(role="system", content=schema_instruction),
            *working_messages,
        ]

    else:
        # Tier 3: prompt-only; append an instruction asking for JSON.
        schema_instruction = (
            f"Respond ONLY with a valid JSON object matching this schema "
            f"(no markdown, no explanation):\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        working_messages = [
            *working_messages,
            ChatMessage(role="user", content=schema_instruction),
        ]

    last_error: str = ""
    total_attempts = 1 + max_reask

    for attempt in range(total_attempts):
        if attempt > 0 and last_error:
            # Append the previous error as a user message to guide correction.
            working_messages = [
                *working_messages,
                ChatMessage(
                    role="user",
                    content=(
                        f"The previous response was invalid. Error: {last_error}\n"
                        f"Please respond ONLY with valid JSON matching the schema."
                    ),
                ),
            ]

        result: ChatResult = await client.agenerate(
            working_messages,
            model=model,
            response_format=response_format,
        )

        try:
            parsed: M = schema.model_validate_json(result.text)
            return parsed
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)

    raise StructuredError(
        f"Failed to parse a valid {schema.__name__} after {total_attempts} attempt(s). "
        f"Last error: {last_error}"
    )
