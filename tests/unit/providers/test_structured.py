"""Structured output: request JSON the configured way, validate, re-ask on failure."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from morgan_brain.providers.structured import StructuredError, generate_structured
from tests.fakes import FakeChatClient


class Person(BaseModel):
    name: str
    age: int


async def test_prompted_mode_parses_valid_json() -> None:
    c = FakeChatClient(reply='{"name": "Sam", "age": 40}')
    out = await generate_structured(c, [], model="m", schema=Person, json_mode="prompted")
    assert out.name == "Sam" and out.age == 40
    assert c.last_response_format is None
    assert "schema" in c.last_messages[-1].content


async def test_reask_on_invalid_then_valid() -> None:
    c = FakeChatClient(replies=["not json", '{"name":"Sam","age":40}'])
    out = await generate_structured(c, [], model="m", schema=Person, json_mode="prompted")
    assert out.name == "Sam"
    assert c.calls == 2
    assert "invalid" in c.last_messages[-1].content.lower()


async def test_exhausted_reasks_raise() -> None:
    c = FakeChatClient(reply="never json")
    with pytest.raises(StructuredError):
        await generate_structured(c, [], model="m", schema=Person, max_reask=1)
    assert c.calls == 2


async def test_json_schema_mode_sends_the_schema_natively() -> None:
    c = FakeChatClient(reply='{"name": "Jo", "age": 25}')
    out = await generate_structured(c, [], model="m", schema=Person, json_mode="json_schema")
    assert out.name == "Jo"
    assert c.last_response_format is not None
    assert c.last_response_format["type"] == "json_schema"
    assert c.last_response_format["json_schema"]["name"] == "Person"


async def test_json_object_mode_sends_object_mode_and_the_schema_in_the_prompt() -> None:
    c = FakeChatClient(reply='{"name": "Alex", "age": 30}')
    out = await generate_structured(c, [], model="m", schema=Person, json_mode="json_object")
    assert out.name == "Alex"
    assert c.last_response_format == {"type": "json_object"}
    assert c.last_messages[0].role == "system"


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompted"])
@pytest.mark.parametrize("reason", ["length", "tool_calls"])
async def test_incomplete_valid_json_is_never_accepted(mode, reason):
    from morgan_brain.providers.wire import ChatResult

    class Client:
        calls = 0

        async def agenerate(self, *args, **kwargs):
            self.calls += 1
            return ChatResult(text='{"name":"Sam","age":40}', finish_reason=reason)

    client = Client()
    with pytest.raises(StructuredError, match="incomplete"):
        await generate_structured(client, [], model="m", schema=Person, json_mode=mode)
    assert client.calls == 1


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompted"])
async def test_actual_mode_request_byte_boundary(mode):
    from morgan_brain.providers.structured import structured_input_size, structured_request
    from morgan_brain.providers.wire import ChatMessage

    messages = [ChatMessage(role="user", content="")]
    prepared, response_format = structured_request(messages, schema=Person, json_mode=mode)
    overhead = structured_input_size(prepared, response_format, model="m")
    messages = [ChatMessage(role="user", content="x" * (49152 - overhead))]
    client = FakeChatClient(reply='{"name":"Sam","age":40}')
    await generate_structured(
        client, messages, model="m", schema=Person, json_mode=mode, max_input_bytes=49152
    )
    assert (
        structured_input_size(client.last_messages, client.last_response_format, model="m") == 49152
    )
    messages[0] = ChatMessage(role="user", content=messages[0].content + "x")
    refused = FakeChatClient(reply='{"name":"Sam","age":40}')
    with pytest.raises(ValueError, match="byte limit"):
        await generate_structured(
            refused, messages, model="m", schema=Person, json_mode=mode, max_input_bytes=49152
        )
    assert refused.calls == 0
