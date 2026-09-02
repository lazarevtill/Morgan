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
