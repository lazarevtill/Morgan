"""Tests for the structured-output ladder (generate_structured)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from morgan_brain.providers.structured import StructuredError, generate_structured
from morgan_brain.providers.capability import CapabilityDescriptor, JsonMode
from morgan_brain.providers.adapters.fake import FakeChatClient


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_prompted_json_tier_parses_valid_json() -> None:
    """NONE tier: instruction appended, valid JSON parsed on first call."""
    c = FakeChatClient(reply='{"name": "Sam", "age": 40}')
    d = CapabilityDescriptor(provider="p", model="m", json_mode=JsonMode.NONE)
    out = await generate_structured(c, [], model="m", schema=Person, descriptor=d)
    assert out.name == "Sam" and out.age == 40


@pytest.mark.asyncio
async def test_invalid_then_valid_triggers_one_reask() -> None:
    """First reply is invalid JSON; second is valid → one re-ask, calls == 2."""
    c = FakeChatClient(replies=["not json", '{"name":"Sam","age":40}'])
    d = CapabilityDescriptor(provider="p", model="m", json_mode=JsonMode.NONE)
    out = await generate_structured(c, [], model="m", schema=Person, descriptor=d, max_reask=2)
    assert out.age == 40
    assert c.calls == 2


@pytest.mark.asyncio
async def test_exhausted_reask_raises() -> None:
    """Reply is never valid JSON → StructuredError raised after max_reask attempts."""
    c = FakeChatClient(reply="never json")
    d = CapabilityDescriptor(provider="p", model="m", json_mode=JsonMode.NONE)
    with pytest.raises(StructuredError):
        await generate_structured(c, [], model="m", schema=Person, descriptor=d, max_reask=1)


@pytest.mark.asyncio
async def test_json_object_tier_passes_response_format() -> None:
    """JSON_OBJECT tier: response_format is passed to client, valid JSON parsed."""
    c = FakeChatClient(reply='{"name": "Jo", "age": 25}')
    d = CapabilityDescriptor(provider="p", model="m", json_mode=JsonMode.JSON_OBJECT)
    out = await generate_structured(c, [], model="m", schema=Person, descriptor=d)
    assert out.name == "Jo"
    # Verify the client received a response_format dict
    assert c.last_response_format is not None
    assert c.last_response_format.get("type") == "json_object"


@pytest.mark.asyncio
async def test_json_schema_tier_passes_json_schema_format() -> None:
    """JSON_SCHEMA tier: response_format includes json_schema key."""
    c = FakeChatClient(reply='{"name": "Alex", "age": 30}')
    d = CapabilityDescriptor(provider="p", model="m", json_mode=JsonMode.JSON_SCHEMA)
    out = await generate_structured(c, [], model="m", schema=Person, descriptor=d)
    assert out.name == "Alex"
    assert c.last_response_format is not None
    assert c.last_response_format.get("type") == "json_schema"
    js = c.last_response_format.get("json_schema", {})
    assert js.get("name") == "Person"
