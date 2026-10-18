"""Real SDK wire tests via offline MockTransport; no provider calls."""

import json
from dataclasses import FrozenInstanceError

import httpx
import openai
import pytest
from pydantic import BaseModel

from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.request_budget import (
    ChatRequestOptions,
    StructuredRequestTooLarge,
    serialized_request_bytes,
)
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage, ToolSpec


class Answer(BaseModel):
    answer: str


async def adapter_with_wire(options, replies, *, stream=False):
    captured = []

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        assert request.headers["authorization"] == "Bearer local-test"
        if stream:
            chunk = {
                "id": "offline",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "trusted-model",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
            data = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)
        text = replies[len(captured) - 1]
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    adapter = OpenAICompatAdapter(
        "http://offline/v1", "local-test", "llamacpp", setting="test", request_options=options
    )
    await adapter._client.close()
    adapter._client = openai.AsyncOpenAI(
        base_url="http://offline/v1",
        api_key="local-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return adapter, captured


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "512"])
def test_reject_invalid_output_limit(value):
    with pytest.raises(ValueError, match="positive integer"):
        ChatRequestOptions(max_output_tokens=value)


@pytest.mark.parametrize("value", [0, 1, "false", {}, []])
def test_reject_non_boolean_thinking(value):
    with pytest.raises(ValueError, match="boolean"):
        ChatRequestOptions(enable_thinking=value)


def test_frozen_closed_options():
    options = ChatRequestOptions(max_output_tokens=512, enable_thinking=False)
    with pytest.raises(FrozenInstanceError):
        options.max_output_tokens = 1
    with pytest.raises(TypeError):
        ChatRequestOptions(**json.loads('{"model":"overwrite"}'))
    with pytest.raises(TypeError, match="ChatRequestOptions"):
        OpenAICompatAdapter(
            "http://offline/v1", "k", "p", setting="test", request_options={"model": "overwrite"}
        )


@pytest.mark.parametrize("options", [None, ChatRequestOptions()])
async def test_default_payload_unchanged(options):
    adapter, wire = await adapter_with_wire(options, ["ok"])
    try:
        await adapter.agenerate([ChatMessage(role="user", content="source")], model="trusted-model")
        assert wire == [
            {"model": "trusted-model", "messages": [{"role": "user", "content": "source"}]}
        ]
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("thinking", [False, True, None])
async def test_typed_options_reach_wire_without_overwriting_tools_or_schema(thinking):
    adapter, wire = await adapter_with_wire(
        ChatRequestOptions(max_output_tokens=512, enable_thinking=thinking), ["ok"]
    )
    try:
        await adapter.agenerate(
            [ChatMessage(role="user", content="источник")],
            model="trusted-model",
            tools=[ToolSpec(name="read", parameters={"type": "object"})],
            response_format={"type": "json_object"},
        )
        assert wire[0]["model"] == "trusted-model"
        assert wire[0]["messages"][0]["content"] == "источник"
        assert wire[0]["tools"][0]["function"]["name"] == "read"
        assert wire[0]["response_format"] == {"type": "json_object"}
        assert wire[0]["max_tokens"] == 512
        assert "extra_body" not in wire[0]
        if thinking is None:
            assert "chat_template_kwargs" not in wire[0]
        else:
            assert wire[0]["chat_template_kwargs"] == {"enable_thinking": thinking}
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("temperature", [None, 0.0])
@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompted"])
@pytest.mark.parametrize("text", ['source "quoted"\n', "источник «данные»\n"])
async def test_complete_wire_exact_byte_ceiling_with_controls(mode, text, temperature):
    options = ChatRequestOptions(
        max_output_tokens=512, enable_thinking=False, temperature=temperature
    )
    messages = [ChatMessage(role="user", content=text)]
    probe, observed = await adapter_with_wire(options, ['{"answer":"ok"}'])
    try:
        await generate_structured(
            probe, messages, model="trusted-model", schema=Answer, json_mode=mode
        )
        # Independent measurement from the JSON received by the actual SDK transport.
        size = len(json.dumps(observed[0], ensure_ascii=False, separators=(",", ":")).encode())
        assert size == serialized_request_bytes(observed[0])
        assert size > len(text.encode())
    finally:
        await probe._client.close()
    adapter, captured = await adapter_with_wire(options, ['{"answer":"ok"}'])
    try:
        with pytest.raises(StructuredRequestTooLarge) as refused:
            await generate_structured(
                adapter,
                messages,
                model="trusted-model",
                schema=Answer,
                json_mode=mode,
                request_byte_limit=size - 1,
            )
        assert refused.value.measured_bytes == size and refused.value.attempt == 1
        assert captured == []
        result = await generate_structured(
            adapter,
            messages,
            model="trusted-model",
            schema=Answer,
            json_mode=mode,
            request_byte_limit=size,
        )
        assert result.answer == "ok" and captured == observed
    finally:
        await adapter._client.close()


async def test_reask_controls_consistent_and_complete_second_request_refused():
    options = ChatRequestOptions(max_output_tokens=512, enable_thinking=False, temperature=0.0)
    messages = [ChatMessage(role="user", content="source")]
    probe, wire = await adapter_with_wire(options, ["invalid", '{"answer":"ok"}'])
    try:
        result = await generate_structured(probe, messages, model="trusted-model", schema=Answer)
        assert result.answer == "ok" and len(wire) == 2
        for body in wire:
            assert body["max_tokens"] == 512
            assert body["temperature"] == 0.0
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
        first_size, second_size = [serialized_request_bytes(body) for body in wire]
        assert second_size > first_size
    finally:
        await probe._client.close()
    adapter, captured = await adapter_with_wire(options, ["invalid", '{"answer":"ok"}'])
    try:
        with pytest.raises(StructuredRequestTooLarge) as refused:
            await generate_structured(
                adapter,
                messages,
                model="trusted-model",
                schema=Answer,
                request_byte_limit=first_size,
            )
        assert refused.value.attempt == 2 and refused.value.measured_bytes == second_size
        assert len(captured) == 1
    finally:
        await adapter._client.close()


async def test_stream_uses_same_controls_and_tools():
    adapter, wire = await adapter_with_wire(
        ChatRequestOptions(max_output_tokens=128, enable_thinking=False, temperature=0.0),
        [],
        stream=True,
    )
    try:
        deltas = [
            delta
            async for delta in adapter.astream(
                [ChatMessage(role="user", content="source")],
                model="trusted-model",
                tools=[ToolSpec(name="read")],
            )
        ]
        assert any(delta.text == "ok" for delta in deltas)
        assert wire[0]["stream"] is True and wire[0]["max_tokens"] == 128
        assert wire[0]["temperature"] == 0.0
        assert wire[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert wire[0]["tools"][0]["function"]["name"] == "read"
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("value", [True, False, -0.1, 2.1, float("nan"), float("inf"), "0", {}])
def test_reject_invalid_temperature(value):
    with pytest.raises(ValueError, match="finite number"):
        ChatRequestOptions(temperature=value)


@pytest.mark.parametrize("value", [0, 0.0, 0.6, 2.0])
async def test_explicit_temperature_wire(value):
    adapter, wire = await adapter_with_wire(ChatRequestOptions(temperature=value), ["ok"])
    try:
        await adapter.agenerate([ChatMessage(role="user", content="source")], model="trusted-model")
        assert wire[0]["temperature"] == value
        assert wire[0]["model"] == "trusted-model"
        assert "max_tokens" not in wire[0] and "chat_template_kwargs" not in wire[0]
    finally:
        await adapter._client.close()
