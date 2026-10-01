import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.context import StrictRequest, request_fingerprint
from morgan_brain.providers.factory import build_strict_chat_backend
from morgan_brain.providers.llama_strict import LlamaStrictBackend, StrictBackendUnavailable
from morgan_brain.providers.wire import ChatMessage

FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "test", "strict": True, "schema": {"type": "object"}},
}
MESSAGES = [ChatMessage(role="user", content="Synthetic source")]
REQUEST = StrictRequest("ornith15", 32, FORMAT)


def backend(handler, **kwargs):
    return LlamaStrictBackend(
        base_url="http://fixture/v1",
        model="ornith15",
        template_id="calibration-v1",
        response_format=FORMAT,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def handler_factory(changes=None):
    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.url.path, body))
        data = {
            "/v1/models": {"data": [{"id": "ornith15"}]},
            "/apply-template": {"prompt": "whole rendered template"},
            "/tokenize": {"tokens": [1, 2, 3]},
            "/v1/chat/completions": {
                "model": "ornith15",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}
                ],
            },
        }[request.url.path]
        if changes and request.url.path in changes:
            data = changes[request.url.path]
        return httpx.Response(200, json=data)

    return handler, calls


@pytest.mark.asyncio
async def test_count_generation_shape_and_option_binding():
    handler, calls = handler_factory()
    adapter = backend(handler)
    try:
        count = await adapter.count_request(MESSAGES, request=REQUEST)
        result = await adapter.generate_counted(MESSAGES, request=REQUEST, count=count)
        assert result.usage.input_tokens == count.input_tokens == 3
        template = calls[1][1]
        generation = calls[3][1]
        assert (
            template["chat_template_kwargs"]
            == generation["chat_template_kwargs"]
            == {"enable_thinking": False}
        )
        assert template["response_format"] == generation["response_format"] == FORMAT
        assert (
            generation["max_tokens"] == 32
            and generation["temperature"] == 0
            and generation["seed"] == 42
        )
        assert calls[2][1]["parse_special"] is True and calls[2][1]["add_special"] is False
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_request",
    [
        replace(REQUEST, model="unknown"),
        replace(REQUEST, seed=43),
        replace(REQUEST, enable_thinking=True),
        replace(REQUEST, temperature=0.5),
        replace(REQUEST, response_format={"type": "json_object"}),
    ],
)
async def test_uncalibrated_request_fails_without_network(case_request):
    handler, calls = handler_factory()
    adapter = backend(handler)
    try:
        with pytest.raises(StrictBackendUnavailable):
            await adapter.count_request(MESSAGES, request=case_request)
        assert not calls
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_forged_or_changed_count_cannot_trigger_generation():
    handler, calls = handler_factory()
    adapter = backend(handler)
    try:
        count = await adapter.count_request(MESSAGES, request=REQUEST)
        with pytest.raises(StrictBackendUnavailable):
            await adapter.generate_counted(
                MESSAGES, request=REQUEST, count=replace(count, input_tokens=1)
            )
        with pytest.raises(StrictBackendUnavailable):
            await adapter.generate_counted(
                [*MESSAGES, ChatMessage(role="user", content="changed")],
                request=REQUEST,
                count=count,
            )
        assert len(calls) == 3
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"model": "other", "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        {"model": "ornith15", "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        {"model": "ornith15", "usage": {"prompt_tokens": True, "completion_tokens": 2}},
        {"model": "ornith15", "usage": {"prompt_tokens": 3, "completion_tokens": 33}},
        {"model": "ornith15", "usage": {"prompt_tokens": 3, "completion_tokens": -1}},
        {
            "model": "ornith15",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "choices": [
                {"finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}
            ],
        },
    ],
)
async def test_generation_identity_usage_and_truncation_refused(response):
    handler, _ = handler_factory({"/v1/chat/completions": response})
    adapter = backend(handler)
    try:
        count = await adapter.count_request(MESSAGES, request=REQUEST)
        with pytest.raises(StrictBackendUnavailable):
            await adapter.generate_counted(MESSAGES, request=REQUEST, count=count)
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,response",
    [
        ("/v1/models", {"data": []}),
        ("/apply-template", []),
        ("/apply-template", {"prompt": ""}),
        ("/tokenize", {"tokens": []}),
        ("/tokenize", {"tokens": [True]}),
    ],
)
async def test_malformed_count_capability_refused(route, response):
    handler, _ = handler_factory({route: response})
    adapter = backend(handler)
    try:
        with pytest.raises(StrictBackendUnavailable):
            await adapter.count_request(MESSAGES, request=REQUEST)
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_serialized_request_bound_and_tool_messages_refused():
    handler, calls = handler_factory()
    adapter = backend(handler, max_request_bytes=64)
    try:
        with pytest.raises(StrictBackendUnavailable):
            await adapter.count_request(MESSAGES, request=REQUEST)
        assert all(route == "/v1/models" for route, _ in calls)
        with pytest.raises(StrictBackendUnavailable):
            await adapter.count_request([ChatMessage(role="tool", content="x")], request=REQUEST)
    finally:
        await adapter.aclose()


def test_sampling_options_are_part_of_full_request_identity():
    original = request_fingerprint(MESSAGES, REQUEST)
    assert request_fingerprint(MESSAGES, replace(REQUEST, seed=43)) != original
    assert request_fingerprint(MESSAGES, replace(REQUEST, enable_thinking=True)) != original
    assert request_fingerprint(MESSAGES, replace(REQUEST, temperature=0.5)) != original


def test_unknown_generic_backend_cannot_advertise_exact_capability():
    with pytest.raises(ValueError):
        LlamaStrictBackend(
            base_url="https://fixture/v1",
            model="ornith15",
            template_id="v1",
            response_format=FORMAT,
            provider="openai",
        )


def test_factory_disabled_is_lazy_and_unknown_model_refuses():
    settings = Settings(llm_model="unknown")
    assert build_strict_chat_backend(settings) is None
    object.__setattr__(settings, "strict_context_backend", "llamacpp")
    with pytest.raises(ValueError):
        build_strict_chat_backend(settings, response_format=FORMAT)


@pytest.mark.asyncio
async def test_factory_enabled_constructs_without_network():
    settings = Settings(llm_model="ornith15", llm_endpoint="http://fixture/v1")
    object.__setattr__(settings, "strict_context_backend", "llamacpp")
    adapter = build_strict_chat_backend(settings, response_format=FORMAT)
    assert isinstance(adapter, LlamaStrictBackend)
    await adapter.aclose()


@pytest.mark.asyncio
async def test_count_snapshots_before_delayed_model_lookup():
    arrived, release = asyncio.Event(), asyncio.Event()
    normal, calls = handler_factory()

    async def delayed(req):
        if req.url.path == "/v1/models":
            arrived.set()
            await release.wait()
        return normal(req)

    adapter = backend(delayed)
    messages = [ChatMessage(role="user", content="original")]
    original_format = json.loads(json.dumps(FORMAT))
    strict_request = replace(REQUEST, response_format=original_format)
    original_fingerprint = request_fingerprint(messages, strict_request)
    try:
        task = asyncio.create_task(adapter.count_request(messages, request=strict_request))
        await arrived.wait()
        messages[0].content = "mutated"
        original_format["json_schema"]["name"] = "mutated"
        release.set()
        count = await task
        assert count.request_fingerprint == original_fingerprint
        assert calls[1][1]["messages"][0]["content"] == "original"
        assert calls[1][1]["response_format"]["json_schema"]["name"] == "test"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_generation_snapshots_before_lock_wait():
    handler, calls = handler_factory()
    adapter = backend(handler)
    messages = [ChatMessage(role="user", content="original")]
    original_format = json.loads(json.dumps(FORMAT))
    strict_request = replace(REQUEST, response_format=original_format)
    try:
        count = await adapter.count_request(messages, request=strict_request)
        await adapter._lock.acquire()
        task = asyncio.create_task(
            adapter.generate_counted(messages, request=strict_request, count=count)
        )
        await asyncio.sleep(0)
        messages[0].content = "mutated"
        original_format["json_schema"]["name"] = "mutated"
        adapter._lock.release()
        await task
        assert calls[-1][1]["messages"][0]["content"] == "original"
        assert calls[-1][1]["response_format"]["json_schema"]["name"] == "test"
    finally:
        if adapter._lock.locked():
            adapter._lock.release()
        await adapter.aclose()
