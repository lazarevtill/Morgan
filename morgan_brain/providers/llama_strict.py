"""Opt-in llama-compatible whole-template count and bounded generation capability.

No automatic discovery or generic OpenAI fallback. Callers supply the calibrated model,
template identity and schema. Changed backend/templates require renewed usage checks.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, TypeGuard
from urllib.parse import urlsplit

import httpx

from morgan_brain.providers.context import RequestCount, StrictRequest, request_fingerprint
from morgan_brain.providers.wire import ChatMessage, ChatResult, Usage


class StrictBackendUnavailable(RuntimeError):
    """The explicit counted request cannot safely be generated or accepted."""


def _integer(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


class LlamaStrictBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        template_id: str,
        response_format: dict[str, Any],
        provider: str = "llamacpp",
        api_key: str | None = None,
        max_request_bytes: int = 262144,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            provider != "llamacpp"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("An explicit llama-compatible HTTP endpoint is required")
        if (
            not model.strip()
            or not template_id.strip()
            or not _integer(max_request_bytes)
            or not 1 <= max_request_bytes <= 262144
        ):
            raise ValueError("Invalid calibrated identity or request bound")
        if (
            not isinstance(response_format, dict)
            or response_format.get("type") != "json_schema"
            or not isinstance(response_format.get("json_schema"), dict)
        ):
            raise ValueError("A calibrated JSON schema response format is required")
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._model = model
        self._template_id = template_id
        self._schema = self._encode(response_format)
        self._max_request_bytes = max_request_bytes
        self._client = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        self._lock = asyncio.Lock()
        self._verified_model = False
        self._issued: OrderedDict[str, RequestCount] = OrderedDict()
        self._issued_at: dict[str, float] = {}

    @staticmethod
    def _encode(value: Any) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    async def aclose(self) -> None:
        await self._client.aclose()

    def _snapshot(
        self, messages: list[ChatMessage], request: StrictRequest
    ) -> tuple[list[ChatMessage], StrictRequest]:
        try:
            copied_messages = [message.model_copy(deep=True) for message in messages]
            copied_request = replace(
                request, response_format=json.loads(self._encode(request.response_format))
            )
        except (TypeError, ValueError) as error:
            raise StrictBackendUnavailable("Unsupported request snapshot") from error
        self._validate(copied_messages, copied_request)
        return copied_messages, copied_request

    def _validate(self, messages: list[ChatMessage], request: StrictRequest) -> None:
        if (
            request.model != self._model
            or not _integer(request.output_tokens)
            or not 1 <= request.output_tokens <= 4096
            or not isinstance(request.temperature, (int, float))
            or isinstance(request.temperature, bool)
            or request.temperature != 0
            or not _integer(request.seed)
            or request.seed != 42
            or request.enable_thinking is not False
            or self._encode(request.response_format) != self._schema
        ):
            raise StrictBackendUnavailable(
                "Uncalibrated request model, schema or generation options"
            )
        if not messages or any(
            m.role not in {"system", "user", "assistant"} or m.tool_calls or m.tool_call_id
            for m in messages
        ):
            raise StrictBackendUnavailable("Only calibrated plain-text messages are supported")

    async def _http(
        self, route: str, payload: dict[str, Any] | None, request_seconds: int
    ) -> dict[str, Any]:
        body = self._encode(payload) if payload is not None else None
        if body is not None and len(body) > self._max_request_bytes:
            raise StrictBackendUnavailable("Serialized request exceeded configured bound")

        async def receive() -> dict[str, Any]:
            async with self._client.stream(
                "POST" if body is not None else "GET",
                self._base_url + route,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=request_seconds,
            ) as response:
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 4_000_000:
                        raise StrictBackendUnavailable("Backend response exceeded 4MB bound")
                result = json.loads(data)
                if not isinstance(result, dict):
                    raise StrictBackendUnavailable("Backend response is not an object")
                return result

        try:
            return await asyncio.wait_for(receive(), timeout=request_seconds)
        except StrictBackendUnavailable:
            raise
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            raise StrictBackendUnavailable("Counted backend request failed") from error

    async def count_request(
        self, messages: list[ChatMessage], *, request: StrictRequest
    ) -> RequestCount:
        messages, request = self._snapshot(messages, request)
        try:
            return await asyncio.wait_for(
                self._count_request(messages, request=request), timeout=10
            )
        except TimeoutError as error:
            raise StrictBackendUnavailable("Whole-request counting timed out") from error

    async def _count_request(
        self, messages: list[ChatMessage], *, request: StrictRequest
    ) -> RequestCount:
        self._validate(messages, request)
        fingerprint = request_fingerprint(messages, request)
        async with self._lock:
            if not self._verified_model:
                listing = await self._http("/v1/models", None, 10)
                models = listing.get("data")
                if not isinstance(models, list) or not any(
                    isinstance(item, dict) and item.get("id") == self._model for item in models
                ):
                    raise StrictBackendUnavailable("Configured model identity is unavailable")
                self._verified_model = True
            template = await self._http(
                "/apply-template",
                {
                    "model": request.model,
                    "messages": [m.to_openai() for m in messages],
                    "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
                    "response_format": request.response_format,
                    "add_generation_prompt": True,
                },
                10,
            )
            prompt = template.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise StrictBackendUnavailable("Template response has no nonempty prompt")
            tokenized = await self._http(
                "/tokenize",
                {
                    "model": request.model,
                    "content": prompt,
                    "parse_special": True,
                    "add_special": False,
                },
                10,
            )
            ids = tokenized.get("tokens")
            if not isinstance(ids, list) or not ids or any(not _integer(i) or i < 0 for i in ids):
                raise StrictBackendUnavailable("Tokenizer response has no valid tokens")
            count = RequestCount(len(ids), True, request.model, self._template_id, fingerprint)
            self._issued[fingerprint] = count
            self._issued_at[fingerprint] = time.monotonic()
            self._issued.move_to_end(fingerprint)
            while len(self._issued) > 16:
                evicted, _ = self._issued.popitem(last=False)
                self._issued_at.pop(evicted, None)
            return count

    async def generate_counted(
        self, messages: list[ChatMessage], *, request: StrictRequest, count: RequestCount
    ) -> ChatResult:
        messages, request = self._snapshot(messages, request)
        try:
            return await asyncio.wait_for(
                self._generate_counted(messages, request=request, count=count), timeout=60
            )
        except TimeoutError as error:
            raise StrictBackendUnavailable("Bounded generation timed out") from error

    async def _generate_counted(
        self, messages: list[ChatMessage], *, request: StrictRequest, count: RequestCount
    ) -> ChatResult:
        self._validate(messages, request)
        fingerprint = request_fingerprint(messages, request)
        async with self._lock:
            if (
                not isinstance(count, RequestCount)
                or count.exact is not True
                or not _integer(count.input_tokens)
                or count.input_tokens <= 0
                or count.request_fingerprint != fingerprint
                or count != self._issued.get(fingerprint)
                or time.monotonic() - self._issued_at.get(fingerprint, float("-inf")) > 300
            ):
                raise StrictBackendUnavailable(
                    "Generation requires this backend's matching issued count"
                )
            response = await self._http(
                "/v1/chat/completions",
                {
                    "model": request.model,
                    "messages": [m.to_openai() for m in messages],
                    "max_tokens": request.output_tokens,
                    "temperature": request.temperature,
                    "seed": request.seed,
                    "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
                    "response_format": request.response_format,
                },
                60,
            )
            usage = response.get("usage")
            choices = response.get("choices")
            if (
                response.get("model") != self._model
                or not isinstance(usage, dict)
                or not _integer(usage.get("prompt_tokens"))
                or usage["prompt_tokens"] != count.input_tokens
            ):
                self._issued.clear()
                self._issued_at.clear()
                self._verified_model = False
                raise StrictBackendUnavailable(
                    "Generation identity/token usage differs from counted request"
                )
            output = usage.get("completion_tokens")
            if (
                not _integer(output)
                or not 0 <= output <= request.output_tokens
                or not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise StrictBackendUnavailable("Invalid bounded generation usage/choice")
            choice = choices[0]
            message = choice.get("message")
            if (
                choice.get("finish_reason") != "stop"
                or not isinstance(message, dict)
                or message.get("role") != "assistant"
                or not isinstance(message.get("content"), str)
                or message.get("tool_calls")
            ):
                raise StrictBackendUnavailable(
                    "Generation was truncated or used unsupported tools/content"
                )
            return ChatResult(
                text=message["content"],
                model=self._model,
                usage=Usage(input_tokens=count.input_tokens, output_tokens=output),
                finish_reason="stop",
            )
