"""OpenAI-compatible adapter — wraps the official ``openai`` SDK with a configurable base_url.

This covers Ollama /v1, vLLM, llama.cpp, LM Studio, OpenRouter, and remote OpenAI.
SDK internal retries are disabled (max_retries=0) — fallback is handled by RoleFallback.

Implements the ``ChatClient`` protocol. Embeddings live in ``embeddings.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from morgan_brain.interfaces.llm import ProviderUnreachable
from morgan_brain.providers.wire import (
    ChatMessage,
    ChatResult,
    StreamDelta,
    ToolCall,
    ToolSpec,
    Usage,
)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert wire ChatMessages to openai SDK message dicts."""
    return [m.to_openai() for m in messages]


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert wire ToolSpecs to openai SDK tool dicts."""
    return [t.to_openai() for t in tools]


def _from_openai_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    """Convert openai SDK tool_calls on a message to wire ToolCall objects."""
    if not raw_tool_calls:
        return []
    result = []
    for tc in raw_tool_calls:
        fn = tc.function
        try:
            arguments: dict[str, Any] = json.loads(fn.arguments) if fn.arguments else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        result.append(ToolCall(id=tc.id or "", name=fn.name or "", arguments=arguments))
    return result


class OpenAICompatAdapter:
    """ChatClient + Embedder over the OpenAI-compatible REST API.

    Args:
        base_url: Base URL of the OpenAI-compatible endpoint (e.g. ``http://localhost:8081/v1``).
        api_key:  API key; use any non-empty string for servers that don't enforce one.
        provider: Provider name for capability registry lookups (e.g. ``"llamacpp"``).
        timeout:  Request timeout in seconds. Default is sized for a remote server reached
                  over a network (a homelab GPU box under load), not a loopback socket.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider: str,
        timeout: float = 120.0,
    ) -> None:
        # Import here so that the openai SDK is only required if this adapter is used.
        import openai

        self._provider = provider
        self._base_url = base_url
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout,
        )
        # Connection refused, DNS failure, and a request that timed out all subclass this
        # one SDK error. Held on the instance so the SDK stays imported in exactly one place.
        self._connection_error: type[Exception] = openai.APIConnectionError

    # ------------------------------------------------------------------
    # ChatClient protocol
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Generate a chat completion and return a ``ChatResult``."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages),
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except self._connection_error as exc:
            raise ProviderUnreachable(self._base_url, str(exc)) from exc
        choice = response.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls = _from_openai_tool_calls(getattr(msg, "tool_calls", None))

        usage = Usage()
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        finish_reason = choice.finish_reason or "stop"

        return ChatResult(
            text=text,
            model=model,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    def astream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream chat completion chunks as ``StreamDelta`` objects."""
        return self._astream_impl(messages, model=model, tools=tools)

    async def _astream_impl(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except self._connection_error as exc:
            raise ProviderUnreachable(self._base_url, str(exc)) from exc
        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    # usage chunk (some providers send a final chunk with usage only)
                    if hasattr(chunk, "usage") and chunk.usage:
                        yield StreamDelta(
                            kind="usage",
                            usage=Usage(
                                input_tokens=chunk.usage.prompt_tokens or 0,
                                output_tokens=chunk.usage.completion_tokens or 0,
                            ),
                        )
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Text delta
                if delta.content:
                    yield StreamDelta(kind="text_delta", text=delta.content)

                # Tool call delta
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        fn = tc_delta.function
                        try:
                            args: dict[str, Any] = json.loads(fn.arguments) if fn.arguments else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        yield StreamDelta(
                            kind="tool_call_delta",
                            tool_call=ToolCall(
                                id=tc_delta.id or "",
                                name=fn.name or "",
                                arguments=args,
                            ),
                        )

                # Finish
                if choice.finish_reason:
                    yield StreamDelta(
                        kind="finish",
                        finish_reason=choice.finish_reason,
                    )
