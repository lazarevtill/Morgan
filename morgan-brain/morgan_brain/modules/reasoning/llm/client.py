"""LLMClient: chat completion. OllamaLLMClient hits the OpenAI-compatible /v1/chat/completions
endpoint; FakeLLMClient returns a scripted reply and records inputs for assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[ChatMessage], *, model: str) -> str: ...


class FakeLLMClient:
    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.last_messages: list[ChatMessage] | None = None
        self.last_model: str | None = None

    async def complete(self, messages: list[ChatMessage], *, model: str) -> str:
        self.last_messages = messages
        self.last_model = model
        return self._reply


class OllamaLLMClient:
    def __init__(self, endpoint: str, timeout: float = 120.0) -> None:
        self._url = endpoint.rstrip("/") + "/chat/completions"
        self._timeout = timeout

    async def complete(self, messages: list[ChatMessage], *, model: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        return content
