"""ReasoningModule — interfaces.Reasoner.

Builds the context window from a ``ReasoningRequest``, routes to a model via
``RoleRouter``, calls the model, and returns a ``ReasoningResult``.

The module depends on the provider seam (``RoleRouter``) rather than a concrete
LLM client, so adapters can be swapped (Ollama, OpenAI, vLLM…) without touching
this code.
"""
from __future__ import annotations

from typing import AsyncIterator

from morgan_brain.interfaces.reasoning import ReasoningRequest, ReasoningResult
from morgan_brain.modules.reasoning.context.builder import build_messages
from morgan_brain.providers.router import RoleRouter


class ReasoningModule:
    def __init__(self, *, router: RoleRouter, role: str = "strong") -> None:
        self._router = router
        self._role = role

    async def generate(self, request: ReasoningRequest) -> ReasoningResult:
        needs_tools = bool(getattr(request, "tools", None))
        client, model = self._router.chat_for(self._role, needs_tools=needs_tools)
        messages = build_messages(request)
        result = await client.agenerate(messages, model=model)
        return ReasoningResult(text=result.text, model_used=model, tools_invoked=[])

    async def stream(self, request: ReasoningRequest) -> AsyncIterator[str]:
        needs_tools = bool(getattr(request, "tools", None))
        client, model = self._router.chat_for(self._role, needs_tools=needs_tools)
        messages = build_messages(request)
        async for delta in client.astream(messages, model=model):
            if delta.kind == "text_delta" and delta.text:
                yield delta.text
