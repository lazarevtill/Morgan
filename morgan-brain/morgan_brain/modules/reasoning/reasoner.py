"""ReasoningModule — interfaces.Reasoner. Phase 1: build context, route to a model (only the
strong model is used until planning lands), call the LLM, return the reply. Streaming and
tool-calls arrive in later phases."""
from __future__ import annotations

from typing import AsyncIterator

from morgan_brain.interfaces.reasoning import ReasoningRequest, ReasoningResult
from morgan_brain.modules.reasoning.context.builder import build_messages
from morgan_brain.modules.reasoning.llm.client import LLMClient


class ReasoningModule:
    def __init__(self, *, llm: LLMClient, model: str, fast_model: str) -> None:
        self._llm = llm
        self._model = model
        self._fast_model = fast_model

    async def generate(self, request: ReasoningRequest) -> ReasoningResult:
        messages = build_messages(request)
        text = await self._llm.complete(messages, model=self._model)
        return ReasoningResult(text=text, model_used=self._model, tools_invoked=[])

    async def stream(self, request: ReasoningRequest) -> AsyncIterator[str]:
        result = await self.generate(request)
        yield result.text
