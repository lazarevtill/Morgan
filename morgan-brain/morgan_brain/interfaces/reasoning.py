"""Reasoning contract — the thin LLM orchestrator. Assembles the context window from all
upstream inputs, routes to a model, optionally plans/reflects and calls tools, then generates.
Deliberately not a god class: it coordinates, it does not own memory/learning/personalization.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.models.memory import Memory
from morgan_brain.models.message import Message
from morgan_brain.models.perception import FusedPerception
from morgan_brain.providers.wire import ToolSpec


class ReasoningRequest(BaseModel):
    user_id: str
    perception: FusedPerception
    personalization: PersonalizedContext
    memories: list[Memory] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    skill_prompt: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    system_override: str = ""


class ReasoningResult(BaseModel):
    text: str
    model_used: str = ""
    tools_invoked: list[str] = Field(default_factory=list)


@runtime_checkable
class Reasoner(Protocol):
    async def generate(self, request: ReasoningRequest) -> ReasoningResult: ...

    def stream(self, request: ReasoningRequest) -> AsyncIterator[str]: ...
