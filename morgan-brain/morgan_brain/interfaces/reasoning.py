"""Reasoning contract — the thin LLM orchestrator. Assembles the context window from all
upstream inputs, routes to a model, optionally plans/reflects and calls tools, then generates.
Deliberately not a god class: it coordinates, it does not own memory/learning/personalization.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.models.memory import Memory
from morgan_brain.models.message import Message
from morgan_brain.models.perception import FusedPerception
from morgan_brain.providers.wire import ToolSpec


class ReasoningRequest(BaseModel):
    user_id: str
    #: The turn's project. Required, with no default: tools run inside a turn and must search
    #: the same project the turn is scoped to. ``memory_search`` previously defaulted to
    #: DEFAULT_PROJECT, so asking a question in one repo searched another -- a default here
    #: would let a future call site reintroduce that silently.
    project: str = Field(min_length=1)
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
