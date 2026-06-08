"""Event bus contract. The in-process backend (bus/inproc.py) and the Redis Streams backend
(bus/redis_streams.py) both satisfy this Protocol, so cross-module communication is identical
whether modules share a process or run as separate services.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Perception
    MESSAGE_RECEIVED = "message.received"
    PERCEPTION_COMPLETE = "perception.complete"
    # Memory
    MEMORY_STORED = "memory.stored"
    FACT_UPDATED = "fact.updated"
    # Learning
    TRAIT_EXTRACTED = "trait.extracted"
    USER_MODEL_UPDATED = "user_model.updated"
    SKILL_OPTIMIZED = "skill.optimized"
    # Reasoning
    RESPONSE_GENERATED = "response.generated"
    TOOL_INVOKED = "tool.invoked"
    # Lifecycle
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    HEARTBEAT = "heartbeat"
    # Provider
    LLM_FALLBACK = "llm.fallback"


class Event(BaseModel):
    type: EventType
    user_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    # Additive-optional rule: new fields MUST carry a default so that consumers on an older
    # schema_version can still deserialize without breaking.  Increment when fields are added.
    schema_version: int = 1


Handler = Callable[[Event], Awaitable[None]]


@runtime_checkable
class EventBus(Protocol):
    def subscribe(self, event_type: EventType, handler: Handler) -> None: ...

    async def publish(self, event: Event) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...
