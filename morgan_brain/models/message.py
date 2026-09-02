"""Conversation primitives."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from morgan_brain.models.base import UserScoped
from morgan_brain.models.memory import DEFAULT_PROJECT


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(UserScoped):
    project: str = Field(default=DEFAULT_PROJECT, min_length=1)
    role: Role
    content: str
    session_id: str | None = None


class Conversation(UserScoped):
    project: str = Field(default=DEFAULT_PROJECT, min_length=1)
    session_id: str
    messages: list[Message] = Field(default_factory=list)
