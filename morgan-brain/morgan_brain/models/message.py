"""Conversation primitives."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from morgan_brain.models.base import UserScoped


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(UserScoped):
    role: Role
    content: str
    session_id: str | None = None


class Conversation(UserScoped):
    session_id: str
    messages: list[Message] = Field(default_factory=list)
