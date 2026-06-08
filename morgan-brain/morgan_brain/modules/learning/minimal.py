"""Phase 1 Learner: returns a default UserModel and persists each session message as an episodic
memory (so recall works across turns). Real trait/preference extraction + UserModel maintenance
arrive in Phase 2, running in the learning-worker."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from morgan_brain.interfaces.memory import MemoryStore
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource
from morgan_brain.models.message import Conversation, Role
from morgan_brain.models.user import UserModel


class MinimalLearner:
    def __init__(self, *, memory: MemoryStore, clock: Callable[[], datetime]) -> None:
        self._memory = memory
        self._clock = clock

    async def process_session(self, conversation: Conversation) -> None:
        for msg in conversation.messages:
            source = MemorySource.USER_STATED if msg.role is Role.USER else MemorySource.AGENT_INFERRED
            await self._memory.store(Memory(
                user_id=conversation.user_id,
                kind=MemoryKind.EPISODIC,
                content=msg.content,
                source=source,
                created_at=self._clock(),
            ))

    async def user_model(self, user_id: str) -> UserModel:
        return UserModel(user_id=user_id)

    async def consolidate(self, user_id: str) -> None:
        return None
