"""Learning contract — asynchronous intelligence extraction. Runs in the learning-worker,
off the request path. Reads finished sessions, writes the UserModel + facts via the gate.
Never invoked synchronously during a turn.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from morgan_brain.models.message import Conversation
from morgan_brain.models.user import UserModel


@runtime_checkable
class Learner(Protocol):
    async def process_session(self, conversation: Conversation) -> None:
        """Extract facts/preferences/behaviors from a completed session and persist them."""
        ...

    async def user_model(self, user_id: str) -> UserModel:
        """Return the current stable user model."""
        ...

    async def consolidate(self, user_id: str) -> None:
        """Dedup, decay confidence, curate MEMORY.md, mine behavioral patterns."""
        ...
