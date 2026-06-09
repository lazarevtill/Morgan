"""Phase 2B/C — ConsolidationLearner.

Implements the ``Learner`` Protocol by combining:
- Episodic storage from ``MinimalLearner`` (Phase 1 parity for process_session).
- Real fact consolidation via ``MemoryConsolidator`` (Phase 2B).
- Real UserModel derivation via ``UserProfileBuilder`` (Phase 2C, optional).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable

from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource
from morgan_brain.models.message import Conversation, Role
from morgan_brain.models.user import UserModel
from morgan_brain.security.memory_gate import MemoryGate

if TYPE_CHECKING:
    from morgan_brain.learning.profile import UserProfileBuilder


class ConsolidationLearner:
    """A ``Learner`` that stores episodics **and** runs bi-temporal consolidation.

    Parameters
    ----------
    consolidator:
        The ``MemoryConsolidator`` that runs propose+apply on the off-path.
    gate:
        The ``MemoryGate`` for episodic storage and fact writes.
    clock:
        Injected callable returning the current datetime. Never calls
        ``datetime.now()`` internally — keeps the class deterministic.
    profile_builder:
        Optional ``UserProfileBuilder`` (Phase 2C).  When provided, ``user_model``
        delegates to it; when absent, a default ``UserModel`` is returned.
    """

    def __init__(
        self,
        *,
        consolidator: MemoryConsolidator,
        gate: MemoryGate,
        clock: Callable[[], datetime],
        profile_builder: "UserProfileBuilder | None" = None,
    ) -> None:
        self._consolidator = consolidator
        self._gate = gate
        self._clock = clock
        self._profile_builder = profile_builder

    # ------------------------------------------------------------------
    # Learner Protocol
    # ------------------------------------------------------------------

    async def process_session(self, conversation: Conversation) -> None:
        """Store each message in *conversation* as an episodic memory.

        Mirrors ``MinimalLearner.process_session`` so Phase 1 recall works
        across turns while Phase 2B consolidation enriches the fact base.
        """
        for msg in conversation.messages:
            source = (
                MemorySource.USER_STATED if msg.role is Role.USER else MemorySource.AGENT_INFERRED
            )
            await self._gate.store(
                Memory(
                    user_id=conversation.user_id,
                    kind=MemoryKind.EPISODIC,
                    content=msg.content,
                    source=source,
                    created_at=self._clock(),
                )
            )

    async def user_model(self, user_id: str) -> UserModel:
        """Return the current user model.

        When a ``UserProfileBuilder`` is wired in (Phase 2C), delegates to it so the
        model is derived from currently-valid facts + signals.  Falls back to a default
        model when no profile builder is configured (Phase 2B compat / test simplicity).
        """
        if self._profile_builder is not None:
            return await self._profile_builder.build(user_id)
        return UserModel(user_id=user_id)

    async def consolidate(self, user_id: str) -> None:
        """Run bi-temporal fact consolidation for *user_id*.

        Delegates to ``MemoryConsolidator.consolidate``, which pulls recent
        episodics, proposes fact operations via the LLM, and applies them.
        """
        await self._consolidator.consolidate(user_id)
