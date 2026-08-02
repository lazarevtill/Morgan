"""Phase 2B/C — ConsolidationLearner.

Implements the ``Learner`` Protocol by combining:
- Episodic storage on process_session (Phase 1 parity for cross-turn recall).
- Real fact consolidation via ``MemoryConsolidator`` (Phase 2B).
- Real UserModel derivation via ``UserProfileBuilder`` (Phase 2C, optional).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable

from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.models.memory import DEFAULT_PROJECT, Memory, MemoryKind, MemorySource
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

        Stores each message as an episodic memory so Phase 1 recall works
        across turns while Phase 2B consolidation enriches the fact base.
        Scoped to ``conversation.project`` so a turn served under a real project name
        (e.g. from the CLI) lands where consolidation and recall will actually find it.
        """
        for msg in conversation.messages:
            source = (
                MemorySource.USER_STATED if msg.role is Role.USER else MemorySource.AGENT_INFERRED
            )
            await self._gate.store(
                Memory(
                    user_id=conversation.user_id,
                    project=conversation.project,
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

    async def consolidate(self, user_id: str, *, project: str = DEFAULT_PROJECT) -> None:
        """Run bi-temporal fact consolidation for *user_id*, scoped to *project*.

        Delegates to ``MemoryConsolidator.consolidate``, which pulls recent
        episodics, proposes fact operations via the LLM, and applies them.
        """
        await self._consolidator.consolidate(user_id, project=project)

    async def projects_for_user(self, user_id: str) -> list[str]:
        """Distinct projects *user_id* has written memories under.

        Used by :class:`~morgan_brain.scheduling.learning_jobs.LearningScheduler` to fan
        nightly consolidation out across every project the user actually has, instead of
        the single ``DEFAULT_PROJECT`` bucket -- so anything written under a real project
        name (e.g. by the CLI) is still consolidated.
        """
        return await self._gate.distinct_projects(user_id)

    async def learn_from_edit(
        self, *, user_id: str, project: str = DEFAULT_PROJECT, original: str, edited: str
    ) -> str:
        """CIPHER: distil a preference from an (original, edited) reply pair and persist
        it as agent-inferred ``comm_*`` facts, so the next ``user_model`` reflects it.

        This is the loop that makes "edit my reply → future turns change": the LLM only
        produces the natural-language delta; the mapping to durable facts is deterministic
        and the facts evolve (upsert = supersede), never overwrite. Returns the delta text
        (``""`` when no profile builder is wired). Caller treats failure as non-fatal —
        feedback must never fail because learning did.
        """
        if self._profile_builder is None:
            return ""
        # Local import avoids a module-level cycle (profile imports nothing from here,
        # but keeping it lazy mirrors the TYPE_CHECKING-only import above).
        from morgan_brain.learning.profile import (
            preference_delta_from_edit,
            preference_facts_from_delta,
        )

        delta = await preference_delta_from_edit(self._profile_builder, user_id, original, edited)
        for fact in preference_facts_from_delta(user_id, delta, self._clock(), project=project):
            await self._gate.upsert_fact(fact)
        return delta
