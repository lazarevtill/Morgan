"""Short-horizon persona attribution — the cold-path writer for the right brain.

VoiceMem §3.2 runs affect estimation *during* the turn and edits the persona graph from
it. Morgan cannot: the request path reads learned knowledge and never writes it. So the
same work happens immediately after the reply is sent, in the same cold-path pass that
stores the turn -- the paper's short horizon, on Morgan's side of the invariant.

Attribution is more than sentiment. What the graph needs is affect *plus its target*:
not "the user sounded annoyed" but "the user is impatient **with the weekly Harbor
sync**". A reading with no target cannot become a cross-entity node, and a cross-entity
node is the only kind that can later be generalised on evidence. So an observation
without a target is recorded as intrinsic only when it is a stated preference, and
dropped otherwise.

**A missing model produces nothing, not a guess.** ``NullAttributor`` is the fallback,
and it writes nothing at all. That is deliberate: the alternative -- keyword-matching
affect -- would file confident nonsense about the owner's personality into a store whose
whole purpose is to be right about them, and every wrong node then shapes future turns.
Missing evidence is a gap, and a gap is the honest representation of it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field

from morgan_brain.models.message import Message, Role
from morgan_brain.modules.personalization.persona_graph import PersonaGraph
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage

logger = logging.getLogger(__name__)


class Observation(BaseModel):
    """One affective reading, with the thing it is about."""

    description: str
    #: The real-world thing the affect concerns. Empty means the reading is about the
    #: user in general, which only a *stated* preference can support.
    entity: str = ""
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    #: True only when the user said it about themselves. An inference with no target is
    #: dropped rather than recorded as a disposition -- the same rule `MemorySource`
    #: enforces for facts.
    stated: bool = False


class ObservationBatch(BaseModel):
    observations: list[Observation] = Field(default_factory=list)


class Attributor(Protocol):
    async def observe(self, messages: list[Message]) -> ObservationBatch:
        """Read affect and its target out of a turn. Must not raise."""
        ...


class NullAttributor:
    """Records nothing. The honest fallback when no model is reachable."""

    async def observe(self, messages: list[Message]) -> ObservationBatch:
        return ObservationBatch()


class LLMAttributor:
    """Reads affect and its target from the turn, on the ``reflection`` role."""

    def __init__(
        self,
        *,
        router: RoleRouter,
        capability_registry: CapabilityRegistry,
        role: str = "reflection",
    ) -> None:
        self._router = router
        self._reg = capability_registry
        self._role = role

    async def observe(self, messages: list[Message]) -> ObservationBatch:
        user_text = "\n".join(m.content for m in messages if m.role is Role.USER).strip()
        if not user_text:
            return ObservationBatch()
        try:
            client, model = self._router.chat_for(self._role, needs_json_schema=True)
        except LookupError:
            try:
                client, model = self._router.chat_for(self._role)
            except LookupError:
                return ObservationBatch()

        provider = model.split("/", 1)[0] if "/" in model else "fake"
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "Read what the user's words reveal about their attitude, and about "
                    "WHAT that attitude concerns. Always name the target in `entity` when "
                    "there is one -- a reading with no target says something about the "
                    "person, and only counts when they stated it about themselves "
                    "(`stated`: true). Report nothing rather than guessing. `valence` runs "
                    "-1 (negative) to 1 (positive)."
                ),
            ),
            ChatMessage(role="user", content=user_text),
        ]
        try:
            return await generate_structured(
                client,
                prompt,
                model=model,
                schema=ObservationBatch,
                descriptor=self._reg.get(provider, model),
            )
        except Exception:
            logger.exception("persona-attribution: reading failed; recording nothing this turn")
            return ObservationBatch()


class PersonaAttributor:
    """Turns a turn's observations into persona-graph writes."""

    def __init__(self, *, graph: PersonaGraph, attributor: Attributor) -> None:
        self._graph = graph
        self._attributor = attributor

    async def attribute(
        self,
        *,
        user_id: str,
        project: str,
        session_id: str,
        messages: list[Message],
        now: datetime,
    ) -> int:
        """Record this turn's observations. Returns how many were written."""
        batch = await self._attributor.observe(messages)
        written = 0
        for obs in batch.observations:
            description = obs.description.strip()
            if not description:
                continue
            anchor = obs.entity.strip()
            if not anchor and not obs.stated:
                # An untargeted inference is exactly the situational-read-as-trait error
                # the graph exists to prevent. Only the user's own statement about
                # themselves may enter unanchored.
                logger.debug("persona-attribution: dropping untargeted inference %r", description)
                continue
            self._graph.observe(
                user_id=user_id,
                project=project,
                description=description,
                entity=anchor or None,
                valence=obs.valence,
                session_id=session_id,
                now=now,
            )
            written += 1
        return written
