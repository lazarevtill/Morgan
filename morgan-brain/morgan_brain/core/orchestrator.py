"""The cognitive loop — one turn, end to end (design spec §6).

This class is deliberately thin: it *coordinates* the modules through their Protocols and owns
no domain logic itself. It depends only on interfaces, so every collaborator is swappable and
mockable. The discipline it enforces:

* steps 2–6 only READ learned knowledge (hot path),
* step 7 only WRITES and never blocks the response (cold path).
"""
from __future__ import annotations

from typing import AsyncIterator

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.interfaces.learning import Learner
from morgan_brain.interfaces.perception import Perception
from morgan_brain.interfaces.personalization import Personalizer
from morgan_brain.interfaces.reasoning import Reasoner, ReasoningRequest, ReasoningResult
from morgan_brain.interfaces.skills import SkillEngine
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.security.memory_gate import MemoryGate


class Orchestrator:
    def __init__(
        self,
        *,
        perception: Perception,
        personalizer: Personalizer,
        memory: MemoryGate,
        skills: SkillEngine,
        reasoner: Reasoner,
        learner: Learner,
        bus: EventBus,
    ) -> None:
        self._perception = perception
        self._personalizer = personalizer
        self._memory = memory
        self._skills = skills
        self._reasoner = reasoner
        self._learner = learner
        self._bus = bus

    async def handle_turn(
        self, *, user_id: str, text: str, session_id: str | None = None
    ) -> ReasoningResult:
        # 2. Perception
        perception = await self._perception.analyze(user_id=user_id, text=text)
        await self._bus.publish(
            Event(type=EventType.PERCEPTION_COMPLETE, user_id=user_id, payload={"text": text})
        )

        # 3. Personalization (reads UserModel; biases everything that follows)
        user_model = await self._learner.user_model(user_id)
        personalization = await self._personalizer.build(
            user_model=user_model, perception=perception
        )

        # 4. Memory recall (multi-signal, currently-valid facts)
        memories = await self._memory.recall(
            MemoryQuery(user_id=user_id, text=perception.text)
        )

        # 5. Skill selection
        skills = await self._skills.select(perception)
        skill_prompt = "\n\n".join(s.body for s in skills)

        # 6. Reasoning
        result = await self._reasoner.generate(
            ReasoningRequest(
                user_id=user_id,
                perception=perception,
                personalization=personalization,
                memories=memories,
                skill_prompt=skill_prompt,
            )
        )

        # 7. Post-turn (cold path): announce; the learning-worker consumes this off-path.
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={"session_id": session_id, "request": text, "response": result.text},
            )
        )
        return result

    async def stream_turn(
        self, *, user_id: str, text: str, session_id: str | None = None
    ) -> AsyncIterator[str]:
        """Streaming variant of handle_turn.

        Runs the same pre-reasoning pipeline (perception → personalization → memory recall
        → skill selection) synchronously, then yields LLM token deltas as they arrive.
        After the stream is exhausted, publishes RESPONSE_GENERATED so turn-storage and
        learning still fire on the cold path — identical behaviour to handle_turn.
        """
        # 2. Perception
        perception = await self._perception.analyze(user_id=user_id, text=text)
        await self._bus.publish(
            Event(type=EventType.PERCEPTION_COMPLETE, user_id=user_id, payload={"text": text})
        )

        # 3. Personalization
        user_model = await self._learner.user_model(user_id)
        personalization = await self._personalizer.build(
            user_model=user_model, perception=perception
        )

        # 4. Memory recall
        memories = await self._memory.recall(
            MemoryQuery(user_id=user_id, text=perception.text)
        )

        # 5. Skill selection
        skills = await self._skills.select(perception)
        skill_prompt = "\n\n".join(s.body for s in skills)

        # 6. Stream reasoning — accumulate full text for cold-path storage.
        request = ReasoningRequest(
            user_id=user_id,
            perception=perception,
            personalization=personalization,
            memories=memories,
            skill_prompt=skill_prompt,
        )
        chunks: list[str] = []
        async for delta in self._reasoner.stream(request):
            chunks.append(delta)
            yield delta

        # 7. Post-turn cold path — same as handle_turn.
        full_text = "".join(chunks)
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={"session_id": session_id, "request": text, "response": full_text},
            )
        )
