"""The cognitive loop — one turn, end to end (design spec §6).

This class is deliberately thin: it *coordinates* the modules through their Protocols and owns
no domain logic itself. It depends only on interfaces, so every collaborator is swappable and
mockable. The discipline it enforces:

* steps 2–6 only READ learned knowledge (hot path),
* step 7 only WRITES and never blocks the response (cold path).
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.interfaces.learning import Learner
from morgan_brain.interfaces.perception import Perception
from morgan_brain.interfaces.personalization import Personalizer
from morgan_brain.interfaces.reasoning import Reasoner, ReasoningRequest, ReasoningResult
from morgan_brain.interfaces.skills import SkillEngine
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.providers.wire import ToolSpec
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
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self._perception = perception
        self._personalizer = personalizer
        self._memory = memory
        self._skills = skills
        self._reasoner = reasoner
        self._learner = learner
        self._bus = bus
        self._tools: list[ToolSpec] = tools or []

    def _scoped_tools(self, selected_skills: list[Any]) -> list[ToolSpec]:
        """Return the ToolSpec list to expose for this turn.

        If any selected skill declares a ``tools`` attribute (list of tool names),
        only those specs are included.  Otherwise all registered specs are returned.
        """
        skill_tool_names: set[str] = set()
        for skill in selected_skills:
            declared = getattr(skill, "tools", None)
            if declared:
                skill_tool_names.update(declared)

        if skill_tool_names:
            return [t for t in self._tools if t.name in skill_tool_names]
        return list(self._tools)

    async def handle_turn_with_id(
        self,
        *,
        user_id: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
    ) -> tuple[ReasoningResult, str]:
        """Like handle_turn but also returns the turn_id alongside the result."""
        turn_id = uuid.uuid4().hex

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
        memories = await self._memory.recall(MemoryQuery(user_id=user_id, text=perception.text))

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
                tools=self._scoped_tools(skills),
                history=history or [],
            )
        )

        # 7. Post-turn cold path
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={
                    "session_id": session_id,
                    "request": text,
                    "response": result.text,
                    "turn_id": turn_id,
                },
            )
        )
        return result, turn_id

    async def handle_turn(
        self,
        *,
        user_id: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
    ) -> ReasoningResult:
        turn_id = uuid.uuid4().hex

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
        memories = await self._memory.recall(MemoryQuery(user_id=user_id, text=perception.text))

        # 5. Skill selection
        skills = await self._skills.select(perception)
        skill_prompt = "\n\n".join(s.body for s in skills)

        # 6. Reasoning — include scoped tool specs so the loop can execute tools.
        result = await self._reasoner.generate(
            ReasoningRequest(
                user_id=user_id,
                perception=perception,
                personalization=personalization,
                memories=memories,
                skill_prompt=skill_prompt,
                tools=self._scoped_tools(skills),
                history=history or [],
            )
        )

        # 7. Post-turn (cold path): announce; the learning-worker consumes this off-path.
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={
                    "session_id": session_id,
                    "request": text,
                    "response": result.text,
                    "turn_id": turn_id,
                },
            )
        )
        return result

    async def stream_turn(
        self,
        *,
        user_id: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming variant of handle_turn.

        Runs the same pre-reasoning pipeline (perception → personalization → memory recall
        → skill selection) synchronously, then yields LLM token deltas as they arrive.
        After the stream is exhausted, publishes RESPONSE_GENERATED so turn-storage and
        learning still fire on the cold path — identical behaviour to handle_turn.
        """
        turn_id = uuid.uuid4().hex

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
        memories = await self._memory.recall(MemoryQuery(user_id=user_id, text=perception.text))

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
            tools=self._scoped_tools(skills),
            history=history or [],
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
                payload={
                    "session_id": session_id,
                    "request": text,
                    "response": full_text,
                    "turn_id": turn_id,
                },
            )
        )
