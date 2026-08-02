"""The cognitive loop — one turn, end to end (design spec §6).

This class is deliberately thin: it *coordinates* the modules through their Protocols and owns
no domain logic itself. It depends only on interfaces, so every collaborator is swappable and
mockable. The discipline it enforces:

* steps 2–6 only READ learned knowledge (hot path),
* step 7 only WRITES and never blocks the response (cold path).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.interfaces.learning import Learner
from morgan_brain.interfaces.perception import Perception
from morgan_brain.interfaces.personalization import Personalizer
from morgan_brain.interfaces.reasoning import Reasoner, ReasoningRequest, ReasoningResult
from morgan_brain.interfaces.skills import SkillEngine
from morgan_brain.learning.history import session_key
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.models.message import Message, Role
from morgan_brain.providers.wire import ToolSpec
from morgan_brain.security.memory_gate import MemoryGate

if TYPE_CHECKING:
    from morgan_brain.learning.history import SessionHistoryStore
    from morgan_brain.learning.recorder import SignalRecorder


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
        recorder: "SignalRecorder | None" = None,
        history_store: "SessionHistoryStore | None" = None,
    ) -> None:
        self._perception = perception
        self._personalizer = personalizer
        self._memory = memory
        self._skills = skills
        self._reasoner = reasoner
        self._learner = learner
        self._bus = bus
        self._tools: list[ToolSpec] = tools or []
        self._recorder = recorder
        self._history_store = history_store

    async def _persist_turn(
        self,
        *,
        user_id: str,
        project: str,
        session_id: str | None,
        turn_id: str,
        text: str,
        reply: str,
    ) -> None:
        """Local, **bus-independent** persistence the next turn depends on.

        Session history (so multi-turn threads) and the base interaction signal (so
        feedback/learning has a row to attach to) are written **in-process by whichever
        process served the turn**, synchronously, regardless of the event-bus backend.
        Consolidation stays on the bus (announced via RESPONSE_GENERATED) and runs in the
        learning-worker under Redis — but history and the base signal must not depend on a
        worker consuming an event, or the documented 2-process topology silently degrades
        every turn to turn 1 (the GAP-2 break). No-ops cleanly when deps aren't injected.

        ``project`` is threaded through so ``forget()`` can later erase these rows per
        project instead of only per user — see task-14.
        """
        if self._history_store is not None:
            hkey = session_key(user_id, session_id)
            self._history_store.append(
                hkey, Message(user_id=user_id, role=Role.USER, content=text), project=project
            )
            self._history_store.append(
                hkey,
                Message(user_id=user_id, role=Role.ASSISTANT, content=reply),
                project=project,
            )
        if self._recorder is not None:
            await self._recorder.record_turn(
                user_id=user_id,
                project=project,
                session_id=session_id or "default",
                turn_id=turn_id,
                query=text,
                reply=reply,
            )

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
        project: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
        system_override: str = "",
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
        memories = await self._memory.recall(
            MemoryQuery(user_id=user_id, project=project, text=perception.text)
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
                tools=self._scoped_tools(skills),
                history=history or [],
                system_override=system_override,
            )
        )

        # 7. Post-turn. Local persistence (history + base signal) is written in-process,
        # synchronously; consolidation is announced on the bus and runs off-path.
        await self._persist_turn(
            user_id=user_id,
            project=project,
            session_id=session_id,
            turn_id=turn_id,
            text=text,
            reply=result.text,
        )
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={
                    "session_id": session_id,
                    "project": project,
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
        project: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
        system_override: str = "",
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
        memories = await self._memory.recall(
            MemoryQuery(user_id=user_id, project=project, text=perception.text)
        )

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
                system_override=system_override,
            )
        )

        # 7. Post-turn. Local persistence in-process; consolidation announced off-path.
        await self._persist_turn(
            user_id=user_id,
            project=project,
            session_id=session_id,
            turn_id=turn_id,
            text=text,
            reply=result.text,
        )
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={
                    "session_id": session_id,
                    "project": project,
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
        project: str,
        text: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
        system_override: str = "",
    ) -> AsyncIterator[str]:
        """Streaming variant of handle_turn.

        Runs the same pre-reasoning pipeline (perception → personalization → memory recall
        → skill selection) synchronously, then yields LLM token deltas as they arrive.
        After the stream is exhausted, publishes RESPONSE_GENERATED so turn-storage and
        learning still fire on the cold path — identical behaviour to handle_turn.

        ``system_override`` carries the eval-gated champion preprompt; it must be threaded
        here exactly as in handle_turn or the streamed assistant runs on the base prompt
        instead of the learned one.
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
        memories = await self._memory.recall(
            MemoryQuery(user_id=user_id, project=project, text=perception.text)
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
            tools=self._scoped_tools(skills),
            history=history or [],
            system_override=system_override,
        )
        chunks: list[str] = []
        async for delta in self._reasoner.stream(request):
            chunks.append(delta)
            yield delta

        # 7. Post-turn cold path — same as handle_turn.
        full_text = "".join(chunks)
        await self._persist_turn(
            user_id=user_id,
            project=project,
            session_id=session_id,
            turn_id=turn_id,
            text=text,
            reply=full_text,
        )
        await self._bus.publish(
            Event(
                type=EventType.RESPONSE_GENERATED,
                user_id=user_id,
                payload={
                    "session_id": session_id,
                    "project": project,
                    "request": text,
                    "response": full_text,
                    "turn_id": turn_id,
                },
            )
        )
