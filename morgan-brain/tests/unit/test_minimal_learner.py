from datetime import datetime

from morgan_brain.models.memory import MemoryKind, MemorySource
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.models.user import RelationshipStage
from morgan_brain.modules.learning.minimal import MinimalLearner


class _RecordingMemory:
    def __init__(self):
        self.stored = []

    async def store(self, memory):
        self.stored.append(memory)
        return memory.id

    async def recall(self, query):
        return []

    async def upsert_fact(self, fact):
        return fact.id

    async def current_facts(self, *, user_id, subject=None):
        return []


async def test_user_model_defaults_to_new():
    learner = MinimalLearner(memory=_RecordingMemory(), clock=lambda: datetime(2026, 1, 1))
    um = await learner.user_model("u1")
    assert um.user_id == "u1"
    assert um.relationship_stage is RelationshipStage.NEW


async def test_process_session_stores_each_message_as_episodic():
    mem = _RecordingMemory()
    learner = MinimalLearner(memory=mem, clock=lambda: datetime(2026, 1, 1))
    convo = Conversation(user_id="u1", session_id="s1", messages=[
        Message(user_id="u1", role=Role.USER, content="hello"),
        Message(user_id="u1", role=Role.ASSISTANT, content="hi there"),
    ])
    await learner.process_session(convo)
    assert len(mem.stored) == 2
    assert {m.content for m in mem.stored} == {"hello", "hi there"}
    assert all(m.kind is MemoryKind.EPISODIC for m in mem.stored)
    by_content = {m.content: m.source for m in mem.stored}
    assert by_content["hello"] is MemorySource.USER_STATED
    assert by_content["hi there"] is MemorySource.AGENT_INFERRED
