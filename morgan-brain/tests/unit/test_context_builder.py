from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.models.memory import Memory, MemoryKind
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.context.builder import build_messages


def _request(**kw) -> ReasoningRequest:
    base = {
        "user_id": "u1",
        "project": "p",
        "perception": FusedPerception(text="where do I live?"),
        "personalization": PersonalizedContext(system_fragment="User prefers terse replies."),
        "memories": [
            Memory(user_id="u1", kind=MemoryKind.SEMANTIC, content="User lives in Berlin")
        ],
        "history": [],
        "skill_prompt": "",
    }
    base.update(kw)
    return ReasoningRequest(**base)


def test_system_message_includes_personalization_and_memories():
    msgs = build_messages(_request())
    system = msgs[0]
    assert system.role == "system"
    assert "terse" in system.content
    assert "Berlin" in system.content


def test_last_message_is_the_user_query():
    msgs = build_messages(_request())
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "where do I live?"


def test_skill_prompt_included_when_present():
    msgs = build_messages(_request(skill_prompt="ALWAYS cite memories."))
    assert any("cite memories" in m.content for m in msgs)
