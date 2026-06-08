from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.models.memory import Memory
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.llm.client import FakeLLMClient
from morgan_brain.modules.reasoning.reasoner import ReasoningModule


def _request() -> ReasoningRequest:
    return ReasoningRequest(
        user_id="u1",
        perception=FusedPerception(text="hi"),
        personalization=PersonalizedContext(),
        memories=[Memory(user_id="u1", content="user is called Sam")],
        history=[],
        skill_prompt="",
    )


async def test_generate_returns_llm_reply_and_model():
    llm = FakeLLMClient(reply="Hello Sam!")
    r = ReasoningModule(llm=llm, model="qwen2.5:7b", fast_model="qwen2.5:7b")
    result = await r.generate(_request())
    assert result.text == "Hello Sam!"
    assert result.model_used == "qwen2.5:7b"


async def test_generate_passes_memories_into_context():
    llm = FakeLLMClient(reply="ok")
    r = ReasoningModule(llm=llm, model="m", fast_model="m")
    await r.generate(_request())
    system = llm.last_messages[0]
    assert "Sam" in system.content
