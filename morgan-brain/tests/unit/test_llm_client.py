from morgan_brain.modules.reasoning.llm.client import FakeLLMClient, ChatMessage


async def test_fake_llm_echoes_scripted_reply():
    llm = FakeLLMClient(reply="hello back")
    out = await llm.complete([ChatMessage(role="user", content="hi")], model="m")
    assert out == "hello back"


async def test_fake_llm_records_last_messages():
    llm = FakeLLMClient(reply="ok")
    msgs = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="q")]
    await llm.complete(msgs, model="m")
    assert llm.last_messages == msgs
    assert llm.last_model == "m"
