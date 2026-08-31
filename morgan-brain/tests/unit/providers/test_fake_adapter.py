from morgan_brain.interfaces.llm import ChatClient
from morgan_brain.providers.adapters.fake import FakeChatClient, FakeEmbedder

# ---------------------------------------------------------------------------
# FakeChatClient — single reply
# ---------------------------------------------------------------------------


async def test_fake_chat_agenerate_returns_chat_result():
    c = FakeChatClient(reply="hello")
    from morgan_brain.providers.wire import ChatMessage

    result = await c.agenerate([ChatMessage(role="user", content="hi")], model="m")
    assert result.text == "hello"
    assert result.model == "m"


async def test_fake_chat_records_last_messages_and_model():
    c = FakeChatClient(reply="yo")
    from morgan_brain.providers.wire import ChatMessage

    msgs = [ChatMessage(role="user", content="test")]
    await c.agenerate(msgs, model="fast-model")
    assert c.last_model == "fast-model"
    assert c.last_messages == msgs


async def test_fake_chat_astream_yields_text_delta_then_finish():
    c = FakeChatClient(reply="streamed")
    deltas = [d async for d in c.astream([], model="m")]
    assert len(deltas) == 2
    assert deltas[0].kind == "text_delta" and deltas[0].text == "streamed"
    assert deltas[1].kind == "finish" and deltas[1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# FakeChatClient — replies queue + calls counter (Task 6 dep)
# ---------------------------------------------------------------------------


async def test_fake_chat_replies_queue_and_calls_counter():
    c = FakeChatClient(replies=["a", "b"])
    assert (await c.agenerate([], model="m")).text == "a"
    assert (await c.agenerate([], model="m")).text == "b"
    assert c.calls == 2


async def test_fake_chat_calls_counter_increments_for_single_reply():
    c = FakeChatClient(reply="x")
    await c.agenerate([], model="m")
    await c.agenerate([], model="m")
    assert c.calls == 2


async def test_fake_chat_replies_queue_wraps_when_exhausted():
    """When the queue is exhausted the last reply is repeated (robustness for tests)."""
    c = FakeChatClient(replies=["only"])
    await c.agenerate([], model="m")
    result = await c.agenerate([], model="m")
    assert result.text == "only"  # last reply reused
    assert c.calls == 2


# ---------------------------------------------------------------------------
# FakeChatClient with tool_calls
# ---------------------------------------------------------------------------


async def test_fake_chat_with_tool_calls():
    from morgan_brain.providers.wire import ToolCall

    tc = ToolCall(id="t1", name="search", arguments={"q": "test"})
    c = FakeChatClient(reply="", tool_calls=[tc])
    result = await c.agenerate([], model="m")
    assert result.tool_calls[0].name == "search"


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_fake_chat_satisfies_chat_client_protocol():
    assert isinstance(FakeChatClient(), ChatClient)


# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic + L2-normalized
# ---------------------------------------------------------------------------


async def test_fake_embedder_returns_correct_dim():
    e = FakeEmbedder(dim=8)
    vecs = await e.aembed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 8 for v in vecs)


async def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=16)
    v1 = (await e.aembed(["hello"]))[0]
    v2 = (await e.aembed(["hello"]))[0]
    assert v1 == v2


async def test_fake_embedder_different_texts_differ():
    e = FakeEmbedder(dim=16)
    v1 = (await e.aembed(["hello"]))[0]
    v2 = (await e.aembed(["world"]))[0]
    assert v1 != v2


async def test_fake_embedder_is_l2_normalized():
    import math

    e = FakeEmbedder(dim=16)
    v = (await e.aembed(["test"]))[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6
