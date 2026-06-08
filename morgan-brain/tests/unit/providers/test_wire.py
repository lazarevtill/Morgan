from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, Usage, ToolCall


def test_chatmessage_roundtrips_openai_dict():
    m = ChatMessage(role="user", content="hi")
    assert m.to_openai() == {"role": "user", "content": "hi"}


def test_chatresult_holds_text_and_usage():
    r = ChatResult(text="ok", model="m", usage=Usage(input_tokens=3, output_tokens=1))
    assert r.text == "ok" and r.usage.output_tokens == 1 and r.tool_calls == []


def test_stream_delta_kinds():
    d = StreamDelta(kind="text_delta", text="x")
    assert d.kind == "text_delta" and d.text == "x"
