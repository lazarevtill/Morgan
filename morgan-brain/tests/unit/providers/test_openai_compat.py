"""Unit tests for OpenAICompatAdapter and OllamaAdapter.

Construction and message/tool translation only — NO network calls.
"""

from __future__ import annotations

from morgan_brain.interfaces.llm import ChatClient
from morgan_brain.providers.adapters.openai_compat import (
    OpenAICompatAdapter,
    _to_openai_messages,
    _to_openai_tools,
    _from_openai_tool_calls,
)
from morgan_brain.providers.adapters.ollama import OllamaAdapter
from morgan_brain.providers.wire import ChatMessage, ToolCall, ToolSpec


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_openai_compat_adapter_constructs_without_network():
    adapter = OpenAICompatAdapter(
        base_url="http://x/v1",
        api_key="k",
        provider="openaicompat",
    )
    assert adapter._provider == "openaicompat"
    assert adapter._base_url == "http://x/v1"


def test_openai_compat_adapter_is_chat_client():
    adapter = OpenAICompatAdapter(base_url="http://x/v1", api_key="k", provider="p")
    assert isinstance(adapter, ChatClient)


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------


def test_ollama_adapter_default_base_url():
    adapter = OllamaAdapter()
    assert "11434" in adapter._base_url
    assert adapter._provider == "ollama"


def test_ollama_adapter_custom_base_url():
    adapter = OllamaAdapter(base_url="http://remote:11434/v1")
    assert adapter._base_url == "http://remote:11434/v1"


def test_ollama_adapter_is_chat_client():
    adapter = OllamaAdapter()
    assert isinstance(adapter, ChatClient)


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def test_to_openai_messages_basic():
    msgs = [
        ChatMessage(role="system", content="You are an assistant."),
        ChatMessage(role="user", content="Hello"),
    ]
    result = _to_openai_messages(msgs)
    assert result == [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello"},
    ]


def test_to_openai_messages_with_tool_call_id():
    msg = ChatMessage(role="tool", content="result", tool_call_id="call_abc")
    result = _to_openai_messages([msg])
    assert result[0]["tool_call_id"] == "call_abc"


def test_to_openai_messages_with_tool_calls():
    msg = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})],
    )
    result = _to_openai_messages([msg])
    assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Tool spec translation
# ---------------------------------------------------------------------------


def test_to_openai_tools():
    specs = [
        ToolSpec(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    result = _to_openai_tools(specs)
    assert result == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# Tool call parsing (from openai SDK objects)
# ---------------------------------------------------------------------------


def test_from_openai_tool_calls_none():
    assert _from_openai_tool_calls(None) == []


def test_from_openai_tool_calls_empty():
    assert _from_openai_tool_calls([]) == []


def test_from_openai_tool_calls_valid():
    import json

    class FakeFn:
        name = "my_tool"
        arguments = json.dumps({"x": 1})

    class FakeTc:
        id = "call_1"
        function = FakeFn()

    result = _from_openai_tool_calls([FakeTc()])
    assert len(result) == 1
    assert result[0].name == "my_tool"
    assert result[0].arguments == {"x": 1}
    assert result[0].id == "call_1"


def test_from_openai_tool_calls_invalid_json():
    class FakeFn:
        name = "bad"
        arguments = "{not valid json"

    class FakeTc:
        id = ""
        function = FakeFn()

    result = _from_openai_tool_calls([FakeTc()])
    assert result[0].arguments == {}
