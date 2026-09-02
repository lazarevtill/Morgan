"""A model server that is down is reported by name, from both adapters.

Real sockets, no network: port 1 on loopback is closed, so every connect is refused at once.
"""

from __future__ import annotations

import pytest

from morgan_brain.interfaces.llm import ProviderUnreachable
from morgan_brain.providers.adapters.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.adapters.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.wire import ChatMessage

_CLOSED = "http://127.0.0.1:1/v1"
_MESSAGES = [ChatMessage(role="user", content="hi")]


async def test_chat_names_the_endpoint_it_could_not_reach() -> None:
    adapter = OpenAICompatAdapter(base_url=_CLOSED, api_key="k", provider="p", timeout=2.0)
    with pytest.raises(ProviderUnreachable) as info:
        await adapter.agenerate(_MESSAGES, model="m")
    assert info.value.endpoint == _CLOSED
    assert _CLOSED in str(info.value)
    assert "morgan doctor" in str(info.value)


async def test_stream_raises_before_yielding_anything() -> None:
    adapter = OpenAICompatAdapter(base_url=_CLOSED, api_key="k", provider="p", timeout=2.0)
    received = []
    with pytest.raises(ProviderUnreachable):
        async for delta in adapter.astream(_MESSAGES, model="m"):
            received.append(delta)
    assert received == []


async def test_embedder_names_the_endpoint_it_could_not_reach() -> None:
    embedder = OpenAICompatEmbedder(_CLOSED, "e", timeout=2.0)
    with pytest.raises(ProviderUnreachable) as info:
        await embedder.embed("hi")
    assert info.value.endpoint.startswith(_CLOSED)


def test_it_is_a_connection_error() -> None:
    """Callers that already handle ConnectionError keep working."""
    assert issubclass(ProviderUnreachable, ConnectionError)
