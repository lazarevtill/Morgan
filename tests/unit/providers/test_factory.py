"""The factory's endpoint decisions: chat and embeddings are separately addressable."""

from __future__ import annotations

import pytest

from morgan_brain.config import Settings
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.factory import build_chat_client, build_embedder
from morgan_brain.providers.wire import ChatMessage, ProviderUnreachable

#: Port 1 on loopback is closed, so every connect is refused at once.
_CLOSED = "http://127.0.0.1:1/v1"


async def test_a_separate_embedding_endpoint_that_is_down_is_named_by_its_own_setting():
    """The chat endpoint answers fine here; sending the owner to check it is sending them to
    the one server that is not broken."""
    embedder = build_embedder(
        Settings(
            llm_endpoint="http://chat.invalid/v1",
            embedding_endpoint=_CLOSED,
            embedding_backend="provider",
        )
    )

    with pytest.raises(ProviderUnreachable) as info:
        await embedder.embed("hi")

    assert "MORGAN_EMBEDDING_ENDPOINT" in str(info.value)
    assert "MORGAN_LLM_ENDPOINT" not in str(info.value)


async def test_embeddings_sent_to_the_chat_endpoint_name_the_chat_setting():
    """With no embedding endpoint configured there is no MORGAN_EMBEDDING_ENDPOINT to check:
    the chat endpoint serves both."""
    embedder = build_embedder(
        Settings(llm_endpoint=_CLOSED, embedding_endpoint="", embedding_backend="provider")
    )

    with pytest.raises(ProviderUnreachable) as info:
        await embedder.embed("hi")

    assert "MORGAN_LLM_ENDPOINT" in str(info.value)
    assert "MORGAN_EMBEDDING_ENDPOINT" not in str(info.value)


async def test_a_chat_endpoint_that_is_down_is_named_by_its_setting():
    chat = build_chat_client(
        Settings(llm_endpoint=_CLOSED, embedding_endpoint="http://embed.invalid/v1")
    )

    with pytest.raises(ProviderUnreachable) as info:
        await chat.agenerate([ChatMessage(role="user", content="hi")], model="m")

    assert "MORGAN_LLM_ENDPOINT" in str(info.value)
    assert "MORGAN_EMBEDDING_ENDPOINT" not in str(info.value)


def test_embedder_falls_back_to_the_chat_endpoint_when_unset():
    """One server serving both is the common case and stays zero-configuration."""
    embedder = build_embedder(
        Settings(llm_endpoint="http://chat:8081/v1", embedding_backend="provider")
    )

    assert isinstance(embedder, OpenAICompatEmbedder)
    assert embedder._url == "http://chat:8081/v1/embeddings"


def test_embedder_uses_its_own_endpoint_when_one_is_configured():
    """A chat server without ``--embeddings`` is a normal topology: llama-server serves one
    model per process, so the embedding model is a second server at a second address. With
    a single endpoint setting there is no way to express that, and embeddings 501.
    """
    embedder = build_embedder(
        Settings(
            llm_endpoint="http://chat:8081/v1",
            embedding_endpoint="http://embed:8082/v1",
            embedding_backend="provider",
        )
    )

    assert isinstance(embedder, OpenAICompatEmbedder)
    assert embedder._url == "http://embed:8082/v1/embeddings"


def test_a_model_built_for_a_database_is_checked_against_its_space():
    """The one path every memory command takes: ``build_memory_context`` hands the factory
    its connection, and the model is then checked on its first call."""
    conn = open_db(":memory:")
    try:
        embedder = build_embedder(
            Settings(embedding_endpoint="http://embed:8082/v1", embedding_backend="provider"),
            conn=conn,
        )
    finally:
        conn.close()

    assert isinstance(embedder, CheckedEmbedder)


def test_the_hash_backend_is_never_checked():
    """No model answers it, so there is no space for it to be the wrong model of."""
    conn = open_db(":memory:")
    try:
        embedder = build_embedder(Settings(embedding_backend="hash"), conn=conn)
    finally:
        conn.close()

    assert isinstance(embedder, FakeEmbedder)
