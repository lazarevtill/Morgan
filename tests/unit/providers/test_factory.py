"""The factory's endpoint decisions: chat and embeddings are separately addressable."""

from __future__ import annotations

from morgan_brain.config import Settings
from morgan_brain.providers.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.factory import build_embedder


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
