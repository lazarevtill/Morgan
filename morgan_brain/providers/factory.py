"""Build the chat client and the embedder from ``Settings``.

The only place that turns configuration into a concrete adapter. Everything above depends on
the adapters' interfaces, not on how they were built.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from morgan_brain.config import Settings
from morgan_brain.memory.embedder import Embedder, FakeEmbedder
from morgan_brain.providers.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.openai_compat import OpenAICompatAdapter


class Endpoint(NamedTuple):
    """A model endpoint, and the variable that addresses it."""

    url: str
    setting: str


def chat_endpoint_of(settings: Settings) -> Endpoint:
    return Endpoint(settings.llm_endpoint, "MORGAN_LLM_ENDPOINT")


def embedding_endpoint_of(settings: Settings) -> Endpoint:
    """Where embeddings are sent: their own endpoint when one is configured, otherwise the
    chat endpoint, which then serves both."""
    if settings.embedding_endpoint:
        return Endpoint(settings.embedding_endpoint, "MORGAN_EMBEDDING_ENDPOINT")
    return chat_endpoint_of(settings)


def build_chat_client(settings: Settings) -> OpenAICompatAdapter:
    endpoint = chat_endpoint_of(settings)
    return OpenAICompatAdapter(
        base_url=endpoint.url,
        # llama-server without --api-key still needs SOME non-empty string for the SDK client.
        api_key=settings.llm_api_key or "llamacpp",
        provider="llamacpp",
        timeout=settings.llm_timeout_seconds,
        setting=endpoint.setting,
    )


def build_embedder(settings: Settings) -> Embedder:
    """The single decision between the live embedding endpoint and the deterministic stub.

    The stub reuses ``FakeEmbedder``: sha256 is stable across processes regardless of
    ``PYTHONHASHSEED``, which the CLI (a subprocess per command) and the store need in order
    to agree on vectors for the same text.
    """
    if settings.embedding_backend == "hash":
        return FakeEmbedder(dim=settings.embedding_dim)
    endpoint = embedding_endpoint_of(settings)
    return OpenAICompatEmbedder(
        endpoint.url,
        settings.embedding_model,
        timeout=settings.llm_timeout_seconds,
        api_key=settings.llm_api_key or None,
        setting=endpoint.setting,
    )


async def _answers(
    method: str,
    url: str,
    settings: Settings,
    *,
    # ASYNC109 wants a cancel scope instead of a timeout parameter. That is trio/anyio
    # advice; here the value goes straight to httpx, which is how asyncio expresses it.
    timeout: float,  # noqa: ASYNC109
    body: dict[str, Any] | None = None,
) -> bool:
    """Whether ``url`` answers without a server error. Never raises."""
    import httpx

    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, headers=headers, json=body)
    except Exception:  # noqa: BLE001 -- unreachable is a normal answer, not an error to surface
        return False
    else:
        return resp.status_code < 500


async def check_llm_reachable(settings: Settings, *, timeout: float = 5.0) -> bool:  # noqa: ASYNC109
    """Best-effort reachability check for ``morgan doctor``: GET the ``/models`` listing,
    which llama-server, vLLM and Ollama's ``/v1`` shim all serve. Never raises."""
    url = chat_endpoint_of(settings).url.rstrip("/") + "/models"
    return await _answers("GET", url, settings, timeout=timeout)


async def check_embeddings_reachable(settings: Settings, *, timeout: float = 5.0) -> bool:  # noqa: ASYNC109
    """Best-effort check for ``morgan doctor`` that embeddings are served where they are sent:
    embed one word. Not the ``/models`` listing: a chat server started without embeddings
    lists its models and answers every embedding request with a 501. Never raises."""
    url = embedding_endpoint_of(settings).url.rstrip("/") + "/embeddings"
    body = {"model": settings.embedding_model, "input": "probe"}
    return await _answers("POST", url, settings, timeout=timeout, body=body)
