"""Build the chat client and the embedder from ``Settings``.

The only place that turns configuration into a concrete adapter. Everything above depends on
the adapters' interfaces, not on how they were built.
"""

from __future__ import annotations

from morgan_brain.config import Settings
from morgan_brain.memory.embedder import Embedder, FakeEmbedder
from morgan_brain.providers.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.openai_compat import OpenAICompatAdapter


def build_chat_client(settings: Settings) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        base_url=settings.llm_endpoint,
        # llama-server without --api-key still needs SOME non-empty string for the SDK client.
        api_key=settings.llm_api_key or "llamacpp",
        provider="llamacpp",
        timeout=settings.llm_timeout_seconds,
    )


def build_embedder(settings: Settings) -> Embedder:
    """The single decision between the live embedding endpoint and the deterministic stub.

    The stub reuses ``FakeEmbedder``: sha256 is stable across processes regardless of
    ``PYTHONHASHSEED``, which the CLI (a subprocess per command) and the store need in order
    to agree on vectors for the same text.
    """
    if settings.embedding_backend == "hash":
        return FakeEmbedder(dim=settings.embedding_dim)
    return OpenAICompatEmbedder(
        settings.llm_endpoint,
        settings.embedding_model,
        timeout=settings.llm_timeout_seconds,
        api_key=settings.llm_api_key or None,
    )


async def check_llm_reachable(
    settings: Settings,
    *,
    # ASYNC109 wants a cancel scope instead of a timeout parameter. That is trio/anyio
    # advice; here the value goes straight to httpx, which is how asyncio expresses it.
    timeout: float = 5.0,  # noqa: ASYNC109
) -> bool:
    """Best-effort reachability check for ``morgan doctor``: GET the ``/models`` listing,
    which llama-server, vLLM and Ollama's ``/v1`` shim all serve. Never raises."""
    import httpx

    url = settings.llm_endpoint.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
    except Exception:  # noqa: BLE001 -- unreachable is a normal answer, not an error to surface
        return False
    else:
        return resp.status_code < 500
