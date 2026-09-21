"""Build the chat client and the embedder from ``Settings``.

The only place that turns configuration into a concrete adapter. Everything above depends on
the adapters' interfaces, not on how they were built.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Literal, NamedTuple

from morgan_brain.config import Settings
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.embedder import Embedder, FakeEmbedder
from morgan_brain.providers.embeddings import OpenAICompatEmbedder, RetryBudget
from morgan_brain.providers.openai_compat import OpenAICompatAdapter

#: Which retry budget an embedding call is given: a command's or tool call's, where someone
#: is waiting on the answer, or an import's, which has thousands of calls to make.
Budget = Literal["interactive", "import"]

#: The setting whose value is sent as the key to the model server. Embeddings carry the chat
#: key wherever they are sent, so a refused key is this one whichever endpoint refused it.
_KEY_SETTING = "MORGAN_LLM_API_KEY"


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


def build_embedder(
    settings: Settings,
    *,
    conn: sqlite3.Connection | None = None,
    budget: Budget = "interactive",
) -> Embedder:
    """The single decision between the live embedding endpoint and the deterministic stub.

    The stub reuses ``FakeEmbedder``: sha256 is stable across processes regardless of
    ``PYTHONHASHSEED``, which the CLI (a subprocess per command) and the store need in order
    to agree on vectors for the same text. No model answers it, so it is never checked.

    Given *conn*, the database the vectors are stored in, the live embedder is wrapped in
    ``CheckedEmbedder``: the process's first request also proves the model answering is the
    one that wrote that database's active space. ``build_memory_context`` always passes it.
    Without one -- a measurement run over a bare file -- the model is unchecked.

    *budget* names how long a failing call keeps retrying (``retry_budget_of``). The retry is
    the live adapter's own, below the check, so a retried first request still carries the
    fingerprint strings once. ``MORGAN_LLM_TIMEOUT_SECONDS`` is the chat model's; embeddings
    have ``MORGAN_EMBEDDING_TIMEOUT_SECONDS`` per attempt, inside the budget.
    """
    if settings.embedding_backend == "hash":
        return FakeEmbedder(dim=settings.embedding_dim)
    endpoint = embedding_endpoint_of(settings)
    inner = OpenAICompatEmbedder(
        endpoint.url,
        settings.embedding_model,
        budget=retry_budget_of(settings, budget),
        setting=endpoint.setting,
        key_setting=_KEY_SETTING,
        api_key=settings.llm_api_key or None,
    )
    if conn is None:
        return inner
    return CheckedEmbedder(
        inner, conn=conn, settings=settings, endpoint=endpoint.url, setting=endpoint.setting
    )


def retry_budget_of(settings: Settings, budget: Budget) -> RetryBudget:
    """The settings' retry budget for an ``interactive`` call or an ``import``: they differ
    only in how long a slow host is waited on."""
    return RetryBudget(
        seconds=(
            settings.embedding_import_retry_budget_seconds
            if budget == "import"
            else settings.embedding_retry_budget_seconds
        ),
        unreachable_seconds=settings.embedding_unreachable_budget_seconds,
        backoff_seconds=settings.embedding_retry_backoff_seconds,
        backoff_cap_seconds=settings.embedding_retry_backoff_max_seconds,
        attempt_seconds=settings.embedding_timeout_seconds,
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
