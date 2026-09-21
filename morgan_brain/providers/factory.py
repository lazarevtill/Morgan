"""Build the chat client and the embedder from ``Settings``.

The only place that turns configuration into a concrete adapter. Everything above depends on
the adapters' interfaces, not on how they were built.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any, Literal, NamedTuple

import httpx

from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.embedder import Embedder, FakeEmbedder
from morgan_brain.providers.embeddings import (
    BOUND_GRACE_SECONDS,
    OpenAICompatEmbedder,
    RetryBudget,
)
from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.wire import is_refusal

#: Which retry budget an embedding call is given: a command's or tool call's, where someone
#: is waiting on the answer, or an import's, which has thousands of calls to make.
Budget = Literal["interactive", "import"]


class Endpoint(NamedTuple):
    """A model endpoint, and the variable that addresses it."""

    url: str
    setting: str


class Key(NamedTuple):
    """A key an endpoint's requests carry, and the setting that names it on a 401 or 403."""

    api_key: str | None
    setting: str


def chat_endpoint_of(settings: Settings) -> Endpoint:
    return Endpoint(settings.llm_endpoint, "MORGAN_LLM_ENDPOINT")


def embedding_endpoint_of(settings: Settings) -> Endpoint:
    """Where embeddings are sent: their own endpoint when one is configured, otherwise the
    chat endpoint, which then serves both."""
    if settings.embedding_endpoint:
        return Endpoint(settings.embedding_endpoint, "MORGAN_EMBEDDING_ENDPOINT")
    return chat_endpoint_of(settings)


def embedding_key_of(settings: Settings) -> Key:
    """The key embedding requests carry, and the setting a refusal names for it.

    A separate ``MORGAN_EMBEDDING_ENDPOINT`` gets its own key, ``MORGAN_EMBEDDING_API_KEY``, so
    the owner's chat credential never reaches a host that never needed it -- and never appears
    in that host's logs. Without one, the request goes to the chat host and carries its key,
    ``MORGAN_LLM_API_KEY``, as it must: there is no second host to give a key to. Follows
    ``embedding_endpoint_of(settings).setting`` so this one rule covers both the embedder and
    ``check_embeddings_reachable``. ``api_key`` is ``None`` when empty, so an empty key sends
    no ``Authorization`` header.
    """
    if embedding_endpoint_of(settings).setting == "MORGAN_EMBEDDING_ENDPOINT":
        return Key(settings.embedding_api_key or None, "MORGAN_EMBEDDING_API_KEY")
    return Key(settings.llm_api_key or None, "MORGAN_LLM_API_KEY")


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
    headers: dict[str, str] | None = None,
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

    *headers* rides on every request the returned embedder sends, on top of the key above --
    ``doctor --vectors --clients N`` is the one caller today, tagging each of its concurrent
    clients so a disagreement can be told apart from another client's answer.
    """
    if settings.embedding_backend == "hash":
        return FakeEmbedder(dim=settings.embedding_dim)
    endpoint = embedding_endpoint_of(settings)
    key = embedding_key_of(settings)
    inner = OpenAICompatEmbedder(
        endpoint.url,
        settings.embedding_model,
        budget=retry_budget_of(settings, budget),
        setting=endpoint.setting,
        key_setting=key.setting,
        api_key=key.api_key,
        headers=headers,
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


#: The setting a probe that got no answer in time names.
_PROBE_TIMEOUT_SETTING = "MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS"


class Probe(NamedTuple):
    """What one of ``morgan doctor``'s requests found.

    ``seconds`` is how long the server took to answer, or to fail, counted from the request
    (not from building the HTTP client). ``status`` is the HTTP status it answered with, and
    ``None`` when it gave no answer: refused, unresolved, or silent past the timeout. ``error``
    says what went wrong, ``None`` when nothing did: an exception's class name, an HTTP status,
    and the setting to check -- never a server's own words, which may quote back the key the
    request carried. A 2xx with an ``error`` is an answer that was not what was asked for: an
    embedding probe answered with something other than embeddings. ``vectors`` is an embedding
    probe's answer, one per input, else ``None``.
    """

    seconds: float
    status: int | None
    error: str | None
    vectors: list[list[float]] | None = None


async def _probe(
    method: str,
    url: str,
    *,
    api_key: str | None,
    # ASYNC109 wants a cancel scope instead of a timeout parameter; the value goes to both
    # httpx and asyncio.timeout below, which is that scope.
    timeout: float,  # noqa: ASYNC109
    setting: str,
    key_setting: str,
    body: dict[str, Any] | None = None,
    hints: dict[int, str] | None = None,
) -> tuple[Probe, httpx.Response | None]:
    """One request, timed, and the answer it got. Never raises.

    *setting* is the variable that addresses *url* and *key_setting* the one whose value is
    sent as *api_key*; an error names the one to check. An answer that refuses the request
    (``is_refusal``) names *key_setting* on a 401 or 403 and *setting* on any other status,
    with what *hints* says that status means; one a retry may mend, a 429 or another 5xx, is
    named by its status alone.

    No connection within *timeout* (``ConnectTimeout``) is a host that is off or an address
    that is wrong, and names *setting*; a connection that gave no answer within it names the
    timeout setting. httpx's own timeouts end a stalled connect or read and say which; the
    overall bound comes ``BOUND_GRACE_SECONDS`` later, as the embedder's does, so it catches
    only an answer that trickles in a byte at a time.

    The clock starts once the client is built: building it loads a TLS context, which is no
    part of the server's answer. Building it can fail on its own -- a CA bundle that is not
    there -- which is this machine's problem, not either server's, and is said so with no
    endpoint setting named.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.monotonic()
    try:
        client = httpx.AsyncClient()
    except Exception as exc:  # noqa: BLE001 -- a local failure is a finding, not a crash
        error = f"the HTTP client could not be built ({type(exc).__name__})"
        return Probe(time.monotonic() - started, None, error), None
    async with client:
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout + BOUND_GRACE_SECONDS):
                resp = await client.request(
                    method, url, headers=headers, json=body, timeout=timeout
                )
        except httpx.ConnectTimeout:
            error = f"ConnectTimeout; check {setting}"
            return Probe(time.monotonic() - started, None, error), None
        except (httpx.TimeoutException, TimeoutError):
            error = f"no answer within {timeout:g} s ({_PROBE_TIMEOUT_SETTING})"
            return Probe(time.monotonic() - started, None, error), None
        except Exception as exc:  # noqa: BLE001 -- no answer is a normal finding, not an error
            error = f"{type(exc).__name__}; check {setting}"
            return Probe(time.monotonic() - started, None, error), None
    seconds = time.monotonic() - started
    status = resp.status_code
    if 200 <= status < 300:
        return Probe(seconds, status, None), resp
    if not is_refusal(status):
        return Probe(seconds, status, f"HTTP {status}"), None
    means = f" ({hints[status]})" if hints and status in hints else ""
    check = key_setting if status in (401, 403) else setting
    return Probe(seconds, status, f"HTTP {status}{means}; check {check}"), None


async def check_llm_reachable(settings: Settings) -> Probe:
    """``morgan doctor``'s chat probe: GET the ``/models`` listing, which llama-server, vLLM
    and Ollama's ``/v1`` shim all serve, within ``MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS``.
    Never raises."""
    endpoint = chat_endpoint_of(settings)
    probe, _ = await _probe(
        "GET",
        endpoint.url.rstrip("/") + "/models",
        api_key=settings.llm_api_key or None,
        timeout=settings.doctor_probe_timeout_seconds,
        setting=endpoint.setting,
        key_setting="MORGAN_LLM_API_KEY",
    )
    return probe


async def check_embeddings_reachable(settings: Settings) -> Probe:
    """``morgan doctor``'s embedding probe: embed the five fingerprint strings, in one request,
    where embeddings are sent, within ``MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS``. Never raises.

    Not the ``/models`` listing: a chat server started without embeddings lists its models and
    answers every embedding request with a 501. The five strings rather than one word, because
    their vectors are what the active space's fingerprint was recorded from: the one request
    says both whether embeddings are served and whether the model serving them is the one that
    wrote the stored vectors. One attempt, no retry: how long the one answer took is the
    finding.

    Carries the same key ``build_embedder`` would send (``embedding_key_of``): the chat key
    only when embeddings go to the chat host, its own key when they go to a separate one.
    """
    endpoint = embedding_endpoint_of(settings)
    key = embedding_key_of(settings)
    texts = list(fingerprint.STRINGS)
    probe, resp = await _probe(
        "POST",
        endpoint.url.rstrip("/") + "/embeddings",
        api_key=key.api_key,
        timeout=settings.doctor_probe_timeout_seconds,
        setting=endpoint.setting,
        key_setting=key.setting,
        body={"model": settings.embedding_model, "input": texts},
        hints={501: "the server does not serve embeddings"},
    )
    if resp is None:
        return probe
    # A 200 that is not an embeddings response -- a proxy's HTML page, say -- is the wrong
    # server at that address: named by the endpoint setting, never by what the page said.
    try:
        # Each item names the input it embeds; the order of the list is not promised.
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in data]
    except Exception as exc:  # noqa: BLE001 -- a malformed answer is a finding, not a crash
        return probe._replace(error=_not_embeddings(type(exc).__name__, endpoint.setting))
    if len(vectors) != len(texts):
        found = f"{len(vectors)} vectors for {len(texts)} inputs"
        return probe._replace(error=_not_embeddings(found, endpoint.setting))
    return probe._replace(vectors=vectors)


def _not_embeddings(why: str, setting: str) -> str:
    return f"HTTP 200, not an embeddings response ({why}); check {setting}"
