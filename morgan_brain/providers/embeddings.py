"""The live embedding adapter — an OpenAI-compatible ``/embeddings`` client.

This lives in ``providers/adapters/`` because that is where the invariant puts it: nothing
above the provider layer talks to a model endpoint. It previously sat in
``modules/memory/indexing/embedder.py`` under the name ``OllamaEmbedder``, holding a
module-level ``import httpx`` two layers above the seam, while a correctly-placed
``OpenAICompatEmbedder`` in ``openai_compat.py`` went uncalled from the day it was written.
One of them had to go; the one that shipped is the one that stayed.

``providers/factory.py::build_embedder`` is the only caller — the single decision point
between this and the deterministic hash stub.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx

from morgan_brain.providers.wire import Outcome, ProviderRefused, ProviderUnreachable

#: How much of a refusing server's answer its error quotes, once the key is redacted.
_QUOTED_CHARS = 200

#: A bearer token anywhere in a server's answer (RFC 6750's token characters).
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*")

#: What stands in a quoted answer for a key.
_REDACTED = "[redacted]"

#: What a 501 means to an embedding request: the server has no embedding model to answer with.
_NO_EMBEDDINGS = (
    "the server does not serve embeddings, as a llama-server started without --embeddings "
    "does; point MORGAN_EMBEDDING_ENDPOINT at one that does"
)

#: How much longer than its own timeouts an attempt's overall bound waits. httpx's connect
#: and read timeouts fire first when a connect or the answer stalls, so the failure is named
#: for what stalled; the bound catches only an answer that keeps arriving a byte at a time,
#: which restarts httpx's read timeout with every byte.
_BOUND_GRACE_SECONDS = 0.25


@dataclass(frozen=True)
class RetryBudget:
    """How long one embedding call may take, by what went wrong. Every value is in seconds.

    ``seconds`` bounds the wall time of the whole call, attempts and backoff included: a
    command's budget or an import's. ``unreachable_seconds`` bounds it, counted from the same
    start, while no connection has been made. The first wait between attempts is
    ``backoff_seconds``, doubling each time up to ``backoff_cap_seconds``; the last wait
    shrinks so that one more attempt still has ``backoff_seconds`` of the budget left. One
    attempt takes at most ``attempt_seconds`` and never longer than what is left of the
    call's budget -- but for an answer that trickles in a byte at a time, which is cut off
    ``_BOUND_GRACE_SECONDS`` later.
    """

    seconds: float
    unreachable_seconds: float
    backoff_seconds: float
    backoff_cap_seconds: float
    attempt_seconds: float


class OpenAICompatEmbedder:
    """Embeddings over any OpenAI-compatible ``/embeddings`` endpoint.

    Works against llama-server (the default), Ollama's ``/v1`` shim, vLLM, or a hosted
    provider — the wire format is the same and only the endpoint URL differs.

    Implements the memory layer's ``Embedder`` protocol (``embed`` / ``embed_batch``), not the
    batch-only ``aembed`` shape of the deleted duplicate: the memory store embeds one memory at
    a time on the write path and a list at index-rebuild time.

    A failure that may pass is retried within *budget*, and the error raised once it is spent
    says which kind it was (``ProviderUnreachable.outcome``). **Unreachable** -- no connection
    was made (``ConnectError``, ``ConnectTimeout``, a DNS failure) -- is given
    ``budget.unreachable_seconds``: a host that is off does not start answering. **Slow** -- a
    connection was made, and the answer did not come in time, came as a 5xx or a 429, or was
    cut off (``ReadTimeout``, ``ReadError``, ``RemoteProtocolError``) -- is given
    ``budget.seconds``: a cold host loading its model looks exactly like this. Once an attempt
    of a call has connected, a later connect failure is the same host flapping, and counts as
    slow. Any other 4xx, a 501 (no embeddings are served there) or a redirect is
    ``ProviderRefused`` at once: the same request gets the same answer. The error once a budget
    is spent names the last failure that said what went wrong -- a timeout says only that time
    ran out, so it is named only when every failure was one.

    Args:
        endpoint:    Base URL of the OpenAI-compatible endpoint.
        model:       Embedding model name.
        budget:      How long a call may retry, and how long each attempt may take.
        setting:     The variable that addresses ``endpoint``, named when it fails.
        key_setting: The variable whose value is sent as ``api_key``, named on a 401 or 403.
        api_key:     Outbound bearer token, if the endpoint enforces one (llama-server's
                     ``--api-key``). ``None``/empty sends no ``Authorization`` header. This is
                     ``MORGAN_LLM_API_KEY``, never ``MORGAN_API_KEY`` — opposite directions.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        budget: RetryBudget,
        setting: str,
        key_setting: str,
        api_key: str | None = None,
    ) -> None:
        self._url = endpoint.rstrip("/") + "/embeddings"
        self._model = model
        self._budget = budget
        self._api_key = api_key or ""
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._setting = setting
        self._key_setting = key_setting

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # One client for every attempt of the call, built outside any attempt's timeout:
        # building one loads a TLS context, a fifth of a second on Windows, which is no part of
        # waiting for an answer. The call's budget does count it, from *started*.
        started = time.monotonic()
        async with httpx.AsyncClient() as client:
            return await self._retried(client, texts, started=started)

    async def _retried(
        self, client: httpx.AsyncClient, texts: list[str], *, started: float
    ) -> list[list[float]]:
        """Attempt the request until it succeeds, is refused, or the budget for what went
        wrong is spent. *started* is when the call began: the budget counts from there."""
        budget = self._budget
        connected = False
        backoff = budget.backoff_seconds
        attempts = 0
        named: Exception | None = None
        while True:
            attempts += 1
            elapsed = time.monotonic() - started
            timeout = min(budget.attempt_seconds, budget.seconds - elapsed)
            connect = timeout if connected else min(timeout, budget.unreachable_seconds - elapsed)
            outcome: Outcome
            error: Exception
            try:
                return await self._attempt(client, texts, timeout=timeout, connect_timeout=connect)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                outcome, error = ("slow" if connected else "unreachable"), exc
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                TimeoutError,
            ) as exc:
                connected = True
                outcome, error = "slow", exc
            except httpx.HTTPStatusError as exc:
                connected = True
                status = exc.response.status_code
                if status == 501 or (status != 429 and status < 500):
                    raise self._refused(exc.response) from exc
                outcome, error = "slow", exc
            except httpx.TransportError as exc:
                # An address httpx cannot send to at all (no scheme, a malformed proxy): no
                # retry changes that, and the setting that holds it is what to check.
                raise ProviderUnreachable(
                    self._url, f"{type(exc).__name__}: {exc}", setting=self._setting
                ) from exc

            if named is None or not _timed_out(error) or _timed_out(named):
                named = error
            elapsed = time.monotonic() - started
            allowed = budget.seconds
            if outcome == "unreachable":
                allowed = min(allowed, budget.unreachable_seconds)
            # One more attempt is made while it can still be given the first wait's length of
            # the budget; the wait before it shrinks to leave it that. A backoff that ran past
            # the budget gave up at about half of it, on a host about to answer.
            room = allowed - elapsed - budget.backoff_seconds
            if room <= 0:
                raise ProviderUnreachable.retried(
                    self._url,
                    setting=self._setting,
                    outcome=outcome,
                    error=_name(named),
                    said=_said(named),
                    attempts=attempts,
                    seconds=elapsed,
                ) from error
            await asyncio.sleep(min(backoff, room))
            backoff = min(backoff * 2, budget.backoff_cap_seconds)

    async def _attempt(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
        *,
        # ASYNC109 wants a cancel scope instead of a timeout parameter; the value goes to both
        # httpx and asyncio.timeout below, which is that scope.
        timeout: float,  # noqa: ASYNC109
        connect_timeout: float,
    ) -> list[list[float]]:
        """One request. httpx's timeouts end a stalled connect or read; ``asyncio.timeout``
        bounds the whole attempt, a grace later, because httpx's read timeout restarts with
        every chunk."""
        async with asyncio.timeout(timeout + _BOUND_GRACE_SECONDS):
            resp = await client.post(
                self._url,
                json={"model": self._model, "input": texts},
                headers=self._headers,
                timeout=httpx.Timeout(timeout, connect=connect_timeout),
            )
            resp.raise_for_status()
        # Each item names the input it embeds; the order of the list is not promised. A
        # caller that sends several texts at once -- the first request a process sends
        # carries the fingerprint strings behind the caller's own -- splits them by position.
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def _refused(self, response: httpx.Response) -> ProviderRefused:
        status = response.status_code
        setting = self._key_setting if status in (401, 403) else self._setting
        hint = _NO_EMBEDDINGS if status == 501 else ""
        return ProviderRefused(
            self._url, status, setting, detail=self._quote(response.text), hint=hint
        )

    def _quote(self, answer: str) -> str:
        """The start of a server's *answer*, fit to print: the key this adapter sent, and any
        bearer token, redacted before it is cut, so a gateway that echoes a credential back
        never puts it into a message, a log or a transcript."""
        if self._api_key:
            # As sent, and as a JSON string would escape it.
            for form in {self._api_key, json.dumps(self._api_key)[1:-1]}:
                answer = answer.replace(form, _REDACTED)
        answer = _BEARER.sub(f"Bearer {_REDACTED}", answer)
        return " ".join(answer.split())[:_QUOTED_CHARS]


def _timed_out(error: Exception) -> bool:
    """Whether *error* says only that time ran out: httpx's timeouts and the attempt's bound."""
    return isinstance(error, httpx.TimeoutException | TimeoutError)


def _name(error: Exception) -> str:
    """How a failure is named in its error: ``HTTP 503`` for a status, ``ReadTimeout`` for the
    attempt's bound, else the exception's class, such as ``ConnectError``. The bound fires
    only after httpx's own connect timeout would have, so the connection was made and the
    answer was what did not arrive -- what httpx calls a ``ReadTimeout``."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    if isinstance(error, TimeoutError):
        return "ReadTimeout"
    return type(error).__name__


def _said(error: Exception) -> str:
    """What *error* said, to follow its name: httpx's text for a failed connect or a dropped
    one; nothing for a status, whose name is the whole of it, or a timeout, which is silent."""
    if isinstance(error, httpx.HTTPStatusError):
        return ""
    return " ".join(str(error).split())
