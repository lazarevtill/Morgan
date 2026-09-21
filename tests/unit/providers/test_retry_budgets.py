"""The budget bounds the wait, and which budget applies is the caller's to say.

A host that accepts the connection and never answers used to be waited on for the chat
timeout (``MORGAN_LLM_TIMEOUT_SECONDS``, 120 s), once. Now each attempt is given the lesser of
``MORGAN_EMBEDDING_TIMEOUT_SECONDS`` and what remains of the budget, so the call ends inside the
budget: ``MORGAN_EMBEDDING_RETRY_BUDGET_SECONDS`` for a command or a tool call, the larger
``MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS`` for an import.

The budgets here are a second or more: the call's budget counts building its HTTP client, which
loads a TLS context and takes 0.2-0.5 s on a loaded Windows machine.
"""

from __future__ import annotations

import re
import time

import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderUnreachable
from tests.fakes import model_server, silent_model_server


async def test_a_silent_host_fails_within_the_interactive_budget_not_the_chat_timeout():
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0, llm_timeout=120.0)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable) as exc:
            await build_embedder(_at(settings, url)).embed("x")
        elapsed = time.monotonic() - started
    assert elapsed < 2.5
    # Named as the spec names it once a connection was made, never by Python's builtin.
    assert "answered too slowly or dropped: ReadTimeout after " in str(exc.value)
    assert "TimeoutError" not in str(exc.value)
    assert "unreachable" not in str(exc.value)


async def test_an_answer_that_trickles_in_still_ends_inside_the_budget():
    """httpx's read timeout restarts with every byte, so a body that arrives a byte at a time
    never trips it; the bound on the whole attempt does."""
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0)
    with silent_model_server(trickle_every=0.05) as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable) as exc:
            await build_embedder(_at(settings, url)).embed("x")
        elapsed = time.monotonic() - started
    assert elapsed < 2.2
    assert "answered too slowly or dropped: ReadTimeout after " in str(exc.value)


async def test_the_chat_timeout_no_longer_reaches_the_embedder():
    """A chat cap shorter than the budget would end the first attempt early."""
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0, llm_timeout=0.1)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable) as exc:
            await build_embedder(_at(settings, url)).embed("x")
        elapsed = time.monotonic() - started
    assert elapsed >= 1.1
    assert _attempts(exc.value) == 1


async def test_each_attempt_is_given_at_most_the_attempt_timeout():
    settings = _settings(retry_budget=1.5, attempt_timeout=0.2, backoff=0.05)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_at(settings, url)).embed("x")
    assert _attempts(exc.value) >= 2


async def test_an_import_waits_longer_than_a_command():
    settings = _settings(retry_budget=0.8, import_budget=2.0, attempt_timeout=50.0)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url)).embed("x")
        interactive = time.monotonic() - started

        started = time.monotonic()
        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url), budget="import").embed("x")
        imported = time.monotonic() - started

    assert interactive < 1.5
    assert 1.9 <= imported < 3.5


async def test_an_attempt_is_never_given_less_than_the_first_wait():
    """A budget spent before the first attempt starts -- here, smaller than building the HTTP
    client takes -- still leaves that attempt the first wait's length, the least any attempt
    is given. Without it the attempt had no time at all, and a server that answers was called
    unreachable."""
    settings = _settings(retry_budget=0.01, attempt_timeout=50.0, backoff=1.0)
    with model_server() as url:
        vector = await build_embedder(_at(settings, url)).embed("x")
    assert len(vector) == 1024


def _settings(
    *,
    retry_budget: float,
    attempt_timeout: float,
    import_budget: float = 600.0,
    llm_timeout: float = 120.0,
    backoff: float = 0.5,
) -> Settings:
    return Settings(
        embedding_backend="provider",
        embedding_retry_budget_seconds=retry_budget,
        embedding_import_retry_budget_seconds=import_budget,
        embedding_timeout_seconds=attempt_timeout,
        embedding_retry_backoff_seconds=backoff,
        llm_timeout_seconds=llm_timeout,
    )


def _at(settings: Settings, url: str) -> Settings:
    return settings.model_copy(update={"embedding_endpoint": url})


def _attempts(exc: ProviderUnreachable) -> int:
    found = re.search(r"after (\d+) attempts? over", str(exc))
    assert found, str(exc)
    return int(found.group(1))
