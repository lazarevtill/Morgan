"""The budget bounds the wait, and which budget applies is the caller's to say.

A host that accepts the connection and never answers used to be waited on for the chat
timeout (``MORGAN_LLM_TIMEOUT_SECONDS``, 120 s), once. Now each attempt is given the lesser of
``MORGAN_EMBEDDING_TIMEOUT_SECONDS`` and what remains of the budget, so the call ends inside the
budget: ``MORGAN_EMBEDDING_RETRY_BUDGET_SECONDS`` for a command or a tool call, the larger
``MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS`` for an import.
"""

from __future__ import annotations

import re
import time

import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderUnreachable
from tests.fakes import silent_model_server


async def test_a_silent_host_fails_within_the_interactive_budget_not_the_chat_timeout():
    settings = _settings(retry_budget=0.6, attempt_timeout=50.0, llm_timeout=120.0)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable) as exc:
            await build_embedder(_at(settings, url)).embed("x")
        elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert "answered too slowly or dropped" in str(exc.value)
    assert "unreachable" not in str(exc.value)


async def test_the_chat_timeout_no_longer_reaches_the_embedder():
    """A chat cap shorter than the budget would end the first attempt early."""
    settings = _settings(retry_budget=0.6, attempt_timeout=50.0, llm_timeout=0.1)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable) as exc:
            await build_embedder(_at(settings, url)).embed("x")
        elapsed = time.monotonic() - started
    assert elapsed >= 0.5
    assert _attempts(exc.value) == 1


async def test_each_attempt_is_given_at_most_the_attempt_timeout():
    settings = _settings(retry_budget=0.8, attempt_timeout=0.2, backoff=0.05)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_at(settings, url)).embed("x")
    assert _attempts(exc.value) >= 2


async def test_an_import_waits_longer_than_a_command():
    settings = _settings(retry_budget=0.3, import_budget=0.9, attempt_timeout=50.0)
    with silent_model_server() as url:
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url)).embed("x")
        interactive = time.monotonic() - started

        started = time.monotonic()
        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url), budget="import").embed("x")
        imported = time.monotonic() - started

    assert interactive < 0.8
    assert 0.8 <= imported < 2.5


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
