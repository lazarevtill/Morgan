"""The budget bounds the wait, and which budget applies is the caller's to say.

A host that accepts the connection and never answers used to be waited on for the chat
timeout (``MORGAN_LLM_TIMEOUT_SECONDS``, 120 s), once. Now each attempt is given the lesser of
``MORGAN_EMBEDDING_TIMEOUT_SECONDS`` and what remains of the budget, so the call ends inside the
budget: ``MORGAN_EMBEDDING_RETRY_BUDGET_SECONDS`` for a command or a tool call, the larger
``MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS`` for an import.

Building the call's HTTP client, which loads a TLS context and takes 0.2-0.5 s on a loaded
Windows machine, is no part of waiting for an answer and is not charged to the budget: its
clock starts once the client is ready. The tests that time the budget time it from there.
"""

from __future__ import annotations

import re

import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.embeddings import BOUND_GRACE_SECONDS
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderUnreachable
from tests.fakes import model_server, silent_model_server
from tests.unit.providers.conftest import TOLERANCE


async def test_a_silent_host_fails_within_the_interactive_budget_not_the_chat_timeout(
    client_ready,
):
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0, llm_timeout=120.0)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_at(settings, url)).embed("x")
    assert client_ready.since() < 1.2 + TOLERANCE
    # Named for what happened once a connection was made, never by Python's builtin.
    assert "answered too slowly or dropped: ReadTimeout after " in str(exc.value)
    assert "TimeoutError" not in str(exc.value)
    assert "unreachable" not in str(exc.value)


async def test_an_answer_that_trickles_in_still_ends_inside_the_budget(client_ready):
    """httpx's read timeout restarts with every byte, so a body that arrives a byte at a time
    never trips it; the bound on the whole attempt does, a grace after the budget."""
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0)
    with (
        silent_model_server(trickle_every=0.05) as url,
        pytest.raises(ProviderUnreachable) as exc,
    ):
        await build_embedder(_at(settings, url)).embed("x")
    assert client_ready.since() < 1.2 + BOUND_GRACE_SECONDS + TOLERANCE
    assert "answered too slowly or dropped: ReadTimeout after " in str(exc.value)


async def test_the_chat_timeout_no_longer_reaches_the_embedder(client_ready):
    """A chat cap shorter than the budget would end the first attempt early."""
    settings = _settings(retry_budget=1.2, attempt_timeout=50.0, llm_timeout=0.1)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_at(settings, url)).embed("x")
    assert client_ready.since() >= 1.1
    assert _attempts(exc.value) == 1


async def test_each_attempt_is_given_at_most_the_attempt_timeout():
    settings = _settings(retry_budget=1.5, attempt_timeout=0.2, backoff=0.05)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_at(settings, url)).embed("x")
    assert _attempts(exc.value) >= 2


async def test_an_import_waits_longer_than_a_command(client_ready):
    settings = _settings(retry_budget=0.8, import_budget=2.0, attempt_timeout=50.0)
    with silent_model_server() as url:
        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url)).embed("x")
        interactive = client_ready.since()

        with pytest.raises(ProviderUnreachable):
            await build_embedder(_at(settings, url), budget="import").embed("x")
        imported = client_ready.since()

    assert 0.8 <= interactive < 0.8 + TOLERANCE
    assert 2.0 <= imported < 2.0 + TOLERANCE


async def test_building_the_client_is_not_charged_to_the_budget(client_ready):
    """A client that takes longer to build than the whole budget still leaves the first attempt
    the whole budget: charged for the build, the attempt had no time at all, and a server that
    answers was called unreachable."""
    client_ready.build_seconds = 0.5
    settings = _settings(retry_budget=0.3, attempt_timeout=50.0, backoff=1.0)
    with model_server() as url:
        vector = await build_embedder(_at(settings, url)).embed("x")
    assert len(vector) == 1024


async def test_a_silent_host_fails_within_its_budget_whatever_the_wait(client_ready):
    """The first wait longer than the budget itself -- a legal setting -- neither lengthens an
    attempt nor the call: the budget bounds its wall time."""
    settings = _settings(retry_budget=1.0, attempt_timeout=50.0, backoff=3.0)
    with silent_model_server() as url, pytest.raises(ProviderUnreachable):
        await build_embedder(_at(settings, url)).embed("x")
    assert client_ready.since() < 1.0 + TOLERANCE


async def test_a_closed_port_fails_within_the_unreachable_budget_whatever_the_wait(client_ready):
    settings = _settings(retry_budget=60.0, attempt_timeout=50.0, backoff=2.0)
    settings = settings.model_copy(update={"embedding_unreachable_budget_seconds": 0.5})
    with pytest.raises(ProviderUnreachable, match="unreachable"):
        await build_embedder(_at(settings, "http://127.0.0.1:1/v1")).embed("x")
    assert client_ready.since() < 0.5 + TOLERANCE


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
