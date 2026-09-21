"""A transient failure is retried; a host that is simply off is not waited on.

The difference is whether a connection was made. A cold host that accepts the connection and
then takes seconds to load its model -- 43 s on a first load from disk -- is slow, not
unreachable, and calling it unreachable is what sent the owner to check a server that was
working.
"""

from __future__ import annotations

import re
import time

import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderRefused, ProviderUnreachable
from tests.fakes import Calls, flaky_model_server, silent_model_server

_CLOSED = "http://127.0.0.1:1/v1"


async def test_two_failures_then_an_answer_succeeds(tmp_path):
    with flaky_model_server(fail_times=2, status=502) as url:
        vector = await build_embedder(_settings(url)).embed("x")
    assert len(vector) == 1024


@pytest.mark.parametrize("status", [429, 500, 503, None])
async def test_every_transient_failure_is_retried(status):
    """429 and 5xx are a busy or loading host; ``None`` is a connection dropped mid-request."""
    calls = Calls()
    with flaky_model_server(fail_times=1, status=status, calls=calls) as url:
        vector = await build_embedder(_settings(url)).embed("x")
    assert len(vector) == 1024
    assert calls.total == 2


async def test_a_five_hundred_is_never_a_bare_http_error(tmp_path):
    with (
        flaky_model_server(fail_times=99, status=503) as url,
        pytest.raises(ProviderUnreachable) as exc,
    ):
        await build_embedder(_settings(url, retry_budget=1.0)).embed("x")
    assert "attempts" in str(exc.value) and "MORGAN_EMBEDDING_ENDPOINT" in str(exc.value)
    # A host that answered is never called unreachable.
    assert "answered too slowly or dropped: " in str(exc.value)
    assert "unreachable" not in str(exc.value)
    assert exc.value.outcome == "slow"


async def test_exhaustion_names_the_last_error_and_the_attempts():
    """The backoff leaves the second attempt most of a second: an attempt started at the very
    end of the budget is cut short by it, and would name that cut, not the status."""
    with (
        flaky_model_server(fail_times=99, status=503) as url,
        pytest.raises(ProviderUnreachable) as exc,
    ):
        await build_embedder(_settings(url, retry_budget=1.0, backoff=0.4)).embed("x")
    assert re.search(
        r"answered too slowly or dropped: HTTP 503 after \d+ attempts? over ", str(exc.value)
    )


async def test_a_refused_connection_spends_only_the_unreachable_budget():
    embedder = build_embedder(_settings(_CLOSED, unreachable_budget=1.0, retry_budget=60.0))
    started = time.monotonic()
    with pytest.raises(ProviderUnreachable, match="unreachable"):
        await embedder.embed("x")
    assert time.monotonic() - started < 5.0


async def test_an_unreachable_host_says_so_by_class():
    with pytest.raises(ProviderUnreachable) as exc:
        await build_embedder(_settings(_CLOSED, unreachable_budget=0.5)).embed("x")
    assert "is unreachable: Connect" in str(exc.value)
    assert exc.value.outcome == "unreachable"


async def test_a_four_hundred_is_not_retried():
    calls = Calls()
    with (
        flaky_model_server(fail_times=99, status=400, calls=calls) as url,
        pytest.raises(ProviderRefused) as exc,
    ):
        await build_embedder(_settings(url)).embed("x")
    assert calls.total == 1
    assert exc.value.status == 400
    assert "HTTP 400" in str(exc.value)
    assert "MORGAN_EMBEDDING_ENDPOINT" in str(exc.value)


@pytest.mark.parametrize("status", [401, 403])
async def test_a_four_hundred_is_not_retried_and_a_key_error_names_its_setting(status):
    """The key is named by the setting whose value is sent. Until the embedding host has a key
    of its own, that is the chat key, whichever endpoint embeddings go to."""
    calls = Calls()
    with (
        flaky_model_server(fail_times=99, status=status, calls=calls) as url,
        pytest.raises(ProviderRefused) as exc,
    ):
        await build_embedder(_settings(url)).embed("x")
    assert calls.total == 1
    assert "MORGAN_LLM_API_KEY" in str(exc.value)


async def test_a_host_that_never_answers_fails_inside_the_budget():
    """The point of the budget: not a count of attempts, a bound on the wait."""
    with silent_model_server() as url:  # accepts the connection, never replies
        started = time.monotonic()
        with pytest.raises(ProviderUnreachable, match="too slowly"):
            await build_embedder(_settings(url, retry_budget=3.0, attempt_timeout=50.0)).embed("x")
        assert time.monotonic() - started < 6.0


def _settings(
    url: str,
    *,
    retry_budget: float = 10.0,
    unreachable_budget: float = 1.0,
    attempt_timeout: float = 5.0,
    backoff: float = 0.05,
) -> Settings:
    """The live backend, with embeddings sent to *url* and budgets small enough for a suite."""
    return Settings(
        embedding_backend="provider",
        embedding_endpoint=url,
        embedding_retry_budget_seconds=retry_budget,
        embedding_unreachable_budget_seconds=unreachable_budget,
        embedding_timeout_seconds=attempt_timeout,
        embedding_retry_backoff_seconds=backoff,
    )
