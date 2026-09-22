"""``embed.done`` is emitted exactly once per ``embed_batch`` call, on every outcome.

Any measurement of the embedder's availability -- how many calls are degraded or over budget
-- reads this line, not memory: a call that
never logs cannot be told apart from one that never happened. Nothing here may ever be able to
carry the input text or the key -- only their count. The `outcome` field must never say "ok"
for a call that raised, including one this adapter does not classify into slow/unreachable/
refused -- a malformed reply, say -- which is what the fifth outcome, "error", is for.
"""

from __future__ import annotations

import sqlite3

import pytest
from structlog.testing import capture_logs

from morgan_brain.config import Settings
from morgan_brain.memory.store.spaces import EmbeddingSpaceStore
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderRefused, ProviderUnreachable
from tests.fakes import flaky_model_server, model_server, raw_model_server

_CLOSED = "http://127.0.0.1:1/v1"

#: A distinctive marker that must never reach a logged field, proving `inputs` is a count.
_SECRET_TEXT = "top-secret-marker-only-the-server-should-ever-see-9f3c"


def _settings(
    url: str,
    *,
    retry_budget: float = 10.0,
    unreachable_budget: float = 1.0,
    attempt_timeout: float = 5.0,
    backoff: float = 0.05,
    backoff_cap: float = 2.0,
) -> Settings:
    return Settings(
        embedding_backend="provider",
        embedding_endpoint=url,
        embedding_retry_budget_seconds=retry_budget,
        embedding_unreachable_budget_seconds=unreachable_budget,
        embedding_timeout_seconds=attempt_timeout,
        embedding_retry_backoff_seconds=backoff,
        embedding_retry_backoff_max_seconds=backoff_cap,
    )


def _only_embed_done(logs: list[dict[str, object]]) -> dict[str, object]:
    done = [entry for entry in logs if entry["event"] == "embed.done"]
    assert len(done) == 1, f"expected exactly one embed.done, got {len(done)}: {done}"
    return done[0]


def _assert_common_fields(entry: dict[str, object], *, outcome: str) -> None:
    assert entry["outcome"] == outcome
    assert entry["inputs"] == 1
    assert isinstance(entry["attempts"], int) and entry["attempts"] >= 1
    assert isinstance(entry["latency_ms"], int | float) and entry["latency_ms"] >= 0
    # The field carries a count, never the text, and the text never leaks into any other field.
    assert _SECRET_TEXT not in repr(entry)


async def test_a_successful_call_logs_ok_once(capsys):
    with model_server() as url, capture_logs() as logs:
        vector = await build_embedder(_settings(url)).embed(_SECRET_TEXT)
    assert capsys.readouterr().out == ""
    assert len(vector) == 1024
    entry = _only_embed_done(logs)
    _assert_common_fields(entry, outcome="ok")
    assert entry["attempts"] == 1


async def test_slow_exhausted_logs_slow_once(capsys):
    with (
        flaky_model_server(fail_times=99, status=503) as url,
        capture_logs() as logs,
        pytest.raises(ProviderUnreachable) as exc,
    ):
        await build_embedder(_settings(url, retry_budget=1.5)).embed(_SECRET_TEXT)
    assert capsys.readouterr().out == ""
    assert exc.value.outcome == "slow"
    entry = _only_embed_done(logs)
    _assert_common_fields(entry, outcome="slow")
    assert entry["attempts"] > 1


async def test_unreachable_logs_unreachable_once(capsys):
    with (
        capture_logs() as logs,
        pytest.raises(ProviderUnreachable) as exc,
    ):
        await build_embedder(_settings(_CLOSED, unreachable_budget=1.0)).embed(_SECRET_TEXT)
    assert capsys.readouterr().out == ""
    assert exc.value.outcome == "unreachable"
    entry = _only_embed_done(logs)
    _assert_common_fields(entry, outcome="unreachable")


async def test_refused_logs_refused_once(capsys):
    with (
        flaky_model_server(fail_times=99, status=400) as url,
        capture_logs() as logs,
        pytest.raises(ProviderRefused),
    ):
        await build_embedder(_settings(url)).embed(_SECRET_TEXT)
    assert capsys.readouterr().out == ""
    entry = _only_embed_done(logs)
    _assert_common_fields(entry, outcome="refused")
    assert entry["attempts"] == 1


async def test_a_batch_call_logs_the_count_never_the_texts(capsys):
    texts = [_SECRET_TEXT, "another one", "and a third"]
    with model_server() as url, capture_logs() as logs:
        vectors = await build_embedder(_settings(url)).embed_batch(texts)
    assert capsys.readouterr().out == ""
    assert len(vectors) == 3
    entry = _only_embed_done(logs)
    assert entry["inputs"] == 3
    assert _SECRET_TEXT not in repr(entry)


def _malformed_json_reply(request: bytes) -> bytes:
    """A 200 whose body is not valid JSON: the shape ``resp.raise_for_status()`` never
    catches, and the one no httpx exception class matches either -- exactly what a
    ``KeyError`` or ``JSONDecodeError`` from a malformed server reply looks like."""
    body = b"not json"
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )


async def test_a_malformed_reply_logs_error_once_and_still_raises(capsys):
    """Neither `ProviderRefused` nor `ProviderUnreachable` classifies a 200 with a body that
    is not valid JSON: it is exactly the shape a regression looks like, and the log must not
    call it "ok" just because no known failure class matched."""
    with (
        raw_model_server(_malformed_json_reply) as url,
        capture_logs() as logs,
        pytest.raises(ValueError),  # json.JSONDecodeError, uncaught by this adapter on purpose
    ):
        await build_embedder(_settings(url)).embed(_SECRET_TEXT)
    assert capsys.readouterr().out == ""
    entry = _only_embed_done(logs)
    _assert_common_fields(entry, outcome="error")
    assert entry["attempts"] == 1


async def test_a_checked_embedders_first_call_logs_embed_done_once(capsys):
    """``CheckedEmbedder`` batches the caller's text with the five fingerprint strings into
    one HTTP request on a process's first call -- `embed.done` must count that as the one call
    it is, not the query call plus a separate fingerprint call."""
    conn = sqlite3.connect(":memory:")
    EmbeddingSpaceStore(conn)  # `embedding_spaces` exists; nothing registered active yet

    with model_server() as url, capture_logs() as logs:
        vector = await build_embedder(_settings(url), conn=conn).embed(_SECRET_TEXT)

    assert capsys.readouterr().out == ""
    assert len(vector) == 1024
    entry = _only_embed_done(logs)
    assert entry["inputs"] == 6  # the query plus the five fingerprint strings
    assert entry["outcome"] == "ok"
    assert _SECRET_TEXT not in repr(entry)
