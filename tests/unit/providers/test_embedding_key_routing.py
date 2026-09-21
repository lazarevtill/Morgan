"""The chat key is not sent to the embedding host.

Two hosts, two keys. Morgan sent one key to both, so the owner's chat credential reached a
machine that never needed it -- and appeared in that machine's logs.
"""

from __future__ import annotations

import pytest

from morgan_brain.config import Settings
from morgan_brain.providers.factory import (
    build_embedder,
    check_embeddings_reachable,
    check_llm_reachable,
)
from morgan_brain.providers.wire import ProviderRefused
from tests.fakes import flaky_model_server, header_recording_server


def _settings(**overrides: object) -> Settings:
    return Settings(embedding_backend="provider", **overrides)  # type: ignore[arg-type]


async def test_a_separate_endpoint_never_sees_the_chat_key():
    with header_recording_server() as (url, headers):
        await build_embedder(_settings(embedding_endpoint=url, llm_api_key="chat-key")).embed("x")
    assert "chat-key" not in headers.last.get("authorization", "")


async def test_the_embedding_key_is_sent_when_it_is_set():
    with header_recording_server() as (url, headers):
        await build_embedder(
            _settings(embedding_endpoint=url, embedding_api_key="embed-key")
        ).embed("x")
    assert headers.last["authorization"] == "Bearer embed-key"


async def test_without_a_separate_endpoint_the_chat_key_is_right():
    with header_recording_server() as (url, headers):
        await build_embedder(_settings(llm_endpoint=url, llm_api_key="chat-key")).embed("x")
    assert headers.last["authorization"] == "Bearer chat-key"


async def test_an_empty_key_sends_no_header():
    with header_recording_server() as (url, headers):
        await build_embedder(_settings(embedding_endpoint=url)).embed("x")
    assert "authorization" not in headers.last


async def test_a_401_from_a_separate_embedding_host_names_the_embedding_key():
    """The embedding host has its own key now: a refusal from it must not send the owner to
    check the chat credential, which was never sent there."""
    with (
        flaky_model_server(fail_times=99, status=401) as url,
        pytest.raises(ProviderRefused) as exc,
    ):
        await build_embedder(
            _settings(embedding_endpoint=url, embedding_api_key="embed-key")
        ).embed("x")
    assert "MORGAN_EMBEDDING_API_KEY" in str(exc.value)
    assert "MORGAN_LLM_API_KEY" not in str(exc.value)


async def test_doctors_probes_send_each_endpoint_its_own_key():
    """``check_llm_reachable`` and ``check_embeddings_reachable`` are what ``morgan doctor``
    calls; with the endpoints separate, each must carry its own key, never the other's."""
    with (
        header_recording_server() as (chat_url, chat_headers),
        header_recording_server() as (embedding_url, embedding_headers),
    ):
        settings = _settings(
            llm_endpoint=chat_url,
            embedding_endpoint=embedding_url,
            llm_api_key="chat-key",
            embedding_api_key="embed-key",
        )
        assert (await check_llm_reachable(settings)).error is None
        assert (await check_embeddings_reachable(settings)).error is None

    assert chat_headers.last.get("authorization") == "Bearer chat-key"
    assert embedding_headers.last.get("authorization") == "Bearer embed-key"
