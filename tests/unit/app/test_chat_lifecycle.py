"""Per-command chat transports are released on success, refusal and cleanup failure."""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from morgan_brain.app.chat import Chat
from morgan_brain.composition import build_app_context
from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.wire import ChatClient, ProviderRefused
from morgan_brain.surfaces.cli import commands
from tests.fakes import FakeChatClient, raw_model_server


@pytest.mark.parametrize("command", ["ask", "consolidate"])
@pytest.mark.parametrize("refused", [False, True])
async def test_commands_close_every_owned_chat_client(
    settings_for_tmp, monkeypatch, command, refused
):
    contexts = []
    calls = []

    def build(settings):
        ctx = build_app_context(settings)
        contexts.append(ctx)
        return ctx

    monkeypatch.setattr(commands, "build_app_context", build)

    def reply(request):
        calls.append(request)
        content = "A synthetic answer" if command == "ask" else '{"ops":[]}'
        body = json.dumps(
            {"error": {"message": "synthetic refusal"}}
            if refused
            else {
                "id": "synthetic",
                "model": "synthetic",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        status = 401 if refused else 200
        return (
            f"HTTP/1.1 {status} Synthetic\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )

    with raw_model_server(reply) as url:
        settings = settings_for_tmp.model_copy(update={"llm_endpoint": url})
        await commands.cmd_remember(
            argparse.Namespace(text="A synthetic note", source="user_stated"),
            settings,
            "personal",
        )
        try:
            # Repeated MCP calls use these same command handlers and fresh app contexts.
            for _ in range(2):
                operation = (
                    commands.cmd_ask(argparse.Namespace(text="A question"), settings, "personal")
                    if command == "ask"
                    else commands.cmd_consolidate(
                        argparse.Namespace(all_projects=False), settings, "personal"
                    )
                )
                if refused:
                    with pytest.raises(ProviderRefused):
                        await operation
                else:
                    await operation
                assert contexts[-1].client._client.is_closed()
                with pytest.raises(sqlite3.ProgrammingError):
                    contexts[-1].conn.execute("SELECT 1")
            assert len(contexts) == len(calls) == 2
        finally:
            for ctx in contexts:
                await ctx.client._client.close()
                ctx.conn.close()


async def test_chat_releases_default_client_when_strict_cleanup_raises():
    client = OpenAICompatAdapter(
        base_url="http://127.0.0.1:1/v1",
        api_key="synthetic",
        provider="synthetic",
        setting="MORGAN_LLM_ENDPOINT",
    )

    async def failed_close():
        raise RuntimeError("synthetic cleanup failure")

    chat = Chat(
        gate=None,
        history=None,
        client=client,
        model="synthetic",
        clock=lambda: datetime.now(UTC),
        strict_backend=SimpleNamespace(aclose=failed_close),
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
            await chat.aclose()
        assert client._client.is_closed()
    finally:
        await client._client.close()


async def test_chat_close_preserves_clients_without_optional_cleanup():
    client = FakeChatClient()
    assert isinstance(client, ChatClient)
    chat = Chat(
        gate=None,
        history=None,
        client=client,
        model="synthetic",
        clock=lambda: datetime.now(UTC),
    )
    await chat.aclose()
