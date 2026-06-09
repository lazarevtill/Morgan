"""Unit tests for channels package (commit 3).

All tests are deterministic — no network, no real Telegram/Discord.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.channels.allowlist import ChatAllowlist
from morgan_brain.channels.base import Channel, InboundMessage, OutboundMessage
from morgan_brain.channels.fake import FakeChannel
from morgan_brain.channels.gateway import ChannelGateway
from morgan_brain.composition import build_orchestrator_for_test

_CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731
_REPLY = "gateway reply"


def _make_gateway(*, allowed: set[str] | None = None) -> tuple[ChannelGateway, FakeChannel]:
    orch, _ = build_orchestrator_for_test(reply=_REPLY, clock=_CLOCK)
    allowlist = ChatAllowlist(allowed=allowed)
    gw = ChannelGateway(
        orchestrator=orch,
        allowlist=allowlist,
        owner_user_id="owner",
    )
    ch = FakeChannel(name="test")
    gw.register_channel(ch)
    return gw, ch


# ---------------------------------------------------------------------------
# ChatAllowlist
# ---------------------------------------------------------------------------


def test_allowlist_empty_blocks_all() -> None:
    al = ChatAllowlist()
    assert not al.is_allowed("chat_1")
    assert not al.is_allowed("any")


def test_allowlist_allows_listed_chat() -> None:
    al = ChatAllowlist(allowed={"chat_1", "chat_2"})
    assert al.is_allowed("chat_1")
    assert al.is_allowed("chat_2")


def test_allowlist_blocks_unlisted_chat() -> None:
    al = ChatAllowlist(allowed={"chat_1"})
    assert not al.is_allowed("chat_2")


# ---------------------------------------------------------------------------
# ChannelGateway — allowlisted chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_chat_orchestrator_called_and_reply_returned() -> None:
    gw, _ = _make_gateway(allowed={"chat_42"})
    msg = InboundMessage(channel="test", chat_id="chat_42", user_id="alice", text="hello")
    result = await gw.handle_inbound(msg)
    assert result is not None
    assert result.text == _REPLY
    assert result.chat_id == "chat_42"


@pytest.mark.asyncio
async def test_reply_outbound_message_type() -> None:
    gw, _ = _make_gateway(allowed={"chat_1"})
    msg = InboundMessage(channel="test", chat_id="chat_1", user_id="u1", text="hi")
    result = await gw.handle_inbound(msg)
    assert isinstance(result, OutboundMessage)


# ---------------------------------------------------------------------------
# ChannelGateway — non-allowlisted chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_allowed_chat_returns_none() -> None:
    gw, _ = _make_gateway(allowed={"chat_1"})
    msg = InboundMessage(channel="test", chat_id="BLOCKED", user_id="u1", text="hello")
    result = await gw.handle_inbound(msg)
    assert result is None


@pytest.mark.asyncio
async def test_non_allowed_chat_orchestrator_not_called() -> None:
    """Orchestrator must NOT be invoked for blocked chats."""
    orch, _ = build_orchestrator_for_test(reply=_REPLY, clock=_CLOCK)
    call_count = 0
    original_handle = orch.handle_turn

    async def counting_handle(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return await original_handle(**kwargs)

    orch.handle_turn = counting_handle  # type: ignore[method-assign]

    gw = ChannelGateway(
        orchestrator=orch,
        allowlist=ChatAllowlist(allowed=set()),
        owner_user_id="owner",
    )
    msg = InboundMessage(channel="test", chat_id="blocked", user_id="u1", text="hello")
    await gw.handle_inbound(msg)
    assert call_count == 0


@pytest.mark.asyncio
async def test_empty_allowlist_blocks_all() -> None:
    gw, _ = _make_gateway(allowed=set())
    msg = InboundMessage(channel="test", chat_id="any", user_id="u1", text="hi")
    assert await gw.handle_inbound(msg) is None


# ---------------------------------------------------------------------------
# FakeChannel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_channel_records_sent_messages() -> None:
    ch = FakeChannel(name="fake")
    await ch.start()
    await ch.send(OutboundMessage(chat_id="c1", text="hello"))
    await ch.send(OutboundMessage(chat_id="c1", text="world"))
    assert len(ch.sent) == 2
    assert ch.sent[0].text == "hello"
    assert ch.sent[1].text == "world"


@pytest.mark.asyncio
async def test_fake_channel_start_stop_flags() -> None:
    ch = FakeChannel()
    assert not ch.started
    assert not ch.stopped
    await ch.start()
    assert ch.started
    await ch.stop()
    assert ch.stopped


def test_fake_channel_satisfies_protocol() -> None:
    ch = FakeChannel()
    assert isinstance(ch, Channel)


# ---------------------------------------------------------------------------
# TelegramChannel — importable without the telegram package
# ---------------------------------------------------------------------------


def test_telegram_channel_importable_without_dep() -> None:
    """Importing TelegramChannel must not require python-telegram-bot."""
    from morgan_brain.channels.telegram import TelegramChannel  # noqa: PLC0415

    ch = TelegramChannel(token="dummy", gateway=object())
    assert ch.name == "telegram"
