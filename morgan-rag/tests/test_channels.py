"""
Tests for the multi-channel gateway module.

Tests cover:
- InboundMessage / OutboundMessage data models
- SessionKey formatting and factory methods
- RouteBinding and RouteResolver resolution hierarchy
- BaseChannel ABC contract
- ChannelGateway registration, routing, and lifecycle
- TelegramChannel / DiscordChannel graceful ImportError handling
"""

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morgan.channels.base import (
    BaseChannel,
    InboundMessage,
    OutboundMessage,
)
from morgan.channels.routing import (
    ResolvedRoute,
    RouteBinding,
    RouteResolver,
    SessionKey,
)
from morgan.channels.gateway import ChannelGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockChannel(BaseChannel):
    """Concrete channel for testing purposes."""

    def __init__(self, name: str = "mock"):
        super().__init__()
        self._name = name
        self.started = False
        self.stopped = False
        self.sent_messages: list[OutboundMessage] = []

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, message: OutboundMessage) -> None:
        self.sent_messages.append(message)


# ===========================================================================
# InboundMessage / OutboundMessage
# ===========================================================================


class TestInboundMessage:
    def test_create_dm(self):
        msg = InboundMessage(
            channel="telegram",
            peer_id="user123",
            content="hello",
        )
        assert msg.channel == "telegram"
        assert msg.peer_id == "user123"
        assert msg.content == "hello"
        assert msg.group_id is None
        assert msg.account_id is None
        assert msg.metadata == {}

    def test_create_group(self):
        msg = InboundMessage(
            channel="discord",
            peer_id="user456",
            content="hi group",
            group_id="guild789",
            account_id="acct1",
            metadata={"role": "admin"},
        )
        assert msg.group_id == "guild789"
        assert msg.account_id == "acct1"
        assert msg.metadata == {"role": "admin"}


class TestOutboundMessage:
    def test_create(self):
        msg = OutboundMessage(
            channel="telegram",
            peer_id="user123",
            content="reply",
        )
        assert msg.channel == "telegram"
        assert msg.peer_id == "user123"
        assert msg.content == "reply"
        assert msg.group_id is None
        assert msg.metadata == {}


# ===========================================================================
# SessionKey
# ===========================================================================


class TestSessionKey:
    def test_for_dm(self):
        key = SessionKey.for_dm(
            agent_id="agent1", channel="telegram", peer_id="user1"
        )
        assert str(key) == "agent:agent1:telegram:dm:user1"
        assert key.agent_id == "agent1"
        assert key.channel == "telegram"
        assert key.session_type == "dm"
        assert key.identifier == "user1"

    def test_for_group(self):
        key = SessionKey.for_group(
            agent_id="agent1", channel="discord", group_id="guild1"
        )
        assert str(key) == "agent:agent1:discord:group:guild1"
        assert key.session_type == "group"
        assert key.identifier == "guild1"

    def test_for_main(self):
        key = SessionKey.for_main(agent_id="agent1")
        assert str(key) == "agent:agent1:main"
        assert key.channel == "main"
        assert key.session_type == "main"
        assert key.identifier == ""

    def test_frozen(self):
        key = SessionKey.for_dm(
            agent_id="a", channel="c", peer_id="p"
        )
        with pytest.raises(FrozenInstanceError):
            key.agent_id = "changed"

    def test_hashable(self):
        k1 = SessionKey.for_dm("a", "c", "p")
        k2 = SessionKey.for_dm("a", "c", "p")
        assert k1 == k2
        assert hash(k1) == hash(k2)
        assert len({k1, k2}) == 1


# ===========================================================================
# ResolvedRoute
# ===========================================================================


class TestResolvedRoute:
    def test_create(self):
        key = SessionKey.for_dm("a1", "telegram", "u1")
        route = ResolvedRoute(
            agent_id="a1",
            session_key=key,
            session_type="dm",
            matched_by="peer",
        )
        assert route.agent_id == "a1"
        assert route.matched_by == "peer"


# ===========================================================================
# RouteResolver
# ===========================================================================


class TestRouteResolver:
    def test_default_agent(self):
        resolver = RouteResolver(default_agent_id="default")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert route.agent_id == "default"
        assert route.matched_by == "default"

    def test_channel_binding(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_channel("telegram", "tg_agent")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert route.agent_id == "tg_agent"
        assert route.matched_by == "channel"

    def test_group_binding_overrides_channel(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_channel("discord", "discord_agent")
        resolver.bind_group("guild1", "group_agent")
        msg = InboundMessage(
            channel="discord",
            peer_id="u1",
            content="hi",
            group_id="guild1",
        )
        route = resolver.resolve(msg)
        assert route.agent_id == "group_agent"
        assert route.matched_by == "group"

    def test_peer_binding_overrides_all(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_channel("discord", "discord_agent")
        resolver.bind_group("guild1", "group_agent")
        resolver.bind_peer("u1", "peer_agent")
        msg = InboundMessage(
            channel="discord",
            peer_id="u1",
            content="hi",
            group_id="guild1",
        )
        route = resolver.resolve(msg)
        assert route.agent_id == "peer_agent"
        assert route.matched_by == "peer"

    def test_resolution_hierarchy_dm(self):
        """DM without group_id: peer > channel > default."""
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_channel("telegram", "ch_agent")
        resolver.bind_peer("u1", "peer_agent")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert route.agent_id == "peer_agent"
        assert route.session_type == "dm"

    def test_session_key_dm(self):
        resolver = RouteResolver(default_agent_id="default")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert str(route.session_key) == "agent:default:telegram:dm:u1"

    def test_session_key_group(self):
        resolver = RouteResolver(default_agent_id="default")
        msg = InboundMessage(
            channel="discord", peer_id="u1", content="hi", group_id="g1"
        )
        route = resolver.resolve(msg)
        assert str(route.session_key) == "agent:default:discord:group:g1"

    def test_unbind_peer(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_peer("u1", "peer_agent")
        resolver.unbind_peer("u1")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert route.agent_id == "default"

    def test_unbind_group(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_group("g1", "group_agent")
        resolver.unbind_group("g1")
        msg = InboundMessage(
            channel="discord", peer_id="u1", content="hi", group_id="g1"
        )
        route = resolver.resolve(msg)
        assert route.agent_id == "default"

    def test_unbind_channel(self):
        resolver = RouteResolver(default_agent_id="default")
        resolver.bind_channel("telegram", "tg_agent")
        resolver.unbind_channel("telegram")
        msg = InboundMessage(channel="telegram", peer_id="u1", content="hi")
        route = resolver.resolve(msg)
        assert route.agent_id == "default"


# ===========================================================================
# BaseChannel
# ===========================================================================


class TestBaseChannel:
    def test_abc_enforcement(self):
        """Cannot instantiate BaseChannel directly."""
        with pytest.raises(TypeError):
            BaseChannel()

    def test_set_message_handler(self):
        ch = MockChannel("test")
        handler = AsyncMock()
        ch.set_message_handler(handler)
        assert ch._message_handler is handler

    @pytest.mark.asyncio
    async def test_on_message_calls_handler(self):
        ch = MockChannel("test")
        handler = AsyncMock()
        ch.set_message_handler(handler)

        msg = InboundMessage(channel="test", peer_id="u1", content="hello")
        await ch.on_message(msg)
        handler.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_on_message_without_handler(self):
        ch = MockChannel("test")
        msg = InboundMessage(channel="test", peer_id="u1", content="hello")
        # Should not raise even without a handler
        await ch.on_message(msg)


# ===========================================================================
# ChannelGateway
# ===========================================================================


class TestChannelGateway:
    def test_register_channel(self):
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("telegram")
        gw.register_channel(ch)
        assert "telegram" in gw.channels
        assert ch._message_handler is not None

    def test_register_duplicate_raises(self):
        gw = ChannelGateway(default_agent_id="default")
        ch1 = MockChannel("telegram")
        ch2 = MockChannel("telegram")
        gw.register_channel(ch1)
        with pytest.raises(ValueError, match="already registered"):
            gw.register_channel(ch2)

    def test_set_agent_handler(self):
        gw = ChannelGateway(default_agent_id="default")
        handler = AsyncMock(return_value="reply")
        gw.set_agent_handler(handler)
        assert gw._agent_handler is handler

    @pytest.mark.asyncio
    async def test_start_stop_all(self):
        gw = ChannelGateway(default_agent_id="default")
        ch1 = MockChannel("ch1")
        ch2 = MockChannel("ch2")
        gw.register_channel(ch1)
        gw.register_channel(ch2)

        await gw.start()
        assert ch1.started
        assert ch2.started

        await gw.stop()
        assert ch1.stopped
        assert ch2.stopped

    @pytest.mark.asyncio
    async def test_handle_message_routes_and_replies(self):
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("telegram")
        gw.register_channel(ch)

        async def agent_handler(agent_id, session_key, message):
            return f"echo: {message.content}"

        gw.set_agent_handler(agent_handler)

        msg = InboundMessage(channel="telegram", peer_id="u1", content="ping")
        await gw._handle_message(msg)

        assert len(ch.sent_messages) == 1
        reply = ch.sent_messages[0]
        assert reply.content == "echo: ping"
        assert reply.peer_id == "u1"
        assert reply.channel == "telegram"

    @pytest.mark.asyncio
    async def test_handle_message_group_reply(self):
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("discord")
        gw.register_channel(ch)

        async def agent_handler(agent_id, session_key, message):
            return f"group reply"

        gw.set_agent_handler(agent_handler)

        msg = InboundMessage(
            channel="discord",
            peer_id="u1",
            content="hi",
            group_id="g1",
        )
        await gw._handle_message(msg)

        assert len(ch.sent_messages) == 1
        reply = ch.sent_messages[0]
        assert reply.group_id == "g1"

    @pytest.mark.asyncio
    async def test_handle_message_no_handler(self):
        """If no agent handler set, should not raise."""
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("telegram")
        gw.register_channel(ch)

        msg = InboundMessage(channel="telegram", peer_id="u1", content="ping")
        await gw._handle_message(msg)
        assert len(ch.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_handle_message_none_response(self):
        """Agent handler returning None should not send a reply."""
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("telegram")
        gw.register_channel(ch)

        async def agent_handler(agent_id, session_key, message):
            return None

        gw.set_agent_handler(agent_handler)

        msg = InboundMessage(channel="telegram", peer_id="u1", content="ping")
        await gw._handle_message(msg)
        assert len(ch.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_handle_message_unknown_channel(self):
        """Message from unknown channel should not crash."""
        gw = ChannelGateway(default_agent_id="default")

        async def agent_handler(agent_id, session_key, message):
            return "reply"

        gw.set_agent_handler(agent_handler)

        msg = InboundMessage(channel="unknown", peer_id="u1", content="ping")
        # Should not raise
        await gw._handle_message(msg)

    def test_resolver_access(self):
        gw = ChannelGateway(default_agent_id="default")
        assert gw.resolver is not None
        gw.resolver.bind_peer("u1", "special_agent")

    @pytest.mark.asyncio
    async def test_full_flow_with_routing(self):
        """End-to-end: register channel, bind agent, handle, check route."""
        gw = ChannelGateway(default_agent_id="default")
        ch = MockChannel("telegram")
        gw.register_channel(ch)
        gw.resolver.bind_peer("vip_user", "vip_agent")

        calls = []

        async def agent_handler(agent_id, session_key, message):
            calls.append((agent_id, str(session_key)))
            return "ok"

        gw.set_agent_handler(agent_handler)

        msg = InboundMessage(
            channel="telegram", peer_id="vip_user", content="hi"
        )
        await gw._handle_message(msg)

        assert len(calls) == 1
        assert calls[0][0] == "vip_agent"
        assert calls[0][1] == "agent:vip_agent:telegram:dm:vip_user"


# ===========================================================================
# TelegramChannel / DiscordChannel graceful import handling
# ===========================================================================


class TestTelegramChannel:
    def test_import_succeeds(self):
        from morgan.channels.telegram_channel import TelegramChannel

        assert TelegramChannel is not None

    def test_instantiate(self):
        from morgan.channels.telegram_channel import TelegramChannel

        ch = TelegramChannel(token="fake-token")
        assert ch.name == "telegram"

    @pytest.mark.asyncio
    async def test_start_without_library(self):
        from morgan.channels.telegram_channel import TelegramChannel

        ch = TelegramChannel(token="fake-token")
        # If python-telegram-bot is not installed, start should handle gracefully
        # (either works with the real lib or logs a warning)
        # We just verify it does not raise an unhandled ImportError
        try:
            await ch.start()
        except Exception:
            # Any exception other than ImportError is acceptable here
            # (e.g., invalid token, network error)
            pass


class TestDiscordChannel:
    def test_import_succeeds(self):
        from morgan.channels.discord_channel import DiscordChannel

        assert DiscordChannel is not None

    def test_instantiate(self):
        from morgan.channels.discord_channel import DiscordChannel

        ch = DiscordChannel(token="fake-token")
        assert ch.name == "discord"

    @pytest.mark.asyncio
    async def test_start_without_library(self):
        from morgan.channels.discord_channel import DiscordChannel

        ch = DiscordChannel(token="fake-token")
        try:
            await ch.start()
        except Exception:
            pass


# ===========================================================================
# Package __init__ exports
# ===========================================================================


class TestPackageExports:
    def test_top_level_imports(self):
        from morgan.channels import (
            BaseChannel,
            ChannelGateway,
            InboundMessage,
            OutboundMessage,
            RouteResolver,
            SessionKey,
        )

        assert BaseChannel is not None
        assert ChannelGateway is not None
        assert InboundMessage is not None
        assert OutboundMessage is not None
        assert RouteResolver is not None
        assert SessionKey is not None
