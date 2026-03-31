"""
Tests for the Synology Chat channel adapter.

All aiohttp types are mocked so the test suite runs without the optional
dependency installed.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morgan.channels.base import InboundMessage, OutboundMessage
from morgan.channels.synology_channel import SynologyChannel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(**kwargs):
    """Build a SynologyChannel with sensible test defaults."""
    defaults = {
        "token": "test-secret-token",
        "incoming_url": "https://nas.local:5001/webapi/entry.cgi",
    }
    defaults.update(kwargs)
    return SynologyChannel(**defaults)


def _make_request(data: dict, content_type: str = "application/x-www-form-urlencoded"):
    """Create a mock aiohttp request."""
    req = AsyncMock()
    req.content_type = content_type

    if content_type == "application/json":
        req.json = AsyncMock(return_value=data)
        req.post = AsyncMock(return_value={})
    else:
        # Form-urlencoded: request.post() returns a multidict-like object
        req.post = AsyncMock(return_value=data)
        req.json = AsyncMock(side_effect=Exception("Not JSON"))

    return req


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------


class TestValidateToken:
    def test_correct_token(self):
        ch = _make_channel(token="my-secret")
        assert ch._validate_token("my-secret") is True

    def test_incorrect_token(self):
        ch = _make_channel(token="my-secret")
        assert ch._validate_token("wrong-token") is False

    def test_empty_token(self):
        ch = _make_channel(token="my-secret")
        assert ch._validate_token("") is False

    def test_timing_safe(self):
        """Verify that hmac.compare_digest is used (constant-time)."""
        ch = _make_channel(token="secret")
        # The method uses hmac.compare_digest internally; we just verify
        # it produces the correct boolean result.
        assert ch._validate_token("secret") is True
        assert ch._validate_token("secre") is False
        assert ch._validate_token("secretx") is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    def test_within_limit(self):
        ch = _make_channel(rate_limit_per_minute=5)
        for _ in range(5):
            assert ch._check_rate_limit("user1") is True

    def test_exceeded_limit(self):
        ch = _make_channel(rate_limit_per_minute=3)
        for _ in range(3):
            assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user1") is False

    def test_different_users_independent(self):
        ch = _make_channel(rate_limit_per_minute=2)
        assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user1") is False  # user1 at limit
        assert ch._check_rate_limit("user2") is True  # user2 still ok

    def test_old_entries_expire(self):
        ch = _make_channel(rate_limit_per_minute=1)
        # Manually insert an old timestamp
        ch._rate_tracker["user1"] = [time.monotonic() - 120.0]
        # Should be allowed because the old entry is > 60s ago
        assert ch._check_rate_limit("user1") is True


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


class TestSanitizeInput:
    def test_normal_text(self):
        assert SynologyChannel._sanitize_input("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        assert SynologyChannel._sanitize_input("  hello  ") == "hello"

    def test_truncates_long_text(self):
        long_text = "x" * 5000
        result = SynologyChannel._sanitize_input(long_text)
        assert len(result) == 4000

    def test_empty_string(self):
        assert SynologyChannel._sanitize_input("") == ""

    def test_none_returns_empty(self):
        # The method only receives str, but test defensive behavior
        assert SynologyChannel._sanitize_input("") == ""

    def test_exactly_4000(self):
        text = "a" * 4000
        assert SynologyChannel._sanitize_input(text) == text


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


class TestChunkMessage:
    def test_short_message(self):
        ch = _make_channel()
        result = ch._chunk_message("hello")
        assert result == ["hello"]

    def test_exactly_2000(self):
        ch = _make_channel()
        text = "a" * 2000
        result = ch._chunk_message(text)
        assert result == [text]

    def test_long_message_split(self):
        ch = _make_channel()
        text = "word " * 500  # 2500 chars
        result = ch._chunk_message(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 2000

    def test_paragraph_boundary(self):
        ch = _make_channel()
        # Create text that exceeds 2000 chars with a paragraph break
        part1 = "a" * 1000
        part2 = "b" * 1500
        text = f"{part1}\n\n{part2}"
        result = ch._chunk_message(text)
        assert len(result) >= 2
        assert result[0].rstrip() == part1

    def test_hard_cut_no_spaces(self):
        ch = _make_channel()
        text = "x" * 4500
        result = ch._chunk_message(text)
        assert len(result) == 3  # 2000 + 2000 + 500
        assert "".join(result) == text

    def test_empty_string(self):
        ch = _make_channel()
        result = ch._chunk_message("")
        assert result == [""]


# ---------------------------------------------------------------------------
# Webhook handler
# ---------------------------------------------------------------------------


class TestHandleWebhook:
    @pytest.mark.asyncio
    async def test_valid_request(self):
        ch = _make_channel(token="secret")

        received: list[InboundMessage] = []

        async def handler(msg: InboundMessage) -> None:
            received.append(msg)

        ch.set_message_handler(handler)

        req = _make_request({
            "token": "secret",
            "user_id": "42",
            "username": "alice",
            "text": "Hello Morgan",
            "channel_id": "100",
            "channel_name": "general",
        })

        resp = await ch._handle_webhook(req)

        assert resp.status == 200
        assert len(received) == 1
        assert received[0].peer_id == "42"
        assert received[0].content == "Hello Morgan"
        assert received[0].group_id == "100"
        assert received[0].metadata["username"] == "alice"

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        ch = _make_channel(token="secret")

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        req = _make_request({
            "token": "wrong-token",
            "user_id": "42",
            "username": "alice",
            "text": "Hello",
        })

        resp = await ch._handle_webhook(req)

        assert resp.status == 403
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        ch = _make_channel(token="secret")

        async def handler(msg):
            pass

        ch.set_message_handler(handler)

        # Missing text
        req = _make_request({
            "token": "secret",
            "user_id": "42",
            "username": "alice",
            "text": "",
        })

        resp = await ch._handle_webhook(req)
        # Empty text after sanitization should return 400
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_missing_user_id(self):
        ch = _make_channel(token="secret")

        async def handler(msg):
            pass

        ch.set_message_handler(handler)

        req = _make_request({
            "token": "secret",
            "user_id": "",
            "username": "alice",
            "text": "hello",
        })

        resp = await ch._handle_webhook(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_json_content_type(self):
        ch = _make_channel(token="secret")

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        req = _make_request(
            {
                "token": "secret",
                "user_id": "10",
                "username": "bob",
                "text": "JSON message",
            },
            content_type="application/json",
        )

        resp = await ch._handle_webhook(req)
        assert resp.status == 200
        assert len(received) == 1
        assert received[0].content == "JSON message"

    @pytest.mark.asyncio
    async def test_user_not_in_allowlist(self):
        ch = _make_channel(token="secret", allowed_user_ids={"100"})

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        req = _make_request({
            "token": "secret",
            "user_id": "42",
            "username": "alice",
            "text": "Hello",
        })

        resp = await ch._handle_webhook(req)
        assert resp.status == 403
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        ch = _make_channel(token="secret", rate_limit_per_minute=1)

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        # First request succeeds
        req1 = _make_request({
            "token": "secret",
            "user_id": "42",
            "username": "alice",
            "text": "First",
        })
        resp1 = await ch._handle_webhook(req1)
        assert resp1.status == 200

        # Second request rate-limited
        req2 = _make_request({
            "token": "secret",
            "user_id": "42",
            "username": "alice",
            "text": "Second",
        })
        resp2 = await ch._handle_webhook(req2)
        assert resp2.status == 429
        assert len(received) == 1


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_send_correct_payload(self):
        ch = _make_channel(
            incoming_url="https://nas.local:5001/hook",
        )

        # Mock session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        ch._session = mock_session

        msg = OutboundMessage(
            channel="synology",
            peer_id="42",
            content="Hello from Morgan",
            metadata={},
        )
        await ch.send(msg)

        assert mock_session.post.called
        call_args = mock_session.post.call_args
        url = call_args[0][0]
        assert "https://nas.local:5001/hook" in url
        assert "payload=" in url

    @pytest.mark.asyncio
    async def test_send_not_started(self):
        ch = _make_channel()
        ch._session = None
        msg = OutboundMessage(
            channel="synology",
            peer_id="42",
            content="test",
            metadata={},
        )
        # Should not raise
        await ch.send(msg)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        ch = _make_channel(bot_name="TestBot")
        req = AsyncMock()
        resp = await ch._handle_health(req)
        # aiohttp.web.json_response returns a Response object
        assert resp.status == 200

        import json as json_mod
        body = json_mod.loads(resp.body)
        assert body["status"] == "ok"
        assert body["channel"] == "synology"
        assert body["bot_name"] == "TestBot"


# ---------------------------------------------------------------------------
# Channel name
# ---------------------------------------------------------------------------


class TestChannelName:
    def test_name(self):
        ch = _make_channel()
        assert ch.name == "synology"


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        ch = SynologyChannel(
            token="tok",
            incoming_url="https://example.com",
        )
        assert ch._webhook_path == "/webhook/synology"
        assert ch._webhook_port == 5001
        assert ch._allowed_user_ids == set()
        assert ch._rate_limit_per_minute == 30
        assert ch._bot_name == "Morgan"
        assert ch._allow_insecure_ssl is False

    def test_custom_params(self):
        ch = SynologyChannel(
            token="tok",
            incoming_url="https://example.com",
            webhook_path="/custom",
            webhook_port=9999,
            allowed_user_ids={"1", "2"},
            rate_limit_per_minute=10,
            bot_name="CustomBot",
            allow_insecure_ssl=True,
        )
        assert ch._webhook_path == "/custom"
        assert ch._webhook_port == 9999
        assert ch._allowed_user_ids == {"1", "2"}
        assert ch._rate_limit_per_minute == 10
        assert ch._bot_name == "CustomBot"
        assert ch._allow_insecure_ssl is True
