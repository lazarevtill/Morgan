"""
Tests for the full-featured Telegram channel adapter.

All python-telegram-bot types are mocked so the test suite runs without
the optional dependency installed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morgan.channels.base import InboundMessage, OutboundMessage


# ---------------------------------------------------------------------------
# Helpers to build a TelegramChannel with mocked internals
# ---------------------------------------------------------------------------


def _make_channel(**kwargs):
    """Import and instantiate TelegramChannel with defaults."""
    from morgan.channels.telegram_channel import TelegramChannel

    defaults = {"token": "FAKE_TOKEN"}
    defaults.update(kwargs)
    return TelegramChannel(**defaults)


def _make_update(
    text="hello",
    user_id=123,
    chat_id=456,
    chat_type="private",
    username="testuser",
    bot_username=None,
    reply_to_bot_id=None,
    message_thread_id=None,
    caption=None,
    photo=None,
    document=None,
):
    """Build a minimal mock Update object."""
    user = SimpleNamespace(id=user_id, username=username)

    reply_to = None
    if reply_to_bot_id is not None:
        reply_to = SimpleNamespace(
            from_user=SimpleNamespace(id=reply_to_bot_id)
        )

    message = SimpleNamespace(
        text=text,
        caption=caption,
        message_id=1,
        reply_to_message=reply_to,
        message_thread_id=message_thread_id,
        photo=photo or [],
        document=document,
    )

    chat = SimpleNamespace(id=chat_id, type=chat_type)

    update = SimpleNamespace(
        effective_message=message,
        effective_user=user,
        effective_chat=chat,
        callback_query=None,
    )
    return update


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        ch = _make_channel()
        assert ch.name == "telegram"
        assert ch._max_message_length == 4096
        assert ch._require_mention_in_groups is True
        assert ch._allowed_user_ids == set()
        assert ch._webhook_url is None

    def test_custom_params(self):
        ch = _make_channel(
            allowed_user_ids={"1", "2"},
            require_mention_in_groups=False,
            webhook_url="https://example.com",
            webhook_port=9999,
            max_message_length=2000,
        )
        assert ch._allowed_user_ids == {"1", "2"}
        assert ch._require_mention_in_groups is False
        assert ch._webhook_url == "https://example.com"
        assert ch._webhook_port == 9999
        assert ch._max_message_length == 2000


# ---------------------------------------------------------------------------
# _should_respond_in_group
# ---------------------------------------------------------------------------


class TestShouldRespondInGroup:
    def test_respond_when_mention_not_required(self):
        ch = _make_channel(require_mention_in_groups=False)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        update = _make_update(text="hello", chat_type="group")
        assert ch._should_respond_in_group(update) is True

    def test_respond_when_mentioned(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        update = _make_update(text="@morgan_bot hello", chat_type="group")
        assert ch._should_respond_in_group(update) is True

    def test_respond_when_reply_to_bot(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        update = _make_update(
            text="hello",
            chat_type="group",
            reply_to_bot_id=999,
        )
        assert ch._should_respond_in_group(update) is True

    def test_no_response_without_mention_or_reply(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        update = _make_update(text="hello world", chat_type="group")
        assert ch._should_respond_in_group(update) is False

    def test_no_response_with_no_message(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        update = SimpleNamespace(
            effective_message=None,
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=1, type="group"),
        )
        assert ch._should_respond_in_group(update) is False


# ---------------------------------------------------------------------------
# _chunk_message
# ---------------------------------------------------------------------------


class TestChunkMessage:
    def test_short_message_no_split(self):
        ch = _make_channel(max_message_length=100)
        result = ch._chunk_message("hello")
        assert result == ["hello"]

    def test_exact_length_no_split(self):
        ch = _make_channel(max_message_length=10)
        result = ch._chunk_message("1234567890")
        assert result == ["1234567890"]

    def test_paragraph_split(self):
        ch = _make_channel(max_message_length=20)
        text = "Short para.\n\nSecond paragraph here."
        result = ch._chunk_message(text)
        assert len(result) >= 2
        # First chunk should end at the paragraph boundary
        assert "Short para." in result[0]

    def test_sentence_split(self):
        ch = _make_channel(max_message_length=30)
        text = "First sentence. Second sentence. Third sentence."
        result = ch._chunk_message(text)
        assert len(result) >= 2

    def test_hard_cut_no_boundaries(self):
        ch = _make_channel(max_message_length=10)
        text = "abcdefghijklmnopqrstuvwxyz"
        result = ch._chunk_message(text)
        assert len(result) == 3
        assert "".join(result) == text

    def test_split_on_newline(self):
        ch = _make_channel(max_message_length=20)
        text = "Line one here.\nLine two here."
        result = ch._chunk_message(text)
        assert len(result) >= 2

    def test_empty_string(self):
        ch = _make_channel()
        result = ch._chunk_message("")
        assert result == [""]


# ---------------------------------------------------------------------------
# _escape_markdown
# ---------------------------------------------------------------------------


class TestEscapeMarkdown:
    def test_escapes_special_chars(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._escape_markdown("Hello *world* _test_ [link](url)")
        assert "\\*" in result
        assert "\\_" in result
        assert "\\[" in result
        assert "\\(" in result

    def test_plain_text_unchanged(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._escape_markdown("hello world 123")
        assert result == "hello world 123"

    def test_all_special_chars(self):
        from morgan.channels.telegram_channel import TelegramChannel

        specials = "_*[]()~`>#+-=|{}.!"
        result = TelegramChannel._escape_markdown(specials)
        # Each special char should be preceded by a backslash
        for ch in specials:
            assert f"\\{ch}" in result

    def test_empty_string(self):
        from morgan.channels.telegram_channel import TelegramChannel

        assert TelegramChannel._escape_markdown("") == ""


# ---------------------------------------------------------------------------
# Allowlist filtering
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_empty_allowlist_allows_all(self):
        ch = _make_channel(allowed_user_ids=None)
        assert ch._is_user_allowed("12345") is True
        assert ch._is_user_allowed("99999") is True

    def test_allowlist_allows_listed_user(self):
        ch = _make_channel(allowed_user_ids={"100", "200"})
        assert ch._is_user_allowed("100") is True
        assert ch._is_user_allowed("200") is True

    def test_allowlist_blocks_unlisted_user(self):
        ch = _make_channel(allowed_user_ids={"100", "200"})
        assert ch._is_user_allowed("300") is False


# ---------------------------------------------------------------------------
# send() calls bot.send_message correctly
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_send_calls_bot(self):
        ch = _make_channel()

        # Provide a mock Application with a mock Bot
        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        # Patch _HAS_TELEGRAM
        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            msg = OutboundMessage(
                channel="telegram",
                peer_id="12345",
                content="Hello!",
                metadata={},
            )
            await ch.send(msg)

            # send_message must have been called with correct params.
            # Note: send_chat_action (typing indicator) may be skipped if
            # python-telegram-bot is not installed (ChatAction constant
            # unavailable), which is fine — the indicator is non-critical.
            assert mock_bot.send_message.called

            call_kwargs = mock_bot.send_message.call_args
            assert call_kwargs[1]["chat_id"] == 12345
            assert call_kwargs[1]["text"] == "Hello!"
        finally:
            tg_mod._HAS_TELEGRAM = original

    @pytest.mark.asyncio
    async def test_send_with_group_id(self):
        ch = _make_channel()

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            msg = OutboundMessage(
                channel="telegram",
                peer_id="12345",
                content="Group reply",
                group_id="67890",
                metadata={},
            )
            await ch.send(msg)

            call_kwargs = mock_bot.send_message.call_args
            # Should use group_id as chat_id
            assert call_kwargs[1]["chat_id"] == 67890
        finally:
            tg_mod._HAS_TELEGRAM = original

    @pytest.mark.asyncio
    async def test_send_not_started_logs_warning(self):
        ch = _make_channel()
        ch._app = None

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            msg = OutboundMessage(
                channel="telegram",
                peer_id="1",
                content="test",
                metadata={},
            )
            # Should not raise, just log warning
            await ch.send(msg)
        finally:
            tg_mod._HAS_TELEGRAM = original


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------


class TestCommandHandling:
    @pytest.mark.asyncio
    async def test_start_command(self):
        ch = _make_channel()
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            update = _make_update(text="/start", user_id=1, chat_id=1)
            context = MagicMock()
            await ch._on_command(update, context)

            assert mock_bot.send_message.called
            call_kwargs = mock_bot.send_message.call_args[1]
            assert "Morgan" in call_kwargs["text"] or "Hello" in call_kwargs["text"]
        finally:
            tg_mod._HAS_TELEGRAM = original

    @pytest.mark.asyncio
    async def test_help_command(self):
        ch = _make_channel()
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            update = _make_update(text="/help", user_id=1, chat_id=1)
            context = MagicMock()
            await ch._on_command(update, context)

            assert mock_bot.send_message.called
            call_kwargs = mock_bot.send_message.call_args[1]
            assert "/start" in call_kwargs["text"]
            assert "/help" in call_kwargs["text"]
            assert "/clear" in call_kwargs["text"]
        finally:
            tg_mod._HAS_TELEGRAM = original

    @pytest.mark.asyncio
    async def test_clear_command_sends_inbound_and_confirmation(self):
        ch = _make_channel()
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def mock_handler(msg: InboundMessage) -> None:
            received.append(msg)

        ch.set_message_handler(mock_handler)

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        tg_mod._HAS_TELEGRAM = True
        try:
            update = _make_update(text="/clear", user_id=1, chat_id=1)
            context = MagicMock()
            await ch._on_command(update, context)

            # Should have forwarded a /clear inbound message
            assert len(received) == 1
            assert received[0].content == "/clear"
            assert received[0].metadata.get("command") == "clear"

            # Should have sent a confirmation
            assert mock_bot.send_message.called
            call_kwargs = mock_bot.send_message.call_args[1]
            assert "cleared" in call_kwargs["text"].lower()
        finally:
            tg_mod._HAS_TELEGRAM = original


# ---------------------------------------------------------------------------
# Text message handling with group filtering
# ---------------------------------------------------------------------------


class TestTextMessageHandler:
    @pytest.mark.asyncio
    async def test_private_message_forwarded(self):
        ch = _make_channel()
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="hi there", user_id=42, chat_type="private")
        context = MagicMock()
        await ch._on_text_message(update, context)

        assert len(received) == 1
        assert received[0].content == "hi there"
        assert received[0].peer_id == "42"
        assert received[0].group_id is None

    @pytest.mark.asyncio
    async def test_group_message_ignored_without_mention(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="random chat", chat_type="group")
        context = MagicMock()
        await ch._on_text_message(update, context)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_group_message_with_mention_forwarded(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(
            text="@morgan_bot what is the time?",
            chat_type="group",
        )
        context = MagicMock()
        await ch._on_text_message(update, context)

        assert len(received) == 1
        # The mention should be stripped from the content
        assert "@morgan_bot" not in received[0].content
        assert "what is the time?" in received[0].content

    @pytest.mark.asyncio
    async def test_blocked_user_ignored(self):
        ch = _make_channel(allowed_user_ids={"100"})
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="hi", user_id=999, chat_type="private")
        context = MagicMock()
        await ch._on_text_message(update, context)

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Forum / topic support
# ---------------------------------------------------------------------------


class TestForumTopicSupport:
    @pytest.mark.asyncio
    async def test_thread_id_included_in_group_id(self):
        ch = _make_channel()
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        ch._require_mention_in_groups = False

        mock_bot = AsyncMock()
        mock_app = MagicMock()
        mock_app.bot = mock_bot
        ch._app = mock_app

        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(
            text="topic message",
            chat_type="supergroup",
            message_thread_id=42,
        )
        context = MagicMock()
        await ch._on_text_message(update, context)

        assert len(received) == 1
        # group_id should encode the thread
        assert received[0].group_id == "456:42"
        assert received[0].metadata.get("message_thread_id") == 42
