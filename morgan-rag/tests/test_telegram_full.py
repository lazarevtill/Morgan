"""
Comprehensive tests for the full-featured Telegram channel adapter.

Tests cover:
- Constructor and initialization
- All commands (/start, /help, /clear, /reset, /model, /status, /whoami, /id,
  /tools, /skills, /compact, /stop, /memory, /settings, /activation)
- Text message handling with group filtering
- Photo, document, voice, sticker, edited message handling
- Group chat filtering (mention/reply-to-bot/activation mode)
- Forum/topic support
- Inline keyboard buttons and callback queries
- Markdown-to-HTML conversion
- Message chunking
- Typing indicator management
- Allowlist enforcement
- Error handling (empty response fallback)
- Quick-action button callbacks
- Send helpers and rate limiting
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morgan.channels.base import InboundMessage, OutboundMessage


# ---------------------------------------------------------------------------
# Helpers
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
    first_name="Test",
    last_name="User",
    reply_to_bot_id=None,
    message_thread_id=None,
    caption=None,
    photo=None,
    document=None,
    voice=None,
    sticker=None,
    is_edited=False,
    is_callback=False,
    callback_data=None,
):
    """Build a minimal mock Update object."""
    user = SimpleNamespace(
        id=user_id,
        username=username,
        first_name=first_name,
        last_name=last_name,
        full_name=f"{first_name} {last_name}" if last_name else first_name,
    )

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
        voice=voice,
        sticker=sticker,
    )

    chat = SimpleNamespace(id=chat_id, type=chat_type)

    if is_edited:
        effective_msg = None
        edited_msg = message
    else:
        effective_msg = message
        edited_msg = None

    cb_query = None
    if is_callback:
        cb_query = SimpleNamespace(
            data=callback_data,
            id="cb_123",
            from_user=user,
            answer=AsyncMock(),
            message=message,
        )

    update = SimpleNamespace(
        effective_message=effective_msg,
        edited_message=edited_msg,
        effective_user=user,
        effective_chat=chat,
        callback_query=cb_query,
    )
    return update


def _setup_channel_with_app(**kwargs):
    """Create a channel with a mocked Application and bot.

    The channel's ``_app`` is set so send paths work without needing
    the actual ``python-telegram-bot`` library.
    """
    ch = _make_channel(**kwargs)
    ch._bot_username = "morgan_bot"
    ch._bot_id = 999
    ch._start_time = time.time() - 3661  # ~1h 1m 1s ago

    mock_bot = AsyncMock()
    mock_app = MagicMock()
    mock_app.bot = mock_bot
    ch._app = mock_app

    return ch


# ===========================================================================
# 1. Constructor
# ===========================================================================


class TestConstructor:
    def test_defaults(self):
        ch = _make_channel()
        assert ch.name == "telegram"
        assert ch._max_message_length == 4096
        assert ch._require_mention_in_groups is True
        assert ch._allowed_user_ids == set()
        assert ch._webhook_url is None
        assert ch._webhook_port == 8443
        assert ch._model_name == "qwen3.5:35b"
        assert ch._bot_name == "Morgan"
        assert ch._conversations == {}
        assert ch._processing == {}
        assert ch._group_activation == {}
        assert ch._start_time == 0.0

    def test_custom_params(self):
        ch = _make_channel(
            allowed_user_ids={"1", "2"},
            require_mention_in_groups=False,
            webhook_url="https://example.com",
            webhook_port=9999,
            max_message_length=2000,
            model_name="gpt-4",
            bot_name="CustomBot",
        )
        assert ch._allowed_user_ids == {"1", "2"}
        assert ch._require_mention_in_groups is False
        assert ch._webhook_url == "https://example.com"
        assert ch._webhook_port == 9999
        assert ch._max_message_length == 2000
        assert ch._model_name == "gpt-4"
        assert ch._bot_name == "CustomBot"

    def test_name_property(self):
        ch = _make_channel()
        assert ch.name == "telegram"


# ===========================================================================
# 2. Access control / allowlist
# ===========================================================================


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

    def test_check_access_valid_update(self):
        ch = _make_channel()
        ch._bot_username = "bot"
        ch._bot_id = 1
        update = _make_update()
        assert ch._check_access(update) is True

    def test_check_access_no_user(self):
        ch = _make_channel()
        ch._bot_username = "bot"
        ch._bot_id = 1
        update = _make_update()
        update.effective_user = None
        assert ch._check_access(update) is False

    def test_check_access_denied_by_allowlist(self):
        ch = _make_channel(allowed_user_ids={"100"})
        ch._bot_username = "bot"
        ch._bot_id = 1
        update = _make_update(user_id=999)
        assert ch._check_access(update) is False


# ===========================================================================
# 3. Group filtering / _should_respond_in_group
# ===========================================================================


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
            edited_message=None,
            effective_user=SimpleNamespace(id=1, username="u"),
            effective_chat=SimpleNamespace(id=1, type="group"),
            callback_query=None,
        )
        assert ch._should_respond_in_group(update) is False

    def test_activation_always_overrides(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        ch._group_activation["456"] = "always"
        update = _make_update(text="hello", chat_type="group", chat_id=456)
        assert ch._should_respond_in_group(update) is True

    def test_activation_mention_still_requires_mention(self):
        ch = _make_channel(require_mention_in_groups=True)
        ch._bot_username = "morgan_bot"
        ch._bot_id = 999
        ch._group_activation["456"] = "mention"
        update = _make_update(text="hello", chat_type="group", chat_id=456)
        assert ch._should_respond_in_group(update) is False


# ===========================================================================
# 4. Message chunking
# ===========================================================================


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

    def test_preserves_all_content(self):
        ch = _make_channel(max_message_length=20)
        text = "Hello world. This is a test. Of chunking."
        result = ch._chunk_message(text, max_length=20)
        joined = "".join(result)
        for word in ["Hello", "world", "test", "chunking"]:
            assert word in joined


# ===========================================================================
# 5. Markdown-to-HTML conversion
# ===========================================================================


class TestMarkdownToHtml:
    def test_bold(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("**bold text**")
        assert "<b>bold text</b>" in result

    def test_italic(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("*italic*")
        assert "<i>italic</i>" in result

    def test_inline_code(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("`code`")
        assert "<code>" in result
        assert "code" in result

    def test_code_block_with_language(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("```python\nprint('hi')\n```")
        assert '<pre><code class="language-python">' in result
        assert "print" in result

    def test_code_block_without_language(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("```\nsome code\n```")
        assert "<pre><code>" in result

    def test_link(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("[Google](https://google.com)")
        assert '<a href="https://google.com">Google</a>' in result

    def test_html_entity_escaping(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("a < b & c > d")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_code_block_contents_escaped(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("```\n<script>alert('xss')</script>\n```")
        assert "&lt;script&gt;" in result

    def test_mixed_formatting(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("**bold** and *italic* and `code`")
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result
        assert "<code>code</code>" in result

    def test_plain_text_unchanged(self):
        from morgan.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._markdown_to_html("plain text 123")
        assert "plain text 123" in result


# ===========================================================================
# 6. send() calls bot correctly
# ===========================================================================


class TestSend:
    @pytest.mark.asyncio
    async def test_send_calls_bot(self):
        ch = _setup_channel_with_app()
        msg = OutboundMessage(
            channel="telegram",
            peer_id="12345",
            content="Hello!",
            metadata={},
        )
        await ch.send(msg)

        assert ch._app.bot.send_message.called
        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 12345

    @pytest.mark.asyncio
    async def test_send_with_group_id(self):
        ch = _setup_channel_with_app()
        msg = OutboundMessage(
            channel="telegram",
            peer_id="12345",
            content="Group reply",
            group_id="67890",
            metadata={},
        )
        await ch.send(msg)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 67890

    @pytest.mark.asyncio
    async def test_send_not_started_logs_warning(self):
        ch = _make_channel()
        ch._app = None
        msg = OutboundMessage(
            channel="telegram", peer_id="1", content="test", metadata={},
        )
        # Should not raise, just log warning
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_empty_response_fallback(self):
        ch = _setup_channel_with_app()
        msg = OutboundMessage(
            channel="telegram", peer_id="12345", content="", metadata={},
        )
        await ch.send(msg)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        # Text may be HTML-escaped (e.g., ' -> &#x27;)
        text = call_kwargs["text"].lower()
        assert "generate a response" in text

    @pytest.mark.asyncio
    async def test_send_with_forum_topic(self):
        ch = _setup_channel_with_app()
        msg = OutboundMessage(
            channel="telegram",
            peer_id="12345",
            content="Topic reply",
            group_id="67890:42",
            metadata={"message_thread_id": 42},
        )
        await ch.send(msg)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 67890


# ===========================================================================
# 7. /start command
# ===========================================================================


class TestCommandStart:
    @pytest.mark.asyncio
    async def test_start_command(self):
        ch = _setup_channel_with_app()
        update = _make_update(text="/start")
        ctx = MagicMock()
        await ch._cmd_start(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "Morgan" in call_kwargs["text"]
        assert "emotional intelligence" in call_kwargs["text"]
        assert "/help" in call_kwargs["text"]


# ===========================================================================
# 8. /help command
# ===========================================================================


class TestCommandHelp:
    @pytest.mark.asyncio
    async def test_help_shows_all_commands(self):
        ch = _setup_channel_with_app()
        update = _make_update(text="/help")
        ctx = MagicMock()
        await ch._cmd_help(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        text = call_kwargs["text"]
        for cmd in ["/start", "/help", "/clear", "/reset", "/model",
                    "/status", "/tools", "/skills", "/memory",
                    "/compact", "/whoami", "/id", "/settings",
                    "/stop", "/activation"]:
            assert cmd in text, f"{cmd} not found in help text"


# ===========================================================================
# 9. /clear (/reset) command
# ===========================================================================


class TestCommandClear:
    @pytest.mark.asyncio
    async def test_clear_sends_gateway_message_and_confirmation(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="/clear", user_id=1, chat_id=1)
        ctx = MagicMock()
        await ch._cmd_clear(update, ctx)

        assert len(received) == 1
        assert received[0].metadata.get("action") == "clear"

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "start fresh" in call_kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_clear_removes_conversation_tracking(self):
        ch = _setup_channel_with_app()
        ch._conversations["1"] = ["conv1"]
        ch.set_message_handler(AsyncMock())

        update = _make_update(text="/clear", user_id=1, chat_id=1)
        ctx = MagicMock()
        await ch._cmd_clear(update, ctx)

        assert "1" not in ch._conversations


# ===========================================================================
# 10. /model command
# ===========================================================================


class TestCommandModel:
    @pytest.mark.asyncio
    async def test_model_shows_name(self):
        ch = _setup_channel_with_app(model_name="test-model-v2")
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_model(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "test-model-v2" in call_kwargs["text"]
        assert "Current model:" in call_kwargs["text"]


# ===========================================================================
# 11. /status command
# ===========================================================================


class TestCommandStatus:
    @pytest.mark.asyncio
    async def test_status_shows_all_info(self):
        ch = _setup_channel_with_app(model_name="test-model", bot_name="TestBot")
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_status(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        text = call_kwargs["text"]
        assert "TestBot" in text
        assert "test-model" in text
        assert "Uptime:" in text
        assert "Tools:" in text
        assert "Memory:" in text


# ===========================================================================
# 12. /whoami (/id) command
# ===========================================================================


class TestCommandWhoami:
    @pytest.mark.asyncio
    async def test_whoami_shows_user_info(self):
        ch = _setup_channel_with_app()
        update = _make_update(
            user_id=42, username="johndoe",
            first_name="John", last_name="Doe",
        )
        ctx = MagicMock()
        await ch._cmd_whoami(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        text = call_kwargs["text"]
        assert "42" in text
        assert "John Doe" in text
        assert "@johndoe" in text


# ===========================================================================
# 13. /tools command
# ===========================================================================


class TestCommandTools:
    @pytest.mark.asyncio
    async def test_tools_lists_all(self):
        ch = _setup_channel_with_app()
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_tools(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        text = call_kwargs["text"]
        assert "calculator" in text
        assert "bash" in text
        assert "file_read" in text
        assert "web_search" in text
        assert "memory_search" in text


# ===========================================================================
# 14. /skills command
# ===========================================================================


class TestCommandSkills:
    @pytest.mark.asyncio
    async def test_skills_empty(self):
        ch = _setup_channel_with_app()
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_skills(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "No skills" in call_kwargs["text"]


# ===========================================================================
# 15. /compact command
# ===========================================================================


class TestCommandCompact:
    @pytest.mark.asyncio
    async def test_compact_sends_action(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_compact(update, ctx)

        assert len(received) == 1
        assert received[0].metadata.get("action") == "compact"

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "compaction" in call_kwargs["text"].lower()


# ===========================================================================
# 16. /stop command
# ===========================================================================


class TestCommandStop:
    @pytest.mark.asyncio
    async def test_stop_sets_abort_flag(self):
        ch = _setup_channel_with_app()
        ch._processing["456"] = True

        update = _make_update(chat_id=456)
        ctx = MagicMock()
        await ch._cmd_stop(update, ctx)

        assert ch._processing.get("456") is False

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "cancelled" in call_kwargs["text"].lower()


# ===========================================================================
# 17. /memory command
# ===========================================================================


class TestCommandMemory:
    @pytest.mark.asyncio
    async def test_memory_shows_status(self):
        ch = _setup_channel_with_app()
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_memory(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "Memory Status" in call_kwargs["text"]


# ===========================================================================
# 18. /settings command
# ===========================================================================


class TestCommandSettings:
    @pytest.mark.asyncio
    async def test_settings_shows_config(self):
        ch = _setup_channel_with_app(model_name="test-m", bot_name="TestMorgan")
        update = _make_update()
        ctx = MagicMock()
        await ch._cmd_settings(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        text = call_kwargs["text"]
        assert "Channel Settings" in text
        assert "test-m" in text
        assert "TestMorgan" in text


# ===========================================================================
# 19. /activation command
# ===========================================================================


class TestCommandActivation:
    @pytest.mark.asyncio
    async def test_activation_not_in_group(self):
        ch = _setup_channel_with_app()
        update = _make_update(chat_type="private")
        ctx = MagicMock()
        await ch._cmd_activation(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "only available in group" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_activation_set_always(self):
        ch = _setup_channel_with_app()
        update = _make_update(
            text="/activation always", chat_type="supergroup", chat_id=789,
        )
        ctx = MagicMock()
        await ch._cmd_activation(update, ctx)

        assert ch._group_activation["789"] == "always"

    @pytest.mark.asyncio
    async def test_activation_set_mention(self):
        ch = _setup_channel_with_app()
        update = _make_update(
            text="/activation mention", chat_type="supergroup", chat_id=789,
        )
        ctx = MagicMock()
        await ch._cmd_activation(update, ctx)

        assert ch._group_activation["789"] == "mention"

    @pytest.mark.asyncio
    async def test_activation_no_arg_shows_usage(self):
        ch = _setup_channel_with_app()
        update = _make_update(
            text="/activation", chat_type="supergroup", chat_id=789,
        )
        ctx = MagicMock()
        await ch._cmd_activation(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "Usage:" in call_kwargs["text"]


# ===========================================================================
# 20. Text message forwarding
# ===========================================================================


class TestTextMessageHandler:
    @pytest.mark.asyncio
    async def test_private_message_forwarded(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="hi there", user_id=42, chat_type="private")
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        assert len(received) == 1
        assert received[0].content == "hi there"
        assert received[0].peer_id == "42"
        assert received[0].group_id is None

    @pytest.mark.asyncio
    async def test_group_message_ignored_without_mention(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="random chat", chat_type="group")
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_group_message_with_mention_forwarded(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(
            text="@morgan_bot what is the time?",
            chat_type="group",
        )
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        assert len(received) == 1
        assert "@morgan_bot" not in received[0].content
        assert "what is the time?" in received[0].content

    @pytest.mark.asyncio
    async def test_blocked_user_ignored(self):
        ch = _setup_channel_with_app(allowed_user_ids={"100"})
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="hi", user_id=999, chat_type="private")
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_processing_flag_set_during_handling(self):
        ch = _setup_channel_with_app()
        flag_during = {}

        async def handler(msg):
            flag_during.update(dict(ch._processing))

        ch.set_message_handler(handler)

        # For private chats, chat_key = str(peer_id), not chat_id
        update = _make_update(text="test", user_id=42, chat_id=456)
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        # Private chat: chat_key is str(peer_id) = "42"
        assert "42" in flag_during
        assert flag_during["42"] is True
        # After completion, flag is cleared
        assert "42" not in ch._processing


# ===========================================================================
# 21. Photo handling
# ===========================================================================


class TestPhotoHandler:
    @pytest.mark.asyncio
    async def test_photo_with_caption(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        photo = [SimpleNamespace(file_id="photo_123")]
        update = _make_update(
            caption="Look at this!", photo=photo, chat_type="private",
        )
        ctx = MagicMock()
        await ch._on_photo(update, ctx)

        assert len(received) == 1
        assert received[0].content == "Look at this!"
        assert received[0].metadata["has_photo"] is True
        assert received[0].metadata["media_type"] == "photo"

    @pytest.mark.asyncio
    async def test_photo_without_caption(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        photo = [SimpleNamespace(file_id="photo_123")]
        update = _make_update(photo=photo, caption=None, chat_type="private")
        ctx = MagicMock()
        await ch._on_photo(update, ctx)

        assert len(received) == 1
        assert "[Photo received]" in received[0].content


# ===========================================================================
# 22. Document handling
# ===========================================================================


class TestDocumentHandler:
    @pytest.mark.asyncio
    async def test_document_with_caption(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        doc = SimpleNamespace(file_id="doc_123", file_name="report.pdf")
        update = _make_update(
            document=doc, caption="Here's the file", chat_type="private",
        )
        ctx = MagicMock()
        await ch._on_document(update, ctx)

        assert len(received) == 1
        assert received[0].content == "Here's the file"
        assert received[0].metadata["has_document"] is True
        assert received[0].metadata["document_file_name"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_document_without_caption(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        doc = SimpleNamespace(file_id="doc_123", file_name="data.csv")
        update = _make_update(document=doc, caption=None, chat_type="private")
        ctx = MagicMock()
        await ch._on_document(update, ctx)

        assert len(received) == 1
        assert "data.csv" in received[0].content


# ===========================================================================
# 23. Voice handling
# ===========================================================================


class TestVoiceHandler:
    @pytest.mark.asyncio
    async def test_voice_message(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        voice = SimpleNamespace(file_id="voice_123", duration=5)
        update = _make_update(voice=voice, chat_type="private")
        ctx = MagicMock()
        await ch._on_voice(update, ctx)

        assert len(received) == 1
        assert "[Voice message received]" in received[0].content
        assert received[0].metadata["has_voice"] is True
        assert received[0].metadata["media_type"] == "voice"


# ===========================================================================
# 24. Sticker handling
# ===========================================================================


class TestStickerHandler:
    @pytest.mark.asyncio
    async def test_sticker_acknowledged(self):
        ch = _setup_channel_with_app()
        sticker = SimpleNamespace(file_id="sticker_123")
        update = _make_update(sticker=sticker)
        ctx = MagicMock()
        await ch._on_sticker(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "sticker" in call_kwargs["text"].lower()


# ===========================================================================
# 25. Edited message handling
# ===========================================================================


class TestEditedMessageHandler:
    @pytest.mark.asyncio
    async def test_edited_message_forwarded_with_flag(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(text="edited text", is_edited=True)
        ctx = MagicMock()
        await ch._on_edited_message(update, ctx)

        assert len(received) == 1
        assert received[0].content == "edited text"
        assert received[0].metadata["is_edited"] is True


# ===========================================================================
# 26. Callback queries
# ===========================================================================


class TestCallbackQueries:
    @pytest.mark.asyncio
    async def test_clear_callback(self):
        ch = _setup_channel_with_app()
        ch.set_message_handler(AsyncMock())

        update = _make_update(is_callback=True, callback_data="action:clear")
        ctx = MagicMock()
        await ch._on_callback_query(update, ctx)

        update.callback_query.answer.assert_called_once()
        # Confirmation sent
        assert ch._app.bot.send_message.called
        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "start fresh" in call_kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_compact_callback(self):
        ch = _setup_channel_with_app()
        ch.set_message_handler(AsyncMock())

        update = _make_update(is_callback=True, callback_data="action:compact")
        ctx = MagicMock()
        await ch._on_callback_query(update, ctx)

        call_kwargs = ch._app.bot.send_message.call_args[1]
        assert "compaction" in call_kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_stop_callback(self):
        ch = _setup_channel_with_app()
        ch._processing["456"] = True

        update = _make_update(
            is_callback=True, callback_data="action:stop", chat_id=456,
        )
        ctx = MagicMock()
        await ch._on_callback_query(update, ctx)

        assert ch._processing.get("456") is False

    @pytest.mark.asyncio
    async def test_tool_callback_forwarded(self):
        ch = _setup_channel_with_app()
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(is_callback=True, callback_data="tool:calculator")
        ctx = MagicMock()
        await ch._on_callback_query(update, ctx)

        assert len(received) == 1
        assert received[0].content == "tool:calculator"
        assert received[0].metadata.get("is_callback") is True

    @pytest.mark.asyncio
    async def test_callback_no_query(self):
        ch = _setup_channel_with_app()
        update = _make_update()
        update.callback_query = None
        ctx = MagicMock()
        # Should return early without error
        await ch._on_callback_query(update, ctx)

    @pytest.mark.asyncio
    async def test_callback_blocked_user(self):
        ch = _setup_channel_with_app(allowed_user_ids={"100"})
        update = _make_update(
            is_callback=True, callback_data="action:clear", user_id=999,
        )
        ctx = MagicMock()
        await ch._on_callback_query(update, ctx)

        # Should have answered query but not processed
        update.callback_query.answer.assert_called_once()
        assert not ch._app.bot.send_message.called


# ===========================================================================
# 27. Forum / topic support
# ===========================================================================


class TestForumTopicSupport:
    @pytest.mark.asyncio
    async def test_thread_id_included_in_group_id(self):
        ch = _setup_channel_with_app()
        ch._require_mention_in_groups = False
        received: list[InboundMessage] = []

        async def handler(msg):
            received.append(msg)

        ch.set_message_handler(handler)

        update = _make_update(
            text="topic message",
            chat_type="supergroup",
            message_thread_id=42,
        )
        ctx = MagicMock()
        await ch._on_text_message(update, ctx)

        assert len(received) == 1
        assert received[0].group_id == "456:42"
        assert received[0].metadata.get("message_thread_id") == 42


# ===========================================================================
# 28. Typing indicator
# ===========================================================================


class TestTypingIndicator:
    @pytest.mark.asyncio
    async def test_start_typing_creates_task(self):
        ch = _setup_channel_with_app()
        ch._start_typing(456)
        assert "456" in ch._typing_tasks
        ch._stop_typing("456")

    @pytest.mark.asyncio
    async def test_stop_typing_removes_task(self):
        ch = _setup_channel_with_app()
        ch._start_typing(456)
        ch._stop_typing("456")
        assert "456" not in ch._typing_tasks

    def test_stop_typing_nonexistent(self):
        ch = _setup_channel_with_app()
        ch._stop_typing("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_start_typing_replaces_existing(self):
        ch = _setup_channel_with_app()
        ch._start_typing(456)
        old_task = ch._typing_tasks["456"]
        ch._start_typing(456)
        new_task = ch._typing_tasks["456"]
        assert old_task is not new_task
        ch._stop_typing("456")


# ===========================================================================
# 29. Memory status helpers
# ===========================================================================


class TestMemoryHelpers:
    def test_memory_status_text(self):
        ch = _make_channel()
        status = ch._get_memory_status_text()
        assert isinstance(status, str)
        assert len(status) > 0

    def test_memory_detail_text(self):
        ch = _make_channel()
        detail = ch._get_memory_detail_text()
        assert "Memory Status" in detail
        assert "Active conversations" in detail


# ===========================================================================
# 30. get_chat_id helper
# ===========================================================================


class TestGetChatId:
    def test_from_chat(self):
        ch = _make_channel()
        ch._bot_username = "bot"
        ch._bot_id = 1
        update = _make_update(chat_id=789)
        assert ch._get_chat_id(update) == 789

    def test_from_user_when_no_chat(self):
        ch = _make_channel()
        ch._bot_username = "bot"
        ch._bot_id = 1
        update = _make_update(user_id=42)
        update.effective_chat = None
        assert ch._get_chat_id(update) == 42


# ===========================================================================
# 31. validate_update
# ===========================================================================


class TestValidateUpdate:
    def test_valid_update(self):
        ch = _make_channel()
        update = _make_update()
        assert ch._validate_update(update) is True

    def test_no_message_or_edit_or_callback(self):
        ch = _make_channel()
        update = SimpleNamespace(
            effective_message=None,
            edited_message=None,
            callback_query=None,
            effective_user=SimpleNamespace(id=1),
        )
        assert ch._validate_update(update) is False

    def test_no_user(self):
        ch = _make_channel()
        update = SimpleNamespace(
            effective_message=SimpleNamespace(text="hi", message_id=1),
            edited_message=None,
            callback_query=None,
            effective_user=None,
        )
        assert ch._validate_update(update) is False

    def test_edited_message_is_valid(self):
        ch = _make_channel()
        update = _make_update(is_edited=True)
        assert ch._validate_update(update) is True

    def test_callback_query_is_valid(self):
        ch = _make_channel()
        update = _make_update(is_callback=True, callback_data="test")
        assert ch._validate_update(update) is True


# ===========================================================================
# 32. Quick action buttons
# ===========================================================================


class TestQuickActionButtons:
    def test_returns_none_without_telegram_lib(self):
        from morgan.channels.telegram_channel import _HAS_TELEGRAM as has_tg

        ch = _make_channel()
        result = ch._quick_action_buttons()
        if has_tg:
            assert result is not None
        else:
            assert result is None


# ===========================================================================
# 33. stop() lifecycle
# ===========================================================================


class TestStopLifecycle:
    @pytest.mark.asyncio
    async def test_stop_without_app(self):
        ch = _make_channel()
        await ch.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_cancels_typing_tasks(self):
        ch = _setup_channel_with_app()
        ch._start_typing(100)
        ch._start_typing(200)
        assert len(ch._typing_tasks) == 2

        # Mock updater to None so shutdown doesn't try to stop it
        ch._app.updater = None

        import morgan.channels.telegram_channel as tg_mod
        original = tg_mod._HAS_TELEGRAM
        # Need _HAS_TELEGRAM True for stop() to attempt shutdown
        tg_mod._HAS_TELEGRAM = True
        try:
            await ch.stop()
        finally:
            tg_mod._HAS_TELEGRAM = original

        assert len(ch._typing_tasks) == 0


# ===========================================================================
# 34. MORGAN_TOOLS constant
# ===========================================================================


class TestMorganToolsConstant:
    def test_has_expected_tools(self):
        from morgan.channels.telegram_channel import MORGAN_TOOLS

        tool_names = [t[0] for t in MORGAN_TOOLS]
        assert "calculator" in tool_names
        assert "bash" in tool_names
        assert "file_read" in tool_names
        assert "web_search" in tool_names
        assert "memory_search" in tool_names

    def test_all_have_descriptions(self):
        from morgan.channels.telegram_channel import MORGAN_TOOLS

        for name, desc in MORGAN_TOOLS:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 0


# ===========================================================================
# 35. Import handling
# ===========================================================================


class TestImportHandling:
    def test_module_imports(self):
        from morgan.channels.telegram_channel import TelegramChannel

        assert TelegramChannel is not None

    def test_has_telegram_flag_exists(self):
        from morgan.channels.telegram_channel import _HAS_TELEGRAM

        assert isinstance(_HAS_TELEGRAM, bool)

    def test_channel_can_import_from_package(self):
        from morgan.channels import TelegramChannel

        assert TelegramChannel is not None


# ===========================================================================
# 36. send_with_keyboard helper
# ===========================================================================


class TestSendWithKeyboard:
    @pytest.mark.asyncio
    async def test_send_without_app(self):
        ch = _make_channel()
        ch._app = None
        # Should not raise
        await ch.send_with_keyboard(
            chat_id=123, text="test",
            buttons=[[("btn", "data")]],
        )


# ===========================================================================
# 37. Conversation tracking
# ===========================================================================


class TestConversationTracking:
    def test_initial_conversations_empty(self):
        ch = _make_channel()
        assert ch._conversations == {}

    def test_conversations_can_be_set(self):
        ch = _make_channel()
        ch._conversations["123"] = ["conv_abc"]
        assert ch._conversations["123"] == ["conv_abc"]

    @pytest.mark.asyncio
    async def test_clear_command_resets_group_conversation(self):
        ch = _setup_channel_with_app()
        ch.set_message_handler(AsyncMock())
        # Set up a group conversation
        ch._conversations["789"] = ["conv_1"]

        update = _make_update(
            text="/clear", user_id=1, chat_id=789, chat_type="supergroup",
        )
        ctx = MagicMock()
        await ch._cmd_clear(update, ctx)

        assert "789" not in ch._conversations
