"""
Telegram channel adapter for the multi-channel gateway.

Full-featured Telegram bot using ``python-telegram-bot`` v20+ with
OpenClaw-style features.

Supports:
- Polling (default) and webhook mode
- Full command suite: /start, /help, /clear, /reset, /model, /status,
  /whoami, /id, /tools, /skills, /compact, /stop, /memory, /settings,
  /activation
- Text, photo, document, voice, sticker, and edited message handling
- Group chat with @mention / reply-to-bot filtering and per-group activation
- Forum / topic support (per-topic routing via message_thread_id)
- Persistent typing indicator (re-sent every 4s while processing)
- HTML parse_mode with Markdown-to-HTML conversion and entity escaping
- Message chunking at 4096-char limit
- Inline keyboard buttons for tools, skills, and quick actions
- Per-second rate limiting and retry with exponential backoff
- Rate limit (429) detection with wait-and-retry
- HTML parse error fallback to plain text
- User allowlist (empty = allow all)
- Bot menu registration via set_my_commands
- Conversation tracking and abort flag for /stop
- Graceful shutdown

If ``python-telegram-bot`` is not installed the module still imports; calling
:meth:`start` raises a descriptive :class:`RuntimeError`.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from typing import Any, List, Optional, Tuple

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    from telegram import (
        Bot,
        BotCommand,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ChatAction, ParseMode
    from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler as TGMessageHandler,
        filters,
    )

    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
    logger.debug(
        "python-telegram-bot is not installed. "
        "TelegramChannel will not be able to start."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Available Morgan tools (displayed in /tools)
MORGAN_TOOLS = [
    ("calculator", "Evaluate mathematical expressions"),
    ("bash", "Run shell commands"),
    ("file_read", "Read file contents"),
    ("web_search", "Search the web"),
    ("memory_search", "Search Morgan's memory"),
]

# Default available skills (displayed in /skills)
MORGAN_DEFAULT_SKILLS: list[tuple[str, str]] = []


class TelegramChannel(BaseChannel):
    """Channel adapter for Telegram using python-telegram-bot v20+.

    Full-featured implementation with OpenClaw-style commands, inline
    keyboards, HTML formatting, typing indicators, and group chat support.

    Args:
        token: Telegram Bot API token.
        allowed_user_ids: Set of user-ID strings permitted to interact.
            ``None`` or empty means *all* users are allowed.
        require_mention_in_groups: If ``True`` the bot only responds in
            group chats when @mentioned or replied-to.
        webhook_url: If set, the bot runs in webhook mode at this URL
            instead of long-polling.
        webhook_port: Port for the webhook server (default 8443).
        max_message_length: Maximum characters per Telegram message
            (default 4096, the platform limit).
        model_name: Name of the current LLM model.
        bot_name: Display name for the bot.
    """

    def __init__(
        self,
        token: str,
        allowed_user_ids: Optional[set[str]] = None,
        require_mention_in_groups: bool = True,
        webhook_url: Optional[str] = None,
        webhook_port: int = 8443,
        max_message_length: int = 4096,
        model_name: str = "qwen3.5:35b",
        bot_name: str = "Morgan",
    ) -> None:
        super().__init__()
        self._token = token
        self._allowed_user_ids: set[str] = allowed_user_ids or set()
        self._require_mention_in_groups = require_mention_in_groups
        self._webhook_url = webhook_url
        self._webhook_port = webhook_port
        self._max_message_length = max_message_length
        self._model_name = model_name
        self._bot_name = bot_name

        self._app: Optional[Any] = None  # telegram.ext.Application
        self._bot_username: Optional[str] = None
        self._bot_id: Optional[int] = None

        # Per-chat conversation ID tracking
        self._conversations: dict[str, list] = {}

        # Uptime tracking
        self._start_time: float = 0.0

        # Abort flag per chat for /stop
        self._processing: dict[str, bool] = {}

        # Per-group activation mode: "mention" or "always"
        self._group_activation: dict[str, str] = {}

        # Rate-limiting state: timestamps of recent sends
        self._send_timestamps: list[float] = []
        self._rate_limit_per_second: float = 30.0  # Telegram limit ~30 msg/s

        # Retry configuration
        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0

        # Typing indicator tasks per chat
        self._typing_tasks: dict[str, asyncio.Task] = {}

    # -- BaseChannel interface -----------------------------------------------

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Initialize the bot application, register handlers, register
        BotFather menu commands, and begin polling (or start the webhook)."""
        if not _HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )

        self._start_time = time.time()

        self._app = Application.builder().token(self._token).build()

        # Register command handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("clear", self._cmd_clear))
        self._app.add_handler(CommandHandler("reset", self._cmd_clear))
        self._app.add_handler(CommandHandler("model", self._cmd_model))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("whoami", self._cmd_whoami))
        self._app.add_handler(CommandHandler("id", self._cmd_whoami))
        self._app.add_handler(CommandHandler("tools", self._cmd_tools))
        self._app.add_handler(CommandHandler("skills", self._cmd_skills))
        self._app.add_handler(CommandHandler("compact", self._cmd_compact))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("memory", self._cmd_memory))
        self._app.add_handler(CommandHandler("settings", self._cmd_settings))
        self._app.add_handler(CommandHandler("activation", self._cmd_activation))

        # Text messages (non-command)
        self._app.add_handler(
            TGMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._on_text_message,
            )
        )

        # Photo messages
        self._app.add_handler(
            TGMessageHandler(filters.PHOTO, self._on_photo)
        )

        # Document messages
        self._app.add_handler(
            TGMessageHandler(filters.Document.ALL, self._on_document)
        )

        # Voice messages
        self._app.add_handler(
            TGMessageHandler(filters.VOICE, self._on_voice)
        )

        # Sticker messages
        self._app.add_handler(
            TGMessageHandler(filters.Sticker.ALL, self._on_sticker)
        )

        # Edited messages
        self._app.add_handler(
            TGMessageHandler(
                filters.UpdateType.EDITED_MESSAGE,
                self._on_edited_message,
            )
        )

        # Callback-query handler (inline keyboard buttons)
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))

        # Error handler — python-telegram-bot silently swallows exceptions without this
        async def _error_handler(update, context):
            logger.error("Telegram handler error: %s", context.error, exc_info=context.error)
        self._app.add_error_handler(_error_handler)

        await self._app.initialize()

        # Fetch bot info
        bot_info = await self._app.bot.get_me()
        self._bot_username = bot_info.username
        self._bot_id = bot_info.id
        logger.info(
            "Telegram bot authenticated as @%s (id=%s)",
            self._bot_username,
            self._bot_id,
        )

        # Register BotFather menu commands
        commands = [
            BotCommand("start", "Start chatting with Morgan"),
            BotCommand("help", "Show available commands"),
            BotCommand("clear", "Clear conversation history"),
            BotCommand("model", "Show current AI model"),
            BotCommand("status", "Show bot status"),
            BotCommand("tools", "List available tools"),
            BotCommand("skills", "List available skills"),
            BotCommand("memory", "Show memory status"),
            BotCommand("compact", "Compact conversation context"),
            BotCommand("whoami", "Show your Telegram ID"),
            BotCommand("settings", "Show channel settings"),
            BotCommand("stop", "Stop current processing"),
        ]
        try:
            await self._app.bot.set_my_commands(commands)
            logger.info("Registered %d bot commands with BotFather.", len(commands))
        except Exception:
            logger.warning("Failed to register bot commands with BotFather.", exc_info=True)

        await self._app.start()

        if self._webhook_url:
            logger.info(
                "Starting Telegram webhook at %s (port %d)",
                self._webhook_url,
                self._webhook_port,
            )
            await self._app.updater.start_webhook(
                listen="0.0.0.0",
                port=self._webhook_port,
                url_path=self._token,
                webhook_url=f"{self._webhook_url}/{self._token}",
            )
        else:
            logger.info("Starting Telegram channel polling...")
            await self._app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=False,
            )

    async def stop(self) -> None:
        """Graceful shutdown."""
        # Cancel all typing indicator tasks
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self._app is not None and _HAS_TELEGRAM:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                logger.exception("Error during Telegram channel shutdown")
            logger.info("Telegram channel stopped.")

    async def send(self, message: OutboundMessage) -> None:
        """Send an outbound message with chunking, HTML formatting,
        and typing indicator."""
        if self._app is None:
            logger.warning("Telegram channel is not started; cannot send.")
            return

        chat_id = int(message.group_id.split(":")[0] if message.group_id else message.peer_id)

        # Optional: reply to a specific message_thread_id for forum topics
        thread_id = message.metadata.get("message_thread_id")

        # Stop typing indicator for this chat
        self._stop_typing(str(chat_id))

        content = message.content
        if not content or not content.strip():
            content = "I couldn't generate a response. Please try again."

        # Convert markdown to HTML
        formatted = self._markdown_to_html(content)

        chunks = self._chunk_message(formatted)
        for i, chunk in enumerate(chunks):
            # Add quick-action buttons to the last chunk
            reply_markup = None
            if i == len(chunks) - 1:
                reply_markup = self._quick_action_buttons()

            await self._send_with_retry(
                chat_id=chat_id,
                text=chunk,
                thread_id=thread_id,
                reply_markup=reply_markup,
                parse_mode_str="HTML",
            )

    # -- Command handlers ----------------------------------------------------

    async def _cmd_start(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /start command."""
        logger.info("Received /start from %s", getattr(update.effective_user, "first_name", "?"))
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        await self._send_with_retry(
            chat_id=chat_id,
            text=(
                "Hi! I'm Morgan, your AI assistant with emotional intelligence. "
                "Send me a message to start chatting! Use /help to see all commands."
            ),
        )

    async def _cmd_help(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /help command."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        help_text = (
            "<b>Morgan Commands</b>\n\n"
            "<b>General</b>\n"
            "/start - Start chatting with Morgan\n"
            "/help - Show this help message\n"
            "/status - Show bot status and uptime\n"
            "/settings - Show channel settings\n\n"
            "<b>Conversation</b>\n"
            "/clear - Clear conversation history\n"
            "/reset - Alias for /clear\n"
            "/compact - Compact conversation context\n"
            "/stop - Cancel current processing\n\n"
            "<b>Information</b>\n"
            "/model - Show current AI model\n"
            "/whoami - Show your Telegram ID and name\n"
            "/id - Alias for /whoami\n"
            "/tools - List available tools\n"
            "/skills - List available skills\n"
            "/memory - Show memory status\n\n"
            "<b>Group Settings</b>\n"
            "/activation mention|always - Set group activation mode\n\n"
            "You can also just send me a message and I'll do my best to help!"
        )
        await self._send_with_retry(
            chat_id=chat_id,
            text=help_text,
            parse_mode_str="HTML",
        )

    async def _cmd_clear(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /clear and /reset commands."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        chat_id = self._get_chat_id(update)
        is_group = chat is not None and chat.type in ("group", "supergroup")
        group_id = str(chat.id) if is_group else None

        # Clear local conversation tracking
        conv_key = group_id or peer_id
        self._conversations.pop(conv_key, None)

        # Send clear action through gateway
        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content="/clear",
            group_id=group_id,
            metadata={
                "command": "clear",
                "action": "clear",
                "message_id": effective_message.message_id,
            },
        )
        await self.on_message(msg)

        await self._send_with_retry(
            chat_id=chat_id,
            text="Conversation cleared. Let's start fresh!",
        )

    async def _cmd_model(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /model command."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        await self._send_with_retry(
            chat_id=chat_id,
            text=f"Current model: {self._model_name}",
        )

    async def _cmd_status(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /status command."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)

        uptime_seconds = time.time() - self._start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)

        tools_count = len(MORGAN_TOOLS)
        active_convs = len(self._conversations)

        # Check memory status
        memory_status = self._get_memory_status_text()

        status_text = (
            f"<b>{self._bot_name} Status</b>\n\n"
            f"Uptime: {hours}h {minutes}m {seconds}s\n"
            f"Model: {self._model_name}\n"
            f"Tools: {tools_count} available\n"
            f"Active conversations: {active_convs}\n"
            f"Memory: {memory_status}\n"
            f"Bot: @{self._bot_username or 'unknown'}"
        )
        await self._send_with_retry(
            chat_id=chat_id,
            text=status_text,
            parse_mode_str="HTML",
        )

    async def _cmd_whoami(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /whoami and /id commands."""
        if not self._check_access(update):
            return

        effective_user = update.effective_user
        if effective_user is None:
            return

        chat_id = self._get_chat_id(update)
        user_name = effective_user.full_name or effective_user.username or "Unknown"
        username_str = f"@{effective_user.username}" if effective_user.username else "not set"

        text = (
            f"<b>Your Info</b>\n\n"
            f"Telegram ID: <code>{effective_user.id}</code>\n"
            f"Name: {html.escape(user_name)}\n"
            f"Username: {username_str}"
        )
        await self._send_with_retry(
            chat_id=chat_id,
            text=text,
            parse_mode_str="HTML",
        )

    async def _cmd_tools(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /tools command with inline keyboard buttons."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)

        text = "<b>Available Tools</b>\n\n"
        reply_markup = None
        if _HAS_TELEGRAM:
            try:
                buttons = []
                for tool_name, tool_desc in MORGAN_TOOLS:
                    text += f"<b>{tool_name}</b> - {tool_desc}\n"
                    buttons.append(
                        [InlineKeyboardButton(
                            f"{tool_name}",
                            callback_data=f"tool:{tool_name}",
                        )]
                    )
                reply_markup = InlineKeyboardMarkup(buttons)
            except NameError:
                for tool_name, tool_desc in MORGAN_TOOLS:
                    text += f"<b>{tool_name}</b> - {tool_desc}\n"
        else:
            for tool_name, tool_desc in MORGAN_TOOLS:
                text += f"<b>{tool_name}</b> - {tool_desc}\n"

        await self._send_with_retry(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode_str="HTML",
        )

    async def _cmd_skills(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /skills command with inline keyboard buttons."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)

        # Try to load skills from the skill registry if available
        skills = self._get_available_skills()

        if not skills:
            await self._send_with_retry(
                chat_id=chat_id,
                text="No skills are currently loaded. Skills can be added as markdown files in the skills directory.",
            )
            return

        text = "<b>Available Skills</b>\n\n"
        reply_markup = None
        if _HAS_TELEGRAM:
            try:
                buttons = []
                for skill_name, skill_desc in skills:
                    text += f"<b>{skill_name}</b> - {skill_desc}\n"
                    buttons.append(
                        [InlineKeyboardButton(
                            f"{skill_name}",
                            callback_data=f"skill:{skill_name}",
                        )]
                    )
                reply_markup = InlineKeyboardMarkup(buttons)
            except NameError:
                for skill_name, skill_desc in skills:
                    text += f"<b>{skill_name}</b> - {skill_desc}\n"
        else:
            for skill_name, skill_desc in skills:
                text += f"<b>{skill_name}</b> - {skill_desc}\n"

        await self._send_with_retry(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode_str="HTML",
        )

    async def _cmd_compact(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /compact command - trigger context compaction."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        chat_id = self._get_chat_id(update)
        is_group = chat is not None and chat.type in ("group", "supergroup")
        group_id = str(chat.id) if is_group else None

        # Send compact action through gateway
        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content="/compact",
            group_id=group_id,
            metadata={
                "command": "compact",
                "action": "compact",
                "message_id": effective_message.message_id,
            },
        )
        await self.on_message(msg)

        await self._send_with_retry(
            chat_id=chat_id,
            text="Context compaction triggered. Conversation context has been condensed.",
        )

    async def _cmd_stop(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /stop command - cancel current processing."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        chat_key = str(chat_id)

        # Set abort flag
        self._processing[chat_key] = False

        # Stop typing indicator
        self._stop_typing(chat_key)

        await self._send_with_retry(
            chat_id=chat_id,
            text="Processing cancelled.",
        )

    async def _cmd_memory(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /memory command - show memory status."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        memory_detail = self._get_memory_detail_text()

        await self._send_with_retry(
            chat_id=chat_id,
            text=memory_detail,
            parse_mode_str="HTML",
        )

    async def _cmd_settings(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /settings command - show channel settings."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        chat = update.effective_chat

        is_group = chat is not None and chat.type in ("group", "supergroup")
        group_id = str(chat.id) if is_group else None
        activation_mode = self._group_activation.get(group_id, "mention") if group_id else "N/A (DM)"

        allowlist_str = (
            ", ".join(sorted(self._allowed_user_ids))
            if self._allowed_user_ids
            else "all users allowed"
        )

        text = (
            f"<b>Channel Settings</b>\n\n"
            f"Model: {self._model_name}\n"
            f"Bot name: {self._bot_name}\n"
            f"Mention mode: {'required in groups' if self._require_mention_in_groups else 'respond to all'}\n"
            f"Group activation: {activation_mode}\n"
            f"Allowlist: {allowlist_str}\n"
            f"Max message length: {self._max_message_length}\n"
            f"Webhook: {'enabled' if self._webhook_url else 'polling mode'}"
        )
        await self._send_with_retry(
            chat_id=chat_id,
            text=text,
            parse_mode_str="HTML",
        )

    async def _cmd_activation(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /activation command - toggle group activation mode."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        chat = update.effective_chat
        chat_id = self._get_chat_id(update)

        if effective_message is None:
            return

        is_group = chat is not None and chat.type in ("group", "supergroup")
        if not is_group:
            await self._send_with_retry(
                chat_id=chat_id,
                text="The /activation command is only available in group chats.",
            )
            return

        group_id = str(chat.id)

        # Parse argument
        text = effective_message.text or ""
        parts = text.split()
        if len(parts) < 2 or parts[1].lower() not in ("mention", "always"):
            current = self._group_activation.get(group_id, "mention")
            await self._send_with_retry(
                chat_id=chat_id,
                text=(
                    f"Current activation mode: <b>{current}</b>\n\n"
                    f"Usage: /activation mention|always\n"
                    f"  <b>mention</b> - Only respond when @mentioned or replied to\n"
                    f"  <b>always</b> - Respond to all messages"
                ),
                parse_mode_str="HTML",
            )
            return

        mode = parts[1].lower()
        self._group_activation[group_id] = mode

        await self._send_with_retry(
            chat_id=chat_id,
            text=f"Group activation mode set to: <b>{mode}</b>",
            parse_mode_str="HTML",
        )

    # -- Message handlers ----------------------------------------------------

    async def _on_text_message(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle incoming text messages."""
        logger.info(
            "Received text message from %s: %s",
            getattr(update.effective_user, "first_name", "?"),
            (update.effective_message.text or "")[:100] if update.effective_message else "?",
        )
        if not self._check_access(update):
            logger.info("Access denied for user %s", getattr(update.effective_user, "id", "?"))
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        is_group = chat is not None and chat.type in ("group", "supergroup")

        # Group mention / reply requirement
        if is_group and not self._should_respond_in_group(update):
            return

        text = effective_message.text or ""

        # Strip bot mention from text in groups
        if is_group and self._bot_username:
            text = re.sub(
                rf"@{re.escape(self._bot_username)}\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

        group_id = str(chat.id) if is_group else None

        # Forum / topic support
        thread_id = getattr(effective_message, "message_thread_id", None)
        if thread_id is not None and group_id is not None:
            group_id = f"{chat.id}:{thread_id}"

        chat_id = int(group_id.split(":")[0] if group_id else peer_id)
        chat_key = str(chat_id)

        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
        }
        if thread_id is not None:
            metadata["message_thread_id"] = thread_id

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=text,
            group_id=group_id,
            metadata=metadata,
        )

        # Set processing flag and start typing indicator
        self._processing[chat_key] = True
        self._start_typing(chat_id, thread_id)

        try:
            await self.on_message(msg)
        finally:
            self._processing.pop(chat_key, None)
            self._stop_typing(chat_key)

    async def _on_photo(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle photo messages - extract caption and note photo in metadata."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        is_group = chat is not None and chat.type in ("group", "supergroup")
        if is_group and not self._should_respond_in_group(update):
            return

        caption = effective_message.caption or ""
        group_id = str(chat.id) if is_group else None

        thread_id = getattr(effective_message, "message_thread_id", None)
        if thread_id is not None and group_id is not None:
            group_id = f"{chat.id}:{thread_id}"

        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
            "has_photo": True,
            "media_type": "photo",
        }
        if thread_id is not None:
            metadata["message_thread_id"] = thread_id
        if effective_message.photo:
            metadata["photo_file_id"] = effective_message.photo[-1].file_id

        content = caption if caption else "[Photo received]"

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=content,
            group_id=group_id,
            metadata=metadata,
        )

        chat_id = int(group_id.split(":")[0] if group_id else peer_id)
        self._start_typing(chat_id, thread_id)
        try:
            await self.on_message(msg)
        finally:
            self._stop_typing(str(chat_id))

    async def _on_document(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle document messages - extract caption and note document name."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        is_group = chat is not None and chat.type in ("group", "supergroup")
        if is_group and not self._should_respond_in_group(update):
            return

        caption = effective_message.caption or ""
        group_id = str(chat.id) if is_group else None

        thread_id = getattr(effective_message, "message_thread_id", None)
        if thread_id is not None and group_id is not None:
            group_id = f"{chat.id}:{thread_id}"

        doc = effective_message.document
        doc_name = doc.file_name if doc else "unknown"

        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
            "has_document": True,
            "media_type": "document",
            "document_file_name": doc_name,
        }
        if thread_id is not None:
            metadata["message_thread_id"] = thread_id
        if doc:
            metadata["document_file_id"] = doc.file_id

        content = caption if caption else f"[Document received: {doc_name}]"

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=content,
            group_id=group_id,
            metadata=metadata,
        )

        chat_id = int(group_id.split(":")[0] if group_id else peer_id)
        self._start_typing(chat_id, thread_id)
        try:
            await self.on_message(msg)
        finally:
            self._stop_typing(str(chat_id))

    async def _on_voice(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle voice messages - note voice received (future: transcription)."""
        if not self._check_access(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        is_group = chat is not None and chat.type in ("group", "supergroup")
        if is_group and not self._should_respond_in_group(update):
            return

        group_id = str(chat.id) if is_group else None

        thread_id = getattr(effective_message, "message_thread_id", None)
        if thread_id is not None and group_id is not None:
            group_id = f"{chat.id}:{thread_id}"

        voice = effective_message.voice
        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
            "has_voice": True,
            "media_type": "voice",
        }
        if thread_id is not None:
            metadata["message_thread_id"] = thread_id
        if voice:
            metadata["voice_file_id"] = voice.file_id
            metadata["voice_duration"] = voice.duration

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content="[Voice message received]",
            group_id=group_id,
            metadata=metadata,
        )

        chat_id = int(group_id.split(":")[0] if group_id else peer_id)
        self._start_typing(chat_id, thread_id)
        try:
            await self.on_message(msg)
        finally:
            self._stop_typing(str(chat_id))

    async def _on_sticker(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle sticker messages - acknowledge but don't process."""
        if not self._check_access(update):
            return

        chat_id = self._get_chat_id(update)
        await self._send_with_retry(
            chat_id=chat_id,
            text="I received your sticker! \U0001f60a",
        )

    async def _on_edited_message(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle edited messages - process as new messages with edit flag."""
        if not self._check_access(update):
            return

        effective_message = update.edited_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        is_group = chat is not None and chat.type in ("group", "supergroup")

        if is_group and not self._should_respond_in_group(update):
            return

        text = effective_message.text or ""

        if is_group and self._bot_username:
            text = re.sub(
                rf"@{re.escape(self._bot_username)}\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

        group_id = str(chat.id) if is_group else None

        thread_id = getattr(effective_message, "message_thread_id", None)
        if thread_id is not None and group_id is not None:
            group_id = f"{chat.id}:{thread_id}"

        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
            "is_edited": True,
        }
        if thread_id is not None:
            metadata["message_thread_id"] = thread_id

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=text,
            group_id=group_id,
            metadata=metadata,
        )

        chat_id = int(group_id.split(":")[0] if group_id else peer_id)
        chat_key = str(chat_id)
        self._processing[chat_key] = True
        self._start_typing(chat_id, thread_id)

        try:
            await self.on_message(msg)
        finally:
            self._processing.pop(chat_key, None)
            self._stop_typing(chat_key)

    async def _on_callback_query(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        if query is None:
            return

        await query.answer()

        user = query.from_user
        if user is None:
            return

        peer_id = str(user.id)
        if not self._is_user_allowed(peer_id):
            return

        chat = update.effective_chat
        is_group = chat is not None and chat.type in ("group", "supergroup")
        group_id = str(chat.id) if is_group else None
        chat_id = self._get_chat_id(update)

        callback_data = query.data or ""

        # Handle quick-action buttons directly
        if callback_data == "action:clear":
            # Simulate /clear command
            self._conversations.pop(group_id or peer_id, None)
            msg = InboundMessage(
                channel="telegram",
                peer_id=peer_id,
                content="/clear",
                group_id=group_id,
                metadata={"command": "clear", "action": "clear"},
            )
            await self.on_message(msg)
            await self._send_with_retry(
                chat_id=chat_id,
                text="Conversation cleared. Let's start fresh!",
            )
            return

        if callback_data == "action:compact":
            msg = InboundMessage(
                channel="telegram",
                peer_id=peer_id,
                content="/compact",
                group_id=group_id,
                metadata={"command": "compact", "action": "compact"},
            )
            await self.on_message(msg)
            await self._send_with_retry(
                chat_id=chat_id,
                text="Context compaction triggered. Conversation context has been condensed.",
            )
            return

        if callback_data == "action:stop":
            chat_key = str(chat_id)
            self._processing[chat_key] = False
            self._stop_typing(chat_key)
            await self._send_with_retry(
                chat_id=chat_id,
                text="Processing cancelled.",
            )
            return

        # Forward tool/skill callbacks as messages to the gateway
        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=callback_data,
            group_id=group_id,
            metadata={
                "callback_query_id": query.id,
                "message_id": query.message.message_id if query.message else None,
                "is_callback": True,
            },
        )
        await self.on_message(msg)

    # -- Public helpers ------------------------------------------------------

    async def send_with_keyboard(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[tuple[str, str]]],
        thread_id: Optional[int] = None,
    ) -> None:
        """Send a message with an inline keyboard.

        Args:
            chat_id: Target chat.
            text: Message text.
            buttons: Rows of (label, callback_data) tuples.
            thread_id: Optional forum topic thread id.
        """
        if not _HAS_TELEGRAM or self._app is None:
            return

        keyboard = [
            [InlineKeyboardButton(label, callback_data=data) for label, data in row]
            for row in buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await self._send_with_retry(
            chat_id=chat_id,
            text=text,
            thread_id=thread_id,
            reply_markup=reply_markup,
        )

    # -- Internal helpers ----------------------------------------------------

    def _check_access(self, update: "Update") -> bool:
        """Validate update and check user allowlist. Returns True if allowed."""
        if not self._validate_update(update):
            return False
        effective_user = update.effective_user
        if effective_user is None:
            return False
        peer_id = str(effective_user.id)
        if not self._is_user_allowed(peer_id):
            logger.debug("User %s not in allowlist; ignoring.", peer_id)
            return False
        return True

    def _get_chat_id(self, update: "Update") -> int:
        """Extract the chat ID from an update."""
        chat = update.effective_chat
        if chat is not None:
            return chat.id
        effective_user = update.effective_user
        if effective_user is not None:
            return effective_user.id
        return 0

    def _validate_update(self, update: "Update") -> bool:
        """Return True if the update contains the minimum required fields."""
        return (
            update is not None
            and (update.effective_message is not None or update.edited_message is not None
                 or update.callback_query is not None)
            and update.effective_user is not None
        )

    def _is_user_allowed(self, peer_id: str) -> bool:
        """Check if a user is in the allowlist (empty list = allow all)."""
        if not self._allowed_user_ids:
            return True
        return peer_id in self._allowed_user_ids

    def _should_respond_in_group(self, update: "Update") -> bool:
        """Determine whether the bot should respond in a group context.

        Checks per-group activation mode first, then falls back to the
        default require_mention_in_groups setting.

        Returns ``True`` if:
        - Per-group activation is "always", or
        - ``require_mention_in_groups`` is ``False``, or
        - the message contains ``@bot_username``, or
        - the message is a reply to the bot.
        """
        chat = update.effective_chat
        if chat is not None:
            group_id = str(chat.id)
            activation = self._group_activation.get(group_id)
            if activation == "always":
                return True

        if not self._require_mention_in_groups:
            return True

        effective_message = update.effective_message or update.edited_message
        if effective_message is None:
            return False

        # Check @mention
        text = effective_message.text or effective_message.caption or ""
        if self._bot_username and f"@{self._bot_username}" in text:
            return True

        # Check reply-to-bot
        reply = effective_message.reply_to_message
        if reply and reply.from_user and reply.from_user.id == self._bot_id:
            return True

        return False

    def _quick_action_buttons(self) -> Any:
        """Create quick-action inline keyboard buttons."""
        if not _HAS_TELEGRAM:
            return None
        try:
            return InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Clear", callback_data="action:clear"),
                    InlineKeyboardButton("Compact", callback_data="action:compact"),
                    InlineKeyboardButton("Stop", callback_data="action:stop"),
                ]
            ])
        except NameError:
            return None

    def _get_available_skills(self) -> list[tuple[str, str]]:
        """Try to load available skills from the skill registry."""
        try:
            from morgan.skills import SkillRegistry
            # If skills are loaded globally, try to access them
            # For now, return the default list
            return MORGAN_DEFAULT_SKILLS
        except ImportError:
            return MORGAN_DEFAULT_SKILLS

    def _get_memory_status_text(self) -> str:
        """Get a short memory status string."""
        try:
            from morgan.memory_consolidation import DailyLogManager
            from morgan.workspace import WorkspaceManager

            workspace = WorkspaceManager()
            memory_path = workspace.workspace_path / "MEMORY.md"
            if memory_path.exists():
                size = memory_path.stat().st_size
                if size > 1024:
                    return f"MEMORY.md: {size / 1024:.1f} KB"
                return f"MEMORY.md: {size} bytes"
            return "MEMORY.md: not found"
        except (ImportError, Exception):
            return "available"

    def _get_memory_detail_text(self) -> str:
        """Get detailed memory status for /memory command."""
        lines = ["<b>Memory Status</b>\n"]

        try:
            from morgan.workspace import WorkspaceManager
            workspace = WorkspaceManager()
            memory_path = workspace.workspace_path / "MEMORY.md"
            if memory_path.exists():
                size = memory_path.stat().st_size
                line_count = memory_path.read_text().count("\n")
                lines.append(f"MEMORY.md: {size / 1024:.1f} KB ({line_count} lines)")
            else:
                lines.append("MEMORY.md: not found")
        except (ImportError, Exception):
            lines.append("MEMORY.md: status unavailable")

        try:
            from morgan.memory_consolidation import DailyLogManager
            dlm = DailyLogManager()
            log_count = len(list(dlm.log_dir.glob("*.md"))) if dlm.log_dir.exists() else 0
            lines.append(f"Daily logs: {log_count} entries")
        except (ImportError, Exception):
            lines.append("Daily logs: status unavailable")

        lines.append(f"\nActive conversations: {len(self._conversations)}")
        return "\n".join(lines)

    # -- Typing indicator ----------------------------------------------------

    def _start_typing(self, chat_id: int, thread_id: Optional[int] = None) -> None:
        """Start a persistent typing indicator that re-sends every 4 seconds."""
        chat_key = str(chat_id)
        # Cancel existing typing task for this chat
        self._stop_typing(chat_key)

        async def _typing_loop() -> None:
            try:
                while True:
                    if self._app is None:
                        break
                    try:
                        await self._app.bot.send_chat_action(
                            chat_id=chat_id,
                            action=ChatAction.TYPING,
                            message_thread_id=thread_id,
                        )
                    except Exception:
                        pass  # Non-critical
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass

        self._typing_tasks[chat_key] = asyncio.create_task(_typing_loop())

    def _stop_typing(self, chat_key: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_key, None)
        if task is not None and not task.done():
            task.cancel()

    # -- Markdown-to-HTML conversion ----------------------------------------

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Convert common Markdown formatting to Telegram HTML.

        Handles code blocks, inline code, bold, italic, and links while
        escaping HTML entities in regular text.
        """
        # First, extract and preserve code blocks to avoid processing their contents
        code_blocks: list[str] = []

        def _save_code_block(match: re.Match) -> str:
            lang = match.group(1) or ""
            code = match.group(2)
            idx = len(code_blocks)
            if lang:
                code_blocks.append(
                    f'<pre><code class="language-{html.escape(lang)}">'
                    f"{html.escape(code)}</code></pre>"
                )
            else:
                code_blocks.append(f"<pre><code>{html.escape(code)}</code></pre>")
            return f"\x00CODEBLOCK{idx}\x00"

        # Extract fenced code blocks: ```lang\n...\n```
        text = re.sub(
            r"```(\w*)\n(.*?)```",
            _save_code_block,
            text,
            flags=re.DOTALL,
        )

        # Extract inline code: `...`
        inline_codes: list[str] = []

        def _save_inline_code(match: re.Match) -> str:
            code = match.group(1)
            idx = len(inline_codes)
            inline_codes.append(f"<code>{html.escape(code)}</code>")
            return f"\x00INLINE{idx}\x00"

        text = re.sub(r"`([^`]+)`", _save_inline_code, text)

        # Now escape HTML entities in the remaining text
        text = html.escape(text)

        # Convert markdown formatting to HTML
        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

        # Italic: *text* or _text_
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)

        # Links: [text](url)
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)

        # Restore code blocks
        for idx, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODEBLOCK{idx}\x00", block)

        # Restore inline codes
        for idx, code in enumerate(inline_codes):
            text = text.replace(f"\x00INLINE{idx}\x00", code)

        return text

    # -- Message chunking ----------------------------------------------------

    def _chunk_message(self, text: str, max_length: Optional[int] = None) -> list[str]:
        """Split *text* into chunks of at most *max_length* characters.

        Splits on paragraph boundaries first, then sentence boundaries,
        then hard-cuts as a last resort.
        """
        limit = max_length or self._max_message_length
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            # Try to split on a paragraph boundary (\n\n)
            cut = remaining[:limit].rfind("\n\n")
            if cut > 0:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip("\n")
                continue

            # Try to split on a single newline
            cut = remaining[:limit].rfind("\n")
            if cut > 0:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip("\n")
                continue

            # Try to split on a sentence boundary (. ! ?)
            for sep in (". ", "! ", "? "):
                cut = remaining[:limit].rfind(sep)
                if cut > 0:
                    cut += 1  # include the punctuation
                    chunks.append(remaining[:cut].rstrip())
                    remaining = remaining[cut:].lstrip()
                    break
            else:
                # Try space boundary
                cut = remaining[:limit].rfind(" ")
                if cut > 0:
                    chunks.append(remaining[:cut])
                    remaining = remaining[cut:].lstrip()
                else:
                    # Hard cut
                    chunks.append(remaining[:limit])
                    remaining = remaining[limit:]

        return chunks

    # -- Send with retry -----------------------------------------------------

    async def _send_with_retry(
        self,
        chat_id: int,
        text: str,
        thread_id: Optional[int] = None,
        reply_markup: Optional[Any] = None,
        parse_mode_str: Optional[str] = None,
    ) -> None:
        """Send a single message with rate-limit awareness and retry logic.

        Handles:
        - Rate limiting (429 RetryAfter)
        - Network errors with exponential backoff
        - HTML parse errors with plain text fallback
        """
        if self._app is None:
            return

        # Rate limiting
        await self._enforce_rate_limit()

        parse_mode = None
        if parse_mode_str == "HTML":
            try:
                parse_mode = ParseMode.HTML
            except NameError:
                # ParseMode not available (telegram lib not installed)
                pass

        kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                await self._app.bot.send_message(**kwargs)
                self._send_timestamps.append(time.monotonic())
                return
            except Exception as exc:
                last_exc = exc

                # Handle rate limiting (429)
                if _HAS_TELEGRAM and isinstance(exc, RetryAfter):
                    wait_time = exc.retry_after
                    logger.warning(
                        "Rate limited by Telegram. Waiting %ds before retry.",
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # Handle HTML parse errors - fall back to plain text
                if _HAS_TELEGRAM and isinstance(exc, BadRequest):
                    error_msg = str(exc).lower()
                    if "parse" in error_msg or "can't parse" in error_msg:
                        logger.warning(
                            "HTML parse error, falling back to plain text: %s", exc
                        )
                        kwargs.pop("parse_mode", None)
                        # Strip HTML tags for plain text fallback
                        kwargs["text"] = re.sub(r"<[^>]+>", "", text)
                        continue

                delay = self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Telegram send attempt %d/%d failed: %s. Retrying in %.1fs.",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        logger.error(
            "Failed to send Telegram message after %d attempts: %s",
            self._max_retries,
            last_exc,
        )

    async def _enforce_rate_limit(self) -> None:
        """Delay if we are sending faster than the per-second limit."""
        now = time.monotonic()
        # Prune old timestamps (older than 1 second)
        self._send_timestamps = [
            ts for ts in self._send_timestamps if now - ts < 1.0
        ]
        if len(self._send_timestamps) >= self._rate_limit_per_second:
            wait = 1.0 - (now - self._send_timestamps[0])
            if wait > 0:
                await asyncio.sleep(wait)
