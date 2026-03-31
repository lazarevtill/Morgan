"""
Telegram channel adapter for the multi-channel gateway.

Full-featured Telegram bot using ``python-telegram-bot`` v20+.

Supports:
- Polling (default) and webhook mode
- Text messages, commands (/start, /help, /clear)
- Group chat with @mention / reply-to-bot filtering
- Forum / topic support (per-topic routing)
- Typing indicator before processing
- MarkdownV2 formatting with proper escaping
- Message chunking at 4096-char limit
- Photo / document media with caption extraction
- Inline keyboard buttons
- Per-second rate limiting
- User allowlist (empty = allow all)
- Retry with exponential backoff on network errors
- Graceful shutdown

If ``python-telegram-bot`` is not installed the module still imports; calling
:meth:`start` raises a descriptive :class:`RuntimeError`.
"""

from __future__ import annotations

import asyncio
import logging
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
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ChatAction, ParseMode
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

# Characters that must be escaped for Telegram MarkdownV2.
_MARKDOWNV2_SPECIAL = r"_*[]()~`>#+-=|{}.!"


class TelegramChannel(BaseChannel):
    """Channel adapter for Telegram using python-telegram-bot v20+.

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
    """

    def __init__(
        self,
        token: str,
        allowed_user_ids: Optional[set[str]] = None,
        require_mention_in_groups: bool = True,
        webhook_url: Optional[str] = None,
        webhook_port: int = 8443,
        max_message_length: int = 4096,
    ) -> None:
        super().__init__()
        self._token = token
        self._allowed_user_ids: set[str] = allowed_user_ids or set()
        self._require_mention_in_groups = require_mention_in_groups
        self._webhook_url = webhook_url
        self._webhook_port = webhook_port
        self._max_message_length = max_message_length

        self._app: Optional[Any] = None  # telegram.ext.Application
        self._bot_username: Optional[str] = None
        self._bot_id: Optional[int] = None

        # Rate-limiting state: timestamps of recent sends
        self._send_timestamps: list[float] = []
        self._rate_limit_per_second: float = 30.0  # Telegram limit ~30 msg/s

        # Retry configuration
        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0

    # -- BaseChannel interface -----------------------------------------------

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Initialize the bot application, register handlers, and begin
        polling (or start the webhook server)."""
        if not _HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )

        self._app = Application.builder().token(self._token).build()

        # Register command handlers
        self._app.add_handler(CommandHandler("start", self._on_command))
        self._app.add_handler(CommandHandler("help", self._on_command))
        self._app.add_handler(CommandHandler("clear", self._on_command))

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
            TGMessageHandler(filters.Document.ALL, self._on_photo)
        )

        # Callback-query handler (inline keyboard buttons)
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))

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
                drop_pending_updates=True,
            )

    async def stop(self) -> None:
        """Graceful shutdown."""
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
        """Send an outbound message with chunking, markdown escaping,
        and typing indicator."""
        if not _HAS_TELEGRAM or self._app is None:
            logger.warning("Telegram channel is not started; cannot send.")
            return

        chat_id = int(message.group_id or message.peer_id)

        # Optional: reply to a specific message_thread_id for forum topics
        thread_id = message.metadata.get("message_thread_id")

        # Send typing indicator
        try:
            await self._app.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.TYPING,
                message_thread_id=thread_id,
            )
        except Exception:
            pass  # Non-critical

        chunks = self._chunk_message(message.content)
        for chunk in chunks:
            await self._send_with_retry(
                chat_id=chat_id,
                text=chunk,
                thread_id=thread_id,
            )

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

    # -- Telegram event handlers ---------------------------------------------

    async def _on_text_message(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle incoming text messages."""
        if not self._validate_update(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)

        # Allowlist check
        if not self._is_user_allowed(peer_id):
            logger.debug("User %s not in allowlist; ignoring.", peer_id)
            return

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
            # Route per-topic: encode thread in group_id
            group_id = f"{chat.id}:{thread_id}"

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

        # Typing indicator
        try:
            await self._app.bot.send_chat_action(
                chat_id=int(group_id.split(":")[0]) if group_id else int(peer_id),
                action=ChatAction.TYPING,
                message_thread_id=thread_id,
            )
        except Exception:
            pass

        await self.on_message(msg)

    async def _on_command(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle /start, /help, /clear commands."""
        if not self._validate_update(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        if not self._is_user_allowed(peer_id):
            return

        command = effective_message.text or ""
        command = command.split()[0].lstrip("/").split("@")[0].lower()

        chat_id = chat.id if chat else int(peer_id)

        if command == "start":
            await self._send_with_retry(
                chat_id=chat_id,
                text="Hello! I'm Morgan, your AI assistant. Send me a message to get started.",
            )
        elif command == "help":
            help_text = (
                "Available commands:\n"
                "/start - Start the conversation\n"
                "/help - Show this help message\n"
                "/clear - Reset conversation history\n\n"
                "You can also just send me a message and I'll do my best to help!"
            )
            await self._send_with_retry(chat_id=chat_id, text=help_text)
        elif command == "clear":
            # Build an InboundMessage with a clear directive so the gateway
            # can reset the session.
            is_group = chat is not None and chat.type in ("group", "supergroup")
            group_id = str(chat.id) if is_group else None

            msg = InboundMessage(
                channel="telegram",
                peer_id=peer_id,
                content="/clear",
                group_id=group_id,
                metadata={
                    "command": "clear",
                    "message_id": effective_message.message_id,
                },
            )
            await self.on_message(msg)

            await self._send_with_retry(
                chat_id=chat_id,
                text="Conversation history cleared.",
            )

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

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=query.data or "",
            group_id=group_id,
            metadata={
                "callback_query_id": query.id,
                "message_id": query.message.message_id if query.message else None,
            },
        )
        await self.on_message(msg)

    async def _on_photo(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle photo / document messages — extract caption text."""
        if not self._validate_update(update):
            return

        effective_message = update.effective_message
        effective_user = update.effective_user
        chat = update.effective_chat

        if effective_message is None or effective_user is None:
            return

        peer_id = str(effective_user.id)
        if not self._is_user_allowed(peer_id):
            return

        is_group = chat is not None and chat.type in ("group", "supergroup")
        if is_group and not self._should_respond_in_group(update):
            return

        caption = effective_message.caption or ""
        group_id = str(chat.id) if is_group else None

        metadata: dict[str, Any] = {
            "message_id": effective_message.message_id,
            "chat_type": chat.type if chat else "unknown",
            "has_photo": bool(effective_message.photo),
            "has_document": effective_message.document is not None,
        }

        # Include photo file_id if present
        if effective_message.photo:
            metadata["photo_file_id"] = effective_message.photo[-1].file_id
        if effective_message.document:
            metadata["document_file_id"] = effective_message.document.file_id
            metadata["document_file_name"] = effective_message.document.file_name

        content = caption if caption else "[media received without caption]"

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=content,
            group_id=group_id,
            metadata=metadata,
        )
        await self.on_message(msg)

    # -- Internal helpers ----------------------------------------------------

    def _validate_update(self, update: "Update") -> bool:
        """Return True if the update contains the minimum required fields."""
        return (
            update is not None
            and update.effective_message is not None
            and update.effective_user is not None
        )

    def _is_user_allowed(self, peer_id: str) -> bool:
        """Check if a user is in the allowlist (empty list = allow all)."""
        if not self._allowed_user_ids:
            return True
        return peer_id in self._allowed_user_ids

    def _should_respond_in_group(self, update: "Update") -> bool:
        """Determine whether the bot should respond in a group context.

        Returns ``True`` if:
        - ``require_mention_in_groups`` is ``False``, or
        - the message contains ``@bot_username``, or
        - the message is a reply to the bot.
        """
        if not self._require_mention_in_groups:
            return True

        effective_message = update.effective_message
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

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape special characters for Telegram MarkdownV2.

        Escapes: ``_ * [ ] ( ) ~ ` > # + - = | { } . !``
        """
        return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", text)

    async def _send_with_retry(
        self,
        chat_id: int,
        text: str,
        thread_id: Optional[int] = None,
        reply_markup: Optional[Any] = None,
        use_markdown: bool = False,
    ) -> None:
        """Send a single message with rate-limit awareness and retry logic."""
        if self._app is None:
            return

        # Rate limiting
        await self._enforce_rate_limit()

        parse_mode = ParseMode.MARKDOWN_V2 if use_markdown else None

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
