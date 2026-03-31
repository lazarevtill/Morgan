"""
Telegram channel adapter for the multi-channel gateway.

Uses the ``python-telegram-bot`` library.  If the library is not installed,
the module still imports successfully but :meth:`TelegramChannel.start` will
raise a descriptive :class:`RuntimeError`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

# Attempt to import python-telegram-bot (optional dependency).
try:
    from telegram import Update
    from telegram.ext import (
        Application,
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


class TelegramChannel(BaseChannel):
    """Channel adapter for Telegram using python-telegram-bot.

    Args:
        token: The Telegram Bot API token.
    """

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token
        self._app: Optional[object] = None  # telegram.ext.Application

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        if not _HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )

        self._app = (
            Application.builder().token(self._token).build()
        )

        # Register a handler for all text messages.
        self._app.add_handler(
            TGMessageHandler(
                filters.TEXT & ~filters.COMMAND, self._on_telegram_message
            )
        )

        logger.info("Starting Telegram channel polling...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        if self._app is not None and _HAS_TELEGRAM:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram channel stopped.")

    async def send(self, message: OutboundMessage) -> None:
        if not _HAS_TELEGRAM or self._app is None:
            logger.warning("Telegram channel is not started; cannot send.")
            return

        chat_id = message.group_id or message.peer_id
        await self._app.bot.send_message(
            chat_id=int(chat_id), text=message.content
        )

    # -- Internal Telegram callback ------------------------------------------

    async def _on_telegram_message(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Called by python-telegram-bot for every text message."""
        if update.effective_message is None or update.effective_user is None:
            return

        peer_id = str(update.effective_user.id)
        chat = update.effective_chat
        group_id = (
            str(chat.id) if chat is not None and chat.type != "private" else None
        )

        msg = InboundMessage(
            channel="telegram",
            peer_id=peer_id,
            content=update.effective_message.text or "",
            group_id=group_id,
            metadata={
                "message_id": update.effective_message.message_id,
                "chat_type": chat.type if chat else "unknown",
            },
        )
        await self.on_message(msg)
