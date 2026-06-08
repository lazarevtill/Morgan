"""TelegramChannel — long-polling Telegram adapter.

Deployment note (platform ADR 2026-06-08)
------------------------------------------
Telegram polling runs **inside the Tailscale VPN**; no public port is exposed.
Per-chat allowlist enforcement (``ChatAllowlist``) happens in ``ChannelGateway.handle_inbound``
before the message ever reaches the Orchestrator.

Optional dependency
-------------------
This module can be **imported** without ``python-telegram-bot`` installed.
The heavy import is deferred to ``start()``, which raises a clear ``ImportError``
if the ``[channels]`` extra is absent:

    pip install morgan-brain[channels]

Status: SEAM — long-polling integration is deferred (GPU / channel deployment phase).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from morgan_brain.channels.base import OutboundMessage

if TYPE_CHECKING:
    # Only for static analysis; never executed at runtime unless the dep is present.
    import telegram  # noqa: F401


class TelegramChannel:
    """Telegram long-polling channel adapter.

    Delegates inbound updates to ``ChannelGateway.handle_inbound``.
    Outbound sends use the Telegram ``send_message`` API.

    This class is importable without ``python-telegram-bot`` installed.
    The actual Telegram objects are imported lazily inside ``start()``.

    Args:
        token:      Telegram bot token (``MORGAN_TELEGRAM_TOKEN``).
        gateway:    ``ChannelGateway`` instance to route inbound messages through.
    """

    _NAME = "telegram"

    def __init__(self, *, token: str, gateway: object) -> None:
        self._token = token
        self._gateway = gateway
        self._app: object | None = None  # python-telegram-bot Application

    @property
    def name(self) -> str:
        return self._NAME

    async def start(self) -> None:
        """Initialize the Telegram Application and start long-polling.

        Raises:
            ImportError: If ``python-telegram-bot`` is not installed.
        """
        try:
            from telegram.ext import Application  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "python-telegram-bot is required for TelegramChannel. "
                "Install it with: pip install morgan-brain[channels]"
            ) from exc

        self._app = Application.builder().token(self._token).build()
        # TODO: register message handler that calls self._gateway.handle_inbound()
        # and then calls self.send() with the OutboundMessage.
        # Deferred — GPU/channel deployment phase.

    async def stop(self) -> None:
        """Stop the Telegram Application if running."""
        if self._app is not None:
            try:
                await self._app.stop()  # type: ignore[attr-defined]
                await self._app.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._app = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a reply back to the originating Telegram chat.

        Raises:
            RuntimeError: If called before ``start()``.
        """
        if self._app is None:
            raise RuntimeError("TelegramChannel.start() must be called before send().")
        # TODO: self._app.bot.send_message(chat_id=msg.chat_id, text=msg.text)
        # Deferred — GPU/channel deployment phase.
