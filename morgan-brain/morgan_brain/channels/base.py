"""Core channel contracts.

``InboundMessage``  — message arriving from a channel (Telegram, Discord, etc.).
``OutboundMessage`` — reply to send back into a channel.
``Channel``         — Protocol every channel adapter must satisfy.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class InboundMessage(BaseModel):
    """A message received from a specific channel."""

    channel: str
    """Name of the originating channel (e.g. ``"telegram"``, ``"discord"``)."""

    chat_id: str
    """Channel-native identifier for the chat/room/thread."""

    user_id: str
    """Sender identifier within the channel."""

    text: str
    """Plain-text content of the message."""


class OutboundMessage(BaseModel):
    """A reply to be delivered into a channel."""

    chat_id: str
    """Destination chat/room/thread — must match the inbound chat_id."""

    text: str
    """Reply text."""


@runtime_checkable
class Channel(Protocol):
    """Adapter contract for a messaging channel.

    Implementations handle transport specifics (polling, webhooks, sockets) and
    call ``ChannelGateway.handle_inbound`` for every arriving message.
    They must be importable without their optional dependency installed — the
    heavy import is deferred to ``start()``.
    """

    @property
    def name(self) -> str:
        """Unique channel name (e.g. ``"telegram"``)."""
        ...

    async def start(self) -> None:
        """Connect / start polling. Called once at service startup."""
        ...

    async def stop(self) -> None:
        """Disconnect / stop polling. Called once at service shutdown."""
        ...

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver ``msg`` to the channel."""
        ...
