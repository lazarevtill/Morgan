"""
Base channel abstractions for the multi-channel gateway.

Defines the message data models and the abstract base class that every
concrete channel adapter (Telegram, Discord, etc.) must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Type alias for the async callback that the gateway sets on each channel.
MessageHandler = Callable[["InboundMessage"], Coroutine[Any, Any, None]]


@dataclass
class InboundMessage:
    """A message received from an external channel.

    Attributes:
        channel: Name of the originating channel (e.g. "telegram", "discord").
        peer_id: Unique identifier of the message author within the channel.
        content: Text body of the message.
        group_id: Identifier of the group/guild/chat if the message was sent
            in a group context.  ``None`` for direct messages.
        account_id: Optional account-level identifier (e.g. bot account).
        metadata: Arbitrary extra data supplied by the channel adapter.
    """

    channel: str
    peer_id: str
    content: str
    group_id: Optional[str] = None
    account_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """A message to be sent through a channel.

    Attributes:
        channel: Target channel name.
        peer_id: Recipient peer identifier.
        content: Text body.
        group_id: Target group identifier, if replying to a group message.
        metadata: Arbitrary extra data for the channel adapter.
    """

    channel: str
    peer_id: str
    content: str
    group_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChannel(ABC):
    """Abstract base class for a communication channel adapter.

    Concrete subclasses must implement :pyattr:`name`, :pymeth:`start`,
    :pymeth:`stop`, and :pymeth:`send`.

    The gateway calls :pymeth:`set_message_handler` after registration so
    that the channel can forward inbound messages to the gateway for routing.
    """

    def __init__(self) -> None:
        self._message_handler: Optional[MessageHandler] = None

    # -- Abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this channel (e.g. ``"telegram"``)."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start listening for inbound messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""
        ...

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """Send an outbound message through this channel."""
        ...

    # -- Handler plumbing ----------------------------------------------------

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the callback that receives every inbound message.

        Called by :class:`ChannelGateway` during channel registration.
        """
        self._message_handler = handler

    async def on_message(self, message: InboundMessage) -> None:
        """Forward an inbound message to the registered handler.

        Channel adapters should call this method when they receive a new
        message from the external service.
        """
        if self._message_handler is not None:
            await self._message_handler(message)
        else:
            logger.warning(
                "Channel '%s' received a message but no handler is set.",
                self.name,
            )
