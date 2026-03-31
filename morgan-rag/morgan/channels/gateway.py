"""
Multi-channel gateway that orchestrates channel adapters and message routing.

Usage::

    gateway = ChannelGateway(default_agent_id="main")
    gateway.register_channel(telegram_channel)
    gateway.set_agent_handler(my_async_handler)
    await gateway.start()
    # ... messages are routed automatically ...
    await gateway.stop()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Optional

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage
from morgan.channels.routing import RouteResolver, SessionKey

logger = logging.getLogger(__name__)

# Signature: async def handler(agent_id, session_key, message) -> Optional[str]
AgentHandler = Callable[
    [str, SessionKey, InboundMessage],
    Coroutine[Any, Any, Optional[str]],
]


class ChannelGateway:
    """Central gateway that manages channel adapters and routes messages.

    Flow:
        channel receives message -> on_message -> gateway._handle_message
        -> resolver.resolve -> agent_handler -> channel.send(response)

    Attributes:
        channels: Mapping of channel name to channel instance.
        resolver: The :class:`RouteResolver` used for message routing.
    """

    def __init__(self, default_agent_id: str) -> None:
        self.channels: dict[str, BaseChannel] = {}
        self.resolver = RouteResolver(default_agent_id=default_agent_id)
        self._agent_handler: Optional[AgentHandler] = None

    # -- Channel management --------------------------------------------------

    def register_channel(self, channel: BaseChannel) -> None:
        """Register a channel adapter with the gateway.

        Sets the channel's message handler to :meth:`_handle_message` so
        inbound messages are automatically routed.

        Raises:
            ValueError: If a channel with the same name is already registered.
        """
        if channel.name in self.channels:
            raise ValueError(
                f"Channel '{channel.name}' is already registered."
            )
        channel.set_message_handler(self._handle_message)
        self.channels[channel.name] = channel
        logger.info("Registered channel: %s", channel.name)

    def set_agent_handler(self, handler: AgentHandler) -> None:
        """Set the async function that processes routed messages.

        The handler signature is::

            async def handler(
                agent_id: str,
                session_key: SessionKey,
                message: InboundMessage,
            ) -> Optional[str]:
                ...

        Return a string to send a reply, or ``None`` to send nothing.
        """
        self._agent_handler = handler

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start all registered channels."""
        for name, channel in self.channels.items():
            logger.info("Starting channel: %s", name)
            await channel.start()
        logger.info(
            "Gateway started with %d channel(s).", len(self.channels)
        )

    async def stop(self) -> None:
        """Stop all registered channels."""
        for name, channel in self.channels.items():
            logger.info("Stopping channel: %s", name)
            await channel.stop()
        logger.info("Gateway stopped.")

    # -- Internal routing ----------------------------------------------------

    async def _handle_message(self, message: InboundMessage) -> None:
        """Route an inbound message and dispatch to the agent handler.

        This method is set as the message handler on each registered channel.
        """
        if self._agent_handler is None:
            logger.warning(
                "No agent handler set; dropping message from %s/%s.",
                message.channel,
                message.peer_id,
            )
            return

        route = self.resolver.resolve(message)
        logger.debug(
            "Resolved route: agent=%s, session=%s, matched_by=%s",
            route.agent_id,
            route.session_key,
            route.matched_by,
        )

        response_text = await self._agent_handler(
            route.agent_id, route.session_key, message
        )

        if response_text is None:
            return

        # Find the originating channel to send the reply
        channel = self.channels.get(message.channel)
        if channel is None:
            logger.error(
                "Cannot send reply: channel '%s' not registered.",
                message.channel,
            )
            return

        reply = OutboundMessage(
            channel=message.channel,
            peer_id=message.peer_id,
            content=response_text,
            group_id=message.group_id,
        )
        await channel.send(reply)
