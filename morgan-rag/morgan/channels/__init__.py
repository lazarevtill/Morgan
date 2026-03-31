"""
Multi-Channel Gateway for Morgan AI Assistant.

Provides a unified gateway for receiving and sending messages across
multiple communication channels (Telegram, Discord, etc.), with
configurable routing to different agent instances.

Ported from OpenClaw's routing and channel patterns.

Usage:
    from morgan.channels import ChannelGateway, InboundMessage, OutboundMessage

    gateway = ChannelGateway(default_agent_id="main")
    gateway.register_channel(telegram_channel)
    gateway.set_agent_handler(my_handler)
    await gateway.start()
"""

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage
from morgan.channels.gateway import ChannelGateway
from morgan.channels.routing import ResolvedRoute, RouteBinding, RouteResolver, SessionKey
from morgan.channels.telegram_channel import TelegramChannel
from morgan.channels.synology_channel import SynologyChannel

__all__ = [
    "BaseChannel",
    "ChannelGateway",
    "InboundMessage",
    "OutboundMessage",
    "ResolvedRoute",
    "RouteBinding",
    "RouteResolver",
    "SessionKey",
    "SynologyChannel",
    "TelegramChannel",
]
