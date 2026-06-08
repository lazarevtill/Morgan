"""Channel gateway package — multi-channel inbound/outbound routing.

Exposes the public surface:
* ``InboundMessage`` / ``OutboundMessage`` — wire models.
* ``Channel`` — Protocol every adapter must satisfy.
* ``ChatAllowlist`` — per-chat default-deny filter.
* ``ChannelGateway`` — routes inbound messages through the orchestrator.
* ``FakeChannel`` — in-process fake for tests.
* ``TelegramChannel`` — stub (lazy import; needs ``[channels]`` extra).
"""
from morgan_brain.channels.base import Channel, InboundMessage, OutboundMessage
from morgan_brain.channels.allowlist import ChatAllowlist
from morgan_brain.channels.gateway import ChannelGateway
from morgan_brain.channels.fake import FakeChannel

__all__ = [
    "Channel",
    "InboundMessage",
    "OutboundMessage",
    "ChatAllowlist",
    "ChannelGateway",
    "FakeChannel",
]
