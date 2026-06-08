"""Channel gateway — routes inbound messages through the Orchestrator.

Architecture
------------
* Single-owner model: every inbound message is mapped to ``owner_user_id`` regardless
  of the channel-native sender.  The owner may later be resolved by channel identity
  when multi-tenant support lands.
* Default-deny: the ``ChatAllowlist`` blocks any chat_id not explicitly permitted.
  This is the per-platform ADR security boundary.
* Channels are registered via ``register_channel()``.  The gateway does not start or
  stop them — the application lifespan does.

Usage::

    gw = ChannelGateway(
        orchestrator=orch,
        allowlist=ChatAllowlist(allowed={"chat_42"}),
        owner_user_id="owner",
    )
    # inbound from FakeChannel / Telegram / Discord:
    reply = await gw.handle_inbound(InboundMessage(
        channel="telegram", chat_id="chat_42", user_id="some_user", text="hello"
    ))
"""
from __future__ import annotations

from morgan_brain.channels.allowlist import ChatAllowlist
from morgan_brain.channels.base import Channel, InboundMessage, OutboundMessage
from morgan_brain.core.orchestrator import Orchestrator


class ChannelGateway:
    """Routes inbound channel messages through the Orchestrator.

    Args:
        orchestrator:   The cognitive-loop orchestrator.
        allowlist:      Per-chat default-deny filter.
        owner_user_id:  User ID injected into every turn (single-owner mode).
    """

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        allowlist: ChatAllowlist,
        owner_user_id: str,
    ) -> None:
        self._orchestrator = orchestrator
        self._allowlist = allowlist
        self._owner_user_id = owner_user_id
        self._channels: dict[str, Channel] = {}

    def register_channel(self, channel: Channel) -> None:
        """Register a channel adapter. Duplicate names overwrite."""
        self._channels[channel.name] = channel

    async def handle_inbound(self, msg: InboundMessage) -> OutboundMessage | None:
        """Process an inbound message.

        Returns an ``OutboundMessage`` if the chat is allowed and the orchestrator
        produces a reply, or ``None`` if the message is silently dropped.
        """
        if not self._allowlist.is_allowed(msg.chat_id):
            # Default-deny: chat not in allowlist — drop silently.
            return None

        result = await self._orchestrator.handle_turn(
            user_id=self._owner_user_id,
            text=msg.text,
            session_id=msg.chat_id,
        )
        return OutboundMessage(chat_id=msg.chat_id, text=result.text)
