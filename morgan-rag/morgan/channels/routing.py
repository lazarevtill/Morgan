"""
Session-key generation and route resolution for the multi-channel gateway.

The resolver uses a binding hierarchy to decide which agent should handle
a given inbound message:

    1. Binding by ``peer_id`` (direct match)
    2. Binding by ``group_id``
    3. Binding by ``channel``
    4. Default agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from morgan.channels.base import InboundMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionKey:
    """Immutable, hashable key that uniquely identifies a conversation session.

    Format (without thread):
        ``agent:{agent_id}:{channel}:{type}:{identifier}``
    Format (with thread):
        ``agent:{agent_id}:{channel}:{type}:{identifier}:thread:{thread_id}``

    The optional ``thread_id`` field provides explicit per-topic isolation
    for forum/thread-based channels (Telegram supergroups with topics).
    Each thread gets its own session key, conversation history, and context.
    """

    agent_id: str
    channel: str
    session_type: str  # "dm", "group", or "main"
    identifier: str  # peer_id, group_id, or empty string
    thread_id: Optional[str] = None  # Forum topic / thread ID

    # -- Factory methods -----------------------------------------------------

    @staticmethod
    def for_dm(agent_id: str, channel: str, peer_id: str) -> SessionKey:
        return SessionKey(
            agent_id=agent_id,
            channel=channel,
            session_type="dm",
            identifier=peer_id,
        )

    @staticmethod
    def for_group(
        agent_id: str,
        channel: str,
        group_id: str,
        thread_id: Optional[str] = None,
    ) -> SessionKey:
        return SessionKey(
            agent_id=agent_id,
            channel=channel,
            session_type="group",
            identifier=group_id,
            thread_id=thread_id,
        )

    @staticmethod
    def for_main(agent_id: str) -> SessionKey:
        return SessionKey(
            agent_id=agent_id,
            channel="main",
            session_type="main",
            identifier="",
        )

    # -- String representation -----------------------------------------------

    def __str__(self) -> str:
        if self.session_type == "main":
            return f"agent:{self.agent_id}:main"
        base = (
            f"agent:{self.agent_id}:{self.channel}"
            f":{self.session_type}:{self.identifier}"
        )
        if self.thread_id is not None:
            return f"{base}:thread:{self.thread_id}"
        return base


# ---------------------------------------------------------------------------
# ResolvedRoute
# ---------------------------------------------------------------------------


@dataclass
class ResolvedRoute:
    """The result of resolving a route for an inbound message.

    Attributes:
        agent_id: The agent that should handle the message.
        session_key: The session key for the conversation.
        session_type: ``"dm"`` or ``"group"``.
        matched_by: Which level of the hierarchy produced this match
            (``"peer"``, ``"group"``, ``"channel"``, or ``"default"``).
    """

    agent_id: str
    session_key: SessionKey
    session_type: str
    matched_by: str  # "peer" | "group" | "channel" | "default"


# ---------------------------------------------------------------------------
# RouteBinding
# ---------------------------------------------------------------------------


@dataclass
class RouteBinding:
    """A single binding entry that maps a key to an agent.

    Used internally by :class:`RouteResolver`.
    """

    key: str
    agent_id: str
    level: str  # "peer" | "group" | "channel"


# ---------------------------------------------------------------------------
# RouteResolver
# ---------------------------------------------------------------------------


class RouteResolver:
    """Resolves which agent should handle an inbound message.

    Resolution hierarchy (first match wins):
        1. ``peer_id`` binding
        2. ``group_id`` binding
        3. ``channel`` binding
        4. ``default_agent_id``

    Usage::

        resolver = RouteResolver(default_agent_id="main")
        resolver.bind_peer("user123", "vip_agent")
        route = resolver.resolve(inbound_message)
    """

    def __init__(self, default_agent_id: str) -> None:
        self._default_agent_id = default_agent_id
        self._peer_bindings: dict[str, str] = {}
        self._group_bindings: dict[str, str] = {}
        self._channel_bindings: dict[str, str] = {}

    # -- Bind / unbind -------------------------------------------------------

    def bind_peer(self, peer_id: str, agent_id: str) -> None:
        """Route all messages from *peer_id* to *agent_id*."""
        self._peer_bindings[peer_id] = agent_id

    def unbind_peer(self, peer_id: str) -> None:
        self._peer_bindings.pop(peer_id, None)

    def bind_group(self, group_id: str, agent_id: str) -> None:
        """Route all messages in *group_id* to *agent_id*."""
        self._group_bindings[group_id] = agent_id

    def unbind_group(self, group_id: str) -> None:
        self._group_bindings.pop(group_id, None)

    def bind_channel(self, channel: str, agent_id: str) -> None:
        """Route all messages from *channel* to *agent_id*."""
        self._channel_bindings[channel] = agent_id

    def unbind_channel(self, channel: str) -> None:
        self._channel_bindings.pop(channel, None)

    # -- Resolution ----------------------------------------------------------

    def resolve(self, message: InboundMessage) -> ResolvedRoute:
        """Resolve the route for *message* according to the binding hierarchy."""
        agent_id: str
        matched_by: str

        # 1. peer_id
        if message.peer_id in self._peer_bindings:
            agent_id = self._peer_bindings[message.peer_id]
            matched_by = "peer"
        # 2. group_id
        elif (
            message.group_id is not None
            and message.group_id in self._group_bindings
        ):
            agent_id = self._group_bindings[message.group_id]
            matched_by = "group"
        # 3. channel
        elif message.channel in self._channel_bindings:
            agent_id = self._channel_bindings[message.channel]
            matched_by = "channel"
        # 4. default
        else:
            agent_id = self._default_agent_id
            matched_by = "default"

        # Extract thread_id from metadata (set by channel adapters)
        thread_id = message.metadata.get("message_thread_id")
        thread_id_str = str(thread_id) if thread_id is not None else None

        # Build session key — thread_id creates per-topic isolation
        if message.group_id is not None:
            session_type = "group"
            # Use base group_id (without thread embedded) + explicit thread_id
            base_group_id = str(message.group_id).split(":")[0]
            session_key = SessionKey.for_group(
                agent_id=agent_id,
                channel=message.channel,
                group_id=base_group_id,
                thread_id=thread_id_str,
            )
        else:
            session_type = "dm"
            session_key = SessionKey.for_dm(
                agent_id=agent_id,
                channel=message.channel,
                peer_id=message.peer_id,
            )

        return ResolvedRoute(
            agent_id=agent_id,
            session_key=session_key,
            session_type=session_type,
            matched_by=matched_by,
        )
