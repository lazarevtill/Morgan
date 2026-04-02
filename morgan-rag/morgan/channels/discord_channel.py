"""
Discord channel adapter for the multi-channel gateway.

Uses the ``discord.py`` library.  If the library is not installed, the module
still imports successfully but :meth:`DiscordChannel.start` will raise a
descriptive :class:`RuntimeError`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

# Attempt to import discord.py (optional dependency).
try:
    import discord

    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False
    logger.debug(
        "discord.py is not installed. "
        "DiscordChannel will not be able to start."
    )


class DiscordChannel(BaseChannel):
    """Channel adapter for Discord using discord.py.

    Args:
        token: The Discord Bot token.
        intents: Optional :class:`discord.Intents` instance.  Defaults to
            ``discord.Intents.default()`` with ``message_content`` enabled.
    """

    def __init__(
        self,
        token: str,
        intents: Optional[object] = None,
    ) -> None:
        super().__init__()
        self._token = token
        self._intents = intents
        self._client: Optional[object] = None  # discord.Client
        self._run_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        if not _HAS_DISCORD:
            raise RuntimeError(
                "discord.py is not installed. "
                "Install it with: pip install discord.py"
            )

        intents = self._intents
        if intents is None:
            intents = discord.Intents.default()
            intents.message_content = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            # Ignore messages from the bot itself.
            if message.author == self._client.user:
                return
            await self._on_discord_message(message)

        # Start the client in a background task so start() returns promptly.
        self._run_task = asyncio.create_task(
            self._client.start(self._token)
        )
        logger.info("Discord channel starting...")

    async def stop(self) -> None:
        if self._client is not None and _HAS_DISCORD:
            await self._client.close()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
            logger.info("Discord channel stopped.")

    async def send(self, message: OutboundMessage) -> None:
        if not _HAS_DISCORD or self._client is None:
            logger.warning("Discord channel is not started; cannot send.")
            return

        # Resolve the target channel/DM.
        target_id = message.group_id or message.peer_id
        channel = self._client.get_channel(int(target_id))

        if channel is None:
            # Attempt to send a DM.
            try:
                user = await self._client.fetch_user(int(message.peer_id))
                dm = await user.create_dm()
                await dm.send(message.content)
            except Exception:
                logger.exception(
                    "Failed to send Discord message to %s", message.peer_id
                )
        else:
            await channel.send(message.content)

    # -- Internal Discord callback -------------------------------------------

    async def _on_discord_message(self, message: "discord.Message") -> None:
        """Translate a Discord message into an InboundMessage."""
        peer_id = str(message.author.id)
        guild = message.guild
        group_id = str(message.channel.id) if guild is not None else None

        msg = InboundMessage(
            channel="discord",
            peer_id=peer_id,
            content=message.content,
            group_id=group_id,
            metadata={
                "message_id": message.id,
                "guild_id": str(guild.id) if guild else None,
                "channel_name": getattr(message.channel, "name", None),
            },
        )
        await self.on_message(msg)
