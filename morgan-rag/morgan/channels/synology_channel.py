"""
Synology Chat channel adapter for the multi-channel gateway.

Integrates with Synology Chat via incoming/outgoing webhooks:

- **Outgoing webhook** (Synology -> Morgan): Synology Chat POSTs to Morgan's
  webhook endpoint when a user sends a message.
- **Incoming webhook** (Morgan -> Synology): Morgan POSTs to Synology Chat's
  incoming webhook URL to deliver replies.

Authentication uses a shared token validated with constant-time comparison
(``hmac.compare_digest``) to prevent timing attacks.

If ``aiohttp`` is not installed the module still imports; calling
:meth:`start` raises a descriptive :class:`RuntimeError`.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import ssl
from collections import defaultdict
from typing import Any, Optional
from urllib.parse import urlencode

from morgan.channels.base import BaseChannel, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    import aiohttp
    from aiohttp import web

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False
    logger.debug(
        "aiohttp is not installed. "
        "SynologyChannel will not be able to start."
    )


class SynologyChannel(BaseChannel):
    """Channel adapter for Synology Chat via webhooks.

    Args:
        token: Shared secret token for webhook authentication.
        incoming_url: Synology Chat incoming webhook URL for sending messages.
        webhook_path: URL path where this server listens for Synology
            outgoing-webhook POSTs (default ``/webhook/synology``).
        webhook_port: Port to bind the webhook listener (default 5001).
        allowed_user_ids: Set of Synology user-ID strings permitted to
            interact.  ``None`` or empty means *all* users allowed.
        rate_limit_per_minute: Maximum messages per user per minute
            (default 30).
        bot_name: Display name used in log messages (default ``"Morgan"``).
        allow_insecure_ssl: If ``True``, skip SSL verification for
            outgoing HTTP requests (useful for self-signed certs on a
            Synology NAS).
    """

    # Synology Chat message length limit
    MAX_MESSAGE_LENGTH = 2000

    def __init__(
        self,
        token: str,
        incoming_url: str,
        webhook_path: str = "/webhook/synology",
        webhook_port: int = 5001,
        allowed_user_ids: Optional[set[str]] = None,
        rate_limit_per_minute: int = 30,
        bot_name: str = "Morgan",
        allow_insecure_ssl: bool = False,
    ) -> None:
        super().__init__()
        self._token = token
        self._incoming_url = incoming_url
        self._webhook_path = webhook_path
        self._webhook_port = webhook_port
        self._allowed_user_ids: set[str] = allowed_user_ids or set()
        self._rate_limit_per_minute = rate_limit_per_minute
        self._bot_name = bot_name
        self._allow_insecure_ssl = allow_insecure_ssl

        # Rate-limit tracking: user_id -> list of timestamps
        self._rate_tracker: dict[str, list[float]] = defaultdict(list)

        # aiohttp server / session state
        self._web_app: Optional[Any] = None  # aiohttp.web.Application
        self._runner: Optional[Any] = None  # aiohttp.web.AppRunner
        self._site: Optional[Any] = None  # aiohttp.web.TCPSite
        self._session: Optional[Any] = None  # aiohttp.ClientSession

    # -- BaseChannel interface -----------------------------------------------

    @property
    def name(self) -> str:
        return "synology"

    async def start(self) -> None:
        """Start the aiohttp web server to receive outgoing-webhook POSTs
        from Synology Chat, and prepare an HTTP client session for sending
        messages."""
        if not _HAS_AIOHTTP:
            raise RuntimeError(
                "aiohttp is not installed. "
                "Install it with: pip install aiohttp"
            )

        # Build aiohttp web application
        self._web_app = web.Application()
        self._web_app.router.add_post(self._webhook_path, self._handle_webhook)
        self._web_app.router.add_get(
            f"{self._webhook_path}/health", self._handle_health
        )

        self._runner = web.AppRunner(self._web_app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self._webhook_port)
        await self._site.start()

        # Prepare outgoing HTTP session
        connector_kwargs: dict[str, Any] = {}
        if self._allow_insecure_ssl:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            connector_kwargs["ssl"] = ssl_ctx

        connector = aiohttp.TCPConnector(**connector_kwargs)
        self._session = aiohttp.ClientSession(connector=connector)

        logger.info(
            "Synology Chat channel started — webhook at 0.0.0.0:%d%s",
            self._webhook_port,
            self._webhook_path,
        )

    async def stop(self) -> None:
        """Gracefully shut down the webhook server and HTTP client."""
        if self._session is not None:
            await self._session.close()
            self._session = None

        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()

        logger.info("Synology Chat channel stopped.")

    async def send(self, message: OutboundMessage) -> None:
        """Send a reply to Synology Chat via the incoming webhook URL.

        Long messages are chunked to stay within the 2000-char limit.
        The payload is sent as a query-string parameter ``payload=<json>``.
        """
        if self._session is None:
            logger.warning("Synology channel not started; cannot send.")
            return

        user_ids = []
        if message.peer_id:
            try:
                user_ids = [int(message.peer_id)]
            except (ValueError, TypeError):
                pass

        chunks = self._chunk_message(message.content)
        for chunk in chunks:
            payload = json.dumps(
                {"text": chunk, "user_ids": user_ids},
                ensure_ascii=False,
            )
            encoded = urlencode({"payload": payload})
            url = f"{self._incoming_url}?{encoded}"

            try:
                async with self._session.post(url) as resp:
                    if resp.status not in (200, 201):
                        body = await resp.text()
                        logger.warning(
                            "Synology Chat incoming webhook returned %d: %s",
                            resp.status,
                            body[:200],
                        )
            except Exception:
                logger.exception("Failed to send message to Synology Chat.")

    # -- Webhook handlers ----------------------------------------------------

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """Process an incoming POST from Synology Chat (outgoing webhook).

        The payload can arrive as:
        - ``application/x-www-form-urlencoded``
        - ``application/json``

        Expected fields: ``token``, ``user_id``, ``username``, ``text``.
        Optional: ``channel_id``, ``channel_name``.
        """
        try:
            content_type = request.content_type
            if content_type == "application/json":
                data = await request.json()
            else:
                data = await request.post()
                # Convert multidict to plain dict
                data = {k: v for k, v in data.items()}
        except Exception:
            logger.exception("Failed to parse Synology webhook payload.")
            return web.Response(status=400, text="Bad Request")

        # Token validation
        incoming_token = data.get("token", "")
        if not self._validate_token(str(incoming_token)):
            logger.warning("Invalid Synology webhook token received.")
            return web.Response(status=403, text="Forbidden")

        # Extract fields
        user_id = str(data.get("user_id", ""))
        username = str(data.get("username", ""))
        text = str(data.get("text", ""))
        channel_id = str(data.get("channel_id", "")) or None
        channel_name = str(data.get("channel_name", "")) or None

        if not user_id or not text:
            return web.Response(status=400, text="Missing required fields")

        # Allowlist check
        if self._allowed_user_ids and user_id not in self._allowed_user_ids:
            logger.debug("Synology user %s not in allowlist; ignoring.", user_id)
            return web.Response(status=403, text="User not allowed")

        # Rate limit check
        if not self._check_rate_limit(user_id):
            logger.warning("Rate limit exceeded for Synology user %s.", user_id)
            return web.Response(status=429, text="Rate limit exceeded")

        # Sanitize input
        text = self._sanitize_input(text)
        if not text:
            return web.Response(status=400, text="Empty message after sanitization")

        msg = InboundMessage(
            channel="synology",
            peer_id=user_id,
            content=text,
            group_id=channel_id,
            metadata={
                "username": username,
                "channel_name": channel_name,
            },
        )
        await self.on_message(msg)

        return web.Response(status=200, text="OK")

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """Simple health-check endpoint."""
        return web.json_response(
            {"status": "ok", "channel": self.name, "bot_name": self._bot_name}
        )

    # -- Validation & rate limiting ------------------------------------------

    def _validate_token(self, token: str) -> bool:
        """Constant-time token comparison to prevent timing attacks."""
        return hmac.compare_digest(token, self._token)

    def _check_rate_limit(self, user_id: str) -> bool:
        """Return ``True`` if the user is within rate limits."""
        now = time.monotonic()
        window = 60.0  # one minute

        # Prune old entries
        self._rate_tracker[user_id] = [
            ts for ts in self._rate_tracker[user_id] if now - ts < window
        ]

        if len(self._rate_tracker[user_id]) >= self._rate_limit_per_minute:
            return False

        self._rate_tracker[user_id].append(now)
        return True

    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Sanitize incoming text: strip whitespace, truncate to 4000 chars."""
        if not text:
            return ""
        text = text.strip()
        if len(text) > 4000:
            text = text[:4000]
        return text

    def _chunk_message(
        self, text: str, max_length: Optional[int] = None
    ) -> list[str]:
        """Split *text* into chunks within the Synology Chat message limit.

        Splits on paragraph boundaries first, then sentences, then spaces,
        then hard-cuts.
        """
        limit = max_length or self.MAX_MESSAGE_LENGTH
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            # Paragraph boundary
            cut = remaining[:limit].rfind("\n\n")
            if cut > 0:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip("\n")
                continue

            # Single newline
            cut = remaining[:limit].rfind("\n")
            if cut > 0:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip("\n")
                continue

            # Sentence boundary
            for sep in (". ", "! ", "? "):
                cut = remaining[:limit].rfind(sep)
                if cut > 0:
                    cut += 1
                    chunks.append(remaining[:cut].rstrip())
                    remaining = remaining[cut:].lstrip()
                    break
            else:
                # Space
                cut = remaining[:limit].rfind(" ")
                if cut > 0:
                    chunks.append(remaining[:cut])
                    remaining = remaining[cut:].lstrip()
                else:
                    # Hard cut
                    chunks.append(remaining[:limit])
                    remaining = remaining[limit:]

        return chunks
