"""Redis Streams event bus — cross-service backend (brain-api ↔ learning-worker).
Same EventBus Protocol as the in-process bus.

Publish: ``xadd`` the whole :class:`~morgan_brain.interfaces.events.Event` as a
single ``data`` JSON field.

Consume: a consumer group (``xreadgroup``) runs in a background asyncio task so
that multiple worker processes can each receive every message once.  Handlers for
a given :class:`~morgan_brain.interfaces.events.EventType` are dispatched
sequentially; per-handler exceptions are caught and logged so one bad handler
cannot kill the loop or block acknowledgement.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from typing import Any

from morgan_brain.interfaces.events import Event, EventType, Handler

_STREAM = "morgan:events"
_GROUP = "morgan"
_READ_COUNT = 10
_BLOCK_MS = 1_000  # block up to 1 s per xreadgroup call

logger = logging.getLogger(__name__)


class RedisStreamsBus:
    """Redis Streams-backed event bus.

    Parameters
    ----------
    redis_url:
        ``redis://`` URL passed to ``redis.asyncio.Redis.from_url``.
    client:
        Inject a pre-built (or fake) async redis client.  When *None* the real
        client is created lazily on first use so that importing this module has
        no side-effects.
    stream:
        Redis stream key (default ``"morgan:events"``).
    group:
        Consumer-group name (default ``"morgan"``).
    consumer:
        Consumer name within the group.  Defaults to a UUID hex string so that
        each running process gets its own cursor.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        client: Any | None = None,
        stream: str = _STREAM,
        group: str = _GROUP,
        consumer: str | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer or uuid.uuid4().hex
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False
        self._consume_task: asyncio.Task[None] | None = None
        # Client storage: None means "not yet created".
        # _client_owned tracks whether WE created it (so stop() closes it).
        self._client: Any | None = client
        self._client_owned: bool = client is None  # we own it if we'll create it

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the redis client, creating it lazily if needed."""
        if self._client is None:
            import redis.asyncio as aioredis  # noqa: PLC0415

            self._client = aioredis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    # ------------------------------------------------------------------
    # EventBus Protocol
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called whenever *event_type* is published."""
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Serialize *event* to JSON and append it to the Redis stream."""
        await self._get_client().xadd(self._stream, {"data": event.model_dump_json()})

    async def start(self) -> None:
        """Create the consumer group (if absent) and start the consume loop."""
        client = self._get_client()

        # Create group; swallow BUSYGROUP if it already exists.
        try:
            await client.xgroup_create(self._stream, self._group, id="$", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise

        self._running = True
        self._consume_task = asyncio.ensure_future(self._consume_loop())

    async def stop(self) -> None:
        """Stop the consume loop and (if we own the client) close it."""
        self._running = False
        if self._consume_task is not None:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._consume_task = None
        if self._client_owned and self._client is not None:
            # redis.asyncio exposes aclose() in recent versions; fall back to close().
            close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            self._client = None

    # ------------------------------------------------------------------
    # Consume loop & dispatch
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """Background task: read messages from the stream and dispatch them."""
        client = self._get_client()
        while self._running:
            try:
                results: list[Any] = await client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=_READ_COUNT,
                    block=_BLOCK_MS,
                )
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001
                logger.exception("RedisStreamsBus: xreadgroup error")
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    try:
                        await self._handle_message(msg_id, fields)
                    except Exception:  # noqa: BLE001
                        logger.exception("RedisStreamsBus: unhandled error for msg %s", msg_id)

    async def _handle_message(self, msg_id: str, fields: dict[str, str]) -> None:
        """Parse a single stream message and dispatch to registered handlers.

        This method is intentionally public-facing (single-underscore) so that
        unit tests can drive dispatch without a running consume loop.
        """
        raw_data = fields.get("data", "")
        try:
            event = Event.model_validate_json(raw_data)
        except Exception:  # noqa: BLE001
            logger.exception(
                "RedisStreamsBus: failed to parse event from msg %s: %r", msg_id, raw_data
            )
            # Still ack so we don't re-deliver a permanently-bad message.
            await self._get_client().xack(self._stream, self._group, msg_id)
            return

        for handler in list(self._handlers.get(event.type, [])):
            try:
                await handler(event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "RedisStreamsBus: handler %s raised for event %s",
                    handler,
                    event.type,
                )

        await self._get_client().xack(self._stream, self._group, msg_id)
