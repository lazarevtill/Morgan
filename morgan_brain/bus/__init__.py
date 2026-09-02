"""Event bus backends. Select via MORGAN_EVENT_BUS (inproc | redis)."""

from __future__ import annotations

from morgan_brain.config import get_settings
from morgan_brain.interfaces.events import EventBus


def get_event_bus() -> EventBus:
    """Build a new event bus instance from settings.event_bus.

    Despite the name, this is NOT a singleton: every call constructs a fresh
    ``InProcessBus``/``RedisStreamsBus``. A caller that subscribes handlers and later calls
    ``start()``/``stop()`` must call this once and share that one instance everywhere it's
    needed (composition.py's ``build_app_context``/``build_worker_context`` do this) --
    calling it again elsewhere silently produces a second, disconnected bus that never sees
    the first one's subscriptions.
    """
    settings = get_settings()
    if settings.event_bus == "redis":
        from morgan_brain.bus.redis_streams import RedisStreamsBus

        return RedisStreamsBus(settings.redis_url)
    from morgan_brain.bus.inproc import InProcessBus

    return InProcessBus()
