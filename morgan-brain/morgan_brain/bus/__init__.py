"""Event bus backends. Select via MORGAN_EVENT_BUS (inproc | redis)."""

from __future__ import annotations

from morgan_brain.config import get_settings
from morgan_brain.interfaces.events import EventBus


def get_event_bus() -> EventBus:
    """Return the configured event bus singleton."""
    settings = get_settings()
    if settings.event_bus == "redis":
        from morgan_brain.bus.redis_streams import RedisStreamsBus

        return RedisStreamsBus(settings.redis_url)
    from morgan_brain.bus.inproc import InProcessBus

    return InProcessBus()
