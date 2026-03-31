"""Hook manager that supports sync and async handlers with short-circuit."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Any, Callable

from morgan.hook_system.types import HookType


class HookManager:
    """Register, unregister, and trigger hook handlers.

    Handlers are fired in registration order. If any handler returns a dict
    containing ``{"abort": True}``, subsequent handlers are skipped and the
    abort result is returned immediately.

    Both sync and async handlers are supported -- sync handlers are called
    directly when ``trigger`` is awaited.
    """

    def __init__(self) -> None:
        self._handlers: dict[HookType, list[Callable]] = defaultdict(list)

    def register(self, hook_type: HookType, handler: Callable) -> None:
        """Add a handler for *hook_type*."""
        self._handlers[hook_type].append(handler)

    def unregister(self, hook_type: HookType, handler: Callable) -> None:
        """Remove a previously registered handler."""
        try:
            self._handlers[hook_type].remove(handler)
        except ValueError:
            pass  # handler was not registered -- silently ignore

    async def trigger(self, hook_type: HookType, context: Any = None) -> list[Any]:
        """Fire all handlers for *hook_type* in order.

        Returns a list of handler results. Short-circuits on ``{"abort": True}``.
        """
        results: list[Any] = []
        for handler in list(self._handlers.get(hook_type, [])):
            if inspect.iscoroutinefunction(handler):
                result = await handler(context)
            else:
                result = handler(context)
            results.append(result)
            if isinstance(result, dict) and result.get("abort") is True:
                break
        return results
