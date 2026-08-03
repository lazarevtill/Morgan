"""Tool registry + permission-gated executor.

``ToolRegistry`` is a plain name → BaseTool mapping with schema introspection.
``ToolExecutorImpl`` implements the ``ToolExecutor`` Protocol, wires in the
``PermissionGate`` for default-deny admission control, and publishes a
``TOOL_INVOKED`` audit event on an optional bus.
"""

from __future__ import annotations

from typing import Any

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.interfaces.tools import BaseTool, ToolResult
from morgan_brain.security.permissions import PermissionGate


class ToolRegistry:
    """Mutable name → BaseTool registry shared by the executor."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add *tool* to the registry (replaces any prior tool with the same name)."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under *name*, or ``None``."""
        return self._tools.get(name)

    def list_specs(self) -> list[dict[str, Any]]:
        """Return a list of ``{name, description, schema}`` dicts for every registered tool."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.schema(),
            }
            for t in self._tools.values()
        ]


class ToolExecutorImpl:
    """Permission-gated tool executor implementing the ``ToolExecutor`` Protocol.

    Parameters
    ----------
    registry:
        The ``ToolRegistry`` that holds registered tools.  The executor delegates
        ``register`` and ``list`` to it.
    gate:
        The ``PermissionGate`` consulted before every ``execute`` call.
        Uses ``gate.check(name, params=list(kwargs))``; if the gate returns
        False the call is short-circuited with ``ToolResult(ok=False, error="permission denied")``.
    bus:
        Optional ``EventBus``.  When present, a ``TOOL_INVOKED`` event is published
        after each execution (success or failure).
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gate: PermissionGate,
        bus: EventBus | None = None,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._bus = bus

    # ------------------------------------------------------------------
    # ToolExecutor Protocol
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register *tool* in the underlying registry."""
        self._registry.register(tool)

    def list(self) -> list[dict[str, Any]]:
        """Return tool specs from the underlying registry."""
        return self._registry.list_specs()

    async def execute(self, name: str, *, user_id: str, project: str, **kwargs: Any) -> ToolResult:
        """Execute *name* after gate-checking; publish audit event; catch tool exceptions.

        Steps
        -----
        1. Look up the tool — unknown tool → ``ToolResult(ok=False, error="unknown tool: ...")``.
        2. ``gate.check(name, params=list(kwargs))`` — denied → ``ToolResult(ok=False, error="permission denied")``.
        3. ``await tool.run(user_id=user_id, project=project, **kwargs)`` — exception caught →
           ``ToolResult(ok=False, error=...)``.
        4. Publish ``TOOL_INVOKED`` event if a bus is configured.

        ``project`` is a required keyword and comes from the turn, not from the model. The
        reasoner strips any ``project`` the model put in its tool-call arguments before
        calling this — choosing which project to search is a scoping decision, and the
        assistant does not get to make it.
        """
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}")

        if not self._gate.check(name, params=list(kwargs)):
            await self._emit(name, user_id=user_id, ok=False)
            return ToolResult(ok=False, error="permission denied")

        try:
            result = await tool.run(user_id=user_id, project=project, **kwargs)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(ok=False, error=str(exc))

        await self._emit(name, user_id=user_id, ok=result.ok)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit(self, tool_name: str, *, user_id: str, ok: bool) -> None:
        if self._bus is not None:
            await self._bus.publish(
                Event(
                    type=EventType.TOOL_INVOKED,
                    user_id=user_id,
                    payload={"tool": tool_name, "ok": ok},
                )
            )
