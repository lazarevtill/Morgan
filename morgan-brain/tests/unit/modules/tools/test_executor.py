"""Unit tests for ToolRegistry and ToolExecutorImpl.

All tests are deterministic and in-process — no network, no filesystem side-effects.
"""

from __future__ import annotations

from typing import Any


from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.interfaces.tools import ToolResult
from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
from morgan_brain.security.permissions import Grant, PermissionGate, PermissionMode


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


class _EchoTool:
    """A minimal BaseTool that echoes back the kwargs it received."""

    name = "echo"
    description = "Echoes kwargs."

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"message": {"type": "string"}}}

    async def run(self, *, user_id: str, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output={"user_id": user_id, **kwargs})


class _BombTool:
    """A BaseTool that always raises."""

    name = "bomb"
    description = "Always explodes."

    def schema(self) -> dict[str, Any]:
        return {}

    async def run(self, *, user_id: str, **kwargs: Any) -> ToolResult:
        raise RuntimeError("kaboom")


def _auto_gate() -> PermissionGate:
    return PermissionGate(default=PermissionMode.AUTO)


def _ask_gate_with_grant(tool: str) -> PermissionGate:
    gate = PermissionGate(default=PermissionMode.ASK)
    gate.grant(Grant(tool=tool))
    return gate


def _deny_gate() -> PermissionGate:
    return PermissionGate(default=PermissionMode.DENY)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_list() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    specs = reg.list_specs()
    assert len(specs) == 1
    assert specs[0]["name"] == "echo"
    assert specs[0]["description"] == "Echoes kwargs."
    assert "properties" in specs[0]["schema"]


def test_registry_get_known() -> None:
    reg = ToolRegistry()
    tool = _EchoTool()
    reg.register(tool)  # type: ignore[arg-type]
    assert reg.get("echo") is tool


def test_registry_get_unknown_returns_none() -> None:
    reg = ToolRegistry()
    assert reg.get("nope") is None


def test_registry_register_replaces_existing() -> None:
    reg = ToolRegistry()
    t1 = _EchoTool()
    t2 = _EchoTool()
    reg.register(t1)  # type: ignore[arg-type]
    reg.register(t2)  # type: ignore[arg-type]
    assert reg.get("echo") is t2


# ---------------------------------------------------------------------------
# ToolExecutorImpl — register + list delegate to registry
# ---------------------------------------------------------------------------


def test_executor_register_and_list() -> None:
    reg = ToolRegistry()
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    executor.register(_EchoTool())  # type: ignore[arg-type]
    specs = executor.list()
    assert len(specs) == 1
    assert specs[0]["name"] == "echo"


# ---------------------------------------------------------------------------
# execute — unknown tool
# ---------------------------------------------------------------------------


async def test_execute_unknown_tool_returns_error() -> None:
    reg = ToolRegistry()
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    result = await executor.execute("missing", user_id="u1")
    assert result.ok is False
    assert "unknown tool" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# execute — permission denied
# ---------------------------------------------------------------------------


async def test_execute_denied_tool_returns_ok_false_not_raised() -> None:
    """A denied tool must return ToolResult(ok=False), not raise."""
    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_deny_gate())
    result = await executor.execute("echo", user_id="u1", message="hi")
    assert result.ok is False
    assert "permission denied" in (result.error or "")


# ---------------------------------------------------------------------------
# execute — allowed tool runs and returns output
# ---------------------------------------------------------------------------


async def test_execute_allowed_tool_returns_output() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    result = await executor.execute("echo", user_id="u1", message="hello")
    assert result.ok is True
    assert result.output == {"user_id": "u1", "message": "hello"}


async def test_execute_with_explicit_grant() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_ask_gate_with_grant("echo"))
    result = await executor.execute("echo", user_id="u2", message="world")
    assert result.ok is True


# ---------------------------------------------------------------------------
# execute — tool exception → ToolResult(ok=False)
# ---------------------------------------------------------------------------


async def test_execute_tool_exception_returns_ok_false() -> None:
    reg = ToolRegistry()
    reg.register(_BombTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    result = await executor.execute("bomb", user_id="u1")
    assert result.ok is False
    assert "kaboom" in (result.error or "")


# ---------------------------------------------------------------------------
# execute — TOOL_INVOKED event emitted on bus
# ---------------------------------------------------------------------------


async def test_execute_emits_tool_invoked_event() -> None:
    """Successful execution must publish a TOOL_INVOKED event with ok=True."""
    bus = InProcessBus()
    captured: list[Event] = []

    async def _capture(event: Event) -> None:
        captured.append(event)

    bus.subscribe(EventType.TOOL_INVOKED, _capture)

    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate(), bus=bus)
    await executor.execute("echo", user_id="u1", message="ping")

    assert len(captured) == 1
    evt = captured[0]
    assert evt.type is EventType.TOOL_INVOKED
    assert evt.user_id == "u1"
    assert evt.payload["tool"] == "echo"
    assert evt.payload["ok"] is True


async def test_execute_denied_emits_tool_invoked_event_with_ok_false() -> None:
    """A denied call must still publish a TOOL_INVOKED event with ok=False."""
    bus = InProcessBus()
    captured: list[Event] = []

    async def _capture(event: Event) -> None:
        captured.append(event)

    bus.subscribe(EventType.TOOL_INVOKED, _capture)

    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_deny_gate(), bus=bus)
    await executor.execute("echo", user_id="u1")

    assert len(captured) == 1
    assert captured[0].payload["ok"] is False


async def test_execute_no_bus_does_not_raise() -> None:
    """Executor without a bus must work silently."""
    reg = ToolRegistry()
    reg.register(_EchoTool())  # type: ignore[arg-type]
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    result = await executor.execute("echo", user_id="u1")
    assert result.ok is True


# ---------------------------------------------------------------------------
# Protocols satisfied
# ---------------------------------------------------------------------------


def test_executor_satisfies_tool_executor_protocol() -> None:
    from morgan_brain.interfaces.tools import ToolExecutor

    reg = ToolRegistry()
    executor = ToolExecutorImpl(registry=reg, gate=_auto_gate())
    assert isinstance(executor, ToolExecutor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> None:
    pass
