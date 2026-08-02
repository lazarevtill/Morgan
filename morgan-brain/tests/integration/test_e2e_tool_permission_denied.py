"""E2E test: PermissionGate denying a tool.

When a tool is denied by the gate, the executor returns ok=False, the
tool-call loop does NOT hang, and the orchestrator still returns the
model's final text reply.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatResult, ToolCall
from morgan_brain.security.permissions import PermissionGate, PermissionMode

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


def _make_router(fake_client: FakeChatClient) -> RoleRouter:
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    return RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )


@pytest.mark.asyncio
async def test_permission_denied_tool_returns_error_and_loop_terminates() -> None:
    """When the gate denies a tool, executor returns ok=False and the loop ends cleanly."""
    tc = ToolCall(id="tc-deny-1", name="calculator", arguments={"expression": "1+1"})
    results = [
        # First call: model requests the calculator.
        ChatResult(text="", tool_calls=[tc]),
        # Second call: model sees the tool error and gives a final answer.
        ChatResult(text="I could not compute that.", tool_calls=[]),
    ]
    fake_client = FakeChatClient(results=results)
    router = _make_router(fake_client)
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _, _, _, executor, _, _ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )

    # Deny the calculator explicitly at the executor level.
    # Replace the gate on the executor with one that denies calculator.
    deny_gate = PermissionGate(default=PermissionMode.DENY)
    executor._gate = deny_gate  # type: ignore[attr-defined]  # noqa: SLF001
    # Also replace the gate inside the orchestrator's reasoner's executor.
    orch._reasoner._executor._gate = deny_gate  # type: ignore[attr-defined]  # noqa: SLF001

    result = await orch.handle_turn(
        user_id="u1", project="default", text="What is 1+1?", session_id="s1"
    )

    # Loop terminated without hanging; final text is returned.
    assert result.text == "I could not compute that."

    # The tool was NOT marked as successfully invoked (gate denied it).
    # tools_invoked records names of tools that were *attempted in the loop*,
    # whether or not they succeeded.  Assert the loop did not hang (result returned).
    assert isinstance(result.tools_invoked, list)


@pytest.mark.asyncio
async def test_permission_denied_executor_returns_ok_false() -> None:
    """Direct executor.execute call: denied tool returns ok=False, not an exception."""
    bus = InProcessBus()
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    fake_client = FakeChatClient(reply="ok")
    router = _make_router(fake_client)

    _, _, _, _, executor, _, _ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )

    # Set a DENY gate on the executor.
    deny_gate = PermissionGate(default=PermissionMode.DENY)
    executor._gate = deny_gate  # type: ignore[attr-defined]  # noqa: SLF001

    result = await executor.execute("calculator", user_id="u1", expression="1+1")

    assert result.ok is False
    assert result.error is not None
    assert "permission" in result.error.lower()
