"""Tests for RoleRouter (capability gating) and RoleFallback (event-driven failover)."""

from __future__ import annotations

import pytest

from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.interfaces.events import EventType
from morgan_brain.bus.inproc import InProcessBus


def _reg() -> CapabilityRegistry:
    return CapabilityRegistry.from_seed(
        {
            "p/weak": {"supports_tools": False, "json_mode": "none"},
            "p/strong": {"supports_tools": True, "json_mode": "json_schema"},
        }
    )


@pytest.mark.asyncio
async def test_router_returns_bound_client_for_role() -> None:
    strong = FakeChatClient(reply="s")
    r = RoleRouter(reg=_reg(), bindings={"strong": [Binding("p", "strong", strong)]})
    client, model = r.chat_for("strong")
    assert model == "strong"
    out = await client.agenerate([], model=model)
    assert out.text == "s"


@pytest.mark.asyncio
async def test_router_capability_gate_skips_incapable_binding() -> None:
    weak = FakeChatClient(reply="w")
    strong = FakeChatClient(reply="s")
    r = RoleRouter(
        reg=_reg(),
        bindings={"strong": [Binding("p", "weak", weak), Binding("p", "strong", strong)]},
    )
    client, model = r.chat_for("strong", needs_tools=True)  # weak gated out
    assert model == "strong"


@pytest.mark.asyncio
async def test_router_raises_when_no_capable_binding() -> None:
    weak = FakeChatClient(reply="w")
    r = RoleRouter(reg=_reg(), bindings={"strong": [Binding("p", "weak", weak)]})
    with pytest.raises(LookupError):
        r.chat_for("strong", needs_tools=True)


@pytest.mark.asyncio
async def test_router_raises_for_unknown_role() -> None:
    r = RoleRouter(reg=_reg(), bindings={})
    with pytest.raises(LookupError):
        r.chat_for("missing_role")


@pytest.mark.asyncio
async def test_role_fallback_advances_on_failure_and_emits_event() -> None:
    """First binding raises; second succeeds. One LLM_FALLBACK event emitted."""
    from morgan_brain.providers.resilience import RoleFallback

    failing = FakeChatClient(reply="fail")
    succeeding = FakeChatClient(reply="ok")

    # Make the failing client raise on agenerate
    _orig = failing.agenerate

    async def _raise(*a, **kw):  # type: ignore[override]
        raise RuntimeError("simulated failure")

    failing.agenerate = _raise  # type: ignore[method-assign]

    reg = CapabilityRegistry.from_seed(
        {
            "p/failing": {"supports_tools": False, "json_mode": "none"},
            "p/succeeding": {"supports_tools": False, "json_mode": "none"},
        }
    )
    bindings = [Binding("p", "failing", failing), Binding("p", "succeeding", succeeding)]
    router = RoleRouter(reg=reg, bindings={"chat": bindings})

    bus = InProcessBus()
    events: list = []

    async def capture(e):  # type: ignore[no-untyped-def]
        events.append(e)

    bus.subscribe(EventType.LLM_FALLBACK, capture)

    fb = RoleFallback(router=router, bus=bus)

    async def call_fn(client, model):  # type: ignore[no-untyped-def]
        return await client.agenerate([], model=model)

    result = await fb.call("chat", call_fn)
    assert result.text == "ok"
    assert len(events) == 1
    assert events[0].type == EventType.LLM_FALLBACK


@pytest.mark.asyncio
async def test_role_fallback_raises_when_all_bindings_fail() -> None:
    """All bindings fail → RoleFallback should propagate/raise."""
    from morgan_brain.providers.resilience import RoleFallback

    failing = FakeChatClient(reply="fail")

    async def _raise(*a, **kw):  # type: ignore[override]
        raise RuntimeError("always fails")

    failing.agenerate = _raise  # type: ignore[method-assign]

    reg = CapabilityRegistry.from_seed({"p/failing": {"supports_tools": False}})
    router = RoleRouter(reg=reg, bindings={"chat": [Binding("p", "failing", failing)]})
    bus = InProcessBus()
    fb = RoleFallback(router=router, bus=bus)

    async def call_fn(client, model):  # type: ignore[no-untyped-def]
        return await client.agenerate([], model=model)

    with pytest.raises(RuntimeError):
        await fb.call("chat", call_fn)
