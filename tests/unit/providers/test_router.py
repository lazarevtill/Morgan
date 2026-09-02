"""Tests for RoleRouter (capability gating)."""

from __future__ import annotations

import pytest

from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter


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
    _client, model = r.chat_for("strong", needs_tools=True)  # weak gated out
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
