"""Tests for config-driven event bus in composition (commit 1).

Verifies that:
- With default settings (event_bus="inproc"), _assemble wires an InProcessBus.
- The bus type can be confirmed via the orchestrator's _bus attribute.
- Explicitly injecting a bus is honoured (test-helper path).
"""

from __future__ import annotations

from datetime import datetime

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter


def _fake_router() -> RoleRouter:
    client = FakeChatClient(reply="ok")
    reg = CapabilityRegistry.from_seed(
        {"fake/m": {"supports_tools": False, "json_mode": "none", "context_window": 4096}}
    )
    return RoleRouter(reg=reg, bindings={"strong": [Binding("fake", "m", client)]})


def _fake_clock() -> datetime:
    return datetime(2026, 1, 1)


def test_assemble_default_uses_inprocess_bus() -> None:
    """With default settings (event_bus='inproc'), _assemble wires an InProcessBus."""
    settings = Settings(llm_model="m", llm_fast_model="m")
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=4),
        router=_fake_router(),
        settings=settings,
        clock=_fake_clock,
        temporal_path=":memory:",
    )
    assert isinstance(orch._bus, InProcessBus)  # noqa: SLF001


def test_assemble_injected_bus_is_used() -> None:
    """An explicitly injected bus is wired into the orchestrator."""
    settings = Settings(llm_model="m", llm_fast_model="m")
    injected = InProcessBus()
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=4),
        router=_fake_router(),
        settings=settings,
        clock=_fake_clock,
        temporal_path=":memory:",
        bus=injected,
    )
    assert orch._bus is injected  # noqa: SLF001
