"""Tests for the hook system."""

import asyncio

import pytest

from morgan.hook_system.types import HookType
from morgan.hook_system.manager import HookManager


@pytest.fixture
def manager():
    return HookManager()


# --- HookType enum ---

class TestHookType:
    def test_all_nine_values(self):
        assert len(HookType) == 9

    def test_values(self):
        expected = {
            "message_inbound",
            "message_reply",
            "pre_tool_use",
            "post_tool_use",
            "session_start",
            "session_end",
            "pre_compact",
            "post_compact",
            "config_change",
        }
        assert {h.value for h in HookType} == expected


# --- HookManager ---

class TestHookManager:
    @pytest.mark.asyncio
    async def test_register_and_trigger_sync(self, manager):
        calls = []

        def on_start(ctx):
            calls.append(ctx)
            return "ok"

        manager.register(HookType.SESSION_START, on_start)
        results = await manager.trigger(HookType.SESSION_START, "hello")
        assert results == ["ok"]
        assert calls == ["hello"]

    @pytest.mark.asyncio
    async def test_register_and_trigger_async(self, manager):
        async def on_end(ctx):
            return f"ended-{ctx}"

        manager.register(HookType.SESSION_END, on_end)
        results = await manager.trigger(HookType.SESSION_END, "s1")
        assert results == ["ended-s1"]

    @pytest.mark.asyncio
    async def test_multiple_handlers_fire_in_order(self, manager):
        order = []

        def h1(ctx):
            order.append(1)

        def h2(ctx):
            order.append(2)

        manager.register(HookType.MESSAGE_INBOUND, h1)
        manager.register(HookType.MESSAGE_INBOUND, h2)
        await manager.trigger(HookType.MESSAGE_INBOUND)
        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_short_circuit_on_abort(self, manager):
        calls = []

        def h1(ctx):
            calls.append(1)
            return {"abort": True}

        def h2(ctx):
            calls.append(2)

        manager.register(HookType.PRE_TOOL_USE, h1)
        manager.register(HookType.PRE_TOOL_USE, h2)
        results = await manager.trigger(HookType.PRE_TOOL_USE)
        assert calls == [1]  # h2 never called
        assert results == [{"abort": True}]

    @pytest.mark.asyncio
    async def test_unregister(self, manager):
        def handler(ctx):
            return "fired"

        manager.register(HookType.CONFIG_CHANGE, handler)
        manager.unregister(HookType.CONFIG_CHANGE, handler)
        results = await manager.trigger(HookType.CONFIG_CHANGE)
        assert results == []

    @pytest.mark.asyncio
    async def test_unregister_missing_is_silent(self, manager):
        def handler(ctx):
            pass

        manager.unregister(HookType.CONFIG_CHANGE, handler)  # should not raise

    @pytest.mark.asyncio
    async def test_trigger_no_handlers(self, manager):
        results = await manager.trigger(HookType.POST_COMPACT, {"data": 1})
        assert results == []

    @pytest.mark.asyncio
    async def test_mixed_sync_async(self, manager):
        def sync_h(ctx):
            return "sync"

        async def async_h(ctx):
            return "async"

        manager.register(HookType.MESSAGE_REPLY, sync_h)
        manager.register(HookType.MESSAGE_REPLY, async_h)
        results = await manager.trigger(HookType.MESSAGE_REPLY)
        assert results == ["sync", "async"]
