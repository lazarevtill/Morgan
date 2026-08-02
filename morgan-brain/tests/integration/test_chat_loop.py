from datetime import datetime

from morgan_brain.composition import build_orchestrator_for_test


async def test_chat_loop_generates_and_recalls():
    orch, mem = build_orchestrator_for_test(
        reply="Nice to meet you!", clock=lambda: datetime(2026, 1, 1)
    )
    await mem.bus.start()

    first = await orch.handle_turn(
        user_id="u1", project="default", text="My name is Sam", session_id="s1"
    )
    assert first.text == "Nice to meet you!"

    # publish() now enqueues rather than running the storage subscriber inline (Task 15) —
    # drain the bus before asserting on what it stored.
    await mem.bus.drain()
    hits = await mem.recall_raw(user_id="u1", text="Sam")
    assert any("Sam" in h.content for h in hits)
    await mem.bus.stop()


async def test_chat_loop_is_user_scoped():
    orch, mem = build_orchestrator_for_test(reply="ok", clock=lambda: datetime(2026, 1, 1))
    await mem.bus.start()
    await orch.handle_turn(user_id="u1", project="default", text="secret for u1", session_id="s1")
    await mem.bus.drain()
    other = await mem.recall_raw(user_id="u2", text="secret")
    assert other == []
    await mem.bus.stop()
