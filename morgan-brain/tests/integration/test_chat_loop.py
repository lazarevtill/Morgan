from datetime import datetime

from morgan_brain.composition import build_orchestrator_for_test


async def test_chat_loop_generates_and_recalls():
    orch, mem = build_orchestrator_for_test(reply="Nice to meet you!", clock=lambda: datetime(2026, 1, 1))

    first = await orch.handle_turn(user_id="u1", text="My name is Sam", session_id="s1")
    assert first.text == "Nice to meet you!"

    hits = await mem.recall_raw(user_id="u1", text="Sam")
    assert any("Sam" in h.content for h in hits)


async def test_chat_loop_is_user_scoped():
    orch, mem = build_orchestrator_for_test(reply="ok", clock=lambda: datetime(2026, 1, 1))
    await orch.handle_turn(user_id="u1", text="secret for u1", session_id="s1")
    other = await mem.recall_raw(user_id="u2", text="secret")
    assert other == []
