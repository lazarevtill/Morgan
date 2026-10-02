import pytest

from morgan_brain.app.continuation import ContinuationRequest, continuation_messages, resume_work
from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.providers.structured import structured_input_size
from tests.unit.app.test_working_context import NOW, event, stack


async def test_escaped_continuation_payload_refuses_before_generation_and_writes():
    quote = "\\" * 240

    def span(i):
        return {"event_id": f"source-{i}", "quote": quote}

    selection = {
        "title": "Escaped input",
        "decisions": [{"choice": span(i), "reason": span(i + 4)} for i in range(4)],
        "open_questions": [span(i) for i in range(8, 12)],
        "intentions": [span(i) for i in range(12, 16)],
        "progress": [span(i) for i in range(4)],
    }
    conn, gate, _, service = stack(selection=selection)
    try:
        for i in range(16):
            await event(gate, f"source-{i}", quote)
        preview = await service.preview(
            "escaped",
            context=CheckpointContext(user_id="owner"),
            event_ids=[f"source-{i}" for i in range(16)],
        )
        await service.apply(preview)
        view = await gate.get_working_context("escaped", user_id="owner")
        request = '"' * 8192
        messages = continuation_messages(view, request)
        assert sum(len(m.content.encode("utf-8")) for m in messages) < 49152
        assert structured_input_size(messages, None, model="synthetic") > 49152

        class NoGeneration:
            async def agenerate(self, *args, **kwargs):
                raise AssertionError("Overflow must be refused before generation")

        history = SessionHistoryStore(conn, clock=lambda: NOW)
        changes = conn.total_changes
        with pytest.raises(ValueError, match="49152 bytes"):
            await resume_work(
                gate=gate,
                history=history,
                client=NoGeneration(),
                clock=lambda: NOW,
                request=ContinuationRequest(
                    context_id="escaped",
                    user_id="owner",
                    project="personal",
                    text=request,
                    session_id="independent",
                    model="synthetic",
                ),
            )
        assert conn.total_changes == changes
        assert conn.execute("SELECT count(*) FROM session_history").fetchone()[0] == 0
    finally:
        conn.close()
