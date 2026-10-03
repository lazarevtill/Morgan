import json

import pytest

from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.working_context import WorkingContextPreview
from tests.unit.app.test_working_context import event, stack


@pytest.mark.parametrize("tamper", ["drop", "replace", "context", "head", "generation"])
async def test_input_attestation_refuses_editable_packet_tampering(tamper):
    conn, gate, _, service = stack()
    try:
        await event(gate)
        await event(gate, "unselected", "Old valid limit 41 cm")
        proposal = await service.preview(
            "context",
            context=CheckpointContext(user_id="owner"),
            event_ids=["decision", "unselected"],
        )
        packet = proposal.model_dump(mode="json")
        if tamper == "drop":
            packet["evidence_basis"] = [
                r for r in packet["evidence_basis"] if r["id"] == "decision"
            ]
            await event(gate, "updated", "New limit 88 cm", revises_event_ids=["unselected"])
        elif tamper == "replace":
            packet["evidence_basis"][1]["content"] = "Different model input"
        elif tamper == "context":
            packet["context_id"] = "other"
        elif tamper == "head":
            packet["expected_fact_id"] = "caller-supplied-head"
        else:
            packet["generation"] += 1
        with pytest.raises(ValueError, match="input basis changed"):
            await service.apply(WorkingContextPreview.model_validate(packet))
        assert await gate.get_working_context("context", user_id="owner") is None
    finally:
        conn.close()


async def test_seal_survives_restart_and_selected_output_edits_without_preview_writes(tmp_path):
    path = tmp_path / "sealed.db"
    conn, gate, _, service = stack(path)
    await event(gate)
    await event(gate, "unselected", "Retain the untouched candidate")
    changes = conn.total_changes
    proposal = await service.preview(
        "context", context=CheckpointContext(user_id="owner"), event_ids=["decision", "unselected"]
    )
    assert conn.total_changes == changes
    packet = json.loads(proposal.model_dump_json())
    packet["state"]["title"] = "Owner-edited draft"
    conn.close()
    conn, gate, _, service = stack(path)
    try:
        await service.apply(WorkingContextPreview.model_validate(packet))
        view = await gate.get_working_context("context", user_id="owner")
        assert view.state.title == "Owner-edited draft"
        fact = conn.execute(
            "SELECT support_event_ids FROM facts WHERE id=?", (view.fact_id,)
        ).fetchone()
        assert json.loads(fact[0]) == ["decision"]
        assert "unselected" not in view.state.event_ids()
    finally:
        conn.close()


async def test_independently_initialized_database_cannot_verify_another_database_packet():
    first, gate, _, service = stack()
    second, other_gate, _, other_service = stack()
    try:
        await event(gate)
        await event(other_gate)
        proposal = await service.preview(
            "context", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
        )
        with pytest.raises(ValueError, match="input basis changed"):
            await other_service.apply(proposal)
        assert await other_gate.get_working_context("context", user_id="owner") is None
    finally:
        first.close()
        second.close()
