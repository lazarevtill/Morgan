"""Eight authored RU/EN lifecycle views, without inference or token claims."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import Checkpoint, ReportedProgress
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource

FIXTURE = Path(__file__).with_name("resume_cases_v1.json")
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_frozen_checkpoint_lifecycle(case, tmp_path):
    clock = [datetime.fromisoformat("2026-09-15T00:00:00+00:00")]
    conn = open_db(str(tmp_path / "frozen.db"))
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: clock[0])
    )
    try:
        support = []
        progress = []
        initial = [event for event in case["events"] if not event.get("revises_event_ids")]
        for event in initial:
            await gate.store(
                Memory(
                    id=event["id"],
                    user_id="owner",
                    content=event["text"],
                    source=MemorySource(event.get("source", "user_stated")),
                    author_id="fixture-owner",
                    effective_at=datetime.fromisoformat(
                        event.get("effective_at", "2026-09-01T00:00:00Z")
                    ),
                )
            )
            if event.get("source") == "agent_inferred":
                progress.append(ReportedProgress(text=event["text"], reference_ids=[event["id"]]))
            else:
                support.append(event["id"])
        checkpoint = Checkpoint(
            kind="goal" if case["id"].startswith("language") else "task",
            title=case["checkpoint_id"],
            objective=initial[0]["text"],
            subject_entity_id=case["subject_entity_id"],
            applies_to=case["applies_to"],
            progress=progress,
        )
        identity = await gate.put_checkpoint(
            checkpoint,
            checkpoint_id=case["checkpoint_id"],
            user_id="owner",
            support_event_ids=support,
        )
        for event in case["events"]:
            if not event.get("revises_event_ids"):
                continue
            await gate.store(
                Memory(
                    id=event["id"],
                    user_id="owner",
                    content=event["text"],
                    source=MemorySource.USER_STATED,
                    author_id="fixture-owner",
                    effective_at=datetime.fromisoformat(event["effective_at"]),
                    revises_event_ids=event["revises_event_ids"],
                )
            )
        for observation in case["resume_observations"]:
            clock[0] = datetime.fromisoformat(observation["as_of"])
            result = await gate.get_checkpoint(case["checkpoint_id"], user_id="owner")
            expected = observation.get("old_checkpoint_state", "needs_rebuild")
            assert result.eligibility == expected
            assert result.fact.id == identity
            if expected == "needs_rebuild":
                assert result.state is None
            else:
                assert result.state.progress[0].verification == "unverified_agent_report"
                assert not set(result.state.progress[0].reference_ids) & set(
                    result.fact.support_event_ids
                )
        # Source JSON and ordinary files can preserve precisely the same data.
        # They require explicit lifecycle checking; no model/packing superiority is claimed.
        path = tmp_path / "plain-checkpoint.json"
        path.write_text(checkpoint.encode(support), encoding="utf-8")
        assert Checkpoint.model_validate_json(path.read_text(encoding="utf-8")) == checkpoint
    finally:
        conn.close()


def test_fixture_bytes_and_real_cyrillic():
    assert len([c for c in FIXTURE.read_text(encoding="utf-8") if "\u0400" <= c <= "\u04ff"]) == 850
    assert (
        hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        == "b680a67f269187e50ea2020726a9b218b5681fca1a2f28d2688718c36b82d479"
    )
