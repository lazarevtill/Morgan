"""Synthetic constraints: current scoped evidence, exact boundaries, and immutable recovery."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from morgan_brain.availability import Unavailability
from morgan_brain.composition import build_memory_module, open_evidence_context
from morgan_brain.daily_brief import BriefRequest, prepare_brief
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource, MemoryStatus
from morgan_brain.tasks import TaskKey


@pytest.fixture
def context(tmp_path):
    conn = open_db(str(tmp_path / "memory.db"))
    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4))
    yield tmp_path / "memory.db", gate
    conn.close()


def interval(**changes):
    return {
        "version": "morgan.unavailability.v1",
        "starts_at": "2027-04-10T00:00:00+02:00",
        "ends_at": "2027-04-12T00:00:00+02:00",
        "timezone": "Europe/Paris",
        "status": "active",
        "source_ids": ["statement"],
        **changes,
    }


def record(identity, content, **changes):
    return Memory(
        id=identity,
        user_id="demo",
        project="pilot",
        source=MemorySource.USER_STATED,
        author_id="demo",
        created_at=datetime(2027, 4, 1, tzinfo=UTC),
        content=content if isinstance(content, str) else json.dumps(content),
        **changes,
    )


async def seed(gate):
    await gate.store(record("statement", "Unavailable on the reviewed dates, no location claim."))
    await gate.store(record("constraint", interval()))
    await gate.store(
        record(
            "task",
            {
                "version": "morgan.commitment.v1",
                "title": "Read a chapter",
                "due": "2027-04-09",
                "minutes": 20,
                "priority": "normal",
                "status": "open",
            },
        )
    )


def request(day=date(2027, 4, 10)):
    return BriefRequest(
        TaskKey("demo", "pilot", "plan"),
        ("task",),
        day,
        30,
        unavailability_ids=("constraint",),
        timezone="Europe/Paris",
    )


CUTOFF = datetime(2027, 4, 8, tzinfo=UTC)


@pytest.mark.parametrize("day,blocked", [(9, False), (10, True), (11, True), (12, False)])
async def test_half_open_days_and_source_explanation(context, day, blocked):
    _, gate = context
    await seed(gate)
    draft = await prepare_brief(gate, request(date(2027, 4, day)), as_of=CUTOFF)
    assert draft.blocked_ids == (("task",) if blocked else ())
    assert draft.planned_minutes == (0 if blocked else 20)
    assert "statement" in draft.task.payload and "constraint" in draft.task.payload
    assert (
        "Europe/Paris" in draft.task.payload and "does not describe location" in draft.task.payload
    )
    assert "[start, end)" in draft.task.payload


@pytest.mark.parametrize(
    "patch",
    [
        {"starts_at": "2027-04-10T00:00:00"},
        {"ends_at": "2027-04-10T00:00:00+02:00"},
        {"ends_at": "2029-04-10T00:00:00+02:00"},
        {"timezone": "Not/AZone"},
        {"starts_at": "2027-04-10T00:00:00+01:00"},
        {"starts_at": "2027-03-28T02:30:00+01:00"},
        {"source_ids": []},
        {"source_ids": ["statement", "statement"]},
        {"source_ids": ["bad\nsource"]},
        {"location": "somewhere"},
    ],
)
def test_invalid_or_location_fields_are_rejected(patch):
    with pytest.raises(ValueError):
        Unavailability.model_validate_json(json.dumps(interval(**patch)))


@pytest.mark.parametrize(
    "start,end,day,expected",
    [
        ("2027-03-28T00:00:00+01:00", "2027-03-29T00:00:00+02:00", "2027-03-28", True),
        ("2027-03-28T00:00:00+01:00", "2027-03-29T00:00:00+02:00", "2027-03-29", False),
        ("2027-10-31T02:30:00+02:00", "2027-10-31T02:30:00+01:00", "2027-10-31", True),
        ("2027-04-10T23:59:59+02:00", "2027-04-11T00:00:00+02:00", "2027-04-11", False),
    ],
)
def test_dst_fold_gap_and_exact_boundary(start, end, day, expected):
    constraint = Unavailability.model_validate_json(
        json.dumps(interval(starts_at=start, ends_at=end))
    )
    assert constraint.overlaps_day(date.fromisoformat(day), "Europe/Paris") is expected


def test_planning_timezone_changes_calendar_day():
    constraint = Unavailability.model_validate_json(
        json.dumps(
            interval(
                starts_at="2027-04-10T00:00:00+02:00",
                ends_at="2027-04-10T01:00:00+02:00",
            )
        )
    )
    assert constraint.overlaps_day(date(2027, 4, 9), "UTC")
    assert not constraint.overlaps_day(date(2027, 4, 10), "UTC")


async def test_partial_day_conservatively_blocks_all_candidates(context):
    _, gate = context
    await seed(gate)
    correction = record(
        "corrected",
        interval(
            starts_at="2027-04-10T12:00:00+02:00",
            ends_at="2027-04-10T12:01:00+02:00",
        ),
        revises_event_ids=["constraint"],
    )
    correction.created_at = datetime(2027, 4, 2, tzinfo=UTC)
    await gate.store(correction)
    draft = await prepare_brief(gate, request(), as_of=CUTOFF)
    assert draft.blocked_ids == ("task",) and draft.planned_minutes == 0
    assert draft.unavailability_resolved_ids == (("constraint", "corrected"),)
    assert "does not claim that the entire day is unavailable" in draft.task.payload


async def test_cancellation_persists_across_reopen_without_rewriting_history(context):
    path, gate = context
    await seed(gate)
    cancelled = record("cancelled", interval(status="cancelled"), revises_event_ids=["constraint"])
    cancelled.created_at = datetime(2027, 4, 2, tzinfo=UTC)
    await gate.store(cancelled)
    reopened = open_evidence_context(path)
    try:
        for _ in range(2):
            draft = await prepare_brief(reopened.gate, request(), as_of=CUTOFF)
            assert draft.selected_ids == ("task",) and not draft.blocked_ids
            assert draft.unavailability_resolved_ids == (("constraint", "cancelled"),)
        assert reopened.conn.total_changes == 0
    finally:
        reopened.conn.close()


@pytest.mark.parametrize("target", ["statement", "constraint"])
@pytest.mark.parametrize(
    "case", ["missing", "scope", "owner", "inferred", "quarantine", "instruction"]
)
async def test_untrusted_or_missing_constraints_and_sources_fail_closed(context, target, case):
    _, gate = context
    for item in [
        record("statement", "Unavailable."),
        record("constraint", interval()),
        record(
            "task",
            {
                "version": "morgan.commitment.v1",
                "title": "Read",
                "due": "2027-04-09",
                "minutes": 20,
                "priority": "normal",
                "status": "open",
            },
        ),
    ]:
        if item.id == target:
            if case == "missing":
                continue
            if case == "scope":
                item.project = "other"
            if case == "owner":
                item.user_id = "other"
            if case == "inferred":
                item.source = MemorySource.AGENT_INFERRED
            if case == "quarantine":
                item.status = MemoryStatus.QUARANTINED
            if case == "instruction":
                item.instruction_like = True
        await gate.store(item)
    with pytest.raises(ValueError):
        await prepare_brief(gate, request(), as_of=CUTOFF)


async def test_conflicting_cancellations_do_not_silently_unblock(context):
    _, gate = context
    await seed(gate)
    for identity in ["branch-a", "branch-b"]:
        item = record(identity, interval(status="cancelled"), revises_event_ids=["constraint"])
        item.created_at = datetime(2027, 4, 2, tzinfo=UTC)
        await gate.store(item)
    with pytest.raises(ValueError, match="ambiguous"):
        await prepare_brief(gate, request(), as_of=CUTOFF)


async def test_revised_source_requires_new_reviewed_constraint(context):
    _, gate = context
    await seed(gate)
    correction = record("statement-v2", "Changed availability.", revises_event_ids=["statement"])
    correction.created_at = datetime(2027, 4, 2, tzinfo=UTC)
    await gate.store(correction)
    with pytest.raises(ValueError, match="source evidence"):
        await prepare_brief(gate, request(), as_of=CUTOFF)


def test_missing_timezone_and_duplicate_ids_fail():
    with pytest.raises(ValueError):
        replace(request(), timezone=None)
    with pytest.raises(ValueError):
        replace(request(), unavailability_ids=("constraint", "constraint"))


async def test_cli_writes_private_blocked_plan_without_mutating_memory(context, tmp_path):
    import asyncio
    import os
    import subprocess
    import sys

    path, gate = context
    await seed(gate)
    # The fixture's event times are deliberately in 2027; choose an explicit past version
    # for this real-clock CLI test rather than monkeypatching the subprocess clock.
    from datetime import timedelta

    for identity in ("statement", "constraint", "task"):
        result = await gate.evidence(user_id="demo", project="pilot", evidence_ids=[identity])
        item = record(f"cli-{identity}", result.records[0].content)
        item.created_at = datetime.now(UTC) - timedelta(days=1)
        if identity == "constraint":
            content = interval(source_ids=["cli-statement"])
            item.content = json.dumps(content)
        await gate.store(item)
    output = tmp_path / "output" / "draft.md"
    args = [
        sys.executable,
        "-m",
        "morgan_brain.daily_brief_cli",
        "--memory-db",
        str(path),
        "--user",
        "demo",
        "--project",
        "pilot",
        "--task",
        "draft",
        "--evidence",
        "cli-task",
        "--day",
        "2027-04-10",
        "--budget",
        "30",
        "--unavailability",
        "cli-constraint",
        "--timezone",
        "Europe/Paris",
        "--output",
        str(output),
    ]
    result = await asyncio.to_thread(
        subprocess.run,
        args,
        capture_output=True,
        text=True,
        timeout=10,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["blocked"] == 1 and payload["planned_minutes"] == 0
    assert payload["resolved_unavailability"] == {"cli-constraint": "cli-constraint"}
    assert r"cli\-statement" in output.read_text()
    assert not payload["approved"]


@pytest.mark.parametrize("case", ["future", "wrong-schema", "self-source"])
async def test_constraint_must_be_current_typed_and_independently_sourced(context, case):
    _, gate = context
    await seed(gate)
    changed = record("changed", interval(), revises_event_ids=["constraint"])
    changed.created_at = datetime(2027, 4, 2, tzinfo=UTC)
    if case == "future":
        changed.created_at = datetime(2028, 1, 1, tzinfo=UTC)
    if case == "wrong-schema":
        changed.content = '{"version":"unrecognized"}'
    if case == "self-source":
        changed.content = json.dumps(interval(source_ids=["changed"]))
    await gate.store(changed)
    if case == "future":
        # A future correction must not prematurely release the current restriction.
        assert (await prepare_brief(gate, request(), as_of=CUTOFF)).blocked_ids == ("task",)
    else:
        with pytest.raises(ValueError):
            await prepare_brief(gate, request(), as_of=CUTOFF)


def test_in_memory_nonexistent_zoneinfo_wall_time_is_rejected():
    from zoneinfo import ZoneInfo

    with pytest.raises(ValueError):
        Unavailability(
            version="morgan.unavailability.v1",
            starts_at=datetime(2027, 3, 28, 2, 30, tzinfo=ZoneInfo("Europe/Paris")),
            ends_at=datetime(2027, 3, 28, 4, tzinfo=ZoneInfo("Europe/Paris")),
            timezone="Europe/Paris",
            status="active",
            source_ids=["statement"],
        )


def test_nonexistent_calendar_day_is_rejected():
    constraint = Unavailability.model_validate_json(json.dumps(interval()))
    with pytest.raises(ValueError, match="does not exist"):
        constraint.overlaps_day(date(2011, 12, 30), "Pacific/Apia")


async def test_exact_constraint_retry_is_idempotent_and_changed_retry_is_rejected(context):
    _, gate = context
    await seed(gate)
    original = record("constraint", interval())
    await gate.store(original)
    changed = record("constraint", interval(status="cancelled"))
    with pytest.raises(ValueError):
        await gate.store(changed)
    draft = await prepare_brief(gate, request(), as_of=CUTOFF)
    assert draft.blocked_ids == ("task",)
