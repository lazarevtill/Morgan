"""Personal planning from durable scoped memory through exact approved note delivery."""

import json
from datetime import UTC, date, datetime

import pytest

from morgan_brain.composition import build_memory_module, open_evidence_context
from morgan_brain.daily_brief import BriefRequest, prepare_brief
from morgan_brain.local_notes import NoteWorkspace, approve, run_local
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource
from morgan_brain.tasks import TaskKey

DAY = date(2026, 11, 6)
CUTOFF = datetime(2026, 11, 6, 12, tzinfo=UTC)


@pytest.fixture
def context(tmp_path):
    """Use a real isolated SQLite memory with deterministic storage-only embeddings."""
    path = tmp_path / "memory.db"
    conn = open_db(str(path))
    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4))
    yield path, gate
    conn.close()


def memory(identity, **changes):
    """Create explicit fixture commitments, not claims about the real user's activities."""
    project = changes.pop("project", "personal")
    source = changes.pop("source", MemorySource.USER_STATED)
    content = {
        "version": "morgan.commitment.v1",
        "title": "Review priorities",
        "minutes": 20,
        "priority": "normal",
        "due": "2026-11-06",
        "status": "open",
        **changes,
    }
    return Memory(
        id=identity,
        user_id="demo",
        project=project,
        source=source,
        author_id="demo",
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
        content=json.dumps(content),
    )


def request(ids, budget=50):
    """Bind one exact source selection to the proposed delivery identity."""
    return BriefRequest(TaskKey("demo", "personal", "daily-brief"), tuple(ids), DAY, budget)


async def test_memory_to_budgeted_brief_to_managed_delivery(context, tmp_path):
    """Plan from persisted evidence, require a separate grant, and skip duplicate delivery."""
    path, gate = context
    records = [
        memory("overdue", minutes=30, priority="high", due="2026-11-05"),
        memory("today", title="Read project notes", minutes=20),
        memory("later", title="Sort shelf", minutes=15, priority="low"),
        memory("future", due="2026-11-07"),
        memory("done", status="done"),
    ]
    for record in records:
        await gate.store(record)
    source = open_evidence_context(path)
    try:
        draft = await prepare_brief(source.gate, request([r.id for r in records]), as_of=CUTOFF)
    finally:
        source.conn.close()
    assert draft.selected_ids == ("overdue", "today")
    assert draft.deferred_ids == ("later",) and draft.excluded_ids == ("future", "done")
    assert draft.planned_minutes == 50 and "no language model" in draft.task.payload
    workspace = NoteWorkspace(tmp_path / "delivery")
    try:
        assert (await run_local(workspace)).actions == 0
        assert not (workspace.path / ".approval.json").exists()
        approve(workspace, [draft.task], ttl_seconds=60)
        assert (await run_local(workspace)).actions == 1
        files = list(workspace.path.glob("note-*.md"))
        assert len(files) == 1 and files[0].read_text() == draft.task.payload
        assert (await run_local(workspace)).actions == 0
    finally:
        workspace.close()


@pytest.mark.parametrize(
    "case", ["missing", "other_project", "other_owner", "inferred", "quarantined", "instructions"]
)
async def test_unavailable_or_untrusted_evidence_fails_closed(context, case):
    """Do not silently omit missing sources or treat inferred/unusable records as commitments."""
    _, gate = context
    record = memory("one")
    if case == "other_project":
        record.project = "work"
    if case == "other_owner":
        record.user_id = "other"
    if case == "inferred":
        record.source = MemorySource.AGENT_INFERRED
    if case == "quarantined":
        from morgan_brain.models import MemoryStatus

        record.status = MemoryStatus.QUARANTINED
    if case == "instructions":
        record.instruction_like = True
    if case != "missing":
        await gate.store(record)
    with pytest.raises(ValueError):
        await prepare_brief(gate, request(["one"]), as_of=CUTOFF)


async def test_revised_old_source_is_not_planned(context):
    """A corrected commitment requires selecting the current evidence rather than stale ancestry."""
    _, gate = context
    original = memory("old")
    await gate.store(original)
    correction = memory("new", title="Updated commitment")
    correction.revises_event_ids = ["old"]
    correction.created_at = datetime(2026, 11, 2, tzinfo=UTC)
    await gate.store(correction)
    with pytest.raises(ValueError):
        await prepare_brief(gate, request(["old"]), as_of=CUTOFF)
    assert (await prepare_brief(gate, request(["new"]), as_of=CUTOFF)).selected_ids == ("new",)


@pytest.mark.parametrize(
    "patch",
    [
        {"minutes": True},
        {"minutes": 0},
        {"minutes": 999},
        {"priority": "urgent"},
        {"title": "line\nbreak"},
        {"extra": "field"},
    ],
)
async def test_malformed_commitment_cannot_become_a_plan(context, patch):
    """Reject unbounded or ambiguous planning data instead of asking a model to repair it."""
    _, gate = context
    record = memory("one")
    raw = json.loads(record.content)
    raw.update(patch)
    record.content = json.dumps(raw)
    await gate.store(record)
    with pytest.raises(ValueError):
        await prepare_brief(gate, request(["one"]), as_of=CUTOFF)


async def test_markdown_titles_are_literal_and_budget_never_overflows(context):
    """Render hostile-looking markup as text and defer tasks that cannot fit the budget."""
    _, gate = context
    await gate.store(memory("one", title="[click](javascript:alert(1)) <script>", minutes=10))
    draft = await prepare_brief(gate, request(["one"], budget=10), as_of=CUTOFF)
    assert "[click](" not in draft.task.payload and "<script>" not in draft.task.payload
    empty = await prepare_brief(gate, request(["one"], budget=1), as_of=CUTOFF)
    assert empty.planned_minutes == 0 and empty.deferred_ids == ("one",)


async def test_drafting_is_read_only_and_does_not_migrate(context):
    """The source-only path does not assemble an embedder or mutate stored evidence."""
    path, gate = context
    await gate.store(memory("one"))
    source = open_evidence_context(path)
    try:
        before = source.conn.execute("PRAGMA user_version").fetchone()[0]
        await prepare_brief(source.gate, request(["one"]), as_of=CUTOFF)
        assert source.conn.execute("PRAGMA user_version").fetchone()[0] == before
        assert source.conn.total_changes == 0
    finally:
        source.conn.close()


async def test_cli_prepares_private_draft_without_approval(context, tmp_path):
    """Exercise the source-only command and prove draft creation does not grant delivery."""
    import asyncio
    import sys

    path, gate = context
    await gate.store(memory("one"))
    destination = tmp_path / "draft.md"
    args = [
        sys.executable,
        "-m",
        "morgan_brain.daily_brief_cli",
        "--memory-db",
        str(path),
        "--user",
        "demo",
        "--project",
        "personal",
        "--task",
        "brief",
        "--evidence",
        "one",
        "--day",
        DAY.isoformat(),
        "--budget",
        "30",
        "--output",
        str(destination),
    ]
    child = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    output, error = await asyncio.wait_for(child.communicate(), 10)
    assert child.returncode == 0, error
    assert json.loads(output)["approved"] is False
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / ".approval.json").exists()
    before = destination.read_bytes()
    repeated = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await asyncio.wait_for(repeated.communicate(), 10)
    assert repeated.returncode != 0 and destination.read_bytes() == before


@pytest.mark.parametrize(
    "ids,budget",
    [([], 30), (["one", "one"], 30), (["one"], 0), (["one"], 481), (["one\nheading"], 30)],
)
def test_invalid_selection_is_refused(ids, budget):
    """Reject implicit, duplicate, nonprintable or unbounded planning requests."""
    with pytest.raises(ValueError):
        request(ids, budget)
