"""Revision eligibility must precede real SQLite vector/FTS limits and the floor."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.revisions import RevisionResolver
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryQuery, MemorySource, MemoryStatus

JAN = datetime(2026, 1, 1, tzinfo=UTC)
JUN = datetime(2026, 6, 1, tzinfo=UTC)
SEP = datetime(2026, 9, 1, tzinfo=UTC)


class RankedEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        if text.startswith("lower"):
            return [0.2, 0.98, 0.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("floor", [None, 0.1])
async def test_superseded_plateau_cannot_hide_current_leaf_or_decide_floor(tmp_path, floor):
    conn = open_db(str(tmp_path / "rank.db"))
    embedder = RankedEmbedder()
    module = build_memory_module(
        conn, embedder=embedder, dim=4, clock=lambda: SEP, floor_margin=floor
    )
    gate = MemoryGate(module)
    try:
        for i in range(7):
            await gate.store(
                Memory(
                    id=f"revision-{i}",
                    user_id="owner",
                    content="needle" if i < 6 else "lower needle padded words",
                    source=MemorySource.USER_STATED,
                    author_id="person:owner",
                    effective_at=JAN if i < 6 else JUN,
                    revises_event_ids=[f"revision-{i - 1}"] if i else [],
                )
            )
        # Both real indexes rank six stale sources above the current leaf before eligibility.
        raw_vector = await module._vectors.search(
            user_id="owner", vector=[1.0, 0.0, 0.0, 0.0], top_k=6
        )
        raw_fts = module._fts.search("needle", user_id="owner", top_k=6)
        assert "revision-6" not in [hit.id for hit in raw_vector]
        assert "revision-6" not in raw_fts
        candidates = RevisionResolver(module._episodics, conn=conn, at=SEP).ranked_candidates(
            user_id="owner", project="personal"
        )
        vector = await module._vectors.search(
            user_id="owner", vector=[1.0, 0.0, 0.0, 0.0], top_k=1, candidates=candidates
        )
        fts = module._fts.search("needle", user_id="owner", top_k=1, candidates=candidates)
        assert [hit.id for hit in vector] == ["revision-6"]
        assert fts == ["revision-6"]
        calls = embedder.calls
        result = await gate.recall(MemoryQuery(user_id="owner", text="needle", top_k=3))
        assert [m.id for m in result.memories] == ["revision-6"]
        assert not result.abstained
        assert embedder.calls == calls + 1
        historical = await gate.recall(
            MemoryQuery(user_id="owner", text="needle", top_k=1, effective_at=JAN)
        )
        assert [m.id for m in historical.memories] == ["revision-5"]
    finally:
        conn.close()


async def test_future_quarantine_and_other_owner_cannot_crowd_out_scoped_forks(tmp_path):
    conn = open_db(str(tmp_path / "scope.db"))
    gate = MemoryGate(
        build_memory_module(conn, embedder=RankedEmbedder(), dim=4, clock=lambda: SEP)
    )
    future = datetime(2026, 12, 1, tzinfo=UTC)

    async def store(
        identity,
        *,
        parents=(),
        at=JAN,
        project="personal",
        owner="owner",
        status=MemoryStatus.STORED,
        content="needle",
    ):
        await gate.store(
            Memory(
                id=identity,
                user_id=owner,
                project=project,
                content=content,
                source=MemorySource.USER_STATED,
                author_id="person:" + owner,
                effective_at=at,
                revises_event_ids=list(parents),
                status=status,
            )
        )

    try:
        await store("root")
        for name in ("left", "right"):
            await store(name, parents=["root"], at=JUN, content="lower needle padded words")
        for i in range(6):
            await store(f"future-{i}", parents=["left"], at=future)
            await store(
                f"quarantine-{i}", parents=["right"], at=JUN, status=MemoryStatus.QUARANTINED
            )
            await store(f"other-owner-{i}", owner="someone-else")
        await store("another-project", project="work", content="lower needle padded words")
        scoped = await gate.recall(MemoryQuery(user_id="owner", text="needle", top_k=2))
        assert {m.id for m in scoped.memories} == {"left", "right"}
        assert all(m.revision_state == "conflicted" for m in scoped.memories)
        assert all(set(m.eligible_leaf_ids) == {"left", "right"} for m in scoped.memories)
        all_projects = await gate.recall(
            MemoryQuery(user_id="owner", text="needle", top_k=3, all_projects=True)
        )
        assert {m.id for m in all_projects.memories} == {"left", "right", "another-project"}
        before = await gate.recall(
            MemoryQuery(user_id="owner", text="needle", top_k=1, effective_at=JAN)
        )
        assert [m.id for m in before.memories] == ["root"]
    finally:
        conn.close()


async def test_empty_eligible_set_is_supported_by_real_vector_and_keyword_queries(tmp_path):
    conn = open_db(str(tmp_path / "future-only.db"))
    gate = MemoryGate(
        build_memory_module(conn, embedder=RankedEmbedder(), dim=4, clock=lambda: SEP)
    )
    try:
        await gate.store(
            Memory(
                id="future",
                user_id="owner",
                content="needle",
                effective_at=datetime(2026, 12, 1, tzinfo=UTC),
            )
        )
        result = await gate.recall(MemoryQuery(user_id="owner", text="needle", top_k=1))
        assert result.memories == []
        assert result.abstained and result.reason == "empty"
    finally:
        conn.close()


async def test_bound_integer_cutoff_preserves_microsecond_boundary_with_offset(tmp_path):
    conn = open_db(str(tmp_path / "precise-cutoff.db"))
    module = build_memory_module(conn, embedder=RankedEmbedder(), dim=4, clock=lambda: SEP)
    gate = MemoryGate(module)
    root_at = JAN + timedelta(microseconds=100)
    child_at = root_at + timedelta(microseconds=1)
    try:
        for identity, at, parents in [("root", root_at, []), ("child", child_at, ["root"])]:
            await gate.store(
                Memory(
                    id=identity,
                    user_id="owner",
                    content="needle",
                    source=MemorySource.USER_STATED,
                    author_id="person:owner",
                    effective_at=at,
                    revises_event_ids=parents,
                )
            )
        offset_cutoff = root_at.astimezone(timezone(timedelta(hours=3)))
        candidates = RevisionResolver(
            module._episodics, conn=conn, at=offset_cutoff
        ).ranked_candidates(user_id="owner", project="personal")
        assert isinstance(candidates.params[-1], int)
        before = await gate.recall(
            MemoryQuery(user_id="owner", text="needle", top_k=1, effective_at=offset_cutoff)
        )
        after = await gate.recall(
            MemoryQuery(
                user_id="owner",
                text="needle",
                top_k=1,
                effective_at=offset_cutoff + timedelta(microseconds=1),
            )
        )
        assert [memory.id for memory in before.memories] == ["root"]
        assert [memory.id for memory in after.memories] == ["child"]
    finally:
        conn.close()
