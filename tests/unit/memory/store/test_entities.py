from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.entities import EntityIndex


def _idx(tmp_path):
    return EntityIndex(open_db(str(tmp_path / "m.db")))


def test_matches_on_entity_name(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor"], user_id="u")
    assert idx.search({"harbor"}, user_id="u", top_k=5) == ["a"]


def test_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor"], user_id="u1")
    idx.add("b", ["Harbor"], user_id="u2")
    assert idx.search({"harbor"}, user_id="u1", top_k=5) == ["a"]


def test_ordering_is_deterministic_by_match_count(tmp_path):
    idx = _idx(tmp_path)
    idx.add("b", ["Harbor"], user_id="u")
    idx.add("a", ["Harbor", "Qdrant"], user_id="u")
    assert idx.search({"harbor", "qdrant"}, user_id="u", top_k=5) == ["a", "b"]


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    EntityIndex(open_db(path)).add("a", ["Harbor"], user_id="u")
    assert EntityIndex(open_db(path)).search({"harbor"}, user_id="u", top_k=5) == ["a"]


def test_delete_removes_all_rows_for_the_memory(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor", "Qdrant"], user_id="u")
    idx.delete(["a"])
    assert idx.search({"harbor"}, user_id="u", top_k=5) == []


async def test_eligible_candidates_filter_before_limit_preserves_scope_and_lifecycle():
    from datetime import UTC, datetime, timedelta

    from morgan_brain.composition import build_memory_module
    from morgan_brain.memory.embedder import FakeEmbedder
    from morgan_brain.memory.gate import MemoryGate
    from morgan_brain.memory.revisions import RevisionResolver
    from morgan_brain.memory.store.episodic import EpisodicStore
    from morgan_brain.models import Entity, Memory, MemoryStatus, OriginKind

    now = datetime(2026, 10, 1, tzinfo=UTC)
    conn = open_db(":memory:")
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: now)
    )
    try:
        controls = [
            (
                "a-procedural",
                "owner",
                "personal",
                now,
                OriginKind.ASK_CONFLICT_GUARD,
                MemoryStatus.STORED,
            ),
            (
                "b-future",
                "owner",
                "personal",
                now + timedelta(days=1),
                OriginKind.REMEMBER,
                MemoryStatus.STORED,
            ),
            (
                "c-quarantined",
                "owner",
                "personal",
                now,
                OriginKind.REMEMBER,
                MemoryStatus.QUARANTINED,
            ),
            (
                "d-foreign-owner",
                "foreign",
                "personal",
                now,
                OriginKind.REMEMBER,
                MemoryStatus.STORED,
            ),
            ("e-foreign-project", "owner", "other", now, OriginKind.REMEMBER, MemoryStatus.STORED),
            ("z-active", "owner", "personal", now, OriginKind.REMEMBER, MemoryStatus.STORED),
        ]
        for identity, owner, project, effective, origin, status in controls:
            await gate.store(
                Memory(
                    id=identity,
                    user_id=owner,
                    project=project,
                    content="Harbor",
                    entities=[Entity(name="Harbor")],
                    source="user_stated",
                    author_id="person",
                    created_at=effective,
                    origin_kind=origin,
                    status=status,
                )
            )
        resolver = RevisionResolver(EpisodicStore(conn), conn=conn, at=now)
        index = EntityIndex(conn)
        assert index.search({"harbor"}, user_id="owner", project="personal", top_k=1) == [
            "a-procedural"
        ]
        assert index.search(
            {"harbor"},
            user_id="owner",
            project="personal",
            top_k=1,
            candidates=resolver.ranked_candidates(user_id="owner", project="personal"),
        ) == ["z-active"]
        assert index.search(
            {"harbor"},
            user_id="owner",
            project=None,
            top_k=2,
            candidates=resolver.ranked_candidates(user_id="owner", project=None),
        ) == ["e-foreign-project", "z-active"]
    finally:
        conn.close()
