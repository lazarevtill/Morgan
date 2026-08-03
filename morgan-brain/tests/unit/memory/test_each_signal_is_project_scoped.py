"""Each retrieval signal filters by project on its own.

Project scoping was covered only in aggregate: removing the FTS filter, the entity filter, or
the post-fusion defense-in-depth filter each left the full suite green, because the remaining
layers still hid the leak. Only removing two or more made anything fail. That means any single
layer could be broken by a refactor without CI noticing.

These tests address each index directly -- no fusion, nothing else to mask a leak. A project
leak is the failure that makes Morgan unsafe to point at company repositories, so every layer
gets its own assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex, VectorRecord


@pytest.fixture
def conn(tmp_path: Path) -> object:
    return open_db(str(tmp_path / "m.db"))


def test_fts_search_does_not_cross_projects(conn: object) -> None:
    idx = FtsIndex(conn)  # type: ignore[arg-type]
    idx.add("a", "the Harbor mirror blocked the deploy", user_id="u", project="plata")
    idx.add("b", "the Harbor mirror blocked the deploy", user_id="u", project="personal")

    assert idx.search("harbor", user_id="u", top_k=5, project="plata") == ["a"]
    assert idx.search("harbor", user_id="u", top_k=5, project="personal") == ["b"]
    assert idx.search("harbor", user_id="u", top_k=5, project="empty") == []
    # None is the explicit cross-project escape hatch, never the default.
    assert sorted(idx.search("harbor", user_id="u", top_k=5, project=None)) == ["a", "b"]


def test_entity_search_does_not_cross_projects(conn: object) -> None:
    idx = EntityIndex(conn)  # type: ignore[arg-type]
    idx.add("a", ["harbor"], user_id="u", project="plata")
    idx.add("b", ["harbor"], user_id="u", project="personal")

    assert idx.search(["harbor"], user_id="u", top_k=5, project="plata") == ["a"]
    assert idx.search(["harbor"], user_id="u", top_k=5, project="personal") == ["b"]
    assert idx.search(["harbor"], user_id="u", top_k=5, project="empty") == []
    assert sorted(idx.search(["harbor"], user_id="u", top_k=5, project=None)) == ["a", "b"]


async def test_vector_search_does_not_cross_projects() -> None:
    idx = InMemoryVectorIndex()
    vector = [1.0, 0.0, 0.0]
    await idx.upsert(VectorRecord(id="a", user_id="u", project="plata", vector=vector))
    await idx.upsert(VectorRecord(id="b", user_id="u", project="personal", vector=vector))

    plata = await idx.search(user_id="u", vector=vector, top_k=5, project="plata")
    personal = await idx.search(user_id="u", vector=vector, top_k=5, project="personal")
    both = await idx.search(user_id="u", vector=vector, top_k=5, project=None)

    assert [h.id for h in plata] == ["a"]
    assert [h.id for h in personal] == ["b"]
    assert sorted(h.id for h in both) == ["a", "b"]


def test_fts_project_scoping_survives_a_user_collision(conn: object) -> None:
    """Both filters must apply: same project name under two owners must not merge."""
    idx = FtsIndex(conn)  # type: ignore[arg-type]
    idx.add("a", "harbor", user_id="u1", project="plata")
    idx.add("b", "harbor", user_id="u2", project="plata")

    assert idx.search("harbor", user_id="u1", top_k=5, project="plata") == ["a"]
    assert idx.search("harbor", user_id="u2", top_k=5, project="plata") == ["b"]
