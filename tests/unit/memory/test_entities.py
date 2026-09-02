from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.stores.db import open_db


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
