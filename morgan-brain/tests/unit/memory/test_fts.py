from morgan_brain.modules.memory.retrieval.fts import FtsIndex, to_match_query
from morgan_brain.modules.memory.stores.db import open_db


def _idx(tmp_path):
    return FtsIndex(open_db(str(tmp_path / "m.db")))


def test_finds_english_term(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "the harbor mirror was misconfigured", user_id="u")
    assert idx.search("harbor", user_id="u", top_k=5) == ["a"]


def test_finds_cyrillic_term(tmp_path):
    """The old [a-z0-9]+ tokenizer dropped Cyrillic entirely."""
    idx = _idx(tmp_path)
    idx.add("a", "реестр Harbor был настроен неверно", user_id="u")
    assert idx.search("реестр", user_id="u", top_k=5) == ["a"]


def test_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u1")
    idx.add("b", "harbor", user_id="u2")
    assert idx.search("harbor", user_id="u1", top_k=5) == ["a"]


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    FtsIndex(open_db(path)).add("a", "harbor mirror", user_id="u")
    assert FtsIndex(open_db(path)).search("harbor", user_id="u", top_k=5) == ["a"]


def test_delete_removes_the_row(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u")
    idx.delete(["a"])
    assert idx.search("harbor", user_id="u", top_k=5) == []


def test_raw_punctuation_does_not_raise():
    """Raw user text is not a valid MATCH expression; it must be tokenised and quoted."""
    assert to_match_query('what about ACME-14802 "quoted" AND?') != ""


def test_query_with_no_indexable_tokens_is_empty(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u")
    assert idx.search("!!! ???", user_id="u", top_k=5) == []
