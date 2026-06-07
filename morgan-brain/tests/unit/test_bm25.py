from morgan_brain.modules.memory.retrieval.bm25 import Bm25Index


def test_ranks_documents_by_keyword_overlap():
    idx = Bm25Index()
    idx.add("d1", "the cat sat on the mat")
    idx.add("d2", "dogs run in the park")
    idx.add("d3", "a cat and a dog")
    ranked = idx.search("cat", top_k=3)
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] in {"d1", "d3"}
    assert "d2" not in ids[:1]


def test_empty_query_returns_nothing():
    idx = Bm25Index()
    idx.add("d1", "hello world")
    assert idx.search("", top_k=5) == []
