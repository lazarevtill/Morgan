from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion


def test_item_ranked_high_across_lists_wins():
    vector = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    entity = ["a", "e"]
    fused = reciprocal_rank_fusion([vector, bm25, entity])
    assert fused[0] == "a"


def test_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]
