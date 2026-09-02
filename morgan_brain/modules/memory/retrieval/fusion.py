"""Reciprocal Rank Fusion — combine several ranked id-lists into one. This is Phase 1's single
rerank layer (CrossEncoder reranking is deferred per the degradation ladder)."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
