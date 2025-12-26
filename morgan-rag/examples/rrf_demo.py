#!/usr/bin/env python3
"""
Reciprocal Rank Fusion (RRF) Algorithm Demo

This demo showcases the enhanced RRF implementation with:
- RRF formula: score = Σ(1 / (k + rank)) where k=60
- Intelligent result deduplication using cosine similarity >95%
- Result ranking and merging system with strategy weighting
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morgan.search.multi_stage_search import MultiStageSearchEngine, SearchResult
from unittest.mock import Mock
import json


def create_mock_embedding_service():
    """Create a mock embedding service that returns diverse embeddings."""
    mock_service = Mock()

    # Define diverse embeddings for different content types
    embeddings_map = {
        "Docker containerization platform": [1.0, 0.0, 0.0, 0.0],
        "Install Docker on Ubuntu": [0.8, 0.6, 0.0, 0.0],
        "Kubernetes orchestration": [0.0, 1.0, 0.0, 0.0],
        "Docker Compose tutorial": [0.7, 0.0, 0.7, 0.0],
        "Container networking guide": [0.0, 0.0, 1.0, 0.0],
        "Docker security best practices": [0.5, 0.0, 0.0, 0.8],
    }

    def mock_encode(text, instruction="document"):
        # Find the best matching embedding based on content
        for key, embedding in embeddings_map.items():
            if key.lower() in text.lower():
                return embedding
        # Default embedding for unknown content
        return [0.1, 0.1, 0.1, 0.1]

    mock_service.encode = mock_encode
    return mock_service


def demo_rrf_fusion():
    """Demonstrate RRF fusion with multiple search strategies."""
    print("=== Reciprocal Rank Fusion (RRF) Demo ===\n")

    # Create search engine
    engine = MultiStageSearchEngine()
    engine.embedding_service = create_mock_embedding_service()

    # Create sample results from different search strategies
    semantic_results = [
        SearchResult(
            "Docker containerization platform", "docker-intro.md", 0.95, "knowledge", {}
        ),
        SearchResult(
            "Install Docker on Ubuntu", "install-guide.md", 0.88, "knowledge", {}
        ),
        SearchResult("Kubernetes orchestration", "k8s-guide.md", 0.82, "knowledge", {}),
    ]

    keyword_results = [
        SearchResult(
            "Install Docker on Ubuntu", "install-guide.md", 0.90, "knowledge", {}
        ),  # Duplicate
        SearchResult(
            "Docker Compose tutorial", "compose-guide.md", 0.85, "knowledge", {}
        ),
        SearchResult(
            "Container networking guide", "networking.md", 0.78, "knowledge", {}
        ),
    ]

    category_results = [
        SearchResult(
            "Docker security best practices", "security.md", 0.87, "knowledge", {}
        ),
        SearchResult(
            "Docker containerization platform", "docker-intro.md", 0.83, "knowledge", {}
        ),  # Duplicate
    ]

    print("Input Results by Strategy:")
    print(f"Semantic: {len(semantic_results)} results")
    for i, result in enumerate(semantic_results):
        print(f"  {i+1}. {result.content} (score: {result.score})")

    print(f"\nKeyword: {len(keyword_results)} results")
    for i, result in enumerate(keyword_results):
        print(f"  {i+1}. {result.content} (score: {result.score})")

    print(f"\nCategory: {len(category_results)} results")
    for i, result in enumerate(category_results):
        print(f"  {i+1}. {result.content} (score: {result.score})")

    # Apply RRF fusion
    print(f"\n=== Applying RRF Fusion (k={engine.rrf_k}) ===")

    strategy_results = [semantic_results, keyword_results, category_results]
    fused_results = engine._fusion_results(strategy_results, 10)

    print(f"\nFused Results: {len(fused_results)} unique results")
    print("Ranked by RRF Score:")

    for i, result in enumerate(fused_results):
        metadata = result.metadata
        print(f"\n{i+1}. {result.content}")
        print(f"   Source: {result.source}")
        print(f"   Original Score: {result.score:.3f}")
        print(f"   RRF Score: {result.rrf_score:.3f}")
        print(f"   Strategy: {result.strategy}")
        print(
            f"   Found in {metadata['strategy_count']} strategies: {metadata['strategies_found_in']}"
        )
        print(f"   Strategy Boost: {metadata['strategy_boost']:.2f}x")
        print(f"   Raw RRF Score: {metadata['rrf_raw_score']:.3f}")

    # Demonstrate deduplication
    print(f"\n=== Deduplication Analysis ===")
    print(f"Similarity Threshold: {engine.similarity_threshold} (95%)")

    # Show which results would be considered duplicates
    total_input_results = sum(len(results) for results in strategy_results)
    unique_results = len(fused_results)
    duplicates_removed = total_input_results - unique_results

    print(f"Total Input Results: {total_input_results}")
    print(f"Unique Results After Fusion: {unique_results}")
    print(f"Duplicates Removed: {duplicates_removed}")

    # Calculate RRF scores manually for verification
    print(f"\n=== RRF Score Calculation Verification ===")
    print(
        "Manual calculation for 'Install Docker on Ubuntu' (appears in semantic #2 and keyword #1):"
    )

    # Calculate strategy weights
    weights = engine._calculate_strategy_weights(strategy_results)

    k = engine.rrf_k
    semantic_rank = 1  # 0-indexed rank in semantic results
    keyword_rank = 0  # 0-indexed rank in keyword results

    semantic_weight = weights[0]
    keyword_weight = weights[1]

    manual_rrf = (semantic_weight * (1.0 / (k + semantic_rank))) + (
        keyword_weight * (1.0 / (k + keyword_rank))
    )
    strategy_boost = 1.0 + (0.1 * (2 - 1))  # 2 strategies found it
    boosted_rrf = manual_rrf * strategy_boost

    print(
        f"  Strategy weights: Semantic={semantic_weight:.3f}, Keyword={keyword_weight:.3f}"
    )
    print(
        f"  Semantic contribution: {semantic_weight:.3f} * 1/({k} + {semantic_rank}) = {semantic_weight * (1.0/(k + semantic_rank)):.6f}"
    )
    print(
        f"  Keyword contribution: {keyword_weight:.3f} * 1/({k} + {keyword_rank}) = {keyword_weight * (1.0/(k + keyword_rank)):.6f}"
    )
    print(f"  Raw RRF score: {manual_rrf:.6f}")
    print(f"  Strategy boost (2 strategies): {strategy_boost:.2f}x")
    print(f"  Boosted RRF score: {boosted_rrf:.6f}")

    # Find the actual result to compare
    install_result = next(
        (r for r in fused_results if "Install Docker" in r.content), None
    )
    if install_result:
        print(f"  Actual raw RRF score: {install_result.metadata['rrf_raw_score']:.6f}")
        print(
            f"  Match: {'✓' if abs(install_result.metadata['rrf_raw_score'] - boosted_rrf) < 0.001 else '✗'}"
        )

        # Show final score calculation
        avg_orig_score = install_result.metadata["avg_original_score"]
        final_score = (0.7 * boosted_rrf) + (0.3 * avg_orig_score)
        print(f"  Average original score: {avg_orig_score:.3f}")
        print(
            f"  Final score: 0.7 * {boosted_rrf:.6f} + 0.3 * {avg_orig_score:.3f} = {final_score:.6f}"
        )
        print(f"  Actual final score: {install_result.rrf_score:.6f}")
        print(
            f"  Final match: {'✓' if abs(install_result.rrf_score - final_score) < 0.001 else '✗'}"
        )


if __name__ == "__main__":
    demo_rrf_fusion()
