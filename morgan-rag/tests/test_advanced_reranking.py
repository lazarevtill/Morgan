#!/usr/bin/env python3
"""
Test script for the advanced local Jina AI reranking engine.

This script demonstrates the enhanced features:
- Local Jina AI model loading and caching
- Automatic model selection based on language detection
- Background reranking for popular queries with precomputation
- Quality metrics tracking and improvement measurement
- Resource monitoring and fallback mechanisms
- Precomputed results caching for sub-100ms response times
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "morgan"))

from morgan.jina.reranking import JinaRerankingService, SearchResult
import time


def create_sample_results() -> list:
    """Create sample search results for testing."""
    return [
        SearchResult(
            content="Python is a high-level programming language with dynamic semantics.",
            score=0.85,
            metadata={"category": "programming", "language": "en"},
            source="doc1.txt",
        ),
        SearchResult(
            content="Machine learning algorithms can be used for data analysis.",
            score=0.78,
            metadata={"category": "ai", "language": "en"},
            source="doc2.txt",
        ),
        SearchResult(
            content="Les algorithmes d'apprentissage automatique sont utilisés pour l'analyse de données.",
            score=0.72,
            metadata={"category": "ai", "language": "fr"},
            source="doc3.txt",
        ),
        SearchResult(
            content="Deep learning is a subset of machine learning methods.",
            score=0.69,
            metadata={"category": "ai", "language": "en"},
            source="doc4.txt",
        ),
    ]


def test_language_detection():
    """Test automatic language detection."""
    print("=== Testing Language Detection ===")

    service = JinaRerankingService(enable_background=False)

    # Test English detection
    english_text = "This is a simple English sentence with common words."
    lang = service.detect_language(english_text)
    print(f"English text detected as: {lang}")

    # Test non-English detection
    french_text = "Ceci est une phrase en français avec des mots communs."
    lang = service.detect_language(french_text)
    print(f"French text detected as: {lang}")

    # Test model selection
    results = create_sample_results()
    english_query = "What is machine learning?"
    model = service.select_reranker_model(english_query, results[:2])
    print(f"Selected model for English query: {model}")

    french_query = "Qu'est-ce que l'apprentissage automatique?"
    model = service.select_reranker_model(french_query, results[2:])
    print(f"Selected model for French query: {model}")


def test_reranking_with_metrics():
    """Test reranking with quality metrics tracking."""
    print("\n=== Testing Reranking with Metrics ===")

    # Disable resource monitoring for testing
    service = JinaRerankingService(
        enable_background=False, enable_resource_monitoring=False
    )

    # Try to preload models
    print("Preloading local models...")
    preload_results = service.preload_models()
    for model, success in preload_results.items():
        print(f"  {model}: {'✓' if success else '✗'}")

    loaded_models = service.get_loaded_models()
    if loaded_models:
        print(f"Loaded models: {loaded_models}")
    else:
        print("No models loaded, will use enhanced similarity fallback")
    results = create_sample_results()

    # Test reranking
    query = "machine learning algorithms"
    reranked_results, metrics = service.rerank_results(query, results)

    print(f"Original results: {len(results)}")
    print(f"Reranked results: {len(reranked_results)}")
    print(f"Model used: {metrics.model_used}")
    print(f"Language detected: {metrics.language_detected}")
    print(f"Processing time: {metrics.processing_time:.3f}s")
    print(f"Improvement score: {metrics.improvement_score:.3f}")

    # Show reranked order
    print("\nReranked results order:")
    for i, result in enumerate(reranked_results):
        print(f"{i+1}. Score: {result.score:.3f} - {result.content[:50]}...")


def test_background_processing():
    """Test background processing features."""
    print("\n=== Testing Background Processing ===")

    # Disable resource monitoring for testing
    service = JinaRerankingService(
        enable_background=True, enable_resource_monitoring=False
    )

    # Simulate popular queries to trigger background processing
    popular_queries = [
        "machine learning algorithms",
        "python programming language",
        "deep learning neural networks",
    ]

    results = create_sample_results()

    # Make multiple requests to simulate popularity
    for query in popular_queries:
        for _ in range(4):  # Make each query popular (4+ requests)
            reranked_results, metrics = service.rerank_results(query, results)
            time.sleep(0.1)  # Small delay

    # Check analytics
    analytics = service.get_reranking_analytics(days=1)
    print(f"Total requests: {analytics['total_requests']}")
    print(f"Average improvement: {analytics['avg_improvement']:.3f}")
    print(f"Popular queries: {len(analytics['popular_queries'])}")

    # Start background reranking
    task_id = service.start_background_reranking(
        collection_name="test_collection",
        query_patterns=popular_queries,
        rerank_schedule="daily",
    )
    print(f"Started background task: {task_id}")

    # Wait a bit and check status
    time.sleep(1)
    analytics = service.get_reranking_analytics(days=1)
    print(f"Background tasks status: {analytics['background_tasks_status']}")


def test_quality_metrics():
    """Test quality improvement tracking."""
    print("\n=== Testing Quality Metrics ===")

    # Disable resource monitoring for testing
    service = JinaRerankingService(
        enable_background=False, enable_resource_monitoring=False
    )
    results = create_sample_results()

    # Perform multiple reranking operations
    queries = [
        "machine learning",
        "python programming",
        "data analysis",
        "artificial intelligence",
        "deep learning",
    ]

    for query in queries:
        reranked_results, metrics = service.rerank_results(query, results)
        print(f"Query: '{query}' - Improvement: {metrics.improvement_score:.3f}")

    # Get quality report
    quality_report = service.get_quality_improvement_report()
    print(f"\nQuality Report:")
    print(f"Mean improvement: {quality_report['improvement_stats']['mean']:.3f}")
    print(f"Median improvement: {quality_report['improvement_stats']['median']:.3f}")
    print(f"95th percentile: {quality_report['improvement_stats']['p95']:.3f}")

    target_achievement = quality_report["target_achievement"]
    print(f"Target achievement rate: {target_achievement['achievement_rate']:.1%}")


def test_model_info():
    """Test model information retrieval."""
    print("\n=== Testing Model Information ===")

    service = JinaRerankingService(enable_background=False)

    models = ["jina-reranker-v3", "jina-reranker-v2-base-multilingual"]

    for model in models:
        info = service.get_model_info(model)
        print(f"\nModel: {model}")
        print(f"Language: {info['language']}")
        print(f"Description: {info['description']}")
        print(f"Use case: {info['use_case']}")
        print(f"Max query length: {info['max_query_length']}")


def test_local_models():
    """Test local model loading and functionality."""
    print("\n=== Testing Local Model Loading ===")

    # Disable resource monitoring for testing
    service = JinaRerankingService(
        enable_background=False, enable_resource_monitoring=False
    )

    # Test model preloading
    print("Testing model preloading...")
    preload_results = service.preload_models()

    for model_name, success in preload_results.items():
        status = "✓ Loaded" if success else "✗ Failed"
        print(f"  {model_name}: {status}")

    # Test model information
    print("\nModel information:")
    for model in ["jina-reranker-v3", "jina-reranker-v2-base-multilingual"]:
        info = service.get_model_info(model)
        print(f"  {model}:")
        print(f"    Language: {info['language']}")
        print(f"    Max query length: {info['max_query_length']}")

    # Test fallback behavior
    print("\nTesting fallback behavior...")
    results = create_sample_results()
    query = "test query for fallback"

    reranked_results, metrics = service.rerank_results(query, results)
    print(f"Reranking completed with model: {metrics.model_used}")
    print(f"Processing time: {metrics.processing_time:.3f}s")

    # Test resource monitoring
    print("\nTesting resource monitoring...")
    should_fallback = service._should_fallback_to_embedding_search()
    print(f"Should fallback to embedding search: {should_fallback}")


def main():
    """Run all tests."""
    print("Advanced Local Jina AI Reranking Engine Test")
    print("=" * 50)

    try:
        test_language_detection()
        test_reranking_with_metrics()
        test_background_processing()
        test_quality_metrics()
        test_model_info()
        test_local_models()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nNote: If models failed to load, install sentence-transformers:")
        print("  pip install sentence-transformers")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
