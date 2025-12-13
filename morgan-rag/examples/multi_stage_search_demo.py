#!/usr/bin/env python3
"""
Multi-Stage Search Engine Demo

Demonstrates the capabilities of Morgan's advanced multi-stage search engine
with hierarchical filtering and result fusion.
"""

import sys
import os
from pathlib import Path

# Add morgan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.search import (
    MultiStageSearchEngine,
    SearchStrategy,
    SearchResult,
    SearchResults
)


def demo_search_strategies():
    """Demonstrate different search strategies."""
    print("🔍 Multi-Stage Search Engine Demo")
    print("=" * 50)
    
    # Initialize search engine
    try:
        engine = MultiStageSearchEngine()
        print("✅ Search engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return
    
    # Demo queries
    queries = [
        "How to deploy Docker containers?",
        "Python function definition syntax",
        "API authentication methods",
        "Database connection error troubleshooting"
    ]
    
    # Demo different strategy combinations
    strategy_combinations = [
        {
            "name": "Semantic Only",
            "strategies": [SearchStrategy.SEMANTIC],
            "description": "Pure semantic similarity search"
        },
        {
            "name": "Multi-Strategy",
            "strategies": [
                SearchStrategy.SEMANTIC,
                SearchStrategy.KEYWORD,
                SearchStrategy.CATEGORY
            ],
            "description": "Combined semantic, keyword, and category search"
        },
        {
            "name": "Full Pipeline",
            "strategies": [
                SearchStrategy.SEMANTIC,
                SearchStrategy.KEYWORD,
                SearchStrategy.CATEGORY,
                SearchStrategy.TEMPORAL,
                SearchStrategy.MEMORY
            ],
            "description": "All strategies with result fusion"
        }
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 40)
        
        for combo in strategy_combinations:
            print(f"\n🔧 Strategy: {combo['name']}")
            print(f"   {combo['description']}")
            
            try:
                # Execute search
                results = engine.search(
                    query=query,
                    max_results=3,
                    strategies=combo['strategies'],
                    use_hierarchical=True
                )
                
                # Display results
                print(f"   📊 Results: {len(results)} found in {results.search_time:.3f}s")
                
                if results.total_candidates > 0:
                    reduction = results.get_reduction_ratio()
                    print(f"   🎯 Candidate reduction: {reduction:.1%}")
                
                if results.fusion_applied:
                    print("   🔀 Result fusion applied")
                
                if results.deduplication_applied:
                    print("   🧹 Deduplication applied")
                
                # Show performance summary
                perf = results.get_performance_summary()
                if perf['results_per_second'] > 0:
                    print(f"   ⚡ Performance: {perf['results_per_second']:.1f} results/sec")
                
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
    
    # Show overall performance stats
    print(f"\n📈 Engine Performance Statistics")
    print("-" * 40)
    stats = engine.get_performance_stats()
    print(f"Total searches: {stats['total_searches']}")
    print(f"Average search time: {stats['average_search_time']:.3f}s")
    print(f"Average candidates: {stats['average_candidates']:.0f}")
    print(f"Average reduction: {stats['average_reduction']:.1%}")


def demo_search_result_features():
    """Demonstrate SearchResult and SearchResults features."""
    print(f"\n🧪 Search Result Features Demo")
    print("=" * 50)
    
    # Create sample results
    sample_results = [
        SearchResult(
            content="Docker is a containerization platform that allows you to package applications with their dependencies.",
            source="docker-guide.md",
            score=0.95,
            result_type="knowledge",
            metadata={"category": "documentation", "author": "Docker Team"},
            strategy="semantic",
            rrf_score=0.87,
            hierarchical_scores={"coarse": 0.9, "medium": 0.92, "fine": 0.95}
        ),
        SearchResult(
            content="To install Docker, download it from docker.com or use your package manager.",
            source="installation.md",
            score=0.88,
            result_type="knowledge",
            metadata={"category": "documentation", "difficulty": "beginner"},
            strategy="keyword",
            rrf_score=0.82
        ),
        SearchResult(
            content="Q: How do I start a Docker container?\nA: Use the 'docker run' command with your image name.",
            source="Conversation (2025-01-15)",
            score=0.85,
            result_type="memory",
            metadata={"conversation_id": "conv123", "feedback_rating": 5},
            strategy="memory"
        )
    ]
    
    # Create SearchResults object
    search_results = SearchResults(
        results=sample_results,
        total_candidates=1000,
        filtered_candidates=3,
        strategies_used=["semantic", "keyword", "memory"],
        search_time=0.245,
        fusion_applied=True,
        deduplication_applied=True
    )
    
    print(f"📋 Search Results Summary:")
    print(f"   Total results: {len(search_results)}")
    print(f"   Candidate reduction: {search_results.get_reduction_ratio():.1%}")
    print(f"   Search time: {search_results.search_time:.3f}s")
    print(f"   Strategies used: {', '.join(search_results.strategies_used)}")
    
    print(f"\n📄 Individual Results:")
    for i, result in enumerate(search_results, 1):
        print(f"\n   Result {i}:")
        print(f"   📝 Content: {result.summary(80)}")
        print(f"   📁 Source: {result.source}")
        print(f"   ⭐ Score: {result.score:.3f}")
        print(f"   🏷️  Type: {result.result_type}")
        print(f"   🔧 Strategy: {result.strategy}")
        
        if result.rrf_score:
            print(f"   🔀 RRF Score: {result.rrf_score:.3f}")
        
        if result.hierarchical_scores:
            print(f"   📊 Hierarchical: {result.hierarchical_scores}")
        
        best_score = result.get_best_score()
        print(f"   🎯 Best Score: {best_score:.3f}")
    
    # Demonstrate filtering by strategy
    print(f"\n🔍 Results by Strategy:")
    for strategy in search_results.strategies_used:
        strategy_results = search_results.get_by_strategy(strategy)
        print(f"   {strategy}: {len(strategy_results)} results")
    
    # Show performance summary
    print(f"\n📊 Performance Summary:")
    perf = search_results.get_performance_summary()
    for key, value in perf.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")


def demo_search_utilities():
    """Demonstrate search utility functions."""
    print(f"\n🛠️  Search Utilities Demo")
    print("=" * 50)
    
    engine = MultiStageSearchEngine()
    
    # Test keyword extraction
    test_queries = [
        "How to install Docker on Ubuntu server?",
        "Python function definition with parameters",
        "API authentication using JWT tokens",
        "Database connection timeout error fix"
    ]
    
    print("🔤 Keyword Extraction:")
    for query in test_queries:
        keywords = engine._extract_keywords(query)
        print(f"   '{query}'")
        print(f"   → Keywords: {', '.join(keywords)}")
    
    # Test category detection
    print(f"\n🏷️  Category Detection:")
    category_queries = [
        ("def calculate_sum(a, b): return a + b", "code"),
        ("How to deploy Docker containers guide", "documentation"),
        ("GET /api/users endpoint returns user list", "api"),
        ("Docker container failed to start error", "troubleshooting"),
        ("Configure environment variables in .env file", "configuration")
    ]
    
    for query, expected in category_queries:
        detected = engine._detect_query_category(query)
        status = "✅" if detected == expected else "❌"
        print(f"   {status} '{query[:50]}...'")
        print(f"      Expected: {expected}, Detected: {detected}")
    
    # Test similarity calculation
    print(f"\n📐 Similarity Calculation:")
    test_vectors = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], "Identical vectors"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], "Orthogonal vectors"),
        ([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], "Partially similar vectors"),
        ([0.8, 0.6, 0.0], [0.6, 0.8, 0.0], "Similar vectors")
    ]
    
    for vec1, vec2, description in test_vectors:
        similarity = engine._calculate_similarity(vec1, vec2)
        print(f"   {description}: {similarity:.3f}")


if __name__ == "__main__":
    try:
        demo_search_strategies()
        demo_search_result_features()
        demo_search_utilities()
        
        print(f"\n🎉 Multi-Stage Search Engine Demo Completed!")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()