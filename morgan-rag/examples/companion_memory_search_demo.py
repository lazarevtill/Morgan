#!/usr/bin/env python3
"""
Companion Memory Search Integration Demo

Demonstrates the enhanced memory search capabilities with emotional context
and companion relationship features.

This showcases the implementation of task 6.2:
- Conversation history search with emotional context
- Memory-based personalization for responses
- Relationship-aware memory retrieval and ranking
"""

import sys
import os
from datetime import datetime, timedelta, timezone

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morgan.search.companion_memory_search import get_companion_memory_search_engine
from morgan.core.search import SmartSearch
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def demo_companion_memory_search():
    """Demonstrate companion-aware memory search capabilities."""
    print("🧠 Companion Memory Search Integration Demo")
    print("=" * 50)

    try:
        # Initialize engines
        companion_engine = get_companion_memory_search_engine()
        smart_search = SmartSearch()

        print("\n1. Testing Emotional Context Search")
        print("-" * 30)

        # Test emotional context search
        emotional_query = "I'm feeling frustrated with Docker deployment"
        print(f"Query: '{emotional_query}'")

        emotional_results = companion_engine.search_with_emotional_context(
            query=emotional_query,
            user_id="demo_user",
            max_results=3,
            include_emotional_moments=True,
        )

        print(f"Found {len(emotional_results)} emotional context results:")
        for i, result in enumerate(emotional_results, 1):
            print(f"  {i}. Score: {result.score:.3f} | Type: {result.memory_type}")
            print(
                f"     Emotional: {result.emotional_context.get('has_emotional_content', False)}"
            )
            print(f"     Relationship: {result.relationship_significance:.2f}")
            print(f"     Summary: {result.get_summary(100)}")
            print()

        print("\n2. Testing Relationship Memory Search")
        print("-" * 30)

        # Test relationship memory search
        relationship_results = companion_engine.get_relationship_memories(
            user_id="demo_user", max_results=3
        )

        print(f"Found {len(relationship_results)} relationship memories:")
        for i, result in enumerate(relationship_results, 1):
            print(
                f"  {i}. Score: {result.score:.3f} | Significance: {result.relationship_significance:.2f}"
            )
            print(f"     Factors: {', '.join(result.personalization_factors)}")
            print(f"     Summary: {result.get_summary(100)}")
            print()

        print("\n3. Testing Similar Conversations")
        print("-" * 30)

        # Test similar conversation search
        similar_query = "How do I configure Docker networking?"
        print(f"Query: '{similar_query}'")

        similar_results = companion_engine.search_similar_conversations(
            current_query=similar_query, user_id="demo_user", max_results=3
        )

        print(f"Found {len(similar_results)} similar conversations:")
        for i, result in enumerate(similar_results, 1):
            print(f"  {i}. Score: {result.score:.3f} | Type: {result.memory_type}")
            print(f"     Engagement: {result.user_engagement_score:.2f}")
            print(f"     Summary: {result.get_summary(100)}")
            print()

        print("\n4. Testing Personalized Memories")
        print("-" * 30)

        # Test personalized memory retrieval
        personalized_results = companion_engine.get_personalized_memories(
            user_id="demo_user", max_results=5, days_back=30
        )

        print(f"Found {len(personalized_results)} personalized memories:")
        for i, result in enumerate(personalized_results, 1):
            print(f"  {i}. Score: {result.score:.3f} | Type: {result.memory_type}")
            print(f"     Age: {(datetime.now(timezone.utc) - result.timestamp).days} days")
            print(f"     Factors: {', '.join(result.personalization_factors)}")
            print(f"     Summary: {result.get_summary(80)}")
            print()

        print("\n5. Testing Enhanced Core Search Integration")
        print("-" * 30)

        # Test enhanced core search with companion features
        core_query = "Docker troubleshooting tips"
        print(f"Query: '{core_query}'")

        # Test conversation search with companion features
        enhanced_results = smart_search.search_conversations(
            query=core_query,
            max_results=3,
            user_id="demo_user",
            include_emotional_context=True,
        )

        print(f"Found {len(enhanced_results)} enhanced conversation results:")
        for i, result in enumerate(enhanced_results, 1):
            is_enhanced = result.metadata.get("companion_enhanced", False)
            print(f"  {i}. Score: {result.score:.3f} | Enhanced: {is_enhanced}")
            if is_enhanced:
                print(
                    f"     Memory Type: {result.metadata.get('memory_type', 'unknown')}"
                )
                print(
                    f"     Relationship: {result.metadata.get('relationship_significance', 0.0):.2f}"
                )
            print(f"     Summary: {result.summary(100)}")
            print()

        # Test relationship memory search through core interface
        print("\n6. Testing Relationship Memory Search (Core Interface)")
        print("-" * 30)

        relationship_core_results = smart_search.search_relationship_memories(
            user_id="demo_user", max_results=3
        )

        print(f"Found {len(relationship_core_results)} relationship memories (core):")
        for i, result in enumerate(relationship_core_results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Factors: {result.metadata.get('personalization_factors', [])}")
            print(f"     Summary: {result.summary(100)}")
            print()

        print("\n7. Testing Performance and Integration")
        print("-" * 30)

        # Test performance with multiple searches
        start_time = datetime.now(timezone.utc)

        test_queries = [
            "Python programming help",
            "API development best practices",
            "Database optimization tips",
        ]

        total_results = 0
        for query in test_queries:
            results = companion_engine.search_with_emotional_context(
                query=query, user_id="demo_user", max_results=5
            )
            total_results += len(results)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        print(f"Processed {len(test_queries)} queries in {duration:.3f}s")
        print(f"Total results: {total_results}")
        print(f"Average time per query: {duration/len(test_queries):.3f}s")

        print("\n" + "=" * 50)
        print("✅ Companion Memory Search Integration Demo Completed!")
        print("\nKey Features Demonstrated:")
        print("- Emotional context-aware memory search")
        print("- Relationship significance weighting")
        print("- Memory-based personalization")
        print("- Similar conversation detection")
        print("- Enhanced core search integration")
        print("- Performance optimization")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Companion memory search demo failed: {e}", exc_info=True)
        return False

    return True


def demo_memory_personalization():
    """Demonstrate memory-based personalization features."""
    print("\n🎯 Memory-Based Personalization Demo")
    print("=" * 40)

    try:
        companion_engine = get_companion_memory_search_engine()

        # Simulate different user scenarios
        scenarios = [
            {
                "user_id": "technical_user",
                "query": "deployment strategies",
                "description": "Technical user interested in deployment",
            },
            {
                "user_id": "beginner_user",
                "query": "getting started guide",
                "description": "Beginner user needing guidance",
            },
            {
                "user_id": "experienced_user",
                "query": "advanced configuration",
                "description": "Experienced user seeking advanced topics",
            },
        ]

        for scenario in scenarios:
            print(f"\nScenario: {scenario['description']}")
            print(f"User ID: {scenario['user_id']}")
            print(f"Query: '{scenario['query']}'")

            results = companion_engine.search_with_emotional_context(
                query=scenario["query"],
                user_id=scenario["user_id"],
                max_results=3,
                include_emotional_moments=True,
            )

            print(f"Personalized results ({len(results)}):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(
                    f"     Personalization: {', '.join(result.personalization_factors)}"
                )
                print(f"     Engagement: {result.user_engagement_score:.2f}")
                print()

        print("✅ Memory personalization demo completed!")

    except Exception as e:
        print(f"❌ Personalization demo failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Starting Companion Memory Search Integration Demo...")

    # Run main demo
    success = demo_companion_memory_search()

    if success:
        # Run personalization demo
        demo_memory_personalization()

        print("\n🎉 All demos completed successfully!")
        print("\nThe companion memory search integration provides:")
        print("- Enhanced conversation history search with emotional awareness")
        print("- Relationship-based memory ranking and retrieval")
        print("- Personalized search results based on user engagement")
        print("- Seamless integration with existing search infrastructure")
    else:
        print("\n⚠️  Demo encountered issues. Check logs for details.")
        sys.exit(1)
