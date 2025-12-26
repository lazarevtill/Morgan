#!/usr/bin/env python3
"""
Enhanced Search Integration Demo

Demonstrates the integration of multi-stage search with hierarchical embeddings,
emotional context, and companion-aware result filtering and personalization.

This demo showcases task 7.2: Implement enhanced search integration.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morgan.core.search import SmartSearch
from morgan.search.multi_stage_search import get_multi_stage_search_engine
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_emotional_context(
    emotion: str = "frustration", intensity: float = 0.8
) -> Dict[str, Any]:
    """Create sample emotional context for testing."""
    return {
        "primary_emotion": emotion,
        "intensity": intensity,
        "confidence": 0.9,
        "secondary_emotions": [],
        "emotional_indicators": ["frustrated", "stuck", "problem"],
        "timestamp": datetime.utcnow().isoformat(),
    }


def demo_enhanced_search_integration():
    """Demonstrate enhanced search integration with emotional context and companion features."""
    print("🔍 Enhanced Search Integration Demo")
    print("=" * 50)

    # Initialize search engines
    smart_search = SmartSearch()
    multi_stage_engine = get_multi_stage_search_engine()

    # Test queries with different emotional contexts
    test_scenarios = [
        {
            "query": "How to deploy Docker containers?",
            "emotional_context": create_sample_emotional_context("frustration", 0.8),
            "user_id": "user123",
            "description": "Frustrated user needing Docker help",
        },
        {
            "query": "Python API best practices",
            "emotional_context": create_sample_emotional_context("curiosity", 0.6),
            "user_id": "user123",
            "description": "Curious user exploring API design",
        },
        {
            "query": "Database connection troubleshooting",
            "emotional_context": create_sample_emotional_context("anxiety", 0.9),
            "user_id": "user456",
            "description": "Anxious user with database issues",
        },
        {
            "query": "Machine learning tutorial",
            "emotional_context": None,
            "user_id": None,
            "description": "Standard search without emotional context",
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print("-" * 40)
        print(f"Query: '{scenario['query']}'")

        if scenario["emotional_context"]:
            emotion = scenario["emotional_context"]["primary_emotion"]
            intensity = scenario["emotional_context"]["intensity"]
            print(f"Emotional Context: {emotion} (intensity: {intensity})")

        if scenario["user_id"]:
            print(f"User ID: {scenario['user_id']}")

        try:
            # Test enhanced search through SmartSearch
            print("\n📊 Enhanced SmartSearch Results:")
            results = smart_search.find_relevant_info(
                query=scenario["query"],
                max_results=3,
                emotional_context=scenario["emotional_context"],
                user_id=scenario["user_id"],
                use_enhanced_search=True,
            )

            if results:
                for j, result in enumerate(results, 1):
                    print(
                        f"  {j}. Score: {result.score:.3f} | Type: {result.result_type}"
                    )
                    print(f"     Source: {result.source}")

                    # Show enhancement information
                    metadata = result.metadata
                    if metadata.get("enhanced_multi_stage_search"):
                        print(f"     Enhanced: Multi-stage search used")
                    if metadata.get("emotional_enhanced"):
                        print(f"     Enhanced: Emotional context applied")
                    if metadata.get("companion_enhanced"):
                        print(f"     Enhanced: Companion personalization applied")

                    enhancement_factors = metadata.get("enhancement_factors", [])
                    if enhancement_factors:
                        print(f"     Enhancements: {', '.join(enhancement_factors)}")

                    print(f"     Content: {result.summary(100)}")
                    print()
            else:
                print("  No results found")

            # Test direct multi-stage search
            print("🎯 Direct Multi-Stage Search Results:")
            search_results = multi_stage_engine.search(
                query=scenario["query"],
                max_results=3,
                strategies=(
                    ["semantic", "memory"]
                    if scenario["emotional_context"]
                    else ["semantic"]
                ),
                use_hierarchical=True,
                emotional_context=scenario["emotional_context"],
                user_id=scenario["user_id"],
            )

            print(f"  Search Time: {search_results.search_time:.3f}s")
            print(f"  Strategies Used: {', '.join(search_results.strategies_used)}")
            print(f"  Fusion Applied: {search_results.fusion_applied}")
            print(f"  Candidate Reduction: {search_results.get_reduction_ratio():.1%}")

            if search_results.results:
                for j, result in enumerate(search_results.results, 1):
                    print(
                        f"  {j}. Score: {result.score:.3f} | Strategy: {result.strategy}"
                    )
                    print(
                        f"     RRF Score: {result.rrf_score:.3f}"
                        if result.rrf_score
                        else "     No RRF Score"
                    )

                    # Show hierarchical scores if available
                    if (
                        hasattr(result, "hierarchical_scores")
                        and result.hierarchical_scores
                    ):
                        hierarchical_info = ", ".join(
                            [
                                f"{scale}: {score:.3f}"
                                for scale, score in result.hierarchical_scores.items()
                            ]
                        )
                        print(f"     Hierarchical: {hierarchical_info}")

                    print(f"     Content: {result.summary(80)}")
                    print()
            else:
                print("  No results found")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            logger.error(f"Demo scenario {i} failed: {e}")

    print("\n✅ Enhanced Search Integration Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("- Multi-stage search with hierarchical embeddings")
    print("- Emotional context integration in result ranking")
    print("- Companion-aware result filtering and personalization")
    print("- Reciprocal Rank Fusion for result merging")
    print("- Performance optimization with candidate reduction")


def demo_hierarchical_search_features():
    """Demonstrate specific hierarchical search features."""
    print("\n🏗️ Hierarchical Search Features Demo")
    print("=" * 50)

    try:
        from morgan.vector_db.client import VectorDBClient
        from morgan.embeddings.service import get_embedding_service

        vector_db = VectorDBClient()
        embedding_service = get_embedding_service()

        # Test hierarchical search with emotional context
        query = "Docker deployment best practices"
        emotional_context = create_sample_emotional_context("determination", 0.7)
        user_id = "demo_user"

        print(f"Query: '{query}'")
        print(
            f"Emotional Context: {emotional_context['primary_emotion']} ({emotional_context['intensity']})"
        )
        print(f"User ID: {user_id}")

        # Generate query embedding
        query_embedding = embedding_service.encode(query, instruction="query")

        # Test enhanced hierarchical search
        print("\n🔍 Enhanced Hierarchical Search:")
        hierarchical_results = vector_db.search_hierarchical(
            collection_name="morgan_knowledge",
            coarse_vector=query_embedding,  # Using same embedding for all scales in demo
            medium_vector=query_embedding,
            fine_vector=query_embedding,
            limit=5,
            emotional_context=emotional_context,
            user_id=user_id,
        )

        if hierarchical_results:
            print(f"Found {len(hierarchical_results)} hierarchical results:")
            for i, result in enumerate(hierarchical_results, 1):
                print(f"  {i}. Score: {result.score:.3f}")

                # Show enhancement information
                payload = result.payload
                original_score = payload.get("original_score", result.score)
                enhancement_factors = payload.get("enhancement_factors", [])

                if original_score != result.score:
                    boost = result.score - original_score
                    print(
                        f"     Original Score: {original_score:.3f} → Enhanced: {result.score:.3f} (+{boost:.3f})"
                    )

                if enhancement_factors:
                    print(f"     Enhancement Factors: {', '.join(enhancement_factors)}")

                print(
                    f"     Emotional Enhanced: {payload.get('emotional_enhanced', False)}"
                )
                print(
                    f"     Companion Enhanced: {payload.get('companion_enhanced', False)}"
                )
                print()
        else:
            print("No hierarchical results found (collection may not exist yet)")

        # Test emotional context search
        print("💭 Emotional Context Search:")
        emotional_results = vector_db.search_with_emotional_context(
            collection_name="morgan_knowledge",
            query_vector=query_embedding,
            emotional_context=emotional_context,
            user_id=user_id,
            limit=3,
        )

        if emotional_results:
            print(f"Found {len(emotional_results)} emotional context results:")
            for i, result in enumerate(emotional_results, 1):
                print(f"  {i}. Score: {result.score:.3f}")

                payload = result.payload
                enhancement_factors = payload.get("enhancement_factors", [])
                if enhancement_factors:
                    print(f"     Enhancements: {', '.join(enhancement_factors)}")
                print()
        else:
            print("No emotional context results found (collection may not exist yet)")

    except Exception as e:
        print(f"❌ Hierarchical search demo failed: {e}")
        logger.error(f"Hierarchical search demo failed: {e}")


def demo_companion_personalization():
    """Demonstrate companion-aware personalization features."""
    print("\n👥 Companion Personalization Demo")
    print("=" * 50)

    try:
        # Test different user profiles
        user_profiles = [
            {
                "user_id": "tech_user",
                "interests": ["python", "docker", "api", "microservices"],
                "description": "Technical user interested in backend development",
            },
            {
                "user_id": "beginner_user",
                "interests": ["tutorial", "guide", "basics", "learning"],
                "description": "Beginner user looking for educational content",
            },
        ]

        query = "API development guide"

        for profile in user_profiles:
            print(f"\n👤 {profile['description']}")
            print(f"User ID: {profile['user_id']}")
            print(f"Interests: {', '.join(profile['interests'])}")

            # Create emotional context based on user type
            if "beginner" in profile["user_id"]:
                emotional_context = create_sample_emotional_context("curiosity", 0.6)
            else:
                emotional_context = create_sample_emotional_context("focus", 0.8)

            try:
                smart_search = SmartSearch()
                results = smart_search.find_relevant_info(
                    query=query,
                    max_results=3,
                    emotional_context=emotional_context,
                    user_id=profile["user_id"],
                    use_enhanced_search=True,
                )

                if results:
                    print(f"Personalized results for {profile['user_id']}:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. Score: {result.score:.3f}")

                        metadata = result.metadata
                        if metadata.get("companion_enhanced"):
                            print(f"     Companion Enhanced: Yes")

                        enhancement_factors = metadata.get("enhancement_factors", [])
                        if enhancement_factors:
                            print(
                                f"     Personalization: {', '.join(enhancement_factors)}"
                            )

                        print(f"     Content: {result.summary(80)}")
                        print()
                else:
                    print(f"No personalized results found for {profile['user_id']}")

            except Exception as e:
                print(f"❌ Personalization failed for {profile['user_id']}: {e}")

    except Exception as e:
        print(f"❌ Companion personalization demo failed: {e}")
        logger.error(f"Companion personalization demo failed: {e}")


if __name__ == "__main__":
    try:
        # Run all demos
        demo_enhanced_search_integration()
        demo_hierarchical_search_features()
        demo_companion_personalization()

        print("\n🎉 All Enhanced Search Integration Demos Complete!")
        print("\nThis demonstrates the successful implementation of:")
        print("✅ Task 7.2: Enhanced search integration")
        print("  - Multi-stage search with hierarchical embeddings")
        print("  - Emotional context in search result ranking")
        print("  - Companion-aware result filtering and personalization")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Enhanced search integration demo failed: {e}")
        sys.exit(1)
