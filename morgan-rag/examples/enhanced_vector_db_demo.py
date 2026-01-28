#!/usr/bin/env python3
"""
Demo script for enhanced vector database client with companion features.

This script demonstrates the new companion-aware functionality added to the
VectorDBClient, including:
- Companion collection initialization
- Hierarchical collection creation
- Batch operations for emotional data
- Memory and companion profile management
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morgan.vector_db.client import VectorDBClient, BatchOperationResult


def demo_companion_collections():
    """Demonstrate companion collection management."""
    print("🔧 Companion Collection Management Demo")
    print("=" * 50)

    try:
        # Create enhanced client
        client = VectorDBClient()
        print("✓ Enhanced VectorDBClient created")

        # Show collection constants
        print(f"📁 Companion Collections:")
        print(f"   - Companions: {client.COMPANIONS_COLLECTION}")
        print(f"   - Memories: {client.MEMORIES_COLLECTION}")
        print(f"   - Emotions: {client.EMOTIONS_COLLECTION}")
        print(f"   - Milestones: {client.MILESTONES_COLLECTION}")

        # Note: We can't actually initialize collections without Qdrant running
        print("\n💡 Note: Collection initialization requires Qdrant server")
        print(
            "   Use client.initialize_companion_collections() when Qdrant is available"
        )

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_hierarchical_collections():
    """Demonstrate hierarchical collection creation."""
    print("\n🏗️ Hierarchical Collection Demo")
    print("=" * 50)

    try:
        client = VectorDBClient()

        print("📊 Hierarchical Collection Configuration:")
        print("   - Coarse embeddings: 384 dimensions (category filtering)")
        print("   - Medium embeddings: 768 dimensions (concept matching)")
        print("   - Fine embeddings: 1536 dimensions (precise retrieval)")

        print("\n💡 Usage: client.create_hierarchical_collection('knowledge_base')")
        print(
            "   Creates collection with multiple named vectors for multi-scale search"
        )

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_batch_operations():
    """Demonstrate batch operation capabilities."""
    print("\n⚡ Batch Operations Demo")
    print("=" * 50)

    try:
        client = VectorDBClient()

        # Create sample memory data
        sample_memories = []
        for i in range(5):
            memory = {
                "id": f"memory_{i}",
                "vector": [0.1 + i * 0.1] * 1536,  # Mock embeddings
                "payload": {
                    "user_id": "demo_user",
                    "content": f"This is sample memory {i}",
                    "importance_score": 0.5 + i * 0.1,
                    "entities": [f"entity_{i}"],
                    "concepts": [f"concept_{i}"],
                    "conversation_id": f"conv_{i}",
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                    "emotional_context": {
                        "primary_emotion": "joy",
                        "intensity": 0.7,
                        "confidence": 0.9,
                    },
                    "user_mood": "happy",
                    "relationship_significance": 0.6,
                },
            }
            sample_memories.append(memory)

        print(f"📦 Created {len(sample_memories)} sample memories")

        # Simulate batch operation (without actual Qdrant)
        print("\n💡 Batch Operation Features:")
        print("   - Automatic batching (default: 100 items per batch)")
        print("   - Retry logic with exponential backoff")
        print("   - Comprehensive error reporting")
        print("   - Performance metrics tracking")

        # Show what a batch result would look like
        mock_result = BatchOperationResult(
            success_count=4,
            failure_count=1,
            total_count=5,
            errors=["Sample error for demo"],
            processing_time=1.23,
        )

        print(f"\n📊 Sample Batch Result:")
        print(f"   - Success Rate: {mock_result.success_rate:.1f}%")
        print(f"   - Processing Time: {mock_result.processing_time:.2f}s")
        print(f"   - Errors: {len(mock_result.errors)}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_companion_features():
    """Demonstrate companion-specific features."""
    print("\n🤖 Companion Features Demo")
    print("=" * 50)

    try:
        client = VectorDBClient()

        # Sample companion profile data
        profile_data = {
            "user_id": "demo_user",
            "preferred_name": "Alex",
            "communication_style": "friendly",
            "relationship_duration_days": 30,
            "interaction_count": 150,
            "trust_level": 0.8,
            "engagement_score": 0.9,
            "topics_of_interest": ["python", "ai", "machine_learning"],
            "learning_goals": ["deep_learning", "nlp"],
            "emotional_patterns": {
                "dominant_emotions": ["joy", "curiosity"],
                "stress_indicators": ["time_pressure", "complex_problems"],
            },
        }

        print("👤 Sample Companion Profile:")
        for key, value in profile_data.items():
            if isinstance(value, (list, dict)):
                print(f"   - {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"   - {key}: {value}")

        # Sample emotional state data
        emotion_data = {
            "user_id": "demo_user",
            "primary_emotion": "joy",
            "intensity": 0.8,
            "confidence": 0.95,
            "secondary_emotions": ["excitement", "curiosity"],
            "emotional_indicators": ["positive_language", "enthusiasm"],
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n😊 Sample Emotional State:")
        for key, value in emotion_data.items():
            print(f"   - {key}: {value}")

        print(f"\n🔍 Advanced Search Capabilities:")
        print(f"   - Memory search by user with emotional filters")
        print(f"   - Hierarchical search with coarse-to-fine filtering")
        print(f"   - Emotional history tracking over time periods")
        print(f"   - Companion profile retrieval and updates")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_search_capabilities():
    """Demonstrate advanced search capabilities."""
    print("\n🔍 Advanced Search Demo")
    print("=" * 50)

    try:
        client = VectorDBClient()

        print("🎯 Search Methods Available:")
        print("   1. search_memories_by_user() - User-specific memory search")
        print("      - Semantic search with query vectors")
        print("      - Importance score filtering")
        print("      - Emotional state filtering")

        print("\n   2. search_hierarchical() - Multi-scale search")
        print("      - Coarse filtering (90% candidate reduction)")
        print("      - Medium refinement (concept-level matching)")
        print("      - Fine matching (precise content retrieval)")

        print("\n   3. get_user_emotional_history() - Mood tracking")
        print("      - Time-based emotional pattern analysis")
        print("      - Configurable lookback periods")
        print("      - Trend identification support")

        print("\n   4. get_companion_profile() - Relationship data")
        print("      - Complete user relationship profile")
        print("      - Communication preferences")
        print("      - Interaction history and milestones")

        print("\n💡 All methods include comprehensive error handling and logging")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all demos."""
    print("🚀 Enhanced Vector Database Client Demo")
    print("=" * 60)
    print("This demo showcases the new companion-aware features added to")
    print("the VectorDBClient for Morgan RAG's emotional intelligence system.")
    print()

    success = True

    success &= demo_companion_collections()
    success &= demo_hierarchical_collections()
    success &= demo_batch_operations()
    success &= demo_companion_features()
    success &= demo_search_capabilities()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All demos completed successfully!")
        print("\n📚 Key Features Implemented:")
        print("   ✓ Companion collection management")
        print("   ✓ Hierarchical embedding support")
        print("   ✓ Batch operations with error handling")
        print("   ✓ Emotional data processing")
        print("   ✓ Advanced search capabilities")
        print("   ✓ Relationship-aware data storage")

        print("\n🔧 Ready for Integration:")
        print("   - Memory processing with emotional context")
        print("   - Companion profile management")
        print("   - Multi-stage search with result fusion")
        print("   - Performance optimization through batching")
    else:
        print("❌ Some demos encountered issues")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
