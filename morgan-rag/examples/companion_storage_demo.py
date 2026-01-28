"""
Companion Storage Demo

Demonstrates the companion data models and storage functionality for
emotional intelligence and relationship management in Morgan RAG.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from morgan.companion.storage import CompanionStorage
from morgan.companion.schema import CompanionDatabaseSchema
from morgan.intelligence.core.models import (
    CompanionProfile,
    EmotionalState,
    UserPreferences,
    RelationshipMilestone,
    EmotionType,
    CommunicationStyle,
    ResponseLength,
    MilestoneType,
)


async def main():
    """Demonstrate companion storage functionality."""
    print("🤖 Morgan RAG - Companion Storage Demo")
    print("=" * 50)

    # Initialize storage (would use real services in production)
    print("\n1. Initializing Companion Storage...")
    try:
        storage = CompanionStorage()
        print("✅ Storage initialized successfully")
    except Exception as e:
        print(f"❌ Storage initialization failed: {e}")
        print("Note: This demo requires Qdrant to be running")
        return

    # Display schema information
    print("\n2. Database Schema Information:")
    schemas = CompanionDatabaseSchema.get_all_schemas()
    for schema in schemas:
        print(f"   📊 {schema.name}: {schema.vector_size}D vectors")

    summary = CompanionDatabaseSchema.get_schema_summary()
    print(f"   📈 Total collections: {summary['total_collections']}")
    print(f"   📈 Total vector dimensions: {summary['total_vector_dimensions']}")

    # Create sample user preferences
    print("\n3. Creating User Preferences...")
    user_preferences = UserPreferences(
        topics_of_interest=[
            "artificial intelligence",
            "machine learning",
            "python programming",
        ],
        communication_style=CommunicationStyle.FRIENDLY,
        preferred_response_length=ResponseLength.DETAILED,
        learning_goals=[
            "Master Python",
            "Understand neural networks",
            "Build AI applications",
        ],
        personal_context={
            "profession": "software developer",
            "experience_level": "intermediate",
            "timezone": "UTC-8",
        },
    )
    print(
        f"   ✅ Preferences created for {user_preferences.communication_style.value} style"
    )

    # Create companion profile
    print("\n4. Creating Companion Profile...")
    companion_profile = CompanionProfile(
        user_id="demo_user_123",
        preferred_name="Alex",
        relationship_duration=timedelta(days=45),
        interaction_count=127,
        communication_preferences=user_preferences,
        emotional_patterns={
            "dominant_emotions": ["joy", "curiosity"],
            "stress_indicators": ["short responses", "technical questions"],
            "engagement_peaks": ["morning", "evening"],
        },
        shared_memories=["memory_001", "memory_042", "memory_089"],
        trust_level=0.85,
        engagement_score=0.92,
    )
    print(f"   ✅ Profile created for {companion_profile.preferred_name}")
    print(f"   📊 Trust level: {companion_profile.trust_level:.2f}")
    print(f"   📊 Engagement score: {companion_profile.engagement_score:.2f}")

    # Store companion profile
    print("\n5. Storing Companion Profile...")
    try:
        success = storage.store_companion_profile(companion_profile)
        if success:
            print("   ✅ Profile stored successfully")
        else:
            print("   ❌ Failed to store profile")
    except Exception as e:
        print(f"   ❌ Storage error: {e}")

    # Create and store emotional state
    print("\n6. Creating Emotional State...")
    emotional_state = EmotionalState(
        primary_emotion=EmotionType.JOY,
        intensity=0.8,
        confidence=0.92,
        secondary_emotions=[EmotionType.SURPRISE, EmotionType.NEUTRAL],
        emotional_indicators=[
            "excited about learning",
            "asking follow-up questions",
            "using positive language",
        ],
    )
    print(f"   ✅ Emotional state: {emotional_state.primary_emotion.value}")
    print(f"   📊 Intensity: {emotional_state.intensity:.2f}")
    print(f"   📊 Confidence: {emotional_state.confidence:.2f}")

    try:
        success = storage.store_emotional_state("demo_user_123", emotional_state)
        if success:
            print("   ✅ Emotional state stored successfully")
        else:
            print("   ❌ Failed to store emotional state")
    except Exception as e:
        print(f"   ❌ Storage error: {e}")

    # Create and store relationship milestone
    print("\n7. Creating Relationship Milestone...")
    milestone = RelationshipMilestone(
        milestone_id="milestone_breakthrough_001",
        milestone_type=MilestoneType.BREAKTHROUGH_MOMENT,
        description="User successfully implemented their first neural network from scratch",
        timestamp=datetime.now(timezone.utc),
        emotional_significance=0.95,
        related_memories=["memory_089", "memory_090", "memory_091"],
        user_feedback="This was amazing! I finally understand how neural networks work.",
    )
    print(f"   ✅ Milestone: {milestone.milestone_type.value}")
    print(f"   📊 Emotional significance: {milestone.emotional_significance:.2f}")
    print(f"   💬 User feedback: {milestone.user_feedback[:50]}...")

    try:
        success = storage.store_relationship_milestone("demo_user_123", milestone)
        if success:
            print("   ✅ Milestone stored successfully")
        else:
            print("   ❌ Failed to store milestone")
    except Exception as e:
        print(f"   ❌ Storage error: {e}")

    # Retrieve stored data
    print("\n8. Retrieving Stored Data...")
    try:
        # Get companion profile
        retrieved_profile = storage.get_companion_profile("demo_user_123")
        if retrieved_profile:
            print(f"   ✅ Retrieved profile for {retrieved_profile.preferred_name}")
            print(f"   📊 Interactions: {retrieved_profile.interaction_count}")
        else:
            print("   ❌ Profile not found")

        # Get emotional history
        emotional_history = storage.get_user_emotional_history("demo_user_123", days=7)
        print(
            f"   ✅ Retrieved {len(emotional_history)} emotional states from last 7 days"
        )

        # Get milestones
        milestones = storage.get_user_milestones("demo_user_123")
        print(f"   ✅ Retrieved {len(milestones)} relationship milestones")

    except Exception as e:
        print(f"   ❌ Retrieval error: {e}")

    # Display storage statistics
    print("\n9. Storage Statistics...")
    try:
        stats = storage.get_storage_stats()
        for collection_name, info in stats.items():
            if "error" not in info:
                print(f"   📊 {collection_name}: {info.get('points_count', 0)} points")
            else:
                print(f"   ❌ {collection_name}: {info['error']}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")

    # Demonstrate schema validation
    print("\n10. Schema Validation Demo...")
    from morgan.companion.schema import (
        validate_companion_payload,
        validate_emotion_payload,
    )

    # Valid companion payload
    valid_companion = {
        "user_id": "test_user",
        "preferred_name": "Test User",
        "communication_style": "friendly",
        "trust_level": 0.7,
        "engagement_score": 0.8,
    }
    errors = validate_companion_payload(valid_companion)
    print(f"   ✅ Valid companion payload: {len(errors)} errors")

    # Invalid companion payload
    invalid_companion = {
        "preferred_name": "Test User",
        "trust_level": 1.5,  # Invalid range
        # Missing required fields
    }
    errors = validate_companion_payload(invalid_companion)
    print(f"   ❌ Invalid companion payload: {len(errors)} errors")
    for error in errors[:2]:  # Show first 2 errors
        print(f"      - {error}")

    # Valid emotion payload
    valid_emotion = {
        "user_id": "test_user",
        "primary_emotion": "joy",
        "intensity": 0.8,
        "confidence": 0.9,
    }
    errors = validate_emotion_payload(valid_emotion)
    print(f"   ✅ Valid emotion payload: {len(errors)} errors")

    print("\n🎉 Companion Storage Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Companion profile storage with relationship metrics")
    print("• Emotional state tracking and analysis")
    print("• Relationship milestone recording")
    print("• Vector database schema management")
    print("• Data validation and error handling")
    print("• Storage statistics and monitoring")


if __name__ == "__main__":
    asyncio.run(main())
