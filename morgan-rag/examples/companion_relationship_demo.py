"""
Companion Relationship Manager Demo.

Demonstrates the core functionality of the companion relationship manager
including profile building, conversation adaptation, milestone tracking,
and personalized interactions.
"""

from datetime import datetime, timedelta, timezone
from morgan.companion.relationship_manager import (
    CompanionRelationshipManager,
    Interaction,
)
from morgan.intelligence.core.models import EmotionalState, EmotionType, ConversationContext


def main():
    """Demonstrate companion relationship manager functionality."""
    print("=== Companion Relationship Manager Demo ===\n")

    # Initialize the relationship manager
    manager = CompanionRelationshipManager()
    user_id = "demo_user"

    # Create sample interactions over time
    interactions = [
        Interaction(
            interaction_id="int_1",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=10),
            message_content="Hi, I'm new here and interested in learning Python",
            topics_discussed=["python", "learning"],
            user_satisfaction=0.7,
        ),
        Interaction(
            interaction_id="int_2",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=8),
            message_content="Could you please help me understand data structures?",
            topics_discussed=["data structures", "programming"],
            user_satisfaction=0.8,
        ),
        Interaction(
            interaction_id="int_3",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=5),
            message_content="Thank you for the explanation! Call me Sarah, by the way.",
            topics_discussed=["feedback"],
            user_satisfaction=0.9,
        ),
        Interaction(
            interaction_id="int_4",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=2),
            message_content="I'm working on a machine learning project now",
            topics_discussed=["machine learning", "projects"],
            user_satisfaction=0.85,
        ),
    ]

    # 1. Build user profile
    print("1. Building User Profile")
    print("-" * 30)
    profile = manager.build_user_profile(user_id, interactions)

    print(f"User ID: {profile.user_id}")
    print(f"Preferred Name: {profile.preferred_name}")
    print(f"Interaction Count: {profile.interaction_count}")
    print(f"Relationship Duration: {profile.relationship_duration.days} days")
    print(f"Trust Level: {profile.trust_level:.2f}")
    print(f"Engagement Score: {profile.engagement_score:.2f}")
    print(
        f"Communication Style: {profile.communication_preferences.communication_style.value}"
    )
    print(f"Topics of Interest: {profile.communication_preferences.topics_of_interest}")
    print()

    # 2. Adapt conversation style based on mood
    print("2. Conversation Style Adaptation")
    print("-" * 35)

    # Happy mood
    happy_mood = EmotionalState(
        primary_emotion=EmotionType.JOY, intensity=0.8, confidence=0.9
    )

    happy_style = manager.adapt_conversation_style(profile, happy_mood)
    print("When user is happy:")
    print(f"  Formality Level: {happy_style.formality_level:.2f}")
    print(f"  Technical Depth: {happy_style.technical_depth:.2f}")
    print(f"  Empathy Emphasis: {happy_style.empathy_emphasis:.2f}")
    print(f"  Personality Traits: {happy_style.personality_traits}")

    # Sad mood
    sad_mood = EmotionalState(
        primary_emotion=EmotionType.SADNESS, intensity=0.7, confidence=0.8
    )

    sad_style = manager.adapt_conversation_style(profile, sad_mood)
    print("\nWhen user is sad:")
    print(f"  Formality Level: {sad_style.formality_level:.2f}")
    print(f"  Technical Depth: {sad_style.technical_depth:.2f}")
    print(f"  Empathy Emphasis: {sad_style.empathy_emphasis:.2f}")
    print(f"  Personality Traits: {sad_style.personality_traits}")
    print()

    # 3. Track relationship milestones
    print("3. Relationship Milestone Tracking")
    print("-" * 37)

    # Track first conversation milestone
    milestone1 = manager.track_relationship_milestones(user_id, "first_conversation")
    print(f"Milestone: {milestone1.description}")
    print(f"Type: {milestone1.milestone_type.value}")
    print(f"Emotional Significance: {milestone1.emotional_significance:.2f}")

    # Track learning milestone
    milestone2 = manager.track_relationship_milestones(user_id, "learning_milestone")
    print(f"\nMilestone: {milestone2.description}")
    print(f"Type: {milestone2.milestone_type.value}")
    print(f"Emotional Significance: {milestone2.emotional_significance:.2f}")

    # Check updated profile
    updated_profile = manager.profiles[user_id]
    print(f"\nTotal Milestones: {len(updated_profile.relationship_milestones)}")
    print(f"Updated Trust Level: {updated_profile.trust_level:.2f}")
    print()

    # 4. Generate personalized greetings
    print("4. Personalized Greeting Generation")
    print("-" * 38)

    # Recent interaction (1 hour ago)
    recent_greeting = manager.generate_personalized_greeting(
        updated_profile, timedelta(hours=1)
    )
    print("Recent interaction (1 hour ago):")
    print(f"  Greeting: {recent_greeting.greeting_text}")
    print(f"  Personalization Level: {recent_greeting.personalization_level:.2f}")
    print(f"  Context Elements: {recent_greeting.context_elements}")

    # Long absence (1 week ago)
    long_greeting = manager.generate_personalized_greeting(
        updated_profile, timedelta(days=7)
    )
    print("\nLong absence (1 week ago):")
    print(f"  Greeting: {long_greeting.greeting_text}")
    print(f"  Personalization Level: {long_greeting.personalization_level:.2f}")
    print(f"  Context Elements: {long_greeting.context_elements}")
    print()

    # 5. Suggest conversation topics
    print("5. Conversation Topic Suggestions")
    print("-" * 36)

    context = ConversationContext(
        user_id=user_id,
        conversation_id="demo_conv",
        message_text="I want to learn more about neural networks",
        timestamp=datetime.now(timezone.utc),
        previous_messages=["Tell me about machine learning algorithms"],
    )

    topics = manager.suggest_conversation_topics(
        updated_profile.communication_preferences.topics_of_interest, context
    )

    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic.topic}")
        print(f"   Category: {topic.category}")
        print(f"   Relevance: {topic.relevance_score:.2f}")
        print(f"   Interest Match: {topic.user_interest_match:.2f}")
        print(f"   Reasoning: {topic.reasoning}")
        print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
