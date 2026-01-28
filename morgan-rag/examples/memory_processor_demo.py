#!/usr/bin/env python3
"""
Demo script for enhanced memory processor with emotional awareness.

This script demonstrates how the memory processor extracts memories from
conversations with emotional context, importance scoring, and personal
preference detection.
"""

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from morgan.memory.memory_processor import get_memory_processor
from morgan.core.memory import ConversationTurn
from morgan.intelligence.core.models import EmotionalState, EmotionType, ConversationContext


def create_sample_conversation_turns():
    """Create sample conversation turns for demonstration."""
    turns = [
        ConversationTurn(
            turn_id="turn_1",
            conversation_id="demo_conversation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="I'm really excited about learning Python! It's been my personal goal for months and I finally have time to focus on it.",
            answer="That's wonderful to hear! Your enthusiasm is contagious. Python is an excellent choice for beginners and has a fantastic community. What specific area of Python interests you most?",
            sources=["python-tutorial.md"],
            feedback_rating=5,
            feedback_comment="Very encouraging response!",
        ),
        ConversationTurn(
            turn_id="turn_2",
            conversation_id="demo_conversation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="I'm particularly interested in machine learning and data science. I work at a tech company and want to transition into an AI role.",
            answer="Machine learning is a fantastic field! Given your tech background, you're already ahead. I'd recommend starting with pandas and numpy for data manipulation, then moving to scikit-learn for ML basics.",
            sources=["ml-guide.md", "career-transition.md"],
            feedback_rating=4,
            feedback_comment="Great roadmap!",
        ),
        ConversationTurn(
            turn_id="turn_3",
            conversation_id="demo_conversation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="I'm feeling a bit overwhelmed by all the options. There's so much to learn - pandas, numpy, scikit-learn, TensorFlow, PyTorch. Where should I really start?",
            answer="I completely understand that feeling! It's natural to feel overwhelmed when starting something new. Let's break it down into manageable steps. Start with just pandas for data manipulation - master that first before moving on.",
            sources=["learning-path.md"],
            feedback_rating=5,
            feedback_comment="This really helped calm my anxiety!",
        ),
        ConversationTurn(
            turn_id="turn_4",
            conversation_id="demo_conversation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="Thank you so much! I really appreciate how you understand my situation. I feel like I can trust your guidance. Can you recommend some specific pandas tutorials?",
            answer="I'm so glad I could help ease your concerns! Trust is important in learning. Here are some excellent pandas resources: the official pandas documentation has great tutorials, and Kaggle Learn has hands-on exercises.",
            sources=["pandas-resources.md"],
            feedback_rating=5,
            feedback_comment="Perfect recommendations!",
        ),
    ]

    return turns


def create_sample_emotional_states():
    """Create sample emotional states corresponding to conversation turns."""
    return [
        EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            secondary_emotions=[EmotionType.SURPRISE],
            emotional_indicators=["excited", "wonderful", "goal", "finally"],
        ),
        EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.6,
            confidence=0.8,
            secondary_emotions=[],
            emotional_indicators=["interested", "want", "transition"],
        ),
        EmotionalState(
            primary_emotion=EmotionType.FEAR,
            intensity=0.7,
            confidence=0.8,
            secondary_emotions=[EmotionType.SADNESS],
            emotional_indicators=["overwhelmed", "anxiety", "so much"],
        ),
        EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.9,
            confidence=0.9,
            secondary_emotions=[],
            emotional_indicators=["thank you", "appreciate", "trust", "glad"],
        ),
    ]


def demonstrate_memory_extraction():
    """Demonstrate memory extraction with emotional awareness."""
    print("🧠 Enhanced Memory Processor Demo")
    print("=" * 50)

    # Get memory processor instance
    try:
        memory_processor = get_memory_processor()
        print("✅ Memory processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize memory processor: {e}")
        return

    # Create sample data
    conversation_turns = create_sample_conversation_turns()
    emotional_states = create_sample_emotional_states()

    print(f"\n📝 Processing {len(conversation_turns)} conversation turns...")

    all_memories = []
    all_insights = []

    # Process each conversation turn
    for i, (turn, emotional_state) in enumerate(
        zip(conversation_turns, emotional_states)
    ):
        print(f"\n--- Turn {i+1} ---")
        print(f"Question: {turn.question[:100]}...")
        print(
            f"Emotion: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.2f})"
        )

        try:
            # Extract memories with emotional context
            result = memory_processor.extract_memories(
                turn,
                feedback_rating=turn.feedback_rating,
                emotional_context=emotional_state,
            )

            print(f"Extracted {len(result.memories)} memories")

            # Display extracted memories
            for j, memory in enumerate(result.memories):
                print(f"  Memory {j+1}:")
                print(f"    Content: {memory.content}")
                print(f"    Importance: {memory.importance_score:.3f}")
                print(f"    Entities: {memory.entities}")
                print(f"    Concepts: {memory.concepts}")
                print(f"    Emotional Context: {memory.emotional_context}")
                print(
                    f"    Relationship Significance: {memory.relationship_significance:.3f}"
                )
                print(f"    Personal Preferences: {memory.personal_preferences}")

                # Store memory (in real usage)
                try:
                    stored = memory_processor.store_memory(memory)
                    if stored:
                        print(f"    ✅ Memory stored successfully")
                    else:
                        print(f"    ❌ Failed to store memory")
                except Exception as e:
                    print(f"    ⚠️  Storage simulation: {e}")

            # Display emotional insights
            print(f"  Emotional Insights: {result.emotional_insights}")
            print(f"  Preference Updates: {result.preference_updates}")
            print(f"  Relationship Indicators: {result.relationship_indicators}")

            all_memories.extend(result.memories)
            all_insights.append(result.emotional_insights)

        except Exception as e:
            print(f"❌ Error processing turn {i+1}: {e}")

    # Demonstrate importance scoring
    print(f"\n🎯 Importance Scoring Demo")
    print("-" * 30)

    test_contents = [
        "This is really important for my career goals!",
        "I just had lunch.",
        "I'm feeling overwhelmed and need help with this crucial project.",
        "The weather is nice today.",
        "I trust your advice and want to share something personal.",
    ]

    for content in test_contents:
        context = {
            "feedback_rating": 4,
            "is_user_message": True,
            "emotional_context": EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.6, confidence=0.8
            ),
        }

        score = memory_processor.score_importance(
            content, context, emotional_weight=1.5
        )
        print(f"Content: '{content[:50]}...'")
        print(f"Importance Score: {score:.3f}")
        print()

    # Demonstrate entity detection
    print(f"\n🏷️  Entity Detection Demo")
    print("-" * 25)

    test_text = "I work at Google using Python and Docker for machine learning projects with TensorFlow."
    entities = memory_processor.detect_entities(test_text)

    print(f"Text: {test_text}")
    print(f"Detected Entities: {entities}")

    # Demonstrate personal preference extraction
    print(f"\n👤 Personal Preference Extraction Demo")
    print("-" * 40)

    try:
        preferences = memory_processor.extract_personal_preferences(conversation_turns)

        print(f"Topics of Interest: {preferences.topics_of_interest}")
        print(f"Communication Style: {preferences.communication_style}")
        print(f"Preferred Response Length: {preferences.preferred_response_length}")
        print(f"Learning Goals: {preferences.learning_goals}")
        print(f"Personal Context: {preferences.personal_context}")

    except Exception as e:
        print(f"❌ Error extracting preferences: {e}")

    # Summary
    print(f"\n📊 Processing Summary")
    print("-" * 20)
    print(f"Total Memories Extracted: {len(all_memories)}")
    print(
        f"Average Importance Score: {sum(m.importance_score for m in all_memories) / len(all_memories):.3f}"
    )
    print(f"Emotional Contexts Analyzed: {len(all_insights)}")

    # Display memory distribution by emotion
    emotion_counts = {}
    for memory in all_memories:
        emotion = memory.user_mood
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print(f"Memory Distribution by Emotion:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} memories")

    # Display relationship significance distribution
    high_significance = [m for m in all_memories if m.relationship_significance > 0.5]
    print(f"High Relationship Significance Memories: {len(high_significance)}")

    print(f"\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demonstrate_memory_extraction()
