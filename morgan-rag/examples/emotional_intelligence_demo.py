#!/usr/bin/env python3
"""
Emotional Intelligence Engine Demo for Morgan RAG.

Demonstrates the core capabilities of the emotional intelligence engine:
- Real-time emotion detection from text
- Mood pattern analysis over time
- Empathetic response generation
- Personal preference learning
- Relationship milestone tracking
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add morgan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.emotional.intelligence_engine import EmotionalIntelligenceEngine
from morgan.emotional.models import (
    ConversationContext, InteractionData, EmotionalState, EmotionType
)


def demo_emotion_detection():
    """Demonstrate emotion detection capabilities."""
    print("="*60)
    print("EMOTION DETECTION DEMO")
    print("="*60)
    
    engine = EmotionalIntelligenceEngine()
    
    # Test various emotional texts
    test_texts = [
        "I'm so excited about this new project! It's going to be amazing!",
        "I'm feeling really down today. Everything seems to be going wrong.",
        "This is absolutely frustrating! Nothing is working as expected.",
        "I'm a bit worried about the upcoming presentation tomorrow.",
        "Wow, I didn't expect that to happen! What a surprise!",
        "Can you help me understand how this algorithm works?",
        "Thank you so much for your help! You've been incredibly supportive."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: \"{text}\"")
        
        # Create conversation context
        context = ConversationContext(
            user_id="demo_user",
            conversation_id=f"demo_conv_{i}",
            message_text=text,
            timestamp=datetime.utcnow()
        )
        
        # Analyze emotion
        emotion = engine.analyze_emotion(text, context)
        
        print(f"   Emotion: {emotion.primary_emotion.value}")
        print(f"   Intensity: {emotion.intensity:.2f}")
        print(f"   Confidence: {emotion.confidence:.2f}")
        if emotion.emotional_indicators:
            print(f"   Indicators: {', '.join(emotion.emotional_indicators)}")


def demo_mood_patterns():
    """Demonstrate mood pattern tracking."""
    print("\n" + "="*60)
    print("MOOD PATTERN ANALYSIS DEMO")
    print("="*60)
    
    engine = EmotionalIntelligenceEngine()
    user_id = "demo_user"
    
    # Simulate emotional states over time
    emotions = [
        (EmotionType.JOY, 0.8, 7),      # 7 days ago
        (EmotionType.JOY, 0.7, 6),      # 6 days ago
        (EmotionType.NEUTRAL, 0.5, 5),  # 5 days ago
        (EmotionType.SADNESS, 0.6, 4),  # 4 days ago
        (EmotionType.ANGER, 0.4, 3),    # 3 days ago
        (EmotionType.JOY, 0.9, 2),      # 2 days ago
        (EmotionType.JOY, 0.8, 1),      # 1 day ago
        (EmotionType.NEUTRAL, 0.5, 0),  # today
    ]
    
    # Add emotions to engine
    for emotion_type, intensity, days_ago in emotions:
        emotional_state = EmotionalState(
            primary_emotion=emotion_type,
            intensity=intensity,
            confidence=0.8,
            timestamp=datetime.utcnow() - timedelta(days=days_ago)
        )
        engine.mood_patterns[user_id].append(emotional_state)
    
    # Analyze patterns
    pattern = engine.track_mood_patterns(user_id, "7d")
    
    print(f"\nMood Pattern Analysis for {user_id}:")
    print(f"  Timeframe: {pattern.timeframe}")
    print(f"  Dominant emotions: {[e.value for e in pattern.dominant_emotions]}")
    print(f"  Average intensity: {pattern.average_intensity:.2f}")
    print(f"  Mood stability: {pattern.mood_stability:.2f}")
    print(f"  Pattern confidence: {pattern.pattern_confidence:.2f}")
    
    if pattern.emotional_trends:
        print(f"  Trends:")
        for key, value in pattern.emotional_trends.items():
            print(f"    {key}: {value}")


def demo_empathetic_responses():
    """Demonstrate empathetic response generation."""
    print("\n" + "="*60)
    print("EMPATHETIC RESPONSE GENERATION DEMO")
    print("="*60)
    
    engine = EmotionalIntelligenceEngine()
    
    # Test scenarios with different emotions
    scenarios = [
        (EmotionType.JOY, 0.8, "User just got a promotion at work"),
        (EmotionType.SADNESS, 0.7, "User is dealing with a difficult breakup"),
        (EmotionType.FEAR, 0.6, "User is nervous about a job interview"),
        (EmotionType.ANGER, 0.5, "User is frustrated with technical issues"),
    ]
    
    for i, (emotion_type, intensity, context) in enumerate(scenarios, 1):
        print(f"\n{i}. Scenario: {context}")
        
        emotional_state = EmotionalState(
            primary_emotion=emotion_type,
            intensity=intensity,
            confidence=0.8
        )
        
        # Generate empathetic response
        response = engine.generate_empathetic_response(emotional_state, context)
        
        print(f"   User emotion: {emotion_type.value} (intensity: {intensity})")
        print(f"   Emotional tone: {response.emotional_tone}")
        print(f"   Empathy level: {response.empathy_level:.2f}")
        print(f"   Response: \"{response.response_text}\"")
        print(f"   Confidence: {response.confidence_score:.2f}")


def demo_relationship_milestones():
    """Demonstrate relationship milestone detection."""
    print("\n" + "="*60)
    print("RELATIONSHIP MILESTONE DETECTION DEMO")
    print("="*60)
    
    engine = EmotionalIntelligenceEngine()
    
    # Simulate conversation history
    conversations = [
        ConversationContext(
            user_id="demo_user",
            conversation_id="conv1",
            message_text="Hi, I'm new here. Can you help me get started?",
            timestamp=datetime.utcnow() - timedelta(days=10)
        ),
        ConversationContext(
            user_id="demo_user",
            conversation_id="conv2",
            message_text="Thank you so much! That explanation really helped me understand the concept. I never thought about it that way before.",
            timestamp=datetime.utcnow() - timedelta(days=8),
            user_feedback=5
        ),
        ConversationContext(
            user_id="demo_user",
            conversation_id="conv3",
            message_text="I learned so much from our previous conversations. Now I know how to approach this problem systematically.",
            timestamp=datetime.utcnow() - timedelta(days=5)
        ),
        ConversationContext(
            user_id="demo_user",
            conversation_id="conv4",
            message_text="I feel comfortable sharing this with you. I've been struggling with confidence in my technical skills.",
            timestamp=datetime.utcnow() - timedelta(days=2)
        ),
    ]
    
    # Detect milestones
    milestones = engine.detect_relationship_milestones(conversations)
    
    print(f"\nDetected {len(milestones)} relationship milestones:")
    
    for i, milestone in enumerate(milestones, 1):
        print(f"\n{i}. {milestone.milestone_type.value.replace('_', ' ').title()}")
        print(f"   Description: {milestone.description}")
        print(f"   Timestamp: {milestone.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Emotional significance: {milestone.emotional_significance:.2f}")


def demo_user_profile_learning():
    """Demonstrate user profile learning and adaptation."""
    print("\n" + "="*60)
    print("USER PROFILE LEARNING DEMO")
    print("="*60)
    
    engine = EmotionalIntelligenceEngine()
    user_id = "demo_user"
    
    # Simulate multiple interactions
    interactions = [
        {
            "message": "I'm really interested in machine learning and AI. Can you explain neural networks in detail?",
            "emotion": EmotionType.JOY,
            "intensity": 0.7,
            "topics": ["machine learning", "AI", "neural networks"],
            "satisfaction": 0.9
        },
        {
            "message": "That was a comprehensive explanation! I prefer detailed responses that cover all aspects of the topic.",
            "emotion": EmotionType.JOY,
            "intensity": 0.8,
            "topics": ["learning", "explanation"],
            "satisfaction": 0.9
        },
        {
            "message": "I'm working on a deep learning project for computer vision. The technical details are fascinating!",
            "emotion": EmotionType.JOY,
            "intensity": 0.9,
            "topics": ["deep learning", "computer vision", "technical"],
            "satisfaction": 0.8
        }
    ]
    
    print(f"Simulating {len(interactions)} interactions for user profile learning...")
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\nInteraction {i}: \"{interaction['message'][:50]}...\"")
        
        # Create interaction data
        context = ConversationContext(
            user_id=user_id,
            conversation_id=f"conv_{i}",
            message_text=interaction["message"],
            timestamp=datetime.utcnow() - timedelta(hours=24-i*8),
            user_feedback=int(interaction["satisfaction"] * 5)
        )
        
        emotional_state = EmotionalState(
            primary_emotion=interaction["emotion"],
            intensity=interaction["intensity"],
            confidence=0.8
        )
        
        interaction_data = InteractionData(
            conversation_context=context,
            emotional_state=emotional_state,
            user_satisfaction=interaction["satisfaction"],
            topics_discussed=interaction["topics"]
        )
        
        # Update profile
        profile = engine.update_user_profile(user_id, interaction_data)
        
        print(f"  Updated profile - Interactions: {profile.interaction_count}")
        print(f"  Trust level: {profile.trust_level:.2f}")
        print(f"  Engagement score: {profile.engagement_score:.2f}")
    
    # Show final profile
    final_profile = engine.user_profiles[user_id]
    print(f"\nFinal User Profile:")
    print(f"  User ID: {final_profile.user_id}")
    print(f"  Total interactions: {final_profile.interaction_count}")
    print(f"  Relationship duration: {final_profile.relationship_duration}")
    print(f"  Trust level: {final_profile.trust_level:.2f}")
    print(f"  Engagement score: {final_profile.engagement_score:.2f}")
    print(f"  Preferred response length: {final_profile.communication_preferences.preferred_response_length.value}")
    print(f"  Topics of interest: {final_profile.communication_preferences.topics_of_interest}")


def main():
    """Run all emotional intelligence demos."""
    print("Morgan RAG - Emotional Intelligence Engine Demo")
    print("Demonstrating companion AI capabilities with emotional awareness")
    
    try:
        demo_emotion_detection()
        demo_mood_patterns()
        demo_empathetic_responses()
        demo_relationship_milestones()
        demo_user_profile_learning()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Emotional Intelligence Engine provides:")
        print("✓ Real-time emotion detection from text")
        print("✓ Mood pattern analysis over time")
        print("✓ Empathetic response generation")
        print("✓ Relationship milestone tracking")
        print("✓ Personal preference learning")
        print("✓ User profile adaptation")
        print("\nThis enables Morgan to be a truly empathetic companion AI!")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())