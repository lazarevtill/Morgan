#!/usr/bin/env python3
"""
Test script for the enhanced Morgan assistant with emotional intelligence.
"""

import sys
import os

# Add the morgan-rag directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'morgan-rag'))

def test_enhanced_assistant():
    """Test the enhanced Morgan assistant functionality."""
    try:
        from morgan.core.assistant import MorganAssistant
        
        print("🤖 Testing Enhanced Morgan Assistant with Emotional Intelligence")
        print("=" * 70)
        
        # Initialize assistant
        morgan = MorganAssistant()
        print(f"✅ Assistant initialized: {morgan}")
        
        # Test basic question with emotional intelligence
        test_user_id = "test_user_123"
        question = "I'm feeling frustrated with Docker. Can you help me understand it?"
        
        print(f"\n📝 Test Question: {question}")
        print(f"👤 User ID: {test_user_id}")
        
        # Ask question with emotional intelligence
        response = morgan.ask(
            question=question,
            user_id=test_user_id
        )
        
        print(f"\n🤖 Response:")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Emotional Tone: {response.emotional_tone}")
        print(f"Empathy Level: {response.empathy_level:.2f}")
        print(f"Personalization Elements: {response.personalization_elements}")
        
        if response.milestone_celebration:
            print(f"🎉 Milestone: {response.milestone_celebration.description}")
        
        # Test relationship insights
        print(f"\n📊 Relationship Insights:")
        insights = morgan.get_relationship_insights(test_user_id)
        for key, value in insights.items():
            print(f"  {key}: {value}")
        
        # Test conversation topics
        print(f"\n💡 Suggested Topics:")
        topics = morgan.suggest_conversation_topics(test_user_id)
        for i, topic in enumerate(topics[:3], 1):
            print(f"  {i}. {topic}")
        
        # Test milestone celebration
        print(f"\n🎊 Testing Milestone Celebration:")
        celebration = morgan.celebrate_milestone(
            test_user_id, 
            "first_conversation",
            "Welcome to our journey together!"
        )
        print(f"Celebration: {celebration}")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_assistant()
    sys.exit(0 if success else 1)