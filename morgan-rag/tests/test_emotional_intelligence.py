"""
Tests for emotional intelligence engine.

Tests core functionality of emotion detection, mood pattern analysis,
empathetic response generation, and user profile management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.emotional.intelligence_engine import EmotionalIntelligenceEngine
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext, InteractionData,
    UserPreferences, CommunicationStyle, ResponseLength
)


class TestEmotionalIntelligenceEngine:
    """Test emotional intelligence engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create emotional intelligence engine for testing."""
        with patch('morgan.emotional.intelligence_engine.get_llm_service'):
            return EmotionalIntelligenceEngine()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling great today!",
            timestamp=datetime.utcnow()
        )
    
    def test_emotion_detection_joy(self, engine, sample_context):
        """Test detection of joy emotion."""
        # Test with clear joy indicators
        joyful_text = "I'm so happy and excited about this amazing news!"
        
        emotion = engine.analyze_emotion(joyful_text, sample_context)
        
        assert emotion.primary_emotion == EmotionType.JOY
        assert emotion.intensity > 0.5
        assert emotion.confidence > 0.5
        assert len(emotion.emotional_indicators) > 0
    
    def test_emotion_detection_sadness(self, engine, sample_context):
        """Test detection of sadness emotion."""
        sad_text = "I'm feeling really down and disappointed about everything"
        sample_context.message_text = sad_text
        
        emotion = engine.analyze_emotion(sad_text, sample_context)
        
        assert emotion.primary_emotion == EmotionType.SADNESS
        assert emotion.intensity > 0.3
        assert emotion.confidence > 0.5
    
    def test_emotion_detection_neutral(self, engine, sample_context):
        """Test detection of neutral emotion."""
        neutral_text = "Can you help me with this technical question?"
        sample_context.message_text = neutral_text
        
        emotion = engine.analyze_emotion(neutral_text, sample_context)
        
        # Should detect neutral or have low intensity
        assert emotion.intensity <= 0.7  # Not highly emotional
        assert emotion.confidence >= 0.3
    
    def test_mood_pattern_tracking(self, engine):
        """Test mood pattern analysis over time."""
        user_id = "test_user"
        
        # Add some emotional states to history
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9,
                timestamp=datetime.utcnow() - timedelta(days=1)
            ),
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.7,
                confidence=0.8,
                timestamp=datetime.utcnow() - timedelta(hours=12)
            ),
            EmotionalState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                confidence=0.7,
                timestamp=datetime.utcnow()
            )
        ]
        
        engine.mood_patterns[user_id] = emotions
        
        # Analyze mood patterns
        pattern = engine.track_mood_patterns(user_id, "7d")
        
        assert pattern.user_id == user_id
        assert EmotionType.JOY in pattern.dominant_emotions
        assert pattern.average_intensity > 0.5
        assert pattern.pattern_confidence > 0.0
    
    def test_relationship_milestone_detection(self, engine):
        """Test detection of relationship milestones."""
        conversations = [
            ConversationContext(
                user_id="test_user",
                conversation_id="conv1",
                message_text="Hello, this is my first time using this system",
                timestamp=datetime.utcnow() - timedelta(days=5)
            ),
            ConversationContext(
                user_id="test_user",
                conversation_id="conv2",
                message_text="Thank you so much! This really helped me understand the concept. I never thought about it that way before.",
                timestamp=datetime.utcnow() - timedelta(days=3),
                user_feedback=5
            ),
            ConversationContext(
                user_id="test_user",
                conversation_id="conv3",
                message_text="I learned so much from our previous conversations. Now I know how to approach this problem.",
                timestamp=datetime.utcnow() - timedelta(days=1)
            )
        ]
        
        milestones = engine.detect_relationship_milestones(conversations)
        
        # Should detect first conversation and breakthrough moments
        assert len(milestones) >= 1
        
        # Check for first conversation milestone
        first_conv_milestones = [m for m in milestones if m.milestone_type.value == "first_conversation"]
        assert len(first_conv_milestones) == 1
    
    def test_empathetic_response_generation(self, engine):
        """Test generation of empathetic responses."""
        # Mock LLM service
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(content="I understand you're feeling happy! That's wonderful to hear.")
        engine.llm_service = mock_llm
        
        user_emotion = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )
        
        context = "User just achieved a personal goal"
        
        response = engine.generate_empathetic_response(user_emotion, context)
        
        assert response.empathy_level > 0.5
        assert response.emotional_tone == "warm and celebratory"
        assert len(response.response_text) > 0
        assert response.confidence_score > 0.5
    
    def test_user_profile_creation_and_update(self, engine):
        """Test user profile creation and updates."""
        user_id = "test_user"
        
        # Create interaction data
        context = ConversationContext(
            user_id=user_id,
            conversation_id="test_conv",
            message_text="I really enjoy learning about machine learning and AI. This is fascinating!",
            timestamp=datetime.utcnow(),
            user_feedback=4
        )
        
        emotional_state = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.7,
            confidence=0.8
        )
        
        interaction_data = InteractionData(
            conversation_context=context,
            emotional_state=emotional_state,
            user_satisfaction=0.8,
            topics_discussed=["machine learning", "AI"],
            learning_indicators=["enjoy learning"]
        )
        
        # Update profile
        profile = engine.update_user_profile(user_id, interaction_data)
        
        assert profile.user_id == user_id
        assert profile.interaction_count == 1
        assert profile.trust_level > 0.1  # Should increase from initial
        assert profile.engagement_score > 0.5
        assert "machine learning" in profile.communication_preferences.topics_of_interest
    
    def test_emotion_intensity_modifiers(self, engine, sample_context):
        """Test that intensity modifiers work correctly."""
        # Test with intensity modifier
        intense_text = "I'm extremely happy about this!"
        moderate_text = "I'm happy about this"
        
        intense_emotion = engine.analyze_emotion(intense_text, sample_context)
        moderate_emotion = engine.analyze_emotion(moderate_text, sample_context)
        
        # Intense emotion should have higher intensity
        assert intense_emotion.intensity >= moderate_emotion.intensity
    
    def test_emotion_caching(self, engine, sample_context):
        """Test that emotion analysis results are cached."""
        text = "I'm feeling great today!"
        
        # First call should compute and cache
        emotion1 = engine.analyze_emotion(text, sample_context)
        
        # Second call should use cache
        emotion2 = engine.analyze_emotion(text, sample_context)
        
        # Results should be identical
        assert emotion1.primary_emotion == emotion2.primary_emotion
        assert emotion1.intensity == emotion2.intensity
        assert emotion1.confidence == emotion2.confidence
    
    def test_fallback_empathetic_response(self, engine):
        """Test fallback empathetic responses when LLM fails."""
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM failed")
        engine.llm_service = mock_llm
        
        user_emotion = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.6,
            confidence=0.8
        )
        
        response = engine.generate_empathetic_response(user_emotion, "test context")
        
        # Should still generate a response using fallback
        assert len(response.response_text) > 0
        assert "support" in response.response_text.lower() or "difficult" in response.response_text.lower()
    
    def test_communication_preference_learning(self, engine):
        """Test that communication preferences are learned from interactions."""
        user_id = "test_user"
        
        # Create interaction with long message (indicates detailed preference)
        long_message = "This is a very detailed message where I explain my thoughts thoroughly and provide extensive context about what I'm thinking and feeling. I like to be comprehensive in my communication and provide lots of details about my situation and needs." * 2
        
        context = ConversationContext(
            user_id=user_id,
            conversation_id="test_conv",
            message_text=long_message,
            timestamp=datetime.utcnow()
        )
        
        emotional_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            confidence=0.7
        )
        
        interaction_data = InteractionData(
            conversation_context=context,
            emotional_state=emotional_state
        )
        
        # Update profile
        profile = engine.update_user_profile(user_id, interaction_data)
        
        # Should learn detailed communication preference
        assert profile.communication_preferences.preferred_response_length == ResponseLength.DETAILED