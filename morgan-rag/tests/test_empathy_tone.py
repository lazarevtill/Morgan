"""
Tests for emotional tone matching module.

Tests tone matching, adaptation, tone-matched response generation,
and user tone preference analysis.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from morgan.intelligence.empathy.tone import EmotionalToneManager, ToneType, ToneIntensity
from morgan.intelligence.core.models import (
    EmotionalState,
    EmotionType,
    ConversationContext,
    CommunicationStyle,
)


class TestEmotionalToneManager:
    """Test emotional tone manager functionality."""

    @pytest.fixture
    def tone_manager(self):
        """Create emotional tone manager for testing."""
        with patch("morgan.intelligence.empathy.tone.get_llm_service") as mock_llm:
            # Configure mock to return proper response with .content attribute
            mock_response = Mock()
            mock_response.content = "This is a wonderfully tone-matched response that shows warmth and support."
            mock_llm.return_value.generate.return_value = mock_response
            with patch("morgan.intelligence.empathy.tone.get_settings"):
                return EmotionalToneManager()

    @pytest.fixture
    def emotional_state_joy(self):
        """Create emotional state - joy."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def emotional_state_sadness(self):
        """Create emotional state - sadness."""
        return EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm having a great day!",
            timestamp=datetime.now(timezone.utc),
        )

    def test_match_emotional_tone_joy(
        self, tone_manager, emotional_state_joy, conversation_context
    ):
        """Test matching emotional tone for joy."""
        tone_config = tone_manager.match_emotional_tone(
            emotional_state_joy, conversation_context
        )

        assert isinstance(tone_config, dict)
        assert "primary_tone" in tone_config
        assert "tone_intensity" in tone_config
        assert "emotional_resonance" in tone_config
        assert tone_config["primary_tone"] == ToneType.ENERGETIC_ENTHUSIASTIC

    def test_match_emotional_tone_sadness(
        self, tone_manager, emotional_state_sadness, conversation_context
    ):
        """Test matching emotional tone for sadness."""
        tone_config = tone_manager.match_emotional_tone(
            emotional_state_sadness, conversation_context
        )

        assert tone_config["primary_tone"] == ToneType.GENTLE_CALMING
        assert isinstance(tone_config["tone_intensity"], ToneIntensity)

    def test_match_emotional_tone_with_communication_style(
        self, tone_manager, emotional_state_joy, conversation_context
    ):
        """Test matching tone with user communication style."""
        tone_config = tone_manager.match_emotional_tone(
            emotional_state_joy,
            conversation_context,
            user_communication_style=CommunicationStyle.FORMAL,
        )

        assert tone_config is not None
        assert "secondary_tone" in tone_config

    def test_adapt_response_tone_minimal(self, tone_manager):
        """Test adapting response tone with minimal strength."""
        response_text = "This is a test response"
        tone_config = {
            "primary_tone": ToneType.WARM_SUPPORTIVE,
            "tone_intensity": ToneIntensity.MODERATE,
            "tone_characteristics": {},
        }

        adapted = tone_manager.adapt_response_tone(
            response_text, tone_config, adaptation_strength=0.2
        )

        assert adapted is not None
        assert isinstance(adapted, str)

    def test_adapt_response_tone_strong(self, tone_manager):
        """Test adapting response tone with strong adaptation."""
        response_text = "This is a test response"
        tone_config = {
            "primary_tone": ToneType.ENERGETIC_ENTHUSIASTIC,
            "tone_intensity": ToneIntensity.STRONG,
            "tone_characteristics": {
                "punctuation_style": "expressive",
                "sentence_starters": ["Wow!", "That's incredible!"],
            },
        }

        adapted = tone_manager.adapt_response_tone(
            response_text, tone_config, adaptation_strength=0.9
        )

        assert adapted is not None

    def test_create_tone_matched_response(
        self, tone_manager, emotional_state_joy, conversation_context
    ):
        """Test creating tone-matched response."""
        content = "That's wonderful news about your success"

        response = tone_manager.create_tone_matched_response(
            content, emotional_state_joy, conversation_context
        )

        assert response is not None
        assert len(response) > 0

    def test_analyze_user_tone_preferences_insufficient_data(self, tone_manager):
        """Test analyzing tone preferences with insufficient data."""
        analysis = tone_manager.analyze_user_tone_preferences([], [])

        assert analysis["preferences"] == "insufficient_data"

    def test_analyze_user_tone_preferences_with_data(
        self, tone_manager, conversation_context
    ):
        """Test analyzing tone preferences with conversation data."""
        conversation_history = [conversation_context] * 5
        feedback_history = [{"rating": 4}, {"rating": 5}]

        analysis = tone_manager.analyze_user_tone_preferences(
            conversation_history, feedback_history
        )

        assert "preferred_tones" in analysis
        assert "communication_style" in analysis
        assert "formality_preference" in analysis
        assert "confidence_score" in analysis

    def test_determine_intensity_level_high(self, tone_manager):
        """Test determining intensity level - high."""
        level = tone_manager._determine_intensity_level(0.85)

        assert level == "high"

    def test_determine_intensity_level_medium(self, tone_manager):
        """Test determining intensity level - medium."""
        level = tone_manager._determine_intensity_level(0.6)

        assert level == "medium"

    def test_determine_intensity_level_low(self, tone_manager):
        """Test determining intensity level - low."""
        level = tone_manager._determine_intensity_level(0.3)

        assert level == "low"

    def test_calculate_emotional_resonance(self, tone_manager, emotional_state_joy):
        """Test calculating emotional resonance."""
        resonance = tone_manager._calculate_emotional_resonance(emotional_state_joy)

        assert resonance >= 0.0
        assert resonance <= 1.0

    def test_calculate_adaptation_confidence(
        self, tone_manager, emotional_state_joy, conversation_context
    ):
        """Test calculating adaptation confidence."""
        confidence = tone_manager._calculate_adaptation_confidence(
            emotional_state_joy, conversation_context
        )

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_get_combined_tone_characteristics(self, tone_manager):
        """Test getting combined tone characteristics."""
        combined = tone_manager._get_combined_tone_characteristics(
            ToneType.WARM_SUPPORTIVE,
            ToneType.ENCOURAGING_UPLIFTING,
            ToneIntensity.MODERATE,
        )

        assert isinstance(combined, dict)
        assert "adjectives" in combined
        assert "phrases" in combined
        assert "punctuation_style" in combined

    def test_analyze_message_characteristics(self, tone_manager, conversation_context):
        """Test analyzing message characteristics."""
        conversation_history = [conversation_context] * 10

        analysis = tone_manager._analyze_message_characteristics(conversation_history)

        assert isinstance(analysis, dict)
        assert "length_preference" in analysis
        assert "formality" in analysis
        assert "expressiveness" in analysis

    def test_analyze_feedback_patterns_no_feedback(self, tone_manager):
        """Test analyzing feedback patterns with no data."""
        analysis = tone_manager._analyze_feedback_patterns([])

        assert analysis["feedback_available"] is False

    def test_analyze_feedback_patterns_with_feedback(self, tone_manager):
        """Test analyzing feedback patterns with data."""
        feedback_history = [
            {"rating": 4, "text": "helpful and clear"},
            {"rating": 5, "text": "very friendly"},
            {"rating": 2, "text": "too technical"},
        ]

        analysis = tone_manager._analyze_feedback_patterns(feedback_history)

        assert analysis["feedback_available"] is True
        assert "avg_rating" in analysis
        assert analysis["total_feedback"] == 3

    def test_determine_preferred_tones(self, tone_manager):
        """Test determining preferred tones from analysis."""
        message_analysis = {"formality": "casual", "expressiveness": "high"}
        feedback_analysis = {"feedback_available": True, "avg_rating": 4.5}

        preferred_tones = tone_manager._determine_preferred_tones(
            message_analysis, feedback_analysis
        )

        assert isinstance(preferred_tones, list)
        assert len(preferred_tones) > 0

    def test_calculate_preference_confidence(self, tone_manager):
        """Test calculating preference confidence."""
        confidence = tone_manager._calculate_preference_confidence(
            conversation_count=15, feedback_count=5
        )

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_select_secondary_tone_with_user_style(
        self, tone_manager, conversation_context
    ):
        """Test selecting secondary tone with user communication style."""
        secondary_tones = [ToneType.WARM_SUPPORTIVE, ToneType.RESPECTFUL_NEUTRAL]

        tone = tone_manager._select_secondary_tone(
            secondary_tones, conversation_context, CommunicationStyle.FORMAL
        )

        assert tone == ToneType.RESPECTFUL_NEUTRAL

    def test_select_secondary_tone_without_user_style(
        self, tone_manager, conversation_context
    ):
        """Test selecting secondary tone without user style."""
        secondary_tones = [ToneType.COMPASSIONATE_CARING, ToneType.WARM_SUPPORTIVE]

        tone = tone_manager._select_secondary_tone(
            secondary_tones, conversation_context, None
        )

        assert tone in secondary_tones
