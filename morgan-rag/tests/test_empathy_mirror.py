"""
Tests for emotional mirroring and reflection module.

Tests emotional mirroring, reflection prompts, emotional echoing,
and comprehensive emotional reflection capabilities.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from morgan.intelligence.empathy.mirror import EmotionalMirror
from morgan.intelligence.core.models import EmotionalState, EmotionType, ConversationContext


class TestEmotionalMirror:
    """Test emotional mirror functionality."""

    @pytest.fixture
    def mirror(self):
        """Create emotional mirror for testing."""
        with patch("morgan.intelligence.empathy.mirror.get_llm_service") as mock_llm:
            # Configure mock to return proper response with .content attribute
            mock_response = Mock()
            mock_response.content = "I can really feel the emotion you're expressing and I'm here with you."
            mock_llm.return_value.generate.return_value = mock_response
            with patch("morgan.intelligence.empathy.mirror.get_settings"):
                return EmotionalMirror()

    @pytest.fixture
    def emotional_state_joy(self):
        """Create sample emotional state - joy."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def emotional_state_anger(self):
        """Create sample emotional state - anger."""
        return EmotionalState(
            primary_emotion=EmotionType.ANGER,
            intensity=0.75,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def conversation_context(self):
        """Create sample conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm so frustrated with this situation!",
            timestamp=datetime.now(timezone.utc),
        )

    def test_mirror_emotion_high_intensity(
        self, mirror, emotional_state_joy, conversation_context
    ):
        """Test mirroring emotion with high intensity."""
        response = mirror.mirror_emotion(
            emotional_state_joy, conversation_context, mirroring_intensity=0.8
        )

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    def test_mirror_emotion_low_intensity(self, mirror, conversation_context):
        """Test mirroring emotion with low intensity."""
        low_intensity_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.3,
            confidence=0.6,
            timestamp=datetime.now(timezone.utc),
        )

        response = mirror.mirror_emotion(
            low_intensity_state, conversation_context, mirroring_intensity=0.5
        )

        assert response is not None
        assert len(response) > 0

    def test_create_reflection_prompt_sadness(self, mirror, conversation_context):
        """Test creating reflection prompt for sadness."""
        sadness_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

        prompt = mirror.create_reflection_prompt(sadness_state, conversation_context)

        assert prompt is not None
        assert len(prompt) > 0
        assert "?" in prompt  # Should be a question

    def test_create_reflection_prompt_joy(
        self, mirror, emotional_state_joy, conversation_context
    ):
        """Test creating reflection prompt for joy."""
        prompt = mirror.create_reflection_prompt(
            emotional_state_joy, conversation_context
        )

        assert prompt is not None
        assert isinstance(prompt, str)

    def test_generate_emotional_reflection_light(
        self, mirror, emotional_state_anger, conversation_context
    ):
        """Test generating light emotional reflection."""
        reflection = mirror.generate_emotional_reflection(
            emotional_state_anger, conversation_context, reflection_depth="light"
        )

        assert isinstance(reflection, dict)
        assert "emotion_mirror" in reflection
        assert "reflection_prompt" in reflection
        assert "emotional_patterns" in reflection

    def test_generate_emotional_reflection_medium(
        self, mirror, emotional_state_anger, conversation_context
    ):
        """Test generating medium emotional reflection."""
        reflection = mirror.generate_emotional_reflection(
            emotional_state_anger, conversation_context, reflection_depth="medium"
        )

        assert isinstance(reflection, dict)
        assert "emotional_triggers" in reflection
        assert "emotional_needs" in reflection
        assert "relationship_impact" in reflection

    def test_generate_emotional_reflection_deep(
        self, mirror, emotional_state_joy, conversation_context
    ):
        """Test generating deep emotional reflection."""
        reflection = mirror.generate_emotional_reflection(
            emotional_state_joy, conversation_context, reflection_depth="deep"
        )

        assert isinstance(reflection, dict)
        assert "underlying_values" in reflection
        assert "personal_growth_insights" in reflection
        assert "future_emotional_preparation" in reflection

    def test_create_empathetic_echo(self, mirror, emotional_state_anger):
        """Test creating empathetic echo."""
        user_words = "I'm feeling really angry about this unfair treatment"

        echo = mirror.create_empathetic_echo(emotional_state_anger, user_words)

        assert echo is not None
        assert len(echo) > 0

    def test_identify_emotional_patterns_high_intensity(
        self, mirror, conversation_context
    ):
        """Test identifying emotional patterns for high intensity."""
        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.FEAR,
            intensity=0.85,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
        )

        patterns = mirror._identify_emotional_patterns(
            high_intensity_state, conversation_context
        )

        assert isinstance(patterns, list)
        assert "high_emotional_intensity" in patterns

    def test_identify_growth_opportunities_sadness(self, mirror):
        """Test identifying growth opportunities for sadness."""
        sadness_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )

        opportunities = mirror._identify_growth_opportunities(sadness_state)

        assert isinstance(opportunities, list)
        assert len(opportunities) > 0

    def test_suggest_coping_strategies_anger(self, mirror, emotional_state_anger):
        """Test suggesting coping strategies for anger."""
        strategies = mirror._suggest_coping_strategies(emotional_state_anger)

        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_calculate_reflection_confidence(self, mirror, emotional_state_joy):
        """Test calculating reflection confidence."""
        confidence = mirror._calculate_reflection_confidence(emotional_state_joy)

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_identify_emotional_triggers(self, mirror, conversation_context):
        """Test identifying emotional triggers from context."""
        triggers = mirror._identify_emotional_triggers(conversation_context)

        assert isinstance(triggers, list)

    def test_identify_emotional_needs_joy(self, mirror, emotional_state_joy):
        """Test identifying emotional needs for joy."""
        needs = mirror._identify_emotional_needs(emotional_state_joy)

        assert isinstance(needs, list)
        assert len(needs) > 0

    def test_identify_emotional_needs_fear(self, mirror):
        """Test identifying emotional needs for fear."""
        fear_state = EmotionalState(
            primary_emotion=EmotionType.FEAR,
            intensity=0.7,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )

        needs = mirror._identify_emotional_needs(fear_state)

        assert "safety" in needs or "reassurance" in needs or "control" in needs

    def test_analyze_relationship_impact(
        self, mirror, emotional_state_anger, conversation_context
    ):
        """Test analyzing relationship impact of emotions."""
        impact = mirror._analyze_relationship_impact(
            emotional_state_anger, conversation_context
        )

        assert isinstance(impact, dict)
        assert "communication_style" in impact
        assert "relationship_needs" in impact
        assert "connection_opportunities" in impact

    def test_extract_emotional_phrases(self, mirror, emotional_state_anger):
        """Test extracting emotional phrases from user words."""
        user_words = "I'm feeling really angry and frustrated right now"

        phrases = mirror._extract_emotional_phrases(user_words, emotional_state_anger)

        assert isinstance(phrases, list)

    def test_get_mirroring_phrase(self, mirror, emotional_state_joy):
        """Test getting mirroring phrase for emotion."""
        phrase = mirror._get_mirroring_phrase(emotional_state_joy)

        assert phrase is not None
        assert len(phrase) > 0

    def test_get_intensity_modifier_high(self, mirror):
        """Test getting intensity modifier for high intensity."""
        modifier = mirror._get_intensity_modifier(0.85)

        assert modifier is not None
        assert "strong" in modifier or "powerful" in modifier

    def test_get_intensity_modifier_low(self, mirror):
        """Test getting intensity modifier for low intensity."""
        modifier = mirror._get_intensity_modifier(0.3)

        assert modifier is not None
        assert "gentle" in modifier or "subtle" in modifier or "quiet" in modifier
