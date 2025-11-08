"""
Tests for behavioral adaptation engine module.

Tests response style adaptation, content selection adaptation,
and overall behavioral adaptation based on user preferences.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.learning.adaptation import (
    BehavioralAdaptationEngine,
    ResponseStyleAdapter,
    ContentSelectionAdapter,
    AdaptationType,
    AdaptationStrategy,
    Adaptation,
)
from morgan.learning.preferences import UserPreferenceProfile, PreferenceCategory
from morgan.emotional.models import (
    ConversationContext,
    CompanionProfile,
    CommunicationStyle,
    ResponseLength,
)


class TestResponseStyleAdapter:
    """Test response style adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create response style adapter."""
        return ResponseStyleAdapter()

    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return CompanionProfile(
            user_id="test_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=30),
            interaction_count=15,
            trust_level=0.7,
            engagement_score=0.75,
        )

    @pytest.fixture
    def preference_profile(self):
        """Create test preference profile."""
        profile = UserPreferenceProfile(
            user_id="test_user",
            preferences={},
            confidence_scores={},
            preference_sources={},
        )
        profile.preferences["communication"] = {
            "formality_level": "casual",
            "technical_depth": "high",
        }
        profile.confidence_scores["communication"] = {
            "formality_level": 0.8,
            "technical_depth": 0.75,
        }
        return profile

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="Can you explain this technical concept?",
            timestamp=datetime.utcnow(),
        )

    def test_adapt_response_style(
        self, adapter, user_profile, preference_profile, conversation_context
    ):
        """Test adapting response style."""
        adaptations = adapter.adapt_response_style(
            "test_user", conversation_context, user_profile, preference_profile
        )

        assert isinstance(adaptations, list)

    def test_adapt_formality_to_casual(self, adapter, user_profile, preference_profile):
        """Test adapting formality to casual."""
        adaptations = adapter._adapt_formality_level(
            "test_user", preference_profile, user_profile
        )

        if adaptations:
            assert all(isinstance(a, Adaptation) for a in adaptations)

    def test_adapt_technical_depth(
        self, adapter, preference_profile, conversation_context
    ):
        """Test adapting technical depth."""
        adaptations = adapter._adapt_technical_depth(
            "test_user", preference_profile, conversation_context
        )

        assert isinstance(adaptations, list)

    def test_adapt_response_length(self, adapter, user_profile, preference_profile):
        """Test adapting response length."""
        preference_profile.preferences["content"] = {"response_length": "brief"}
        preference_profile.confidence_scores["content"] = {"response_length": 0.8}

        adaptations = adapter._adapt_response_length(
            "test_user", preference_profile, user_profile
        )

        assert isinstance(adaptations, list)


class TestContentSelectionAdapter:
    """Test content selection adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create content selection adapter."""
        return ContentSelectionAdapter()

    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return CompanionProfile(
            user_id="test_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=30),
            interaction_count=15,
            trust_level=0.7,
            engagement_score=0.75,
        )

    @pytest.fixture
    def preference_profile(self):
        """Create test preference profile."""
        profile = UserPreferenceProfile(
            user_id="test_user",
            preferences={},
            confidence_scores={},
            preference_sources={},
        )
        profile.preferences["topics"] = {
            "interest_python": 0.8,
            "interest_machine_learning": 0.7,
            "learning_focus": True,
        }
        profile.confidence_scores["topics"] = {
            "interest_python": 0.85,
            "interest_machine_learning": 0.75,
            "learning_focus": 0.9,
        }
        return profile

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="Tell me about Python programming",
            timestamp=datetime.utcnow(),
        )

    def test_adapt_content_selection(
        self, adapter, user_profile, preference_profile, conversation_context
    ):
        """Test adapting content selection."""
        adaptations = adapter.adapt_content_selection(
            "test_user", conversation_context, user_profile, preference_profile
        )

        assert isinstance(adaptations, list)

    def test_adapt_search_weighting_with_interests(
        self, adapter, preference_profile, conversation_context
    ):
        """Test adapting search weighting with topic interests."""
        adaptations = adapter._adapt_search_weighting(
            "test_user", preference_profile, conversation_context
        )

        assert isinstance(adaptations, list)
        if adaptations:
            assert any(
                a.adaptation_type == AdaptationType.SEARCH_WEIGHTING
                for a in adaptations
            )

    def test_adapt_content_filtering(self, adapter, user_profile, preference_profile):
        """Test adapting content filtering."""
        preference_profile.preferences["content"] = {"examples_preferred": True}
        preference_profile.confidence_scores["content"] = {"examples_preferred": 0.8}

        adaptations = adapter._adapt_content_filtering(
            "test_user", preference_profile, user_profile
        )

        assert isinstance(adaptations, list)


class TestBehavioralAdaptationEngine:
    """Test behavioral adaptation engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create behavioral adaptation engine."""
        return BehavioralAdaptationEngine()

    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return CompanionProfile(
            user_id="test_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=30),
            interaction_count=15,
            trust_level=0.7,
            engagement_score=0.75,
        )

    @pytest.fixture
    def preference_profile(self):
        """Create test preference profile."""
        profile = UserPreferenceProfile(
            user_id="test_user",
            preferences={},
            confidence_scores={},
            preference_sources={},
        )
        profile.preferences["communication"] = {
            "formality_level": "casual",
            "technical_depth": "high",
        }
        profile.confidence_scores["communication"] = {
            "formality_level": 0.8,
            "technical_depth": 0.75,
        }
        return profile

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="Can you help me understand this?",
            timestamp=datetime.utcnow(),
        )

    def test_adapt_behavior(
        self, engine, user_profile, preference_profile, conversation_context
    ):
        """Test adapting behavior."""
        result = engine.adapt_behavior(
            "test_user", conversation_context, user_profile, preference_profile
        )

        assert result is not None
        assert result.user_id == "test_user"
        assert isinstance(result.adaptations, list)
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0

    def test_adapt_behavior_empty_preferences(
        self, engine, user_profile, conversation_context
    ):
        """Test adapting behavior with empty preferences."""
        empty_profile = UserPreferenceProfile(
            user_id="test_user",
            preferences={},
            confidence_scores={},
            preference_sources={},
        )

        result = engine.adapt_behavior(
            "test_user", conversation_context, user_profile, empty_profile
        )

        assert result is not None
        assert len(result.adaptations) == 0

    def test_update_strategies(self, engine):
        """Test updating adaptation strategies."""
        changes = [
            {"type": "response_style", "parameter": "formality", "adjustment": 0.2}
        ]

        # Should not raise error
        engine.update_strategies("test_user", changes)

    def test_get_adaptation_history(
        self, engine, user_profile, preference_profile, conversation_context
    ):
        """Test getting adaptation history."""
        # First create some history
        engine.adapt_behavior(
            "test_user", conversation_context, user_profile, preference_profile
        )

        history = engine.get_adaptation_history("test_user")

        assert isinstance(history, list)
        assert len(history) > 0

    def test_get_adaptation_history_no_history(self, engine):
        """Test getting adaptation history for user with no history."""
        history = engine.get_adaptation_history("unknown_user")

        assert isinstance(history, list)
        assert len(history) == 0

    def test_generate_adaptation_reasoning(self, engine):
        """Test generating adaptation reasoning."""
        adaptations = [
            Adaptation(
                adaptation_id="test1",
                adaptation_type=AdaptationType.RESPONSE_STYLE,
                parameter="formality",
                old_value="formal",
                new_value="casual",
                confidence=0.8,
                strategy=AdaptationStrategy.GRADUAL,
                reasoning="Test reasoning",
                expected_impact="Test impact",
            ),
            Adaptation(
                adaptation_id="test2",
                adaptation_type=AdaptationType.CONTENT_SELECTION,
                parameter="examples",
                old_value=False,
                new_value=True,
                confidence=0.7,
                strategy=AdaptationStrategy.IMMEDIATE,
                reasoning="Test reasoning 2",
                expected_impact="Test impact 2",
            ),
        ]

        reasoning = engine._generate_adaptation_reasoning(adaptations)

        assert reasoning is not None
        assert len(reasoning) > 0
        assert "2 total adaptations" in reasoning
