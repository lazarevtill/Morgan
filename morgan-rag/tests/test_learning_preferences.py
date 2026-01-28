"""
Tests for preference extraction and storage module.

Tests preference extraction from interactions, preference storage,
preference profile management, and preference confidence tracking.
"""

import pytest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from morgan.learning.preferences import (
    PreferenceExtractor,
    PreferenceStorage,
    UserPreferenceProfile,
    PreferenceCategory,
    PreferenceSource,
    PreferenceUpdate,
)
from morgan.intelligence.core.models import (
    InteractionData,
    ConversationContext,
    EmotionalState,
    EmotionType,
)


class TestPreferenceExtractor:
    """Test preference extractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create preference extractor."""
        return PreferenceExtractor()

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interactions for testing."""
        interactions = []
        messages = [
            "Please help me with this technical concept",
            "I prefer brief and simple explanations",
            "Can you provide examples?",
            "I'm interested in learning about machine learning",
            "Thank you for the detailed explanation",
        ]

        for i, msg in enumerate(messages):
            context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conv",
                message_text=msg,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            interaction = InteractionData(
                conversation_context=context,
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
            interactions.append(interaction)

        return interactions

    def test_extract_preferences(self, extractor, sample_interactions):
        """Test extracting preferences from interactions."""
        updates = extractor.extract_preferences("test_user", sample_interactions)

        assert isinstance(updates, list)
        assert all(isinstance(u, PreferenceUpdate) for u in updates)

    def test_extract_communication_preferences_formal(self, extractor):
        """Test extracting formal communication preferences."""
        formal_interactions = []
        for i in range(5):
            context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conv",
                message_text="Please kindly assist me with this matter. Thank you.",
                timestamp=datetime.now(timezone.utc),
            )
            interaction = InteractionData(
                conversation_context=context,
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
            formal_interactions.append(interaction)

        updates = extractor._extract_communication_preferences(
            "test_user", formal_interactions
        )

        assert isinstance(updates, list)

    def test_extract_communication_preferences_casual(self, extractor):
        """Test extracting casual communication preferences."""
        casual_interactions = []
        for i in range(5):
            context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conv",
                message_text="Hey! Yeah, that's awesome! Cool!",
                timestamp=datetime.now(timezone.utc),
            )
            interaction = InteractionData(
                conversation_context=context,
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
            casual_interactions.append(interaction)

        updates = extractor._extract_communication_preferences(
            "test_user", casual_interactions
        )

        assert isinstance(updates, list)

    def test_extract_content_preferences(self, extractor, sample_interactions):
        """Test extracting content preferences."""
        updates = extractor._extract_content_preferences(
            "test_user", sample_interactions
        )

        assert isinstance(updates, list)

    def test_extract_topic_preferences(self, extractor, sample_interactions):
        """Test extracting topic preferences."""
        updates = extractor._extract_topic_preferences("test_user", sample_interactions)

        assert isinstance(updates, list)

    def test_extract_learning_preferences(self, extractor, sample_interactions):
        """Test extracting learning preferences."""
        updates = extractor._extract_learning_preferences(
            "test_user", sample_interactions
        )

        assert isinstance(updates, list)

    def test_extract_interaction_preferences(self, extractor, sample_interactions):
        """Test extracting interaction preferences."""
        updates = extractor._extract_interaction_preferences(
            "test_user", sample_interactions
        )

        assert isinstance(updates, list)

    def test_extract_emotional_preferences(self, extractor):
        """Test extracting emotional preferences."""
        interactions = []
        for i in range(5):
            context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conv",
                message_text="Test message",
                timestamp=datetime.now(timezone.utc),
            )
            interaction = InteractionData(
                conversation_context=context,
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.JOY,
                    intensity=0.7,
                    confidence=0.8,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
            interactions.append(interaction)

        updates = extractor._extract_emotional_preferences("test_user", interactions)

        assert isinstance(updates, list)

    def test_calculate_pattern_score(self, extractor):
        """Test calculating pattern match score."""
        messages = ["Please help me", "Thank you", "I appreciate your assistance"]
        patterns = [r"\b(please|thank you|appreciate)\b"]

        score = extractor._calculate_pattern_score(messages, patterns)

        assert score >= 0.0
        assert score <= 1.0


class TestPreferenceStorage:
    """Test preference storage functionality."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create preference storage with temporary directory."""
        with patch("morgan.learning.preferences.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            return PreferenceStorage()

    @pytest.fixture
    def sample_profile(self):
        """Create sample user preference profile."""
        profile = UserPreferenceProfile(
            user_id="test_user",
            preferences={
                "communication": {
                    "formality_level": "casual",
                    "technical_depth": "high",
                }
            },
            confidence_scores={
                "communication": {"formality_level": 0.8, "technical_depth": 0.75}
            },
            preference_sources={},
        )
        return profile

    def test_get_user_preferences_new_user(self, storage):
        """Test getting preferences for new user."""
        profile = storage.get_user_preferences("new_user")

        assert profile is not None
        assert profile.user_id == "new_user"
        assert isinstance(profile.preferences, dict)

    def test_save_and_load_preferences(self, storage, sample_profile):
        """Test saving and loading user preferences."""
        # Save profile
        storage.save_user_preferences(sample_profile)

        # Load profile
        loaded_profile = storage.get_user_preferences("test_user")

        assert loaded_profile.user_id == "test_user"
        assert (
            loaded_profile.preferences["communication"]["formality_level"] == "casual"
        )

    def test_update_preference(self, storage):
        """Test updating a user preference."""
        update = PreferenceUpdate(
            update_id="test_update",
            user_id="test_user",
            category=PreferenceCategory.COMMUNICATION,
            preference_key="formality_level",
            preference_value="formal",
            confidence_score=0.8,
            source=PreferenceSource.EXPLICIT_FEEDBACK,
            evidence=["User stated preference"],
        )

        storage.update_preference("test_user", update)

        profile = storage.get_user_preferences("test_user")
        assert (
            profile.get_preference(PreferenceCategory.COMMUNICATION, "formality_level")
            == "formal"
        )

    def test_get_preference_history(self, storage):
        """Test getting preference update history."""
        update = PreferenceUpdate(
            update_id="test_update",
            user_id="test_user",
            category=PreferenceCategory.CONTENT,
            preference_key="response_length",
            preference_value="brief",
            confidence_score=0.7,
            source=PreferenceSource.CONVERSATION_ANALYSIS,
            evidence=["Brief response pattern detected"],
        )

        storage.update_preference("test_user", update)

        history = storage.get_preference_history("test_user")

        assert isinstance(history, list)

    def test_clear_user_preferences(self, storage, sample_profile):
        """Test clearing user preferences."""
        # First save preferences
        storage.save_user_preferences(sample_profile)

        # Clear preferences
        storage.clear_user_preferences("test_user")

        # Check that new profile is created when getting preferences
        profile = storage.get_user_preferences("test_user")
        assert len(profile.preferences) == 0


class TestUserPreferenceProfile:
    """Test user preference profile functionality."""

    @pytest.fixture
    def profile(self):
        """Create test preference profile."""
        return UserPreferenceProfile(
            user_id="test_user",
            preferences={
                "communication": {
                    "formality_level": "casual",
                    "technical_depth": "high",
                },
                "content": {"response_length": "brief"},
            },
            confidence_scores={
                "communication": {"formality_level": 0.8, "technical_depth": 0.75},
                "content": {"response_length": 0.7},
            },
            preference_sources={
                "communication": {
                    "formality_level": PreferenceSource.CONVERSATION_ANALYSIS,
                    "technical_depth": PreferenceSource.CONVERSATION_ANALYSIS,
                },
                "content": {"response_length": PreferenceSource.CONVERSATION_ANALYSIS},
            },
        )

    def test_get_preference_existing(self, profile):
        """Test getting existing preference."""
        value = profile.get_preference(
            PreferenceCategory.COMMUNICATION, "formality_level"
        )

        assert value == "casual"

    def test_get_preference_missing(self, profile):
        """Test getting missing preference with default."""
        value = profile.get_preference(
            PreferenceCategory.TOPICS, "interest_python", default="unknown"
        )

        assert value == "unknown"

    def test_get_confidence_existing(self, profile):
        """Test getting confidence for existing preference."""
        confidence = profile.get_confidence(
            PreferenceCategory.COMMUNICATION, "formality_level"
        )

        assert confidence == 0.8

    def test_get_confidence_missing(self, profile):
        """Test getting confidence for missing preference."""
        confidence = profile.get_confidence(
            PreferenceCategory.TOPICS, "interest_python"
        )

        assert confidence == 0.0

    def test_set_preference_new_category(self, profile):
        """Test setting preference in new category."""
        profile.set_preference(
            PreferenceCategory.TOPICS,
            "interest_python",
            0.9,
            0.85,
            PreferenceSource.IMPLICIT_BEHAVIOR,
        )

        assert (
            profile.get_preference(PreferenceCategory.TOPICS, "interest_python") == 0.9
        )
        assert (
            profile.get_confidence(PreferenceCategory.TOPICS, "interest_python") == 0.85
        )

    def test_set_preference_update_existing(self, profile):
        """Test updating existing preference."""
        profile.set_preference(
            PreferenceCategory.COMMUNICATION,
            "formality_level",
            "formal",
            0.9,
            PreferenceSource.EXPLICIT_FEEDBACK,
        )

        assert (
            profile.get_preference(PreferenceCategory.COMMUNICATION, "formality_level")
            == "formal"
        )
        assert (
            profile.get_confidence(PreferenceCategory.COMMUNICATION, "formality_level")
            == 0.9
        )


class TestPreferenceUpdate:
    """Test preference update model."""

    def test_preference_update_creation(self):
        """Test creating preference update."""
        update = PreferenceUpdate(
            update_id="test",
            user_id="test_user",
            category=PreferenceCategory.COMMUNICATION,
            preference_key="formality_level",
            preference_value="casual",
            confidence_score=0.8,
            source=PreferenceSource.CONVERSATION_ANALYSIS,
            evidence=["Casual language detected"],
        )

        assert update.update_id == "test"
        assert update.category == PreferenceCategory.COMMUNICATION

    def test_preference_update_auto_id(self):
        """Test preference update with auto-generated ID."""
        update = PreferenceUpdate(
            update_id="",
            user_id="test_user",
            category=PreferenceCategory.CONTENT,
            preference_key="response_length",
            preference_value="brief",
            confidence_score=0.7,
            source=PreferenceSource.IMPLICIT_BEHAVIOR,
            evidence=[],
        )

        assert update.update_id is not None
        assert len(update.update_id) > 0
