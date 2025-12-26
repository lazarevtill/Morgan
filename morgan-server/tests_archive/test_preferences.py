"""
Unit tests for the Preferences module.

Tests cover:
- Preference storage and retrieval
- Preference learning
- Preference application
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from morgan_server.personalization.preferences import (
    PreferenceManager,
    UserPreferences,
    PreferenceScore,
    InteractionFeedback,
    PreferenceType,
)


class TestPreferenceCreation:
    """Test preference creation."""

    def test_create_preferences_basic(self):
        """Test creating basic preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.create_preferences("user123")

            assert prefs.user_id == "user123"
            assert prefs.communication_style_scores == {}
            assert prefs.response_length_scores == {}
            assert prefs.topic_interests == {}
            assert prefs.interaction_history == []

    def test_create_duplicate_preferences_raises_error(self):
        """Test that creating duplicate preferences raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            with pytest.raises(ValueError, match="Preferences already exist"):
                manager.create_preferences("user123")

    def test_preferences_have_timestamps(self):
        """Test that preferences have creation and update timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            before = datetime.now()
            prefs = manager.create_preferences("user123")
            after = datetime.now()

            assert before <= prefs.created_at <= after
            assert before <= prefs.last_updated <= after

    def test_get_or_create_preferences_creates_new(self):
        """Test get_or_create creates new preferences if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.get_or_create_preferences("user123")

            assert prefs.user_id == "user123"
            assert "user123" in manager.preferences

    def test_get_or_create_preferences_returns_existing(self):
        """Test get_or_create returns existing preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Create preferences
            original = manager.create_preferences("user123")

            # Get or create should return existing
            prefs = manager.get_or_create_preferences("user123")

            assert prefs.user_id == "user123"
            assert prefs is original


class TestPreferenceRetrieval:
    """Test preference retrieval."""

    def test_get_preferences_exists(self):
        """Test getting existing preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            prefs = manager.get_preferences("user123")

            assert prefs is not None
            assert prefs.user_id == "user123"

    def test_get_preferences_not_exists(self):
        """Test getting non-existent preferences returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.get_preferences("nonexistent")

            assert prefs is None

    def test_get_preference_summary_exists(self):
        """Test getting preference summary for existing user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            summary = manager.get_preference_summary("user123")

            assert summary["user_id"] == "user123"
            assert summary["exists"] is True
            assert "preferred_communication_style" in summary
            assert "preferred_response_length" in summary
            assert "topic_interests" in summary
            assert "interaction_count" in summary

    def test_get_preference_summary_not_exists(self):
        """Test getting preference summary for non-existent user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            summary = manager.get_preference_summary("nonexistent")

            assert summary["user_id"] == "nonexistent"
            assert summary["exists"] is False


class TestInteractionRecording:
    """Test recording interactions for preference learning."""

    def test_record_interaction_basic(self):
        """Test recording a basic interaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.record_interaction(
                user_id="user123", message_length=100, response_length=300
            )

            assert prefs.user_id == "user123"
            assert len(prefs.interaction_history) == 1
            assert prefs.interaction_history[0].message_length == 100
            assert prefs.interaction_history[0].response_length == 300

    def test_record_interaction_with_topics(self):
        """Test recording interaction with topics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                topics=["AI", "programming"],
            )

            assert len(prefs.interaction_history) == 1
            assert "AI" in prefs.interaction_history[0].topics
            assert "programming" in prefs.interaction_history[0].topics

    def test_record_interaction_with_satisfaction(self):
        """Test recording interaction with satisfaction score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                user_satisfaction=0.8,
            )

            assert prefs.interaction_history[0].user_satisfaction == 0.8

    def test_record_interaction_with_communication_style(self):
        """Test recording interaction with communication style."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                communication_style_used="casual",
            )

            assert prefs.interaction_history[0].communication_style_used == "casual"

    def test_record_interaction_invalid_satisfaction(self):
        """Test that invalid satisfaction score raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            with pytest.raises(ValueError, match="user_satisfaction must be between"):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    user_satisfaction=1.5,
                )

            with pytest.raises(ValueError, match="user_satisfaction must be between"):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    user_satisfaction=-0.1,
                )

    def test_record_multiple_interactions(self):
        """Test recording multiple interactions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.record_interaction(
                user_id="user123", message_length=100, response_length=300
            )
            manager.record_interaction(
                user_id="user123", message_length=150, response_length=400
            )
            manager.record_interaction(
                user_id="user123", message_length=200, response_length=500
            )

            prefs = manager.get_preferences("user123")
            assert len(prefs.interaction_history) == 3

    def test_interaction_history_max_size(self):
        """Test that interaction history respects max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir, max_history_size=5)

            # Record 10 interactions
            for i in range(10):
                manager.record_interaction(
                    user_id="user123", message_length=100 + i, response_length=300 + i
                )

            prefs = manager.get_preferences("user123")
            # Should only keep last 5
            assert len(prefs.interaction_history) == 5
            # Should be the most recent ones
            assert prefs.interaction_history[0].message_length == 105
            assert prefs.interaction_history[-1].message_length == 109


class TestPreferenceLearning:
    """Test preference learning from interactions."""

    def test_learn_communication_style_preference(self):
        """Test learning communication style preference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record multiple interactions with "casual" style
            for _ in range(5):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    communication_style_used="casual",
                    user_satisfaction=0.9,
                )

            prefs = manager.get_preferences("user123")
            assert "casual" in prefs.communication_style_scores
            assert prefs.communication_style_scores["casual"].count == 5
            assert prefs.communication_style_scores["casual"].score > 0.0

    def test_learn_response_length_preference(self):
        """Test learning response length preference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interactions with brief responses
            for _ in range(5):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=150,  # Brief
                    user_satisfaction=0.9,
                )

            prefs = manager.get_preferences("user123")
            assert "brief" in prefs.response_length_scores
            assert prefs.response_length_scores["brief"].count == 5

    def test_learn_topic_interests(self):
        """Test learning topic interests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interactions with AI topic
            for _ in range(5):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    topics=["AI"],
                    user_satisfaction=0.8,
                )

            prefs = manager.get_preferences("user123")
            assert "AI" in prefs.topic_interests
            assert prefs.topic_interests["AI"] > 0.0

    def test_topic_interest_increases_with_satisfaction(self):
        """Test that topic interest increases with high satisfaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interactions with increasing satisfaction
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                topics=["AI"],
                user_satisfaction=0.5,
            )

            initial_score = manager.get_preferences("user123").topic_interests["AI"]

            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                topics=["AI"],
                user_satisfaction=0.9,
            )

            final_score = manager.get_preferences("user123").topic_interests["AI"]

            assert final_score > initial_score

    def test_multiple_topics_tracked(self):
        """Test that multiple topics are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                topics=["AI", "programming", "music"],
                user_satisfaction=0.8,
            )

            prefs = manager.get_preferences("user123")
            assert "AI" in prefs.topic_interests
            assert "programming" in prefs.topic_interests
            assert "music" in prefs.topic_interests


class TestPreferenceRetrieval:
    """Test retrieving learned preferences."""

    def test_get_preferred_communication_style(self):
        """Test getting preferred communication style."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interactions with different styles
            for _ in range(5):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    communication_style_used="casual",
                    user_satisfaction=0.9,
                )

            for _ in range(2):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    communication_style_used="professional",
                    user_satisfaction=0.5,
                )

            preferred = manager.get_preferred_communication_style("user123")
            assert preferred == "casual"

    def test_get_preferred_communication_style_no_preference(self):
        """Test getting communication style when no preference exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            preferred = manager.get_preferred_communication_style("user123")
            assert preferred is None

    def test_get_preferred_response_length(self):
        """Test getting preferred response length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interactions with detailed responses
            for _ in range(5):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=1000,  # Detailed
                    user_satisfaction=0.9,
                )

            preferred = manager.get_preferred_response_length("user123")
            assert preferred == "detailed"

    def test_get_preferred_response_length_no_preference(self):
        """Test getting response length when no preference exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            preferred = manager.get_preferred_response_length("user123")
            assert preferred is None

    def test_get_topic_interests(self):
        """Test getting topic interests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Add topics with different scores
            manager.add_topic_interest("user123", "AI", 0.9)
            manager.add_topic_interest("user123", "programming", 0.7)
            manager.add_topic_interest("user123", "music", 0.5)

            topics = manager.get_topic_interests("user123")

            # Should be sorted by score (descending)
            assert topics == ["AI", "programming", "music"]

    def test_get_topic_interests_with_min_score(self):
        """Test getting topic interests with minimum score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.add_topic_interest("user123", "AI", 0.9)
            manager.add_topic_interest("user123", "programming", 0.7)
            manager.add_topic_interest("user123", "music", 0.3)

            topics = manager.get_topic_interests("user123", min_score=0.5)

            # Should only include topics with score >= 0.5
            assert "AI" in topics
            assert "programming" in topics
            assert "music" not in topics

    def test_get_topic_interests_no_preferences(self):
        """Test getting topic interests when no preferences exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            topics = manager.get_topic_interests("nonexistent")
            assert topics == []


class TestTopicInterestManagement:
    """Test manual topic interest management."""

    def test_add_topic_interest(self):
        """Test adding a topic interest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.add_topic_interest("user123", "AI", 0.8)

            assert "AI" in prefs.topic_interests
            assert prefs.topic_interests["AI"] == 0.8

    def test_add_topic_interest_updates_existing(self):
        """Test that adding topic interest updates existing score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.add_topic_interest("user123", "AI", 0.5)
            prefs = manager.add_topic_interest("user123", "AI", 0.9)

            assert prefs.topic_interests["AI"] == 0.9

    def test_add_topic_interest_invalid_score(self):
        """Test that invalid score raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            with pytest.raises(ValueError, match="score must be between"):
                manager.add_topic_interest("user123", "AI", 1.5)

            with pytest.raises(ValueError, match="score must be between"):
                manager.add_topic_interest("user123", "AI", -0.1)

    def test_remove_topic_interest(self):
        """Test removing a topic interest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.add_topic_interest("user123", "AI", 0.8)
            prefs = manager.remove_topic_interest("user123", "AI")

            assert "AI" not in prefs.topic_interests

    def test_remove_nonexistent_topic(self):
        """Test removing non-existent topic doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")
            prefs = manager.remove_topic_interest("user123", "nonexistent")

            # Should not raise error
            assert "nonexistent" not in prefs.topic_interests

    def test_remove_topic_nonexistent_user(self):
        """Test removing topic for non-existent user raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            with pytest.raises(ValueError, match="Preferences not found"):
                manager.remove_topic_interest("nonexistent", "AI")


class TestPreferencePersistence:
    """Test preference persistence."""

    def test_preferences_saved_to_disk(self):
        """Test that preferences are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            # Check file exists
            prefs_files = list(Path(tmpdir).glob("*_preferences.json"))
            assert len(prefs_files) == 1

    def test_preferences_loaded_from_disk(self):
        """Test that preferences are loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save preferences
            manager1 = PreferenceManager(storage_dir=tmpdir)
            manager1.add_topic_interest("user123", "AI", 0.8)

            # Create new manager (should load from disk)
            manager2 = PreferenceManager(storage_dir=tmpdir)

            prefs = manager2.get_preferences("user123")
            assert prefs is not None
            assert prefs.user_id == "user123"
            assert "AI" in prefs.topic_interests
            assert prefs.topic_interests["AI"] == 0.8

    def test_interaction_history_persisted(self):
        """Test that interaction history is persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record interactions
            manager1 = PreferenceManager(storage_dir=tmpdir)
            manager1.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                topics=["AI"],
            )

            # Load from disk
            manager2 = PreferenceManager(storage_dir=tmpdir)

            prefs = manager2.get_preferences("user123")
            assert len(prefs.interaction_history) == 1
            assert prefs.interaction_history[0].message_length == 100

    def test_learned_preferences_persisted(self):
        """Test that learned preferences are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Learn preferences
            manager1 = PreferenceManager(storage_dir=tmpdir)
            for _ in range(5):
                manager1.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    communication_style_used="casual",
                    user_satisfaction=0.9,
                )

            # Load from disk
            manager2 = PreferenceManager(storage_dir=tmpdir)

            prefs = manager2.get_preferences("user123")
            assert "casual" in prefs.communication_style_scores

    def test_multiple_users_persisted(self):
        """Test that multiple users' preferences are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple preferences
            manager1 = PreferenceManager(storage_dir=tmpdir)
            manager1.add_topic_interest("user1", "AI", 0.8)
            manager1.add_topic_interest("user2", "music", 0.7)
            manager1.add_topic_interest("user3", "sports", 0.9)

            # Load from disk
            manager2 = PreferenceManager(storage_dir=tmpdir)

            assert manager2.get_preferences("user1") is not None
            assert manager2.get_preferences("user2") is not None
            assert manager2.get_preferences("user3") is not None

    def test_preference_deletion_removes_file(self):
        """Test that deleting preferences removes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            # Verify file exists
            prefs_files = list(Path(tmpdir).glob("*_preferences.json"))
            assert len(prefs_files) == 1

            # Delete preferences
            manager.delete_preferences("user123")

            # Verify file removed
            prefs_files = list(Path(tmpdir).glob("*_preferences.json"))
            assert len(prefs_files) == 0

    def test_delete_preferences_success(self):
        """Test successful preference deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            manager.create_preferences("user123")

            result = manager.delete_preferences("user123")

            assert result is True
            assert manager.get_preferences("user123") is None

    def test_delete_nonexistent_preferences(self):
        """Test deleting non-existent preferences returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            result = manager.delete_preferences("nonexistent")

            assert result is False


class TestPreferenceEdgeCases:
    """Test edge cases and error handling."""

    def test_preferences_with_special_characters_in_id(self):
        """Test preferences with special characters in user ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # User ID with special characters
            user_id = "user@example.com"

            prefs = manager.create_preferences(user_id)

            assert prefs.user_id == user_id

            # Should be able to retrieve
            retrieved = manager.get_preferences(user_id)
            assert retrieved is not None
            assert retrieved.user_id == user_id

    def test_empty_topics_list(self):
        """Test recording interaction with empty topics list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            prefs = manager.record_interaction(
                user_id="user123", message_length=100, response_length=300, topics=[]
            )

            assert prefs.interaction_history[0].topics == []

    def test_response_length_categorization(self):
        """Test response length categorization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Brief response
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=150,
                user_satisfaction=0.8,
            )

            # Moderate response
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=500,
                user_satisfaction=0.8,
            )

            # Detailed response
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=1000,
                user_satisfaction=0.8,
            )

            prefs = manager.get_preferences("user123")
            assert "brief" in prefs.response_length_scores
            assert "moderate" in prefs.response_length_scores
            assert "detailed" in prefs.response_length_scores

    def test_preference_score_updates_correctly(self):
        """Test that preference scores update correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record interaction with low satisfaction
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                communication_style_used="casual",
                user_satisfaction=0.3,
            )

            prefs = manager.get_preferences("user123")
            initial_score = prefs.communication_style_scores["casual"].score

            # Record interaction with high satisfaction
            manager.record_interaction(
                user_id="user123",
                message_length=100,
                response_length=300,
                communication_style_used="casual",
                user_satisfaction=0.9,
            )

            prefs = manager.get_preferences("user123")
            final_score = prefs.communication_style_scores["casual"].score

            # Score should increase
            assert final_score > initial_score

    def test_topic_interest_capped_at_one(self):
        """Test that topic interest score is capped at 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreferenceManager(storage_dir=tmpdir)

            # Record many interactions with high satisfaction
            for _ in range(20):
                manager.record_interaction(
                    user_id="user123",
                    message_length=100,
                    response_length=300,
                    topics=["AI"],
                    user_satisfaction=1.0,
                )

            prefs = manager.get_preferences("user123")
            # Score should not exceed 1.0
            assert prefs.topic_interests["AI"] <= 1.0
