"""
Unit tests for the User Profile module.

Tests cover:
- Profile creation and updates
- Profile persistence
- Metrics calculation
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from morgan_server.personalization.profile import (
    ProfileManager,
    UserProfile,
    CommunicationStyle,
    ResponseLength,
)


class TestUserProfileCreation:
    """Test user profile creation."""
    
    def test_create_profile_basic(self):
        """Test creating a basic profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile("user123")
            
            assert profile.user_id == "user123"
            assert profile.preferred_name is None
            assert profile.communication_style == CommunicationStyle.FRIENDLY
            assert profile.response_length == ResponseLength.MODERATE
            assert profile.topics_of_interest == []
            assert profile.trust_level == 0.0
            assert profile.engagement_score == 0.0
            assert profile.interaction_count == 0
    
    def test_create_profile_with_preferences(self):
        """Test creating a profile with preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile(
                user_id="user123",
                preferred_name="Alice",
                communication_style=CommunicationStyle.PROFESSIONAL,
                response_length=ResponseLength.DETAILED,
                topics_of_interest=["AI", "programming"]
            )
            
            assert profile.user_id == "user123"
            assert profile.preferred_name == "Alice"
            assert profile.communication_style == CommunicationStyle.PROFESSIONAL
            assert profile.response_length == ResponseLength.DETAILED
            assert "AI" in profile.topics_of_interest
            assert "programming" in profile.topics_of_interest
    
    def test_create_duplicate_profile_raises_error(self):
        """Test that creating duplicate profile raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile("user123")

            with pytest.raises(
                ValueError, match="Profile already exists"
            ):
                manager.create_profile("user123")
    
    def test_profile_has_timestamps(self):
        """Test that profile has creation and update timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            before = datetime.now()
            profile = manager.create_profile("user123")
            after = datetime.now()
            
            assert before <= profile.created_at <= after
            assert before <= profile.last_updated <= after
    
    def test_get_or_create_profile_creates_new(self):
        """Test get_or_create creates new profile if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.get_or_create_profile("user123")
            
            assert profile.user_id == "user123"
            assert "user123" in manager.profiles
    
    def test_get_or_create_profile_returns_existing(self):
        """Test get_or_create returns existing profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            # Create profile
            original = manager.create_profile(
                "user123", preferred_name="Alice"
            )
            
            # Get or create should return existing
            profile = manager.get_or_create_profile("user123")
            
            assert profile.user_id == "user123"
            assert profile.preferred_name == "Alice"
            assert profile is original


class TestUserProfileRetrieval:
    """Test user profile retrieval."""
    
    def test_get_profile_exists(self):
        """Test getting an existing profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123", preferred_name="Alice")
            
            profile = manager.get_profile("user123")
            
            assert profile is not None
            assert profile.user_id == "user123"
            assert profile.preferred_name == "Alice"
    
    def test_get_profile_not_exists(self):
        """Test getting a non-existent profile returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.get_profile("nonexistent")
            
            assert profile is None
    
    def test_list_profiles(self):
        """Test listing all profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user1")
            manager.create_profile("user2")
            manager.create_profile("user3")
            
            profiles = manager.list_profiles()
            
            assert len(profiles) == 3
            assert "user1" in profiles
            assert "user2" in profiles
            assert "user3" in profiles
    
    def test_list_profiles_empty(self):
        """Test listing profiles when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profiles = manager.list_profiles()
            
            assert profiles == []


class TestUserProfileUpdates:
    """Test user profile updates."""
    
    def test_update_preferred_name(self):
        """Test updating preferred name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123", preferred_name="Alice")
            
            updated = manager.update_profile("user123", preferred_name="Alicia")
            
            assert updated.preferred_name == "Alicia"
    
    def test_update_communication_style(self):
        """Test updating communication style."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_profile(
                "user123",
                communication_style=CommunicationStyle.TECHNICAL
            )
            
            assert updated.communication_style == CommunicationStyle.TECHNICAL
    
    def test_update_response_length(self):
        """Test updating response length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_profile(
                "user123",
                response_length=ResponseLength.BRIEF
            )
            
            assert updated.response_length == ResponseLength.BRIEF
    
    def test_update_topics_of_interest(self):
        """Test updating topics of interest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123", topics_of_interest=["AI"])
            
            updated = manager.update_profile(
                "user123",
                topics_of_interest=["AI", "programming", "music"]
            )
            
            assert len(updated.topics_of_interest) == 3
            assert "AI" in updated.topics_of_interest
            assert "programming" in updated.topics_of_interest
            assert "music" in updated.topics_of_interest
    
    def test_update_metadata(self):
        """Test updating metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_profile(
                "user123",
                metadata={"favorite_color": "blue", "timezone": "UTC"}
            )
            
            assert updated.metadata["favorite_color"] == "blue"
            assert updated.metadata["timezone"] == "UTC"
    
    def test_update_multiple_fields(self):
        """Test updating multiple fields at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_profile(
                "user123",
                preferred_name="Alice",
                communication_style=CommunicationStyle.CASUAL,
                topics_of_interest=["AI"]
            )
            
            assert updated.preferred_name == "Alice"
            assert updated.communication_style == CommunicationStyle.CASUAL
            assert "AI" in updated.topics_of_interest
    
    def test_update_nonexistent_profile_raises_error(self):
        """Test that updating non-existent profile raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            with pytest.raises(ValueError, match="Profile not found"):
                manager.update_profile(
                    "nonexistent", preferred_name="Alice"
                )
    
    def test_update_updates_timestamp(self):
        """Test that update changes last_updated timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile("user123")
            original_timestamp = profile.last_updated
            
            # Wait a tiny bit to ensure timestamp difference
            import time
            time.sleep(0.01)
            
            updated = manager.update_profile("user123", preferred_name="Alice")
            
            assert updated.last_updated > original_timestamp
    
    def test_partial_update_preserves_other_fields(self):
        """Test that partial update preserves other fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile(
                "user123",
                preferred_name="Alice",
                communication_style=CommunicationStyle.PROFESSIONAL,
                topics_of_interest=["AI"]
            )

            # Update only preferred name
            updated = manager.update_profile(
                "user123", preferred_name="Alicia"
            )

            # Other fields should be preserved
            assert updated.preferred_name == "Alicia"
            assert (
                updated.communication_style
                == CommunicationStyle.PROFESSIONAL
            )
            assert "AI" in updated.topics_of_interest


class TestProfileMetrics:
    """Test profile metrics."""
    
    def test_update_trust_level(self):
        """Test updating trust level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_metrics("user123", trust_level=0.75)
            
            assert updated.trust_level == 0.75
    
    def test_update_engagement_score(self):
        """Test updating engagement score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_metrics("user123", engagement_score=0.85)
            
            assert updated.engagement_score == 0.85
    
    def test_update_relationship_age_days(self):
        """Test updating relationship age."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_metrics("user123", relationship_age_days=30)
            
            assert updated.relationship_age_days == 30
    
    def test_update_interaction_count(self):
        """Test updating interaction count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_metrics("user123", interaction_count=50)
            
            assert updated.interaction_count == 50
    
    def test_update_multiple_metrics(self):
        """Test updating multiple metrics at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            updated = manager.update_metrics(
                "user123",
                trust_level=0.8,
                engagement_score=0.9,
                relationship_age_days=45,
                interaction_count=100
            )
            
            assert updated.trust_level == 0.8
            assert updated.engagement_score == 0.9
            assert updated.relationship_age_days == 45
            assert updated.interaction_count == 100
    
    def test_update_metrics_invalid_trust_level(self):
        """Test that invalid trust level raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile("user123")

            with pytest.raises(
                ValueError, match="Trust level must be between"
            ):
                manager.update_metrics("user123", trust_level=1.5)

            with pytest.raises(
                ValueError, match="Trust level must be between"
            ):
                manager.update_metrics("user123", trust_level=-0.1)
    
    def test_update_metrics_invalid_engagement_score(self):
        """Test that invalid engagement score raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile("user123")

            with pytest.raises(
                ValueError, match="Engagement score must be between"
            ):
                manager.update_metrics("user123", engagement_score=1.5)

            with pytest.raises(
                ValueError, match="Engagement score must be between"
            ):
                manager.update_metrics("user123", engagement_score=-0.1)
    
    def test_update_metrics_invalid_relationship_age(self):
        """Test that negative relationship age raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile("user123")

            with pytest.raises(
                ValueError,
                match="Relationship age days must be non-negative"
            ):
                manager.update_metrics("user123", relationship_age_days=-1)
    
    def test_update_metrics_invalid_interaction_count(self):
        """Test that negative interaction count raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)

            manager.create_profile("user123")

            with pytest.raises(
                ValueError,
                match="Interaction count must be non-negative"
            ):
                manager.update_metrics("user123", interaction_count=-1)
    
    def test_update_metrics_nonexistent_profile(self):
        """Test that updating metrics for non-existent profile raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            with pytest.raises(ValueError, match="Profile not found"):
                manager.update_metrics("nonexistent", trust_level=0.5)
    
    def test_calculate_relationship_age(self):
        """Test calculating relationship age."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            # Create profile
            profile = manager.create_profile("user123")
            
            # Manually set created_at to 10 days ago
            profile.created_at = datetime.now() - timedelta(days=10)
            
            age = manager.calculate_relationship_age("user123")
            
            # Should be approximately 10 days
            assert 9 <= age <= 11
    
    def test_calculate_relationship_age_new_profile(self):
        """Test calculating relationship age for new profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            age = manager.calculate_relationship_age("user123")
            
            # Should be 0 for new profile
            assert age == 0
    
    def test_calculate_relationship_age_nonexistent(self):
        """Test calculating relationship age for non-existent profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            age = manager.calculate_relationship_age("nonexistent")
            
            assert age == 0


class TestProfilePersistence:
    """Test profile persistence."""
    
    def test_profile_saved_to_disk(self):
        """Test that profile is saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123", preferred_name="Alice")
            
            # Check file exists
            profile_files = list(Path(tmpdir).glob("*.json"))
            assert len(profile_files) == 1
    
    def test_profile_loaded_from_disk(self):
        """Test that profile is loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save profile
            manager1 = ProfileManager(storage_dir=tmpdir)
            manager1.create_profile("user123", preferred_name="Alice")

            # Create new manager (should load from disk)
            manager2 = ProfileManager(storage_dir=tmpdir)

            profile = manager2.get_profile("user123")
            assert profile is not None
            assert profile.user_id == "user123"
            assert profile.preferred_name == "Alice"
    
    def test_profile_update_persisted(self):
        """Test that profile updates are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and update profile
            manager1 = ProfileManager(storage_dir=tmpdir)
            manager1.create_profile("user123", preferred_name="Alice")
            manager1.update_profile("user123", preferred_name="Alicia")
            
            # Load from disk
            manager2 = ProfileManager(storage_dir=tmpdir)
            
            profile = manager2.get_profile("user123")
            assert profile.preferred_name == "Alicia"
    
    def test_metrics_update_persisted(self):
        """Test that metrics updates are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and update metrics
            manager1 = ProfileManager(storage_dir=tmpdir)
            manager1.create_profile("user123")
            manager1.update_metrics(
                "user123", trust_level=0.75, interaction_count=50
            )

            # Load from disk
            manager2 = ProfileManager(storage_dir=tmpdir)

            profile = manager2.get_profile("user123")
            assert profile.trust_level == 0.75
            assert profile.interaction_count == 50
    
    def test_multiple_profiles_persisted(self):
        """Test that multiple profiles are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple profiles
            manager1 = ProfileManager(storage_dir=tmpdir)
            manager1.create_profile("user1", preferred_name="Alice")
            manager1.create_profile("user2", preferred_name="Bob")
            manager1.create_profile("user3", preferred_name="Charlie")
            
            # Load from disk
            manager2 = ProfileManager(storage_dir=tmpdir)
            
            assert len(manager2.list_profiles()) == 3
            assert manager2.get_profile("user1").preferred_name == "Alice"
            assert manager2.get_profile("user2").preferred_name == "Bob"
            assert manager2.get_profile("user3").preferred_name == "Charlie"
    
    def test_profile_deletion_removes_file(self):
        """Test that deleting profile removes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            # Verify file exists
            profile_files = list(Path(tmpdir).glob("*.json"))
            assert len(profile_files) == 1
            
            # Delete profile
            manager.delete_profile("user123")
            
            # Verify file removed
            profile_files = list(Path(tmpdir).glob("*.json"))
            assert len(profile_files) == 0
    
    def test_delete_profile_success(self):
        """Test successful profile deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            manager.create_profile("user123")
            
            result = manager.delete_profile("user123")
            
            assert result is True
            assert manager.get_profile("user123") is None
    
    def test_delete_nonexistent_profile(self):
        """Test deleting non-existent profile returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            result = manager.delete_profile("nonexistent")
            
            assert result is False
    
    def test_profile_serialization_deserialization(self):
        """Test profile to_dict and from_dict."""
        profile = UserProfile(
            user_id="user123",
            preferred_name="Alice",
            communication_style=CommunicationStyle.PROFESSIONAL,
            response_length=ResponseLength.DETAILED,
            topics_of_interest=["AI", "programming"],
            trust_level=0.75,
            engagement_score=0.85,
            relationship_age_days=30,
            interaction_count=50
        )
        
        # Serialize
        data = profile.to_dict()
        
        # Deserialize
        restored = UserProfile.from_dict(data)
        
        assert restored.user_id == profile.user_id
        assert restored.preferred_name == profile.preferred_name
        assert restored.communication_style == profile.communication_style
        assert restored.response_length == profile.response_length
        assert restored.topics_of_interest == profile.topics_of_interest
        assert restored.trust_level == profile.trust_level
        assert restored.engagement_score == profile.engagement_score
        assert restored.relationship_age_days == profile.relationship_age_days
        assert restored.interaction_count == profile.interaction_count


class TestProfileEdgeCases:
    """Test edge cases and error handling."""
    
    def test_profile_with_special_characters_in_id(self):
        """Test profile with special characters in user ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            # User ID with special characters
            user_id = "user@example.com"
            
            profile = manager.create_profile(user_id)
            
            assert profile.user_id == user_id
            
            # Should be able to retrieve
            retrieved = manager.get_profile(user_id)
            assert retrieved is not None
            assert retrieved.user_id == user_id
    
    def test_profile_with_empty_topics(self):
        """Test profile with empty topics list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile("user123", topics_of_interest=[])
            
            assert profile.topics_of_interest == []
    
    def test_profile_with_none_values(self):
        """Test profile with None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile(
                "user123",
                preferred_name=None
            )
            
            assert profile.preferred_name is None
    
    def test_profile_metadata_empty(self):
        """Test profile with empty metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile("user123")
            
            assert profile.metadata == {}
    
    def test_profile_metadata_update_merges(self):
        """Test that metadata updates merge with existing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(storage_dir=tmpdir)
            
            profile = manager.create_profile("user123")
            manager.update_profile("user123", metadata={"key1": "value1"})
            manager.update_profile("user123", metadata={"key2": "value2"})
            
            profile = manager.get_profile("user123")
            assert profile.metadata["key1"] == "value1"
            assert profile.metadata["key2"] == "value2"
