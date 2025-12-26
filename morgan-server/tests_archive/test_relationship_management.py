"""
Unit tests for the Relationship Management system.
"""

import pytest
from datetime import datetime
from morgan_server.empathic.relationships import (
    RelationshipManager,
    InteractionType,
    MilestoneType,
    Milestone,
)


class TestRelationshipManager:
    """Test suite for RelationshipManager."""

    def test_initialization(self):
        """Test relationship manager initialization."""
        manager = RelationshipManager()
        assert manager.interaction_window_days == 90
        assert len(manager.profiles) == 0

        manager_custom = RelationshipManager(interaction_window_days=30)
        assert manager_custom.interaction_window_days == 30

    def test_track_interaction_creates_profile(self):
        """Test that tracking an interaction creates a profile."""
        manager = RelationshipManager()
        user_id = "user123"

        interaction = manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT, sentiment="positive"
        )

        assert user_id in manager.profiles
        assert interaction.user_id == user_id
        assert interaction.interaction_type == InteractionType.CHAT
        assert interaction.sentiment == "positive"

    def test_track_multiple_interactions(self):
        """Test tracking multiple interactions."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track 5 interactions
        for _ in range(5):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        profile = manager.get_profile(user_id)
        assert profile is not None
        # +1 for the first conversation milestone interaction
        assert len(profile.interactions) == 5

    def test_first_conversation_milestone(self):
        """Test that first conversation milestone is created."""
        manager = RelationshipManager()
        user_id = "user123"

        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        milestones = manager.get_milestones(user_id)
        assert len(milestones) == 1
        assert milestones[0].milestone_type == MilestoneType.FIRST_CONVERSATION

    def test_conversation_count_milestones(self):
        """Test conversation count milestones."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track 10 interactions to trigger milestone
        for _ in range(10):
            manager.track_interaction(
                user_id=user_id, interaction_type=InteractionType.CHAT
            )

        milestones = manager.get_milestones(user_id)
        milestone_types = [m.milestone_type for m in milestones]

        assert MilestoneType.FIRST_CONVERSATION in milestone_types
        assert MilestoneType.CONVERSATIONS_10 in milestone_types

    def test_trust_level_calculation_empty(self):
        """Test trust level calculation with no interactions."""
        manager = RelationshipManager()
        trust = manager.calculate_trust_level("nonexistent_user")
        assert trust == 0.0

    def test_trust_level_increases_with_interactions(self):
        """Test that trust level increases with more interactions."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track 1 interaction
        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT, sentiment="positive"
        )
        trust_1 = manager.calculate_trust_level(user_id)

        # Track 9 more interactions (total 10)
        for _ in range(9):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )
        trust_10 = manager.calculate_trust_level(user_id)

        assert trust_10 > trust_1
        assert 0.0 <= trust_1 <= 1.0
        assert 0.0 <= trust_10 <= 1.0

    def test_trust_level_affected_by_sentiment(self):
        """Test that trust level is affected by sentiment."""
        manager = RelationshipManager()
        user_positive = "user_positive"
        user_negative = "user_negative"

        # Track positive interactions
        for _ in range(10):
            manager.track_interaction(
                user_id=user_positive,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        # Track negative interactions
        for _ in range(10):
            manager.track_interaction(
                user_id=user_negative,
                interaction_type=InteractionType.CHAT,
                sentiment="negative",
            )

        trust_positive = manager.calculate_trust_level(user_positive)
        trust_negative = manager.calculate_trust_level(user_negative)

        assert trust_positive > trust_negative

    def test_trust_level_affected_by_emotional_support(self):
        """Test that emotional support interactions increase trust."""
        manager = RelationshipManager()
        user_with_support = "user_support"
        user_without_support = "user_no_support"

        # Track interactions with emotional support
        for _ in range(5):
            manager.track_interaction(
                user_id=user_with_support,
                interaction_type=InteractionType.EMOTIONAL_SUPPORT,
                sentiment="positive",
            )

        # Track regular interactions
        for _ in range(5):
            manager.track_interaction(
                user_id=user_without_support,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        trust_with_support = manager.calculate_trust_level(user_with_support)
        trust_without_support = manager.calculate_trust_level(user_without_support)

        assert trust_with_support > trust_without_support

    def test_engagement_score_calculation_empty(self):
        """Test engagement score with no interactions."""
        manager = RelationshipManager()
        engagement = manager.calculate_engagement_score("nonexistent_user")
        assert engagement == 0.0

    def test_engagement_score_increases_with_variety(self):
        """Test that engagement increases with variety of interactions."""
        manager = RelationshipManager()
        user_varied = "user_varied"
        user_monotone = "user_monotone"

        # Track varied interactions
        interaction_types = [
            InteractionType.CHAT,
            InteractionType.QUESTION,
            InteractionType.LEARNING,
            InteractionType.FEEDBACK,
        ]
        for interaction_type in interaction_types:
            manager.track_interaction(
                user_id=user_varied, interaction_type=interaction_type
            )

        # Track monotone interactions
        for _ in range(4):
            manager.track_interaction(
                user_id=user_monotone, interaction_type=InteractionType.CHAT
            )

        engagement_varied = manager.calculate_engagement_score(user_varied)
        engagement_monotone = manager.calculate_engagement_score(user_monotone)

        assert engagement_varied > engagement_monotone

    def test_relationship_depth_calculation(self):
        """Test relationship depth calculation."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track some interactions
        for _ in range(10):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        depth = manager.calculate_relationship_depth(user_id)
        assert 0.0 <= depth <= 1.0
        assert depth > 0.0  # Should have some depth with 10 interactions

    def test_relationship_depth_combines_metrics(self):
        """Test that relationship depth combines trust and engagement."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track varied, positive interactions
        interaction_types = [
            InteractionType.CHAT,
            InteractionType.QUESTION,
            InteractionType.LEARNING,
            InteractionType.EMOTIONAL_SUPPORT,
        ]
        for _ in range(5):
            for interaction_type in interaction_types:
                manager.track_interaction(
                    user_id=user_id,
                    interaction_type=interaction_type,
                    sentiment="positive",
                )

        depth = manager.calculate_relationship_depth(user_id)
        trust = manager.calculate_trust_level(user_id)
        engagement = manager.calculate_engagement_score(user_id)

        # Depth should be influenced by both trust and engagement
        assert depth > 0.0
        assert depth <= max(trust, engagement)

    def test_get_milestones(self):
        """Test getting milestones for a user."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track interactions to trigger milestones
        for _ in range(15):
            manager.track_interaction(
                user_id=user_id, interaction_type=InteractionType.CHAT
            )

        milestones = manager.get_milestones(user_id)
        assert len(milestones) >= 2  # First conversation + 10 conversations

    def test_get_recent_milestones(self):
        """Test getting recent milestones."""
        manager = RelationshipManager()
        user_id = "user123"

        # Create profile and add interaction
        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        # Get recent milestones (should include first conversation)
        recent = manager.get_recent_milestones(user_id, days=7)
        assert len(recent) >= 1
        assert recent[0].milestone_type == MilestoneType.FIRST_CONVERSATION

    def test_get_metrics(self):
        """Test getting relationship metrics."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track some interactions
        for _ in range(5):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        metrics = manager.get_metrics(user_id)
        assert metrics is not None
        assert metrics.user_id == user_id
        assert metrics.total_interactions == 5
        assert metrics.trust_level >= 0.0
        assert metrics.engagement_score >= 0.0
        assert metrics.relationship_depth >= 0.0
        assert metrics.interaction_frequency > 0.0
        assert metrics.positive_interaction_ratio > 0.0

    def test_get_profile(self):
        """Test getting complete relationship profile."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track interaction
        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        profile = manager.get_profile(user_id)
        assert profile is not None
        assert profile.user_id == user_id
        assert len(profile.interactions) > 0
        assert len(profile.milestones) > 0
        assert profile.metrics is not None

    def test_get_profile_nonexistent_user(self):
        """Test getting profile for nonexistent user."""
        manager = RelationshipManager()
        profile = manager.get_profile("nonexistent_user")
        assert profile is None

    def test_celebrate_milestone(self):
        """Test milestone celebration message generation."""
        manager = RelationshipManager()

        milestone = Milestone(
            milestone_type=MilestoneType.CONVERSATIONS_10,
            achieved_at=datetime.now(),
            description="10 conversations",
        )

        message = manager.celebrate_milestone(milestone)
        assert isinstance(message, str)
        assert len(message) > 0
        assert "10" in message or "conversations" in message.lower()

    def test_milestone_celebration_messages_unique(self):
        """Test that different milestones have different messages."""
        manager = RelationshipManager()

        milestone_10 = Milestone(
            milestone_type=MilestoneType.CONVERSATIONS_10,
            achieved_at=datetime.now(),
            description="10 conversations",
        )

        milestone_50 = Milestone(
            milestone_type=MilestoneType.CONVERSATIONS_50,
            achieved_at=datetime.now(),
            description="50 conversations",
        )

        message_10 = manager.celebrate_milestone(milestone_10)
        message_50 = manager.celebrate_milestone(milestone_50)

        assert message_10 != message_50

    def test_interaction_with_metadata(self):
        """Test tracking interaction with metadata."""
        manager = RelationshipManager()
        user_id = "user123"

        metadata = {"topic": "programming", "duration_seconds": 120, "quality": "high"}

        interaction = manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT, metadata=metadata
        )

        assert interaction.metadata == metadata

    def test_interaction_with_context(self):
        """Test tracking interaction with context."""
        manager = RelationshipManager()
        user_id = "user123"

        context = "User asked about Python programming"

        interaction = manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.QUESTION, context=context
        )

        assert interaction.context == context

    def test_metrics_updated_after_interaction(self):
        """Test that metrics are updated after each interaction."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track first interaction
        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        metrics_1 = manager.get_metrics(user_id)
        count_1 = metrics_1.total_interactions

        # Track second interaction
        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        metrics_2 = manager.get_metrics(user_id)
        count_2 = metrics_2.total_interactions

        assert count_2 > count_1

    def test_trust_milestone_triggered(self):
        """Test that trust milestones are triggered."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track many positive interactions to build trust
        for _ in range(50):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.EMOTIONAL_SUPPORT,
                sentiment="positive",
            )

        milestones = manager.get_milestones(user_id)
        milestone_types = [m.milestone_type for m in milestones]

        # Should have at least one trust milestone
        trust_milestones = [
            MilestoneType.TRUST_MILESTONE_50,
            MilestoneType.TRUST_MILESTONE_75,
            MilestoneType.TRUST_MILESTONE_90,
        ]

        has_trust_milestone = any(mt in milestone_types for mt in trust_milestones)
        assert has_trust_milestone

    def test_engagement_milestone_triggered(self):
        """Test that engagement milestone is triggered."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track varied interactions frequently
        interaction_types = list(InteractionType)
        for _ in range(20):
            for interaction_type in interaction_types:
                manager.track_interaction(
                    user_id=user_id,
                    interaction_type=interaction_type,
                    sentiment="positive",
                )

        milestones = manager.get_milestones(user_id)
        milestone_types = [m.milestone_type for m in milestones]

        # Should have engagement milestone
        assert MilestoneType.ENGAGEMENT_HIGH in milestone_types

    def test_multiple_users_isolated(self):
        """Test that multiple users are properly isolated."""
        manager = RelationshipManager()
        user1 = "user1"
        user2 = "user2"

        # Track interactions for user1
        for _ in range(5):
            manager.track_interaction(
                user_id=user1, interaction_type=InteractionType.CHAT
            )

        # Track interactions for user2
        for _ in range(10):
            manager.track_interaction(
                user_id=user2, interaction_type=InteractionType.CHAT
            )

        profile1 = manager.get_profile(user1)
        profile2 = manager.get_profile(user2)

        assert profile1.user_id == user1
        assert profile2.user_id == user2
        assert len(profile1.interactions) == 5
        assert len(profile2.interactions) == 10

    def test_relationship_age_days_calculated(self):
        """Test that relationship age in days is calculated."""
        manager = RelationshipManager()
        user_id = "user123"

        manager.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT
        )

        metrics = manager.get_metrics(user_id)
        assert metrics.relationship_age_days >= 1

    def test_interaction_frequency_calculated(self):
        """Test that interaction frequency is calculated."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track 10 interactions
        for _ in range(10):
            manager.track_interaction(
                user_id=user_id, interaction_type=InteractionType.CHAT
            )

        metrics = manager.get_metrics(user_id)
        assert metrics.interaction_frequency > 0.0
        # Should be high since all interactions are on the same day
        assert metrics.interaction_frequency >= 10.0

    def test_positive_interaction_ratio_calculated(self):
        """Test that positive interaction ratio is calculated."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track 7 positive and 3 negative interactions
        for _ in range(7):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="positive",
            )

        for _ in range(3):
            manager.track_interaction(
                user_id=user_id,
                interaction_type=InteractionType.CHAT,
                sentiment="negative",
            )

        metrics = manager.get_metrics(user_id)
        assert 0.6 <= metrics.positive_interaction_ratio <= 0.8

    def test_recent_activity_trend_calculated(self):
        """Test that recent activity trend is calculated."""
        manager = RelationshipManager()
        user_id = "user123"

        # Track enough interactions to calculate trend
        for _ in range(15):
            manager.track_interaction(
                user_id=user_id, interaction_type=InteractionType.CHAT
            )

        metrics = manager.get_metrics(user_id)
        assert metrics.recent_activity_trend in ["increasing", "stable", "decreasing"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
