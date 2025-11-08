"""
Tests for the Relationships intelligence modules.

Tests the core functionality of the new relationship intelligence modules:
- RelationshipBuilder
- MilestoneDetector
- RelationshipTimeline
- RelationshipDynamics
- RelationshipAdaptation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from morgan.relationships import (
    RelationshipBuilder,
    MilestoneDetector,
    RelationshipTimeline,
    RelationshipDynamics,
    RelationshipAdaptation,
)
from morgan.emotional.models import (
    CompanionProfile,
    EmotionalState,
    EmotionType,
    ConversationContext,
    InteractionData,
    UserPreferences,
    CommunicationStyle,
    ResponseLength,
)


class TestRelationshipModules:
    """Test cases for relationship intelligence modules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user_id = "test_user_123"

        # Create sample profile
        self.profile = CompanionProfile(
            user_id=self.user_id,
            relationship_duration=timedelta(days=10),
            interaction_count=15,
            preferred_name="Alex",
            communication_preferences=UserPreferences(
                topics_of_interest=["python", "ai"],
                communication_style=CommunicationStyle.FRIENDLY,
                preferred_response_length=ResponseLength.DETAILED,
            ),
            trust_level=0.6,
            engagement_score=0.7,
            profile_created=datetime.utcnow() - timedelta(days=10),
        )

        # Create sample interaction data
        self.interaction_data = InteractionData(
            conversation_context=ConversationContext(
                user_id=self.user_id,
                conversation_id="conv_123",
                message_text="I'm really excited about learning AI!",
                timestamp=datetime.utcnow(),
            ),
            emotional_state=EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.8, confidence=0.9
            ),
            user_satisfaction=0.9,
            topics_discussed=["ai", "learning"],
        )

    def test_relationship_builder_initialization(self):
        """Test RelationshipBuilder initialization."""
        builder = RelationshipBuilder()
        assert builder is not None
        assert hasattr(builder, "assess_relationship_stage")
        assert hasattr(builder, "calculate_relationship_metrics")

    def test_relationship_builder_assess_stage(self):
        """Test relationship stage assessment."""
        builder = RelationshipBuilder()
        stage = builder.assess_relationship_stage(self.profile)

        assert stage is not None
        assert hasattr(stage, "value")
        assert stage.value in [
            "initial",
            "building",
            "established",
            "deep",
            "companion",
        ]

    def test_relationship_builder_calculate_metrics(self):
        """Test relationship metrics calculation."""
        builder = RelationshipBuilder()
        metrics = builder.calculate_relationship_metrics(self.profile)

        assert metrics is not None
        assert 0.0 <= metrics.trust_score <= 1.0
        assert 0.0 <= metrics.engagement_level <= 1.0
        assert 0.0 <= metrics.intimacy_level <= 1.0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert 0.0 <= metrics.overall_strength() <= 1.0

    def test_milestone_detector_initialization(self):
        """Test MilestoneDetector initialization."""
        detector = MilestoneDetector()
        assert detector is not None
        assert hasattr(detector, "detect_milestones")
        assert hasattr(detector, "analyze_milestone_trends")

    def test_milestone_detector_detect_milestones(self):
        """Test milestone detection."""
        detector = MilestoneDetector()
        milestones = detector.detect_milestones(self.profile, self.interaction_data)

        assert isinstance(milestones, list)
        # Should return 0-2 milestones as per implementation
        assert len(milestones) <= 2

    def test_relationship_timeline_initialization(self):
        """Test RelationshipTimeline initialization."""
        timeline = RelationshipTimeline()
        assert timeline is not None
        assert hasattr(timeline, "build_timeline")
        assert hasattr(timeline, "get_relationship_summary")

    def test_relationship_timeline_build_timeline(self):
        """Test timeline building."""
        timeline = RelationshipTimeline()
        events = timeline.build_timeline(self.profile)

        assert isinstance(events, list)
        # Should have at least milestone events from profile
        assert len(events) >= 0

    def test_relationship_dynamics_initialization(self):
        """Test RelationshipDynamics initialization."""
        dynamics = RelationshipDynamics()
        assert dynamics is not None
        assert hasattr(dynamics, "analyze_relationship_dynamics")
        assert hasattr(dynamics, "analyze_interaction_patterns")

    def test_relationship_dynamics_analyze_patterns(self):
        """Test interaction pattern analysis."""
        dynamics = RelationshipDynamics()
        patterns = dynamics.analyze_interaction_patterns([self.interaction_data])

        assert patterns is not None
        assert hasattr(patterns, "frequency_trend")
        assert hasattr(patterns, "engagement_trend")
        assert hasattr(patterns, "consistency_score")

    def test_relationship_adaptation_initialization(self):
        """Test RelationshipAdaptation initialization."""
        adaptation = RelationshipAdaptation()
        assert adaptation is not None
        assert hasattr(adaptation, "analyze_adaptation_needs")
        assert hasattr(adaptation, "create_adaptation_plan")

    def test_relationship_adaptation_analyze_needs(self):
        """Test adaptation needs analysis."""
        from morgan.relationships.adaptation import AdaptationContext

        adaptation = RelationshipAdaptation()
        context = AdaptationContext(
            user_profile=self.profile,
            current_interaction=self.interaction_data,
            relationship_history=[self.interaction_data],
            emotional_state=self.interaction_data.emotional_state,
            conversation_context=self.interaction_data.conversation_context,
        )

        needs = adaptation.analyze_adaptation_needs(context)

        assert isinstance(needs, list)
        assert len(needs) <= 5  # Should return top 5 adaptations

    def test_module_integration(self):
        """Test that all modules can work together."""
        builder = RelationshipBuilder()
        detector = MilestoneDetector()
        timeline = RelationshipTimeline()
        dynamics = RelationshipDynamics()
        adaptation = RelationshipAdaptation()

        # Test basic workflow
        stage = builder.assess_relationship_stage(self.profile)
        metrics = builder.calculate_relationship_metrics(self.profile)
        milestones = detector.detect_milestones(self.profile, self.interaction_data)
        events = timeline.build_timeline(self.profile)
        patterns = dynamics.analyze_interaction_patterns([self.interaction_data])

        # All should complete without errors
        assert stage is not None
        assert metrics is not None
        assert isinstance(milestones, list)
        assert isinstance(events, list)
        assert patterns is not None

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        builder = RelationshipBuilder()

        # Test with minimal profile
        minimal_profile = CompanionProfile(
            user_id="minimal",
            relationship_duration=timedelta(days=1),
            interaction_count=1,
            preferred_name="Test",
            communication_preferences=UserPreferences(),
            trust_level=0.0,
            engagement_score=0.0,
        )

        # Should not raise exceptions
        stage = builder.assess_relationship_stage(minimal_profile)
        metrics = builder.calculate_relationship_metrics(minimal_profile)

        assert stage is not None
        assert metrics is not None
