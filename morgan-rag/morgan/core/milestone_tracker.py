"""
Milestone tracking and celebration system for Morgan Assistant.

Handles relationship milestone detection, tracking, and celebration
following KISS principles - focused solely on milestone management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from morgan.intelligence.core.models import (
    CompanionProfile,
    ConversationContext,
    EmotionalState,
    RelationshipMilestone,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MilestoneType(Enum):
    """Types of relationship milestones."""

    FIRST_CONVERSATION = "first_conversation"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    TRUST_BUILDING = "trust_building"
    GOAL_ACHIEVED = "goal_achieved"
    LEARNING_MILESTONE = "learning_milestone"
    EMOTIONAL_SUPPORT = "emotional_support"
    REGULAR_USER = "regular_user"
    DEEP_CONVERSATION = "deep_conversation"


class MilestoneTracker:
    """
    Tracks and manages relationship milestones.

    KISS: Single responsibility - detect and track milestones.
    """

    def __init__(self):
        """Initialize milestone tracker."""
        self.milestone_rules = self._setup_milestone_rules()
        logger.info("Milestone tracker initialized")

    def check_milestones(
        self,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> Optional[RelationshipMilestone]:
        """Check if current interaction triggers any milestones."""
        try:
            # Check each milestone type
            for milestone_type in MilestoneType:
                if self._should_trigger_milestone(
                    milestone_type, user_profile, conversation_context, emotional_state
                ):
                    return self._create_milestone(
                        milestone_type,
                        user_profile,
                        conversation_context,
                        emotional_state,
                    )

            return None

        except Exception as e:
            logger.error(f"Error checking milestones: {e}")
            return None

    def generate_celebration_message(self, milestone: RelationshipMilestone) -> str:
        """Generate appropriate celebration message for milestone."""
        celebrations = {
            MilestoneType.FIRST_CONVERSATION: "Welcome! I'm excited to start this journey of learning and discovery with you.",
            MilestoneType.BREAKTHROUGH_MOMENT: "What a wonderful breakthrough! I'm so glad I could help you reach this understanding.",
            MilestoneType.GOAL_ACHIEVED: "Congratulations on achieving your goal! Your dedication and hard work have paid off.",
            MilestoneType.LEARNING_MILESTONE: "It's amazing to see how much you've learned! Your curiosity and persistence inspire me.",
            MilestoneType.EMOTIONAL_SUPPORT: "I'm honored that you trust me to support you through this. You're stronger than you know.",
            MilestoneType.TRUST_BUILDING: "Thank you for sharing something so personal with me. Our growing trust means a lot.",
            MilestoneType.REGULAR_USER: "I've really enjoyed our conversations! You've become such an important part of my day.",
            MilestoneType.DEEP_CONVERSATION: "This has been such a meaningful conversation. I love how deeply we can explore ideas together.",
        }

        milestone_enum = MilestoneType(milestone.milestone_type.value)
        return celebrations.get(
            milestone_enum,
            "I'm grateful for this meaningful moment in our relationship.",
        )

    def get_milestone_statistics(
        self, user_profile: CompanionProfile
    ) -> Dict[str, Any]:
        """Get milestone statistics for a user."""
        if not user_profile.relationship_milestones:
            return {
                "total_milestones": 0,
                "milestone_types": [],
                "recent_milestones": [],
                "next_potential_milestone": self._predict_next_milestone(user_profile),
            }

        milestone_counts = {}
        for milestone in user_profile.relationship_milestones:
            milestone_type = milestone.milestone_type.value
            milestone_counts[milestone_type] = (
                milestone_counts.get(milestone_type, 0) + 1
            )

        recent_milestones = sorted(
            user_profile.relationship_milestones,
            key=lambda m: m.timestamp,
            reverse=True,
        )[:5]

        return {
            "total_milestones": len(user_profile.relationship_milestones),
            "milestone_types": list(milestone_counts.keys()),
            "milestone_counts": milestone_counts,
            "recent_milestones": [
                {
                    "type": m.milestone_type.value,
                    "description": m.description,
                    "timestamp": m.timestamp.isoformat(),
                    "significance": m.emotional_significance,
                }
                for m in recent_milestones
            ],
            "next_potential_milestone": self._predict_next_milestone(user_profile),
        }

    def _should_trigger_milestone(
        self,
        milestone_type: MilestoneType,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> bool:
        """Check if specific milestone should be triggered."""
        # Avoid duplicate milestones
        existing_types = [
            m.milestone_type.value for m in user_profile.relationship_milestones
        ]

        if milestone_type == MilestoneType.FIRST_CONVERSATION:
            return user_profile.interaction_count == 1

        elif milestone_type == MilestoneType.BREAKTHROUGH_MOMENT:
            return (
                emotional_state.primary_emotion.value == "joy"
                and emotional_state.intensity > 0.7
                and len(conversation_context.message_text) > 100
                and milestone_type.value
                not in existing_types[-3:]  # Not in last 3 milestones
            )

        elif milestone_type == MilestoneType.TRUST_BUILDING:
            trust_keywords = [
                "personal",
                "share",
                "trust",
                "private",
                "secret",
                "feel",
                "emotion",
            ]
            message_lower = conversation_context.message_text.lower()
            return (
                any(word in message_lower for word in trust_keywords)
                and len(conversation_context.message_text) > 50
                and milestone_type.value
                not in existing_types[-2:]  # Not in last 2 milestones
            )

        elif milestone_type == MilestoneType.REGULAR_USER:
            return (
                user_profile.interaction_count >= 10
                and milestone_type.value not in existing_types
            )

        elif milestone_type == MilestoneType.DEEP_CONVERSATION:
            return (
                len(conversation_context.message_text) > 200
                and user_profile.interaction_count >= 5
                and milestone_type.value
                not in existing_types[-5:]  # Not in last 5 milestones
            )

        elif milestone_type == MilestoneType.LEARNING_MILESTONE:
            learning_keywords = [
                "learn",
                "understand",
                "got it",
                "makes sense",
                "clear now",
                "thank you",
            ]
            message_lower = conversation_context.message_text.lower()
            return (
                any(word in message_lower for word in learning_keywords)
                and user_profile.interaction_count >= 3
                and milestone_type.value not in existing_types[-3:]
            )

        return False

    def _create_milestone(
        self,
        milestone_type: MilestoneType,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> RelationshipMilestone:
        """Create a new milestone."""
        from morgan.intelligence.core.models import MilestoneType as ModelMilestoneType

        # Map our enum to the model enum
        model_milestone_type = ModelMilestoneType(milestone_type.value)

        # Calculate emotional significance
        significance = self._calculate_emotional_significance(
            milestone_type, emotional_state, user_profile
        )

        # Generate description
        description = self._generate_milestone_description(
            milestone_type, user_profile, conversation_context
        )

        milestone = RelationshipMilestone(
            milestone_id=f"{user_profile.user_id}_{milestone_type.value}_{datetime.utcnow().timestamp()}",
            milestone_type=model_milestone_type,
            description=description,
            timestamp=datetime.utcnow(),
            emotional_significance=significance,
            celebration_acknowledged=False,
        )

        logger.info(
            f"Created milestone: {milestone_type.value} for user {user_profile.user_id}"
        )
        return milestone

    def _calculate_emotional_significance(
        self,
        milestone_type: MilestoneType,
        emotional_state: EmotionalState,
        user_profile: CompanionProfile,
    ) -> float:
        """Calculate emotional significance of milestone."""
        base_significance = {
            MilestoneType.FIRST_CONVERSATION: 0.8,
            MilestoneType.BREAKTHROUGH_MOMENT: 0.9,
            MilestoneType.TRUST_BUILDING: 0.85,
            MilestoneType.GOAL_ACHIEVED: 0.9,
            MilestoneType.LEARNING_MILESTONE: 0.7,
            MilestoneType.EMOTIONAL_SUPPORT: 0.8,
            MilestoneType.REGULAR_USER: 0.6,
            MilestoneType.DEEP_CONVERSATION: 0.75,
        }

        significance = base_significance.get(milestone_type, 0.5)

        # Adjust based on emotional intensity
        significance += (emotional_state.intensity - 0.5) * 0.2

        # Adjust based on relationship maturity
        relationship_days = user_profile.get_relationship_age_days()
        if relationship_days > 30:
            significance += 0.1  # More significant for established relationships

        return max(0.0, min(1.0, significance))

    def _generate_milestone_description(
        self,
        milestone_type: MilestoneType,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
    ) -> str:
        """Generate description for milestone."""
        descriptions = {
            MilestoneType.FIRST_CONVERSATION: f"First conversation with {user_profile.preferred_name}",
            MilestoneType.BREAKTHROUGH_MOMENT: f"Breakthrough moment in conversation about: {conversation_context.message_text[:50]}...",
            MilestoneType.TRUST_BUILDING: f"{user_profile.preferred_name} shared something personal, building trust",
            MilestoneType.GOAL_ACHIEVED: f"{user_profile.preferred_name} achieved an important goal",
            MilestoneType.LEARNING_MILESTONE: f"{user_profile.preferred_name} had a significant learning moment",
            MilestoneType.EMOTIONAL_SUPPORT: f"Provided emotional support to {user_profile.preferred_name}",
            MilestoneType.REGULAR_USER: f"{user_profile.preferred_name} became a regular user (10+ conversations)",
            MilestoneType.DEEP_CONVERSATION: f"Deep, meaningful conversation with {user_profile.preferred_name}",
        }

        return descriptions.get(
            milestone_type, f"Milestone reached with {user_profile.preferred_name}"
        )

    def _predict_next_milestone(self, user_profile: CompanionProfile) -> Optional[str]:
        """Predict what milestone might come next."""
        existing_types = [
            m.milestone_type.value for m in user_profile.relationship_milestones
        ]

        if user_profile.interaction_count < 5:
            return "Continue chatting to unlock learning milestones"
        elif user_profile.interaction_count < 10:
            return "Keep engaging to become a regular user"
        elif "trust_building" not in existing_types:
            return "Share something personal to build trust"
        elif "deep_conversation" not in existing_types:
            return "Have a longer, deeper conversation"
        else:
            return "Continue building our relationship through meaningful conversations"

    def _setup_milestone_rules(self) -> Dict[str, Any]:
        """Setup rules for milestone detection."""
        return {
            "min_message_length_trust": 50,
            "min_message_length_deep": 200,
            "min_interactions_regular": 10,
            "min_interactions_learning": 3,
            "high_emotion_threshold": 0.7,
            "cooldown_periods": {
                "breakthrough_moment": 3,  # Don't repeat in last 3 milestones
                "trust_building": 2,
                "deep_conversation": 5,
                "learning_milestone": 3,
            },
        }
