"""
Milestone Detection for Morgan Assistant.

Advanced milestone detection system that identifies significant moments
in user relationships and tracks relationship progression.

Requirements: 9.4, 9.5, 10.3
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..emotional.models import (
    CompanionProfile,
    ConversationContext,
    EmotionalState,
    InteractionData,
    MilestoneType,
    RelationshipMilestone,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MilestoneCategory(Enum):
    """Categories of relationship milestones."""

    RELATIONSHIP = "relationship"  # Trust, bonding, connection
    LEARNING = "learning"  # Knowledge, understanding, growth
    EMOTIONAL = "emotional"  # Support, empathy, emotional moments
    ACHIEVEMENT = "achievement"  # Goals, accomplishments, success
    COMMUNICATION = "communication"  # Style, depth, quality improvements


@dataclass
class MilestonePattern:
    """Pattern definition for milestone detection."""

    milestone_type: MilestoneType
    category: MilestoneCategory
    detection_rules: Dict[str, Any]
    significance_base: float
    cooldown_interactions: int
    required_conditions: List[str]


@dataclass
class MilestoneContext:
    """Context information for milestone detection."""

    user_profile: CompanionProfile
    interaction_data: InteractionData
    conversation_history: List[ConversationContext]
    emotional_history: List[EmotionalState]
    recent_milestones: List[RelationshipMilestone]


class MilestoneDetector:
    """
    Advanced milestone detection system.

    Identifies significant moments in user relationships through pattern
    analysis, emotional cues, and interaction quality assessment.
    """

    def __init__(self):
        """Initialize milestone detector."""
        self.milestone_patterns = self._initialize_milestone_patterns()
        self.detection_history = {}  # Track detection patterns per user
        logger.info("Milestone detector initialized with advanced patterns")

    def detect_milestones(
        self,
        user_profile: CompanionProfile,
        interaction_data: InteractionData,
        conversation_history: Optional[List[ConversationContext]] = None,
        emotional_history: Optional[List[EmotionalState]] = None,
    ) -> List[RelationshipMilestone]:
        """
        Detect potential milestones from current interaction.

        Args:
            user_profile: User's companion profile
            interaction_data: Current interaction data
            conversation_history: Recent conversation history
            emotional_history: Recent emotional state history

        Returns:
            List of detected milestones
        """
        if conversation_history is None:
            conversation_history = []
        if emotional_history is None:
            emotional_history = []

        context = MilestoneContext(
            user_profile=user_profile,
            interaction_data=interaction_data,
            conversation_history=conversation_history,
            emotional_history=emotional_history,
            recent_milestones=user_profile.relationship_milestones[
                -5:
            ],  # Last 5 milestones
        )

        detected_milestones = []

        # Check each milestone pattern
        for pattern in self.milestone_patterns:
            if self._should_check_milestone(pattern, context):
                milestone = self._evaluate_milestone_pattern(pattern, context)
                if milestone:
                    detected_milestones.append(milestone)
                    logger.info(
                        f"Detected milestone: {milestone.milestone_type.value} "
                        f"for user {user_profile.user_id}"
                    )

        # Sort by emotional significance
        detected_milestones.sort(key=lambda m: m.emotional_significance, reverse=True)

        # Limit to most significant milestones to avoid overwhelming
        return detected_milestones[:2]

    def analyze_milestone_trends(
        self, user_profile: CompanionProfile, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze milestone trends for a user.

        Args:
            user_profile: User's companion profile
            timeframe_days: Analysis timeframe in days

        Returns:
            Dictionary with milestone trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_milestones = [
            m
            for m in user_profile.relationship_milestones
            if m.timestamp >= cutoff_date
        ]

        if not recent_milestones:
            return {
                "milestone_count": 0,
                "milestone_rate": 0.0,
                "dominant_categories": [],
                "trend_direction": "stable",
                "next_predicted": None,
            }

        # Calculate milestone rate
        milestone_rate = len(recent_milestones) / timeframe_days

        # Analyze categories
        category_counts = {}
        for milestone in recent_milestones:
            pattern = self._get_pattern_for_milestone(milestone.milestone_type)
            if pattern:
                category = pattern.category.value
                category_counts[category] = category_counts.get(category, 0) + 1

        dominant_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Determine trend direction
        if len(recent_milestones) >= 4:
            first_half = recent_milestones[: len(recent_milestones) // 2]
            second_half = recent_milestones[len(recent_milestones) // 2 :]

            first_rate = len(first_half) / (timeframe_days / 2)
            second_rate = len(second_half) / (timeframe_days / 2)

            if second_rate > first_rate * 1.2:
                trend_direction = "accelerating"
            elif second_rate < first_rate * 0.8:
                trend_direction = "slowing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "emerging"

        # Predict next milestone
        next_predicted = self._predict_next_milestone(user_profile, recent_milestones)

        return {
            "milestone_count": len(recent_milestones),
            "milestone_rate": milestone_rate,
            "dominant_categories": [cat for cat, count in dominant_categories],
            "category_distribution": dict(dominant_categories),
            "trend_direction": trend_direction,
            "average_significance": sum(
                m.emotional_significance for m in recent_milestones
            )
            / len(recent_milestones),
            "next_predicted": next_predicted,
            "analysis_period": f"{timeframe_days} days",
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    def suggest_milestone_opportunities(
        self, user_profile: CompanionProfile
    ) -> List[Dict[str, Any]]:
        """
        Suggest opportunities to create meaningful milestones.

        Args:
            user_profile: User's companion profile

        Returns:
            List of milestone opportunities
        """
        opportunities = []
        existing_types = [
            m.milestone_type.value for m in user_profile.relationship_milestones
        ]

        # Trust building opportunity
        if (
            "trust_building" not in existing_types
            and user_profile.trust_level > 0.3
            and user_profile.interaction_count >= 5
        ):
            opportunities.append(
                {
                    "milestone_type": "trust_building",
                    "category": "relationship",
                    "opportunity": "Encourage personal sharing to build deeper trust",
                    "readiness_score": user_profile.trust_level,
                    "suggested_approach": "Ask about personal interests, goals, or experiences",
                    "estimated_timeline": "1-2 conversations",
                }
            )

        # Learning milestone opportunity
        if (
            "learning_milestone" not in existing_types
            and user_profile.interaction_count >= 3
        ):
            opportunities.append(
                {
                    "milestone_type": "learning_milestone",
                    "category": "learning",
                    "opportunity": "Create a significant learning moment",
                    "readiness_score": user_profile.engagement_score,
                    "suggested_approach": "Provide deep insights on topics of interest",
                    "estimated_timeline": "1-3 conversations",
                }
            )

        # Deep conversation opportunity
        if (
            "deep_conversation" not in existing_types
            and user_profile.trust_level > 0.4
            and user_profile.interaction_count >= 8
        ):
            opportunities.append(
                {
                    "milestone_type": "deep_conversation",
                    "category": "communication",
                    "opportunity": "Engage in a profound, meaningful conversation",
                    "readiness_score": (
                        user_profile.trust_level + user_profile.engagement_score
                    )
                    / 2,
                    "suggested_approach": "Explore philosophical topics or personal values",
                    "estimated_timeline": "1-2 conversations",
                }
            )

        # Emotional support opportunity
        if "emotional_support" not in existing_types:
            opportunities.append(
                {
                    "milestone_type": "emotional_support",
                    "category": "emotional",
                    "opportunity": "Provide meaningful emotional support",
                    "readiness_score": user_profile.trust_level * 0.8,
                    "suggested_approach": "Be attentive to emotional cues and offer support",
                    "estimated_timeline": "When needed",
                }
            )

        # Goal achievement opportunity
        if "goal_achieved" not in existing_types:
            opportunities.append(
                {
                    "milestone_type": "goal_achieved",
                    "category": "achievement",
                    "opportunity": "Help user achieve a meaningful goal",
                    "readiness_score": user_profile.engagement_score,
                    "suggested_approach": "Identify user goals and provide structured support",
                    "estimated_timeline": "1-4 weeks",
                }
            )

        # Sort by readiness score
        opportunities.sort(key=lambda x: x["readiness_score"], reverse=True)

        logger.info(
            f"Identified {len(opportunities)} milestone opportunities for user {user_profile.user_id}"
        )

        return opportunities[:5]  # Top 5 opportunities

    def validate_milestone(
        self, milestone: RelationshipMilestone, context: MilestoneContext
    ) -> Tuple[bool, float, str]:
        """
        Validate a detected milestone for quality and appropriateness.

        Args:
            milestone: Detected milestone to validate
            context: Milestone detection context

        Returns:
            Tuple of (is_valid, confidence_score, validation_reason)
        """
        validation_score = 0.0
        validation_reasons = []

        # Check emotional appropriateness
        if context.interaction_data.emotional_state.primary_emotion.value in [
            "joy",
            "surprise",
        ]:
            validation_score += 0.3
            validation_reasons.append("positive emotional context")
        elif context.interaction_data.emotional_state.primary_emotion.value in [
            "sadness",
            "fear",
        ]:
            if milestone.milestone_type == MilestoneType.EMOTIONAL_SUPPORT:
                validation_score += 0.4
                validation_reasons.append("appropriate for emotional support")
            else:
                validation_score -= 0.2
                validation_reasons.append("emotional context mismatch")

        # Check interaction quality
        message_length = len(context.interaction_data.conversation_context.message_text)
        if message_length > 100:
            validation_score += 0.2
            validation_reasons.append("substantial interaction")
        elif message_length < 30:
            validation_score -= 0.3
            validation_reasons.append("interaction too brief")

        # Check relationship readiness
        if milestone.milestone_type == MilestoneType.TRUST_BUILDING:
            if context.user_profile.trust_level < 0.2:
                validation_score -= 0.2
                validation_reasons.append("trust level too low")
            else:
                validation_score += 0.2
                validation_reasons.append("trust level appropriate")

        # Check timing appropriateness
        recent_same_type = [
            m
            for m in context.recent_milestones
            if m.milestone_type == milestone.milestone_type
        ]
        if recent_same_type:
            validation_score -= 0.4
            validation_reasons.append("recent duplicate milestone")
        else:
            validation_score += 0.2
            validation_reasons.append("no recent duplicates")

        # Check user satisfaction if available
        if context.interaction_data.user_satisfaction:
            if context.interaction_data.user_satisfaction > 0.7:
                validation_score += 0.3
                validation_reasons.append("high user satisfaction")
            elif context.interaction_data.user_satisfaction < 0.4:
                validation_score -= 0.3
                validation_reasons.append("low user satisfaction")

        # Normalize score
        final_score = max(0.0, min(1.0, validation_score + 0.5))
        is_valid = final_score >= 0.6

        reason = (
            "; ".join(validation_reasons)
            if validation_reasons
            else "standard validation"
        )

        logger.debug(
            f"Milestone validation: {milestone.milestone_type.value} "
            f"- Valid: {is_valid}, Score: {final_score:.2f}, Reason: {reason}"
        )

        return is_valid, final_score, reason

    def _initialize_milestone_patterns(self) -> List[MilestonePattern]:
        """Initialize milestone detection patterns."""
        return [
            # First conversation milestone
            MilestonePattern(
                milestone_type=MilestoneType.FIRST_CONVERSATION,
                category=MilestoneCategory.RELATIONSHIP,
                detection_rules={"interaction_count": 1, "min_message_length": 10},
                significance_base=0.8,
                cooldown_interactions=0,
                required_conditions=["first_interaction"],
            ),
            # Trust building milestone
            MilestonePattern(
                milestone_type=MilestoneType.TRUST_BUILDING,
                category=MilestoneCategory.RELATIONSHIP,
                detection_rules={
                    "min_trust_level": 0.3,
                    "personal_keywords": [
                        "feel",
                        "think",
                        "personal",
                        "share",
                        "trust",
                        "believe",
                    ],
                    "min_message_length": 50,
                    "min_interactions": 3,
                },
                significance_base=0.85,
                cooldown_interactions=5,
                required_conditions=["personal_sharing", "sufficient_trust"],
            ),
            # Learning milestone
            MilestonePattern(
                milestone_type=MilestoneType.LEARNING_MILESTONE,
                category=MilestoneCategory.LEARNING,
                detection_rules={
                    "learning_keywords": [
                        "learn",
                        "understand",
                        "got it",
                        "makes sense",
                        "clear",
                        "thank you",
                    ],
                    "min_engagement": 0.4,
                    "min_interactions": 2,
                },
                significance_base=0.7,
                cooldown_interactions=3,
                required_conditions=["learning_indicators", "engagement"],
            ),
            # Emotional support milestone
            MilestonePattern(
                milestone_type=MilestoneType.EMOTIONAL_SUPPORT,
                category=MilestoneCategory.EMOTIONAL,
                detection_rules={
                    "emotional_keywords": [
                        "sad",
                        "worried",
                        "anxious",
                        "stressed",
                        "difficult",
                        "hard",
                    ],
                    "support_response": True,
                    "emotional_intensity": 0.6,
                },
                significance_base=0.8,
                cooldown_interactions=7,
                required_conditions=["emotional_need", "support_provided"],
            ),
            # Goal achievement milestone
            MilestonePattern(
                milestone_type=MilestoneType.GOAL_ACHIEVED,
                category=MilestoneCategory.ACHIEVEMENT,
                detection_rules={
                    "achievement_keywords": [
                        "achieved",
                        "accomplished",
                        "completed",
                        "success",
                        "done",
                        "finished",
                    ],
                    "positive_emotion": True,
                    "min_engagement": 0.5,
                },
                significance_base=0.9,
                cooldown_interactions=10,
                required_conditions=["achievement_indicators", "positive_emotion"],
            ),
            # Breakthrough moment milestone
            MilestonePattern(
                milestone_type=MilestoneType.BREAKTHROUGH_MOMENT,
                category=MilestoneCategory.LEARNING,
                detection_rules={
                    "breakthrough_keywords": [
                        "breakthrough",
                        "revelation",
                        "realize",
                        "suddenly",
                        "aha",
                        "insight",
                    ],
                    "high_emotion_intensity": 0.7,
                    "positive_emotion": True,
                    "min_message_length": 80,
                },
                significance_base=0.9,
                cooldown_interactions=8,
                required_conditions=[
                    "breakthrough_indicators",
                    "high_emotional_impact",
                ],
            ),
        ]

    def _should_check_milestone(
        self, pattern: MilestonePattern, context: MilestoneContext
    ) -> bool:
        """Check if milestone pattern should be evaluated."""
        # Check cooldown period
        recent_same_type = [
            m
            for m in context.recent_milestones
            if m.milestone_type == pattern.milestone_type
        ]

        if recent_same_type:
            interactions_since = context.user_profile.interaction_count - len(
                context.recent_milestones
            )
            if interactions_since < pattern.cooldown_interactions:
                return False

        # Check basic requirements
        rules = pattern.detection_rules

        if "min_interactions" in rules:
            if context.user_profile.interaction_count < rules["min_interactions"]:
                return False

        if "interaction_count" in rules:
            if context.user_profile.interaction_count != rules["interaction_count"]:
                return False

        return True

    def _evaluate_milestone_pattern(
        self, pattern: MilestonePattern, context: MilestoneContext
    ) -> Optional[RelationshipMilestone]:
        """Evaluate if a milestone pattern matches current context."""
        rules = pattern.detection_rules
        match_score = 0.0
        match_reasons = []

        message_text = (
            context.interaction_data.conversation_context.message_text.lower()
        )
        emotional_state = context.interaction_data.emotional_state

        # Check keyword matches
        for keyword_type in [
            "personal_keywords",
            "learning_keywords",
            "emotional_keywords",
            "achievement_keywords",
            "breakthrough_keywords",
        ]:
            if keyword_type in rules:
                keywords = rules[keyword_type]
                matches = sum(1 for keyword in keywords if keyword in message_text)
                if matches > 0:
                    match_score += 0.3 * (matches / len(keywords))
                    match_reasons.append(f"{keyword_type}: {matches} matches")

        # Check message length
        if "min_message_length" in rules:
            message_length = len(
                context.interaction_data.conversation_context.message_text
            )
            if message_length >= rules["min_message_length"]:
                match_score += 0.2
                match_reasons.append(f"sufficient message length: {message_length}")
            else:
                return None  # Hard requirement

        # Check trust level
        if "min_trust_level" in rules:
            if context.user_profile.trust_level >= rules["min_trust_level"]:
                match_score += 0.2
                match_reasons.append(
                    f"trust level: {context.user_profile.trust_level:.2f}"
                )
            else:
                return None  # Hard requirement

        # Check engagement
        if "min_engagement" in rules:
            if context.user_profile.engagement_score >= rules["min_engagement"]:
                match_score += 0.2
                match_reasons.append(
                    f"engagement: {context.user_profile.engagement_score:.2f}"
                )

        # Check emotional conditions
        if "positive_emotion" in rules and rules["positive_emotion"]:
            if emotional_state.primary_emotion.value in ["joy", "surprise"]:
                match_score += 0.3
                match_reasons.append("positive emotion detected")

        if "emotional_intensity" in rules:
            if emotional_state.intensity >= rules["emotional_intensity"]:
                match_score += 0.2
                match_reasons.append(
                    f"emotional intensity: {emotional_state.intensity:.2f}"
                )

        if "high_emotion_intensity" in rules:
            if emotional_state.intensity >= rules["high_emotion_intensity"]:
                match_score += 0.3
                match_reasons.append(
                    f"high emotional intensity: {emotional_state.intensity:.2f}"
                )

        # Require minimum match score
        if match_score < 0.5:
            return None

        # Calculate emotional significance
        significance = pattern.significance_base

        # Adjust based on emotional intensity
        significance += (emotional_state.intensity - 0.5) * 0.2

        # Adjust based on relationship maturity
        relationship_days = context.user_profile.get_relationship_age_days()
        if relationship_days > 14:
            significance += 0.1

        significance = max(0.0, min(1.0, significance))

        # Create milestone
        milestone = RelationshipMilestone(
            milestone_id=f"{context.user_profile.user_id}_{pattern.milestone_type.value}_{datetime.utcnow().timestamp()}",
            milestone_type=pattern.milestone_type,
            description=self._generate_milestone_description(
                pattern, context, match_reasons
            ),
            timestamp=datetime.utcnow(),
            emotional_significance=significance,
            celebration_acknowledged=False,
        )

        return milestone

    def _generate_milestone_description(
        self,
        pattern: MilestonePattern,
        context: MilestoneContext,
        match_reasons: List[str],
    ) -> str:
        """Generate description for detected milestone."""
        user_name = context.user_profile.preferred_name

        descriptions = {
            MilestoneType.FIRST_CONVERSATION: f"First meaningful conversation with {user_name}",
            MilestoneType.TRUST_BUILDING: f"{user_name} shared personal thoughts, building trust in our relationship",
            MilestoneType.LEARNING_MILESTONE: f"{user_name} had a significant learning breakthrough",
            MilestoneType.EMOTIONAL_SUPPORT: f"Provided emotional support to {user_name} during a difficult moment",
            MilestoneType.GOAL_ACHIEVED: f"{user_name} achieved an important goal with our collaboration",
            MilestoneType.BREAKTHROUGH_MOMENT: f"{user_name} experienced a profound insight or realization",
        }

        base_description = descriptions.get(
            pattern.milestone_type, f"Meaningful milestone reached with {user_name}"
        )

        # Add context if available
        if len(context.interaction_data.conversation_context.message_text) > 50:
            snippet = (
                context.interaction_data.conversation_context.message_text[:50] + "..."
            )
            base_description += f" (Context: {snippet})"

        return base_description

    def _get_pattern_for_milestone(
        self, milestone_type: MilestoneType
    ) -> Optional[MilestonePattern]:
        """Get pattern definition for milestone type."""
        for pattern in self.milestone_patterns:
            if pattern.milestone_type == milestone_type:
                return pattern
        return None

    def _predict_next_milestone(
        self,
        user_profile: CompanionProfile,
        recent_milestones: List[RelationshipMilestone],
    ) -> Optional[str]:
        """Predict the next likely milestone."""
        existing_types = [
            m.milestone_type.value for m in user_profile.relationship_milestones
        ]

        # Progression logic based on relationship development
        if user_profile.interaction_count <= 3:
            if "first_conversation" not in existing_types:
                return "first_conversation"
            else:
                return "learning_milestone"

        elif user_profile.interaction_count <= 10:
            if (
                "trust_building" not in existing_types
                and user_profile.trust_level > 0.3
            ):
                return "trust_building"
            elif "learning_milestone" not in existing_types:
                return "learning_milestone"

        elif user_profile.interaction_count <= 20:
            if "emotional_support" not in existing_types:
                return "emotional_support"
            elif "breakthrough_moment" not in existing_types:
                return "breakthrough_moment"

        else:
            if "goal_achieved" not in existing_types:
                return "goal_achieved"

        return None
