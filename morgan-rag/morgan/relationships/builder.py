"""
Relationship Builder for Morgan Assistant.

Handles the development and building of meaningful relationships with users
through trust building, engagement tracking, and relationship progression.

Requirements: 9.4, 9.5, 10.3
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..intelligence.core.models import CompanionProfile, InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RelationshipStage(Enum):
    """Stages of relationship development."""

    INITIAL = "initial"  # First few interactions
    BUILDING = "building"  # Trust and rapport building
    ESTABLISHED = "established"  # Comfortable, regular interaction
    DEEP = "deep"  # Strong trust and understanding
    COMPANION = "companion"  # Long-term meaningful relationship


@dataclass
class RelationshipMetrics:
    """Metrics for tracking relationship development."""

    trust_score: float  # 0.0 to 1.0
    engagement_level: float  # 0.0 to 1.0
    intimacy_level: float  # 0.0 to 1.0
    consistency_score: float  # 0.0 to 1.0
    growth_rate: float  # Rate of relationship development
    interaction_quality: float  # Average quality of interactions

    def overall_strength(self) -> float:
        """Calculate overall relationship strength."""
        weights = {
            "trust": 0.3,
            "engagement": 0.25,
            "intimacy": 0.2,
            "consistency": 0.15,
            "quality": 0.1,
        }

        return (
            self.trust_score * weights["trust"]
            + self.engagement_level * weights["engagement"]
            + self.intimacy_level * weights["intimacy"]
            + self.consistency_score * weights["consistency"]
            + self.interaction_quality * weights["quality"]
        )


@dataclass
class RelationshipGoals:
    """Goals for relationship development."""

    target_trust_level: float = 0.8
    target_engagement: float = 0.7
    target_intimacy: float = 0.6
    milestone_targets: List[str] = None

    def __post_init__(self):
        if self.milestone_targets is None:
            self.milestone_targets = [
                "first_conversation",
                "trust_building",
                "regular_user",
                "deep_conversation",
            ]


class RelationshipBuilder:
    """
    Builds and develops meaningful relationships with users.

    Focuses on trust building, engagement tracking, and relationship
    progression through natural interaction patterns.
    """

    def __init__(self):
        """Initialize relationship builder."""
        self.relationship_stages = self._define_stage_criteria()
        logger.info("Relationship builder initialized")

    def assess_relationship_stage(self, profile: CompanionProfile) -> RelationshipStage:
        """
        Assess current relationship stage based on profile metrics.

        Args:
            profile: User's companion profile

        Returns:
            RelationshipStage: Current stage of relationship
        """
        metrics = self.calculate_relationship_metrics(profile)
        days = profile.get_relationship_age_days()
        interactions = profile.interaction_count

        # Stage determination logic
        if interactions <= 3 or days <= 1:
            return RelationshipStage.INITIAL

        elif metrics.trust_score >= 0.8 and metrics.intimacy_level >= 0.7:
            return RelationshipStage.COMPANION

        elif metrics.trust_score >= 0.6 and metrics.engagement_level >= 0.7:
            return RelationshipStage.DEEP

        elif metrics.trust_score >= 0.4 and interactions >= 10:
            return RelationshipStage.ESTABLISHED

        else:
            return RelationshipStage.BUILDING

    def calculate_relationship_metrics(
        self, profile: CompanionProfile
    ) -> RelationshipMetrics:
        """
        Calculate comprehensive relationship metrics.

        Args:
            profile: User's companion profile

        Returns:
            RelationshipMetrics: Calculated relationship metrics
        """
        # Trust score (existing profile field)
        trust_score = profile.trust_level

        # Engagement level (existing profile field)
        engagement_level = profile.engagement_score

        # Intimacy level based on personal sharing and emotional depth
        intimacy_level = self._calculate_intimacy_level(profile)

        # Consistency score based on interaction patterns
        consistency_score = self._calculate_consistency_score(profile)

        # Growth rate based on milestone progression
        growth_rate = self._calculate_growth_rate(profile)

        # Interaction quality from emotional patterns
        interaction_quality = self._calculate_interaction_quality(profile)

        return RelationshipMetrics(
            trust_score=trust_score,
            engagement_level=engagement_level,
            intimacy_level=intimacy_level,
            consistency_score=consistency_score,
            growth_rate=growth_rate,
            interaction_quality=interaction_quality,
        )

    def build_trust(
        self, profile: CompanionProfile, interaction_data: InteractionData
    ) -> float:
        """
        Build trust based on interaction quality and consistency.

        Args:
            profile: User's companion profile
            interaction_data: Current interaction data

        Returns:
            float: Updated trust level
        """
        current_trust = profile.trust_level
        trust_factors = []

        # Positive trust factors
        if (
            interaction_data.user_satisfaction
            and interaction_data.user_satisfaction > 0.7
        ):
            trust_factors.append(0.05)  # Good satisfaction boosts trust

        if interaction_data.emotional_state.primary_emotion.value in [
            "joy",
            "surprise",
        ]:
            trust_factors.append(0.03)  # Positive emotions build trust

        if len(interaction_data.conversation_context.message_text) > 100:
            trust_factors.append(0.02)  # Longer messages show engagement

        # Personal sharing indicators
        personal_keywords = ["feel", "think", "believe", "personal", "share", "trust"]
        message_lower = interaction_data.conversation_context.message_text.lower()
        if any(keyword in message_lower for keyword in personal_keywords):
            trust_factors.append(0.04)  # Personal sharing builds trust

        # Consistency bonus
        if profile.interaction_count > 5:
            days_since_last = (datetime.now(timezone.utc) - profile.last_interaction).days
            if days_since_last <= 7:  # Regular interaction
                trust_factors.append(0.02)

        # Calculate trust increase
        trust_increase = sum(trust_factors)

        # Apply diminishing returns for high trust levels
        if current_trust > 0.7:
            trust_increase *= 0.5
        elif current_trust > 0.5:
            trust_increase *= 0.8

        new_trust = min(1.0, current_trust + trust_increase)

        logger.debug(
            f"Trust updated for user {profile.user_id}: "
            f"{current_trust:.3f} -> {new_trust:.3f} "
            f"(+{trust_increase:.3f})"
        )

        return new_trust

    def enhance_engagement(
        self, profile: CompanionProfile, interaction_data: InteractionData
    ) -> float:
        """
        Enhance engagement based on interaction patterns.

        Args:
            profile: User's companion profile
            interaction_data: Current interaction data

        Returns:
            float: Updated engagement level
        """
        current_engagement = profile.engagement_score
        engagement_factors = []

        # Message length indicates engagement
        message_length = len(interaction_data.conversation_context.message_text)
        if message_length > 200:
            engagement_factors.append(0.06)
        elif message_length > 100:
            engagement_factors.append(0.04)
        elif message_length > 50:
            engagement_factors.append(0.02)

        # Question asking shows engagement
        if "?" in interaction_data.conversation_context.message_text:
            engagement_factors.append(0.03)

        # Topic diversity
        if len(interaction_data.topics_discussed) > 2:
            engagement_factors.append(0.03)

        # Emotional investment
        if interaction_data.emotional_state.intensity > 0.6:
            engagement_factors.append(0.04)

        # Learning indicators
        if interaction_data.learning_indicators:
            engagement_factors.append(0.05)

        # Calculate engagement increase
        engagement_increase = sum(engagement_factors)

        # Apply session duration bonus if available
        if interaction_data.conversation_context.session_duration:
            minutes = (
                interaction_data.conversation_context.session_duration.total_seconds()
                / 60
            )
            if minutes > 10:
                engagement_increase += 0.02

        new_engagement = min(1.0, current_engagement + engagement_increase)

        logger.debug(
            f"Engagement updated for user {profile.user_id}: "
            f"{current_engagement:.3f} -> {new_engagement:.3f} "
            f"(+{engagement_increase:.3f})"
        )

        return new_engagement

    def identify_relationship_opportunities(
        self, profile: CompanionProfile
    ) -> List[Dict[str, Any]]:
        """
        Identify opportunities to strengthen the relationship.

        Args:
            profile: User's companion profile

        Returns:
            List of relationship building opportunities
        """
        opportunities = []
        metrics = self.calculate_relationship_metrics(profile)
        stage = self.assess_relationship_stage(profile)

        # Trust building opportunities
        if metrics.trust_score < 0.5:
            opportunities.append(
                {
                    "type": "trust_building",
                    "priority": "high",
                    "suggestion": "Share more personal insights and ask about user preferences",
                    "target_metric": "trust_score",
                    "current_value": metrics.trust_score,
                    "target_value": 0.6,
                }
            )

        # Engagement opportunities
        if metrics.engagement_level < 0.6:
            opportunities.append(
                {
                    "type": "engagement_boost",
                    "priority": "medium",
                    "suggestion": "Ask more engaging questions and explore user interests deeply",
                    "target_metric": "engagement_level",
                    "current_value": metrics.engagement_level,
                    "target_value": 0.7,
                }
            )

        # Intimacy development
        if metrics.intimacy_level < 0.4 and stage != RelationshipStage.INITIAL:
            opportunities.append(
                {
                    "type": "intimacy_development",
                    "priority": "medium",
                    "suggestion": "Encourage personal sharing and provide emotional support",
                    "target_metric": "intimacy_level",
                    "current_value": metrics.intimacy_level,
                    "target_value": 0.5,
                }
            )

        # Consistency improvement
        if metrics.consistency_score < 0.5:
            opportunities.append(
                {
                    "type": "consistency_improvement",
                    "priority": "low",
                    "suggestion": "Maintain regular interaction patterns and follow up on previous conversations",
                    "target_metric": "consistency_score",
                    "current_value": metrics.consistency_score,
                    "target_value": 0.6,
                }
            )

        # Milestone progression
        milestone_types = [
            m.milestone_type.value for m in profile.relationship_milestones
        ]
        if "trust_building" not in milestone_types and metrics.trust_score > 0.3:
            opportunities.append(
                {
                    "type": "milestone_progression",
                    "priority": "high",
                    "suggestion": "Focus on trust-building conversations to unlock trust milestone",
                    "target_metric": "milestone_count",
                    "current_value": len(profile.relationship_milestones),
                    "target_value": len(profile.relationship_milestones) + 1,
                }
            )

        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        opportunities.sort(
            key=lambda x: priority_order.get(x["priority"], 0), reverse=True
        )

        logger.info(
            f"Identified {len(opportunities)} relationship opportunities for user {profile.user_id}"
        )

        return opportunities

    def generate_relationship_strategy(
        self, profile: CompanionProfile, goals: Optional[RelationshipGoals] = None
    ) -> Dict[str, Any]:
        """
        Generate a strategy for relationship development.

        Args:
            profile: User's companion profile
            goals: Optional relationship goals

        Returns:
            Dictionary containing relationship development strategy
        """
        if goals is None:
            goals = RelationshipGoals()

        current_stage = self.assess_relationship_stage(profile)
        metrics = self.calculate_relationship_metrics(profile)
        opportunities = self.identify_relationship_opportunities(profile)

        # Determine next stage target
        stage_progression = {
            RelationshipStage.INITIAL: RelationshipStage.BUILDING,
            RelationshipStage.BUILDING: RelationshipStage.ESTABLISHED,
            RelationshipStage.ESTABLISHED: RelationshipStage.DEEP,
            RelationshipStage.DEEP: RelationshipStage.COMPANION,
        }

        next_stage = stage_progression.get(current_stage, current_stage)

        # Generate action plan
        action_plan = []
        for opportunity in opportunities[:3]:  # Top 3 opportunities
            action_plan.append(
                {
                    "action": opportunity["suggestion"],
                    "priority": opportunity["priority"],
                    "expected_impact": opportunity["target_metric"],
                    "timeline": self._estimate_timeline(opportunity["type"]),
                }
            )

        strategy = {
            "current_stage": current_stage.value,
            "target_stage": next_stage.value,
            "current_metrics": {
                "trust": metrics.trust_score,
                "engagement": metrics.engagement_level,
                "intimacy": metrics.intimacy_level,
                "overall_strength": metrics.overall_strength(),
            },
            "target_metrics": {
                "trust": goals.target_trust_level,
                "engagement": goals.target_engagement,
                "intimacy": goals.target_intimacy,
            },
            "action_plan": action_plan,
            "estimated_timeline": self._estimate_stage_timeline(
                current_stage, next_stage
            ),
            "success_indicators": self._define_success_indicators(next_stage),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Generated relationship strategy for user {profile.user_id}: "
            f"{current_stage.value} -> {next_stage.value}"
        )

        return strategy

    def _calculate_intimacy_level(self, profile: CompanionProfile) -> float:
        """Calculate intimacy level based on personal sharing indicators."""
        base_intimacy = 0.0

        # Milestone-based intimacy
        milestone_types = [
            m.milestone_type.value for m in profile.relationship_milestones
        ]
        intimacy_milestones = [
            "trust_building",
            "emotional_support",
            "deep_conversation",
        ]

        for milestone_type in intimacy_milestones:
            if milestone_type in milestone_types:
                base_intimacy += 0.2

        # Trust level contributes to intimacy
        base_intimacy += profile.trust_level * 0.3

        # Interaction count factor (more interactions = potential for intimacy)
        interaction_factor = min(profile.interaction_count / 20, 0.3)
        base_intimacy += interaction_factor

        return min(1.0, base_intimacy)

    def _calculate_consistency_score(self, profile: CompanionProfile) -> float:
        """Calculate consistency score based on interaction patterns."""
        if profile.interaction_count <= 1:
            return 0.0

        # Base consistency from regular interaction
        days_active = profile.get_relationship_age_days()
        if days_active == 0:
            return 0.5  # Single day, assume good consistency

        interactions_per_day = profile.interaction_count / max(days_active, 1)

        # Ideal range: 0.5 to 2 interactions per day
        if 0.5 <= interactions_per_day <= 2.0:
            consistency_base = 0.8
        elif interactions_per_day > 2.0:
            consistency_base = 0.6  # Too frequent might indicate dependency
        else:
            consistency_base = interactions_per_day * 1.6  # Scale up low frequency

        # Recent activity bonus
        days_since_last = (datetime.now(timezone.utc) - profile.last_interaction).days
        if days_since_last <= 3:
            consistency_base += 0.2
        elif days_since_last <= 7:
            consistency_base += 0.1

        return min(1.0, consistency_base)

    def _calculate_growth_rate(self, profile: CompanionProfile) -> float:
        """Calculate relationship growth rate."""
        days = max(profile.get_relationship_age_days(), 1)
        milestones_per_week = (len(profile.relationship_milestones) / days) * 7

        # Ideal growth: 1-2 milestones per week
        if 1.0 <= milestones_per_week <= 2.0:
            return 0.8
        elif milestones_per_week > 2.0:
            return 0.6  # Too fast might not be sustainable
        else:
            return min(milestones_per_week * 0.8, 0.8)

    def _calculate_interaction_quality(self, profile: CompanionProfile) -> float:
        """Calculate average interaction quality."""
        # Base quality from engagement and trust
        base_quality = (profile.engagement_score + profile.trust_level) / 2

        # Milestone quality bonus
        if profile.relationship_milestones:
            avg_significance = sum(
                m.emotional_significance for m in profile.relationship_milestones
            ) / len(profile.relationship_milestones)
            base_quality = (base_quality + avg_significance) / 2

        return base_quality

    def _define_stage_criteria(self) -> Dict[RelationshipStage, Dict[str, float]]:
        """Define criteria for each relationship stage."""
        return {
            RelationshipStage.INITIAL: {
                "min_interactions": 1,
                "min_trust": 0.0,
                "min_engagement": 0.0,
                "min_days": 0,
            },
            RelationshipStage.BUILDING: {
                "min_interactions": 3,
                "min_trust": 0.2,
                "min_engagement": 0.3,
                "min_days": 1,
            },
            RelationshipStage.ESTABLISHED: {
                "min_interactions": 10,
                "min_trust": 0.4,
                "min_engagement": 0.5,
                "min_days": 7,
            },
            RelationshipStage.DEEP: {
                "min_interactions": 20,
                "min_trust": 0.6,
                "min_engagement": 0.7,
                "min_days": 14,
            },
            RelationshipStage.COMPANION: {
                "min_interactions": 50,
                "min_trust": 0.8,
                "min_engagement": 0.8,
                "min_days": 30,
            },
        }

    def _estimate_timeline(self, opportunity_type: str) -> str:
        """Estimate timeline for relationship opportunity."""
        timelines = {
            "trust_building": "1-2 weeks",
            "engagement_boost": "3-5 days",
            "intimacy_development": "2-3 weeks",
            "consistency_improvement": "1 week",
            "milestone_progression": "1-2 weeks",
        }
        return timelines.get(opportunity_type, "1 week")

    def _estimate_stage_timeline(
        self, current_stage: RelationshipStage, target_stage: RelationshipStage
    ) -> str:
        """Estimate timeline to reach target stage."""
        stage_timelines = {
            (RelationshipStage.INITIAL, RelationshipStage.BUILDING): "3-7 days",
            (RelationshipStage.BUILDING, RelationshipStage.ESTABLISHED): "1-2 weeks",
            (RelationshipStage.ESTABLISHED, RelationshipStage.DEEP): "2-4 weeks",
            (RelationshipStage.DEEP, RelationshipStage.COMPANION): "1-2 months",
        }
        return stage_timelines.get((current_stage, target_stage), "2-3 weeks")

    def _define_success_indicators(self, target_stage: RelationshipStage) -> List[str]:
        """Define success indicators for reaching target stage."""
        indicators = {
            RelationshipStage.BUILDING: [
                "User shares personal preferences",
                "Positive emotional responses increase",
                "Message length and engagement improve",
            ],
            RelationshipStage.ESTABLISHED: [
                "Regular interaction pattern established",
                "Trust milestone achieved",
                "User asks follow-up questions",
            ],
            RelationshipStage.DEEP: [
                "Deep conversation milestone reached",
                "Emotional support provided successfully",
                "User shares personal challenges or goals",
            ],
            RelationshipStage.COMPANION: [
                "Long-term relationship milestones achieved",
                "High trust and intimacy levels maintained",
                "User considers Morgan a valued companion",
            ],
        }
        return indicators.get(target_stage, ["Continued positive interactions"])
