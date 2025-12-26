"""
Relationship Dynamics Analysis for Morgan Assistant.

Analyzes the complex dynamics of user relationships including interaction
patterns, emotional dynamics, communication evolution, and relationship health.

Requirements: 9.4, 9.5, 10.3
"""

import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from ..emotional.models import CompanionProfile, EmotionalState, InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DynamicsPattern(Enum):
    """Types of relationship dynamics patterns."""

    GROWING = "growing"  # Relationship is strengthening
    STABLE = "stable"  # Relationship is consistent
    FLUCTUATING = "fluctuating"  # Relationship has ups and downs
    DECLINING = "declining"  # Relationship is weakening
    RECOVERING = "recovering"  # Relationship is improving after decline
    STAGNANT = "stagnant"  # Relationship lacks progression


class CommunicationStyle(Enum):
    """Communication style evolution patterns."""

    FORMAL_TO_CASUAL = "formal_to_casual"
    CASUAL_TO_FORMAL = "casual_to_formal"
    DEEPENING = "deepening"
    SURFACE_LEVEL = "surface_level"
    TECHNICAL_FOCUS = "technical_focus"
    EMOTIONAL_FOCUS = "emotional_focus"
    STABLE = "stable"


@dataclass
class InteractionPattern:
    """Pattern analysis of user interactions."""

    frequency_trend: str  # "increasing", "decreasing", "stable"
    session_length_trend: str  # "longer", "shorter", "stable"
    engagement_trend: str  # "improving", "declining", "stable"
    topic_diversity: float  # 0.0 to 1.0
    consistency_score: float  # 0.0 to 1.0
    preferred_times: List[int]  # Hours of day (0-23)
    interaction_gaps: List[int]  # Days between interactions


@dataclass
class EmotionalDynamics:
    """Analysis of emotional patterns in relationship."""

    emotional_stability: float  # 0.0 to 1.0
    dominant_emotions: List[str]  # Most frequent emotions
    emotional_range: float  # Breadth of emotions expressed
    positive_ratio: float  # Ratio of positive to negative emotions
    emotional_responsiveness: float  # How well Morgan responds to emotions
    mood_correlation: float  # Correlation between user and Morgan moods
    emotional_growth: str  # "improving", "stable", "concerning"


@dataclass
class TrustDynamics:
    """Analysis of trust development patterns."""

    trust_trajectory: str  # "building", "stable", "declining"
    trust_velocity: float  # Rate of trust change
    trust_milestones: List[str]  # Trust-related milestones achieved
    vulnerability_indicators: int  # Count of vulnerable sharing moments
    consistency_impact: float  # How consistency affects trust
    trust_recovery_ability: float  # Ability to recover from trust issues


@dataclass
class CommunicationDynamics:
    """Analysis of communication evolution."""

    style_evolution: CommunicationStyle
    depth_progression: str  # "deepening", "stable", "superficial"
    topic_evolution: Dict[str, float]  # How topic preferences change
    response_quality_trend: str  # "improving", "stable", "declining"
    mutual_understanding: float  # 0.0 to 1.0
    communication_efficiency: float  # How well they understand each other


@dataclass
class RelationshipHealth:
    """Overall relationship health assessment."""

    health_score: float  # 0.0 to 1.0 overall health
    strength_areas: List[str]  # Areas where relationship is strong
    growth_areas: List[str]  # Areas needing attention
    risk_factors: List[str]  # Potential relationship risks
    resilience_score: float  # Ability to handle challenges
    sustainability_outlook: str  # "excellent", "good", "concerning"


class RelationshipDynamics:
    """
    Analyzes complex relationship dynamics and patterns.

    Provides comprehensive analysis of interaction patterns, emotional
    dynamics, trust development, and overall relationship health.
    """

    def __init__(self):
        """Initialize relationship dynamics analyzer."""
        self.analysis_cache = {}  # Cache analysis results
        self.pattern_thresholds = self._initialize_thresholds()
        logger.info("Relationship dynamics analyzer initialized")

    def analyze_relationship_dynamics(
        self,
        user_profile: CompanionProfile,
        interaction_history: List[InteractionData],
        emotional_history: List[EmotionalState],
    ) -> Dict[str, Any]:
        """
        Perform comprehensive relationship dynamics analysis.

        Args:
            user_profile: User's companion profile
            interaction_history: Historical interaction data
            emotional_history: Historical emotional states

        Returns:
            Comprehensive dynamics analysis
        """
        # Analyze different aspects of relationship dynamics
        interaction_patterns = self.analyze_interaction_patterns(interaction_history)
        emotional_dynamics = self.analyze_emotional_dynamics(
            emotional_history, interaction_history
        )
        trust_dynamics = self.analyze_trust_dynamics(user_profile, interaction_history)
        communication_dynamics = self.analyze_communication_dynamics(
            interaction_history, user_profile
        )
        relationship_health = self.assess_relationship_health(
            user_profile,
            interaction_patterns,
            emotional_dynamics,
            trust_dynamics,
            communication_dynamics,
        )

        # Determine overall dynamics pattern
        overall_pattern = self._determine_overall_pattern(
            interaction_patterns, emotional_dynamics, trust_dynamics
        )

        analysis = {
            "user_id": user_profile.user_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_pattern": overall_pattern.value,
            "interaction_patterns": self._serialize_interaction_patterns(
                interaction_patterns
            ),
            "emotional_dynamics": self._serialize_emotional_dynamics(
                emotional_dynamics
            ),
            "trust_dynamics": self._serialize_trust_dynamics(trust_dynamics),
            "communication_dynamics": self._serialize_communication_dynamics(
                communication_dynamics
            ),
            "relationship_health": self._serialize_relationship_health(
                relationship_health
            ),
            "recommendations": self._generate_recommendations(
                overall_pattern, relationship_health, user_profile
            ),
        }

        # Cache analysis
        self.analysis_cache[user_profile.user_id] = analysis

        logger.info(
            f"Completed relationship dynamics analysis for user {user_profile.user_id}: "
            f"Pattern: {overall_pattern.value}, Health: {relationship_health.health_score:.2f}"
        )

        return analysis

    def analyze_interaction_patterns(
        self, interaction_history: List[InteractionData]
    ) -> InteractionPattern:
        """
        Analyze patterns in user interactions.

        Args:
            interaction_history: Historical interaction data

        Returns:
            InteractionPattern analysis
        """
        if len(interaction_history) < 3:
            return InteractionPattern(
                frequency_trend="insufficient_data",
                session_length_trend="insufficient_data",
                engagement_trend="insufficient_data",
                topic_diversity=0.0,
                consistency_score=0.0,
                preferred_times=[],
                interaction_gaps=[],
            )

        # Analyze frequency trend
        frequency_trend = self._analyze_frequency_trend(interaction_history)

        # Analyze session length trend
        session_length_trend = self._analyze_session_length_trend(interaction_history)

        # Analyze engagement trend
        engagement_trend = self._analyze_engagement_trend(interaction_history)

        # Calculate topic diversity
        topic_diversity = self._calculate_topic_diversity(interaction_history)

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(interaction_history)

        # Identify preferred interaction times
        preferred_times = self._identify_preferred_times(interaction_history)

        # Calculate interaction gaps
        interaction_gaps = self._calculate_interaction_gaps(interaction_history)

        return InteractionPattern(
            frequency_trend=frequency_trend,
            session_length_trend=session_length_trend,
            engagement_trend=engagement_trend,
            topic_diversity=topic_diversity,
            consistency_score=consistency_score,
            preferred_times=preferred_times,
            interaction_gaps=interaction_gaps,
        )

    def analyze_emotional_dynamics(
        self,
        emotional_history: List[EmotionalState],
        interaction_history: List[InteractionData],
    ) -> EmotionalDynamics:
        """
        Analyze emotional dynamics in the relationship.

        Args:
            emotional_history: Historical emotional states
            interaction_history: Historical interaction data

        Returns:
            EmotionalDynamics analysis
        """
        if not emotional_history:
            return EmotionalDynamics(
                emotional_stability=0.5,
                dominant_emotions=[],
                emotional_range=0.0,
                positive_ratio=0.5,
                emotional_responsiveness=0.5,
                mood_correlation=0.0,
                emotional_growth="unknown",
            )

        # Calculate emotional stability
        emotional_stability = self._calculate_emotional_stability(emotional_history)

        # Identify dominant emotions
        dominant_emotions = self._identify_dominant_emotions(emotional_history)

        # Calculate emotional range
        emotional_range = self._calculate_emotional_range(emotional_history)

        # Calculate positive emotion ratio
        positive_ratio = self._calculate_positive_ratio(emotional_history)

        # Assess emotional responsiveness
        emotional_responsiveness = self._assess_emotional_responsiveness(
            emotional_history, interaction_history
        )

        # Calculate mood correlation (simplified)
        mood_correlation = self._calculate_mood_correlation(emotional_history)

        # Determine emotional growth trend
        emotional_growth = self._determine_emotional_growth(emotional_history)

        return EmotionalDynamics(
            emotional_stability=emotional_stability,
            dominant_emotions=dominant_emotions,
            emotional_range=emotional_range,
            positive_ratio=positive_ratio,
            emotional_responsiveness=emotional_responsiveness,
            mood_correlation=mood_correlation,
            emotional_growth=emotional_growth,
        )

    def analyze_trust_dynamics(
        self, user_profile: CompanionProfile, interaction_history: List[InteractionData]
    ) -> TrustDynamics:
        """
        Analyze trust development patterns.

        Args:
            user_profile: User's companion profile
            interaction_history: Historical interaction data

        Returns:
            TrustDynamics analysis
        """
        # Determine trust trajectory
        trust_trajectory = self._determine_trust_trajectory(
            user_profile, interaction_history
        )

        # Calculate trust velocity (rate of change)
        trust_velocity = self._calculate_trust_velocity(
            user_profile, interaction_history
        )

        # Identify trust milestones
        trust_milestones = self._identify_trust_milestones(user_profile)

        # Count vulnerability indicators
        vulnerability_indicators = self._count_vulnerability_indicators(
            interaction_history
        )

        # Assess consistency impact on trust
        consistency_impact = self._assess_consistency_impact(
            user_profile, interaction_history
        )

        # Assess trust recovery ability
        trust_recovery_ability = self._assess_trust_recovery_ability(user_profile)

        return TrustDynamics(
            trust_trajectory=trust_trajectory,
            trust_velocity=trust_velocity,
            trust_milestones=trust_milestones,
            vulnerability_indicators=vulnerability_indicators,
            consistency_impact=consistency_impact,
            trust_recovery_ability=trust_recovery_ability,
        )

    def analyze_communication_dynamics(
        self, interaction_history: List[InteractionData], user_profile: CompanionProfile
    ) -> CommunicationDynamics:
        """
        Analyze communication evolution and dynamics.

        Args:
            interaction_history: Historical interaction data
            user_profile: User's companion profile

        Returns:
            CommunicationDynamics analysis
        """
        # Analyze style evolution
        style_evolution = self._analyze_style_evolution(interaction_history)

        # Analyze depth progression
        depth_progression = self._analyze_depth_progression(interaction_history)

        # Analyze topic evolution
        topic_evolution = self._analyze_topic_evolution(interaction_history)

        # Analyze response quality trend
        response_quality_trend = self._analyze_response_quality_trend(
            interaction_history
        )

        # Assess mutual understanding
        mutual_understanding = self._assess_mutual_understanding(interaction_history)

        # Calculate communication efficiency
        communication_efficiency = self._calculate_communication_efficiency(
            interaction_history, user_profile
        )

        return CommunicationDynamics(
            style_evolution=style_evolution,
            depth_progression=depth_progression,
            topic_evolution=topic_evolution,
            response_quality_trend=response_quality_trend,
            mutual_understanding=mutual_understanding,
            communication_efficiency=communication_efficiency,
        )

    def assess_relationship_health(
        self,
        user_profile: CompanionProfile,
        interaction_patterns: InteractionPattern,
        emotional_dynamics: EmotionalDynamics,
        trust_dynamics: TrustDynamics,
        communication_dynamics: CommunicationDynamics,
    ) -> RelationshipHealth:
        """
        Assess overall relationship health.

        Args:
            user_profile: User's companion profile
            interaction_patterns: Interaction pattern analysis
            emotional_dynamics: Emotional dynamics analysis
            trust_dynamics: Trust dynamics analysis
            communication_dynamics: Communication dynamics analysis

        Returns:
            RelationshipHealth assessment
        """
        # Calculate overall health score
        health_components = {
            "trust": user_profile.trust_level * 0.25,
            "engagement": user_profile.engagement_score * 0.20,
            "emotional_stability": emotional_dynamics.emotional_stability * 0.15,
            "consistency": interaction_patterns.consistency_score * 0.15,
            "communication": communication_dynamics.mutual_understanding * 0.15,
            "positive_emotions": emotional_dynamics.positive_ratio * 0.10,
        }

        health_score = sum(health_components.values())

        # Identify strength areas
        strength_areas = []
        if user_profile.trust_level >= 0.7:
            strength_areas.append("High trust level")
        if user_profile.engagement_score >= 0.7:
            strength_areas.append("Strong engagement")
        if emotional_dynamics.emotional_stability >= 0.7:
            strength_areas.append("Emotional stability")
        if interaction_patterns.consistency_score >= 0.7:
            strength_areas.append("Consistent interaction")
        if communication_dynamics.mutual_understanding >= 0.7:
            strength_areas.append("Good communication")

        # Identify growth areas
        growth_areas = []
        if user_profile.trust_level < 0.5:
            growth_areas.append("Trust building needed")
        if user_profile.engagement_score < 0.5:
            growth_areas.append("Engagement improvement needed")
        if emotional_dynamics.emotional_stability < 0.5:
            growth_areas.append("Emotional support needed")
        if interaction_patterns.consistency_score < 0.5:
            growth_areas.append("More consistent interaction needed")
        if communication_dynamics.mutual_understanding < 0.5:
            growth_areas.append("Communication clarity needed")

        # Identify risk factors
        risk_factors = []
        if trust_dynamics.trust_trajectory == "declining":
            risk_factors.append("Declining trust")
        if emotional_dynamics.positive_ratio < 0.3:
            risk_factors.append("Predominantly negative emotions")
        if interaction_patterns.frequency_trend == "decreasing":
            risk_factors.append("Decreasing interaction frequency")
        if len(interaction_patterns.interaction_gaps) > 0:
            avg_gap = statistics.mean(interaction_patterns.interaction_gaps)
            if avg_gap > 7:
                risk_factors.append("Long gaps between interactions")

        # Calculate resilience score
        resilience_factors = [
            trust_dynamics.trust_recovery_ability,
            emotional_dynamics.emotional_stability,
            interaction_patterns.consistency_score,
            min(
                1.0, len(user_profile.relationship_milestones) / 5
            ),  # Milestone diversity
        ]
        resilience_score = statistics.mean(resilience_factors)

        # Determine sustainability outlook
        if health_score >= 0.8 and resilience_score >= 0.7:
            sustainability_outlook = "excellent"
        elif health_score >= 0.6 and resilience_score >= 0.5:
            sustainability_outlook = "good"
        else:
            sustainability_outlook = "concerning"

        return RelationshipHealth(
            health_score=health_score,
            strength_areas=strength_areas,
            growth_areas=growth_areas,
            risk_factors=risk_factors,
            resilience_score=resilience_score,
            sustainability_outlook=sustainability_outlook,
        )

    def predict_relationship_trajectory(
        self,
        user_profile: CompanionProfile,
        dynamics_analysis: Dict[str, Any],
        prediction_horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Predict relationship trajectory based on current dynamics.

        Args:
            user_profile: User's companion profile
            dynamics_analysis: Current dynamics analysis
            prediction_horizon_days: Prediction timeframe

        Returns:
            Trajectory prediction with confidence intervals
        """
        current_pattern = DynamicsPattern(dynamics_analysis["overall_pattern"])
        health_score = dynamics_analysis["relationship_health"]["health_score"]

        # Base trajectory prediction
        trajectory_predictions = {
            DynamicsPattern.GROWING: {
                "direction": "positive",
                "confidence": 0.8,
                "expected_change": 0.1,
            },
            DynamicsPattern.STABLE: {
                "direction": "neutral",
                "confidence": 0.7,
                "expected_change": 0.0,
            },
            DynamicsPattern.DECLINING: {
                "direction": "negative",
                "confidence": 0.6,
                "expected_change": -0.1,
            },
            DynamicsPattern.RECOVERING: {
                "direction": "positive",
                "confidence": 0.7,
                "expected_change": 0.15,
            },
            DynamicsPattern.FLUCTUATING: {
                "direction": "uncertain",
                "confidence": 0.4,
                "expected_change": 0.0,
            },
            DynamicsPattern.STAGNANT: {
                "direction": "neutral",
                "confidence": 0.6,
                "expected_change": -0.05,
            },
        }

        base_prediction = trajectory_predictions.get(
            current_pattern,
            {"direction": "uncertain", "confidence": 0.3, "expected_change": 0.0},
        )

        # Adjust prediction based on health score
        if health_score >= 0.8:
            base_prediction["confidence"] += 0.1
            base_prediction["expected_change"] += 0.05
        elif health_score <= 0.4:
            base_prediction["confidence"] -= 0.1
            base_prediction["expected_change"] -= 0.05

        # Generate specific predictions
        predictions = {
            "overall_trajectory": base_prediction["direction"],
            "confidence": min(1.0, max(0.0, base_prediction["confidence"])),
            "predicted_health_change": base_prediction["expected_change"],
            "predicted_health_score": min(
                1.0, max(0.0, health_score + base_prediction["expected_change"])
            ),
            "key_factors": self._identify_trajectory_factors(dynamics_analysis),
            "intervention_opportunities": self._identify_intervention_opportunities(
                dynamics_analysis
            ),
            "milestone_predictions": self._predict_milestone_opportunities(
                user_profile, current_pattern
            ),
            "prediction_horizon_days": prediction_horizon_days,
            "generated_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Generated relationship trajectory prediction for user {user_profile.user_id}: "
            f"{predictions['overall_trajectory']} (confidence: {predictions['confidence']:.2f})"
        )

        return predictions

    def _determine_overall_pattern(
        self,
        interaction_patterns: InteractionPattern,
        emotional_dynamics: EmotionalDynamics,
        trust_dynamics: TrustDynamics,
    ) -> DynamicsPattern:
        """Determine overall relationship dynamics pattern."""
        # Score different patterns based on indicators
        pattern_scores = {
            DynamicsPattern.GROWING: 0.0,
            DynamicsPattern.STABLE: 0.0,
            DynamicsPattern.DECLINING: 0.0,
            DynamicsPattern.RECOVERING: 0.0,
            DynamicsPattern.FLUCTUATING: 0.0,
            DynamicsPattern.STAGNANT: 0.0,
        }

        # Interaction pattern indicators
        if interaction_patterns.frequency_trend == "increasing":
            pattern_scores[DynamicsPattern.GROWING] += 0.3
        elif interaction_patterns.frequency_trend == "decreasing":
            pattern_scores[DynamicsPattern.DECLINING] += 0.3
        else:
            pattern_scores[DynamicsPattern.STABLE] += 0.2

        if interaction_patterns.engagement_trend == "improving":
            pattern_scores[DynamicsPattern.GROWING] += 0.2
            pattern_scores[DynamicsPattern.RECOVERING] += 0.1
        elif interaction_patterns.engagement_trend == "declining":
            pattern_scores[DynamicsPattern.DECLINING] += 0.2

        # Trust dynamics indicators
        if trust_dynamics.trust_trajectory == "building":
            pattern_scores[DynamicsPattern.GROWING] += 0.3
        elif trust_dynamics.trust_trajectory == "declining":
            pattern_scores[DynamicsPattern.DECLINING] += 0.3
        elif trust_dynamics.trust_trajectory == "stable":
            pattern_scores[DynamicsPattern.STABLE] += 0.2

        # Emotional dynamics indicators
        if emotional_dynamics.emotional_growth == "improving":
            pattern_scores[DynamicsPattern.GROWING] += 0.2
            pattern_scores[DynamicsPattern.RECOVERING] += 0.1
        elif emotional_dynamics.emotional_growth == "concerning":
            pattern_scores[DynamicsPattern.DECLINING] += 0.2

        if emotional_dynamics.emotional_stability < 0.4:
            pattern_scores[DynamicsPattern.FLUCTUATING] += 0.3

        # Consistency indicators
        if interaction_patterns.consistency_score < 0.3:
            pattern_scores[DynamicsPattern.FLUCTUATING] += 0.2
            pattern_scores[DynamicsPattern.STAGNANT] += 0.1

        # Return pattern with highest score
        return max(pattern_scores.items(), key=lambda x: x[1])[0]

    def _analyze_frequency_trend(
        self, interaction_history: List[InteractionData]
    ) -> str:
        """Analyze interaction frequency trend."""
        if len(interaction_history) < 6:
            return "insufficient_data"

        # Split into first and second half
        mid_point = len(interaction_history) // 2
        first_half = interaction_history[:mid_point]
        second_half = interaction_history[mid_point:]

        # Calculate time spans
        first_span = (
            first_half[-1].conversation_context.timestamp
            - first_half[0].conversation_context.timestamp
        ).days or 1
        second_span = (
            second_half[-1].conversation_context.timestamp
            - second_half[0].conversation_context.timestamp
        ).days or 1

        # Calculate frequencies
        first_freq = len(first_half) / first_span
        second_freq = len(second_half) / second_span

        if second_freq > first_freq * 1.2:
            return "increasing"
        elif second_freq < first_freq * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _analyze_session_length_trend(
        self, interaction_history: List[InteractionData]
    ) -> str:
        """Analyze session length trend."""
        session_lengths = []
        for interaction in interaction_history:
            if interaction.conversation_context.session_duration:
                length = (
                    interaction.conversation_context.session_duration.total_seconds()
                    / 60
                )
                session_lengths.append(length)

        if len(session_lengths) < 3:
            return "insufficient_data"

        # Compare first and last third
        third = len(session_lengths) // 3
        first_third_avg = statistics.mean(session_lengths[:third]) if third > 0 else 0
        last_third_avg = statistics.mean(session_lengths[-third:]) if third > 0 else 0

        if last_third_avg > first_third_avg * 1.3:
            return "longer"
        elif last_third_avg < first_third_avg * 0.7:
            return "shorter"
        else:
            return "stable"

    def _analyze_engagement_trend(
        self, interaction_history: List[InteractionData]
    ) -> str:
        """Analyze engagement trend."""
        engagement_scores = []
        for interaction in interaction_history:
            # Calculate engagement based on message length and satisfaction
            message_length = len(interaction.conversation_context.message_text)
            satisfaction = interaction.user_satisfaction or 0.5

            engagement = min(message_length / 200, 1.0) * 0.6 + satisfaction * 0.4
            engagement_scores.append(engagement)

        if len(engagement_scores) < 3:
            return "insufficient_data"

        # Compare first and last third
        third = len(engagement_scores) // 3
        first_third_avg = statistics.mean(engagement_scores[:third]) if third > 0 else 0
        last_third_avg = statistics.mean(engagement_scores[-third:]) if third > 0 else 0

        if last_third_avg > first_third_avg * 1.2:
            return "improving"
        elif last_third_avg < first_third_avg * 0.8:
            return "declining"
        else:
            return "stable"

    def _calculate_topic_diversity(
        self, interaction_history: List[InteractionData]
    ) -> float:
        """Calculate topic diversity score."""
        all_topics = []
        for interaction in interaction_history:
            all_topics.extend(interaction.topics_discussed)

        if not all_topics:
            return 0.0

        unique_topics = set(all_topics)
        return min(len(unique_topics) / 10, 1.0)  # Normalize to 0-1

    def _calculate_consistency_score(
        self, interaction_history: List[InteractionData]
    ) -> float:
        """Calculate interaction consistency score."""
        if len(interaction_history) < 2:
            return 0.0

        # Calculate gaps between interactions
        gaps = []
        for i in range(1, len(interaction_history)):
            gap = (
                interaction_history[i].conversation_context.timestamp
                - interaction_history[i - 1].conversation_context.timestamp
            ).days
            gaps.append(gap)

        if not gaps:
            return 0.0

        # Consistency is inverse of gap variance
        gap_variance = statistics.variance(gaps) if len(gaps) > 1 else 0
        consistency = 1.0 / (1.0 + gap_variance / 10)  # Normalize

        return min(consistency, 1.0)

    def _identify_preferred_times(
        self, interaction_history: List[InteractionData]
    ) -> List[int]:
        """Identify preferred interaction times."""
        hours = [
            interaction.conversation_context.timestamp.hour
            for interaction in interaction_history
        ]

        # Count frequency of each hour
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # Return hours with above-average frequency
        if not hour_counts:
            return []

        avg_count = sum(hour_counts.values()) / len(hour_counts)
        preferred = [hour for hour, count in hour_counts.items() if count > avg_count]

        return sorted(preferred)

    def _calculate_interaction_gaps(
        self, interaction_history: List[InteractionData]
    ) -> List[int]:
        """Calculate gaps between interactions."""
        gaps = []
        for i in range(1, len(interaction_history)):
            gap = (
                interaction_history[i].conversation_context.timestamp
                - interaction_history[i - 1].conversation_context.timestamp
            ).days
            gaps.append(gap)

        return gaps

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize analysis thresholds."""
        return {
            "high_trust": 0.7,
            "low_trust": 0.3,
            "high_engagement": 0.7,
            "low_engagement": 0.3,
            "stable_emotion_threshold": 0.6,
            "positive_emotion_threshold": 0.6,
            "consistency_threshold": 0.5,
            "health_excellent": 0.8,
            "health_concerning": 0.4,
        }

    # Additional helper methods for emotional, trust, and communication analysis
    # (Implementation continues with similar pattern for remaining methods)

    def _calculate_emotional_stability(
        self, emotional_history: List[EmotionalState]
    ) -> float:
        """Calculate emotional stability score."""
        if len(emotional_history) < 3:
            return 0.5

        # Calculate variance in emotional intensity
        intensities = [state.intensity for state in emotional_history]
        intensity_variance = statistics.variance(intensities)

        # Stability is inverse of variance
        stability = 1.0 / (1.0 + intensity_variance * 2)
        return min(stability, 1.0)

    def _identify_dominant_emotions(
        self, emotional_history: List[EmotionalState]
    ) -> List[str]:
        """Identify dominant emotions."""
        emotion_counts = {}
        for state in emotional_history:
            emotion = state.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Return top 3 emotions
        sorted_emotions = sorted(
            emotion_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [emotion for emotion, count in sorted_emotions[:3]]

    def _calculate_emotional_range(
        self, emotional_history: List[EmotionalState]
    ) -> float:
        """Calculate breadth of emotions expressed."""
        unique_emotions = {state.primary_emotion.value for state in emotional_history}
        return min(len(unique_emotions) / 7, 1.0)  # 7 basic emotions

    def _calculate_positive_ratio(
        self, emotional_history: List[EmotionalState]
    ) -> float:
        """Calculate ratio of positive to negative emotions."""
        positive_emotions = ["joy", "surprise"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]

        positive_count = sum(
            1
            for state in emotional_history
            if state.primary_emotion.value in positive_emotions
        )
        negative_count = sum(
            1
            for state in emotional_history
            if state.primary_emotion.value in negative_emotions
        )

        total_emotional = positive_count + negative_count
        if total_emotional == 0:
            return 0.5

        return positive_count / total_emotional

    # Serialization methods for returning analysis results
    def _serialize_interaction_patterns(
        self, patterns: InteractionPattern
    ) -> Dict[str, Any]:
        """Serialize interaction patterns for JSON response."""
        return {
            "frequency_trend": patterns.frequency_trend,
            "session_length_trend": patterns.session_length_trend,
            "engagement_trend": patterns.engagement_trend,
            "topic_diversity": patterns.topic_diversity,
            "consistency_score": patterns.consistency_score,
            "preferred_times": patterns.preferred_times,
            "average_interaction_gap": (
                statistics.mean(patterns.interaction_gaps)
                if patterns.interaction_gaps
                else 0
            ),
        }

    def _serialize_emotional_dynamics(
        self, dynamics: EmotionalDynamics
    ) -> Dict[str, Any]:
        """Serialize emotional dynamics for JSON response."""
        return {
            "emotional_stability": dynamics.emotional_stability,
            "dominant_emotions": dynamics.dominant_emotions,
            "emotional_range": dynamics.emotional_range,
            "positive_ratio": dynamics.positive_ratio,
            "emotional_responsiveness": dynamics.emotional_responsiveness,
            "mood_correlation": dynamics.mood_correlation,
            "emotional_growth": dynamics.emotional_growth,
        }

    def _serialize_trust_dynamics(self, dynamics: TrustDynamics) -> Dict[str, Any]:
        """Serialize trust dynamics for JSON response."""
        return {
            "trust_trajectory": dynamics.trust_trajectory,
            "trust_velocity": dynamics.trust_velocity,
            "trust_milestones": dynamics.trust_milestones,
            "vulnerability_indicators": dynamics.vulnerability_indicators,
            "consistency_impact": dynamics.consistency_impact,
            "trust_recovery_ability": dynamics.trust_recovery_ability,
        }

    def _serialize_communication_dynamics(
        self, dynamics: CommunicationDynamics
    ) -> Dict[str, Any]:
        """Serialize communication dynamics for JSON response."""
        return {
            "style_evolution": dynamics.style_evolution.value,
            "depth_progression": dynamics.depth_progression,
            "topic_evolution": dynamics.topic_evolution,
            "response_quality_trend": dynamics.response_quality_trend,
            "mutual_understanding": dynamics.mutual_understanding,
            "communication_efficiency": dynamics.communication_efficiency,
        }

    def _serialize_relationship_health(
        self, health: RelationshipHealth
    ) -> Dict[str, Any]:
        """Serialize relationship health for JSON response."""
        return {
            "health_score": health.health_score,
            "strength_areas": health.strength_areas,
            "growth_areas": health.growth_areas,
            "risk_factors": health.risk_factors,
            "resilience_score": health.resilience_score,
            "sustainability_outlook": health.sustainability_outlook,
        }

    def _generate_recommendations(
        self,
        pattern: DynamicsPattern,
        health: RelationshipHealth,
        user_profile: CompanionProfile,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Pattern-based recommendations
        if pattern == DynamicsPattern.DECLINING:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "relationship_recovery",
                    "action": "Focus on rebuilding trust and engagement",
                    "specific_steps": [
                        "Acknowledge any recent issues",
                        "Increase interaction frequency",
                        "Provide more emotional support",
                    ],
                }
            )

        elif pattern == DynamicsPattern.STAGNANT:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "relationship_growth",
                    "action": "Introduce new elements to stimulate growth",
                    "specific_steps": [
                        "Explore new topics of interest",
                        "Set relationship goals together",
                        "Create meaningful milestones",
                    ],
                }
            )

        # Health-based recommendations
        if health.health_score < 0.5:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "health_improvement",
                    "action": "Address fundamental relationship health issues",
                    "specific_steps": health.growth_areas[:3],
                }
            )

        # Risk-based recommendations
        if health.risk_factors:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "risk_mitigation",
                    "action": "Address identified risk factors",
                    "specific_steps": [
                        f"Mitigate: {risk}" for risk in health.risk_factors[:3]
                    ],
                }
            )

        return recommendations

    def _assess_emotional_responsiveness(
        self,
        emotional_history: List[EmotionalState],
        interaction_history: List[InteractionData],
    ) -> float:
        """
        Assess how well Morgan responds to user's emotional states.

        Calculates based on:
        - Correlation between detected emotions and response appropriateness
        - Time to emotional acknowledgment
        - Emotional alignment scores
        """
        if len(emotional_history) < 3 or len(interaction_history) < 3:
            return 0.5  # Neutral default for insufficient data

        responsiveness_scores = []

        for i, interaction in enumerate(interaction_history):
            # Find matching emotional state
            matching_emotions = [
                e
                for e in emotional_history
                if abs(
                    (
                        e.timestamp - interaction.conversation_context.timestamp
                    ).total_seconds()
                )
                < 60
            ]

            if matching_emotions:
                emotion = matching_emotions[0]

                # Score based on user satisfaction relative to emotional intensity
                satisfaction = interaction.user_satisfaction or 0.5

                # Higher score if satisfaction is good during negative emotions
                # (indicating empathetic response)
                if emotion.primary_emotion.value in ["sadness", "fear", "anger"]:
                    if satisfaction > 0.6:
                        responsiveness_scores.append(1.0)  # Good empathetic response
                    elif satisfaction > 0.4:
                        responsiveness_scores.append(0.6)
                    else:
                        responsiveness_scores.append(0.3)
                else:
                    # For positive/neutral emotions, satisfaction directly indicates responsiveness
                    responsiveness_scores.append(satisfaction)

        if responsiveness_scores:
            return min(1.0, statistics.mean(responsiveness_scores))

        return 0.5

    def _calculate_mood_correlation(
        self, emotional_history: List[EmotionalState]
    ) -> float:
        """
        Calculate correlation between consecutive emotional states.

        High correlation indicates stable emotional patterns,
        low correlation indicates volatility.
        """
        if len(emotional_history) < 4:
            return 0.0

        # Calculate intensity changes between consecutive emotions
        intensity_changes = []
        for i in range(1, len(emotional_history)):
            prev_intensity = emotional_history[i - 1].intensity
            curr_intensity = emotional_history[i].intensity
            change = abs(curr_intensity - prev_intensity)
            intensity_changes.append(change)

        # Also consider emotion type changes
        emotion_consistency = []
        for i in range(1, len(emotional_history)):
            prev_emotion = emotional_history[i - 1].primary_emotion.value
            curr_emotion = emotional_history[i].primary_emotion.value
            # Same emotion = 1.0, different = 0.0
            emotion_consistency.append(1.0 if prev_emotion == curr_emotion else 0.0)

        # Correlation is inverse of average change
        avg_change = statistics.mean(intensity_changes) if intensity_changes else 0
        stability_score = 1.0 - min(avg_change, 1.0)

        # Combine with emotion consistency
        consistency_score = (
            statistics.mean(emotion_consistency) if emotion_consistency else 0.5
        )

        return stability_score * 0.6 + consistency_score * 0.4

    def _determine_emotional_growth(
        self, emotional_history: List[EmotionalState]
    ) -> str:
        """
        Determine the trend in emotional patterns over time.

        Returns: "improving", "stable", "concerning"
        """
        if len(emotional_history) < 6:
            return "stable"

        # Split into early and recent periods
        mid_point = len(emotional_history) // 2
        early_period = emotional_history[:mid_point]
        recent_period = emotional_history[mid_point:]

        # Calculate positive emotion ratios
        positive_emotions = ["joy", "surprise", "trust", "anticipation"]

        early_positive = sum(
            1 for e in early_period if e.primary_emotion.value in positive_emotions
        ) / len(early_period)

        recent_positive = sum(
            1 for e in recent_period if e.primary_emotion.value in positive_emotions
        ) / len(recent_period)

        # Calculate intensity trends
        early_avg_intensity = statistics.mean([e.intensity for e in early_period])
        recent_avg_intensity = statistics.mean([e.intensity for e in recent_period])

        # Determine growth direction
        positive_change = recent_positive - early_positive

        if positive_change > 0.1:
            return "improving"
        elif positive_change < -0.1:
            return "concerning"
        else:
            return "stable"

    def _determine_trust_trajectory(
        self, user_profile: CompanionProfile, interaction_history: List[InteractionData]
    ) -> str:
        """
        Determine trust trajectory based on multiple factors.

        Returns: "building", "stable", "declining"
        """
        current_trust = user_profile.trust_level

        if len(interaction_history) < 5:
            return "stable"

        # Analyze trust indicators over time
        # Trust increases with: positive interactions, vulnerability sharing, consistency

        # Check recent satisfaction trend
        recent_interactions = interaction_history[-10:]
        satisfactions = [
            i.user_satisfaction
            for i in recent_interactions
            if i.user_satisfaction is not None
        ]

        if len(satisfactions) >= 3:
            # Split into halves and compare
            mid = len(satisfactions) // 2
            early_avg = statistics.mean(satisfactions[:mid])
            recent_avg = statistics.mean(satisfactions[mid:])

            if recent_avg > early_avg + 0.1:
                return "building"
            elif recent_avg < early_avg - 0.1:
                return "declining"

        # Check milestone achievements
        trust_milestones = [
            m
            for m in user_profile.relationship_milestones
            if "trust" in m.milestone_type.value.lower()
        ]

        if len(trust_milestones) >= 2:
            return "building"

        # Use current trust level as indicator
        if current_trust > 0.7:
            return "stable"  # High trust tends to stabilize
        elif current_trust > 0.4:
            return "building"
        else:
            return "stable"

    def _calculate_trust_velocity(
        self, user_profile: CompanionProfile, interaction_history: List[InteractionData]
    ) -> float:
        """
        Calculate rate of trust change over time.

        Returns: value between -1.0 and 1.0
        Positive = trust increasing, Negative = trust decreasing
        """
        if len(interaction_history) < 5:
            return 0.0

        # Estimate trust change based on interaction quality
        recent_interactions = interaction_history[-20:]

        # Calculate weighted satisfaction scores (more recent = higher weight)
        weighted_scores = []
        for i, interaction in enumerate(recent_interactions):
            if interaction.user_satisfaction is not None:
                weight = (i + 1) / len(recent_interactions)  # Increasing weight
                weighted_scores.append((interaction.user_satisfaction, weight))

        if len(weighted_scores) < 3:
            return 0.0

        # Calculate trend
        total_weight = sum(w for _, w in weighted_scores)
        early_weighted = (
            sum(s * w for s, w in weighted_scores[: len(weighted_scores) // 2])
            / (total_weight / 2)
            if total_weight > 0
            else 0
        )

        late_weighted = (
            sum(s * w for s, w in weighted_scores[len(weighted_scores) // 2 :])
            / (total_weight / 2)
            if total_weight > 0
            else 0
        )

        velocity = late_weighted - early_weighted

        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, velocity * 2))

    def _identify_trust_milestones(self, user_profile: CompanionProfile) -> List[str]:
        """Identify trust-related milestones achieved."""
        trust_keywords = [
            "trust",
            "confidence",
            "reliable",
            "vulnerable",
            "share",
            "open",
        ]

        trust_milestones = []
        for milestone in user_profile.relationship_milestones:
            milestone_name = milestone.milestone_type.value.lower()
            if any(keyword in milestone_name for keyword in trust_keywords):
                trust_milestones.append(milestone.milestone_type.value)

        return trust_milestones

    def _count_vulnerability_indicators(
        self, interaction_history: List[InteractionData]
    ) -> int:
        """
        Count interactions where user showed vulnerability.

        Vulnerability indicators:
        - Personal/emotional topics
        - Questions about difficult topics
        - Expressions of uncertainty or need for support
        """
        vulnerability_keywords = [
            "feel",
            "worried",
            "scared",
            "anxious",
            "lonely",
            "sad",
            "help me",
            "advice",
            "confused",
            "struggling",
            "difficult",
            "personal",
            "private",
            "trust",
            "secret",
            "vulnerable",
        ]

        vulnerability_count = 0

        for interaction in interaction_history:
            message_text = interaction.conversation_context.message_text.lower()

            # Check for vulnerability keywords
            if any(keyword in message_text for keyword in vulnerability_keywords):
                vulnerability_count += 1

            # Check for emotional intensity in detected emotions
            # (would need emotion data attached to interaction)
            if hasattr(interaction, "detected_emotions"):
                for emotion in getattr(interaction, "detected_emotions", []):
                    if emotion.get("intensity", 0) > 0.7:
                        vulnerability_count += 1
                        break

        return vulnerability_count

    def _assess_consistency_impact(
        self, user_profile: CompanionProfile, interaction_history: List[InteractionData]
    ) -> float:
        """
        Assess how consistency affects the relationship.

        High consistency correlates with trust building.
        """
        if len(interaction_history) < 5:
            return 0.5

        # Calculate interaction consistency
        consistency_score = self._calculate_consistency_score(interaction_history)

        # Calculate satisfaction consistency
        satisfactions = [
            i.user_satisfaction
            for i in interaction_history
            if i.user_satisfaction is not None
        ]

        if len(satisfactions) >= 3:
            try:
                satisfaction_variance = statistics.variance(satisfactions)
                satisfaction_consistency = 1.0 / (1.0 + satisfaction_variance * 5)
            except statistics.StatisticsError:
                satisfaction_consistency = 0.5
        else:
            satisfaction_consistency = 0.5

        # Combine consistency measures
        overall_consistency = (consistency_score + satisfaction_consistency) / 2

        # Impact is the correlation between consistency and current trust
        trust = user_profile.trust_level

        # If high consistency and high trust, consistency has positive impact
        if overall_consistency > 0.6 and trust > 0.6:
            return 0.8
        elif overall_consistency < 0.4 and trust < 0.4:
            return 0.3  # Low consistency, low trust - negative impact
        else:
            return 0.5  # Neutral

    def _assess_trust_recovery_ability(self, user_profile: CompanionProfile) -> float:
        """
        Assess ability to recover from trust issues.

        Based on:
        - Relationship resilience (milestone count, duration)
        - Past recovery patterns
        - Current trust buffer
        """
        # Milestone diversity indicates relationship resilience
        milestone_count = len(user_profile.relationship_milestones)
        milestone_factor = min(milestone_count / 5, 1.0)

        # Relationship age provides stability
        relationship_days = user_profile.get_relationship_age_days()
        age_factor = min(relationship_days / 90, 1.0)  # Cap at 90 days

        # Current trust level provides buffer
        trust_buffer = user_profile.trust_level

        # Engagement indicates investment in relationship
        engagement_factor = user_profile.engagement_score

        # Combine factors
        recovery_ability = (
            milestone_factor * 0.25
            + age_factor * 0.25
            + trust_buffer * 0.25
            + engagement_factor * 0.25
        )

        return min(1.0, recovery_ability)

    def _analyze_style_evolution(
        self, interaction_history: List[InteractionData]
    ) -> CommunicationStyle:
        """
        Analyze how communication style has evolved.
        """
        if len(interaction_history) < 6:
            return CommunicationStyle.STABLE

        # Split into early and recent
        mid_point = len(interaction_history) // 2
        early_interactions = interaction_history[:mid_point]
        recent_interactions = interaction_history[mid_point:]

        # Analyze message characteristics
        def analyze_formality(interactions):
            formal_indicators = 0
            casual_indicators = 0

            for interaction in interactions:
                text = interaction.conversation_context.message_text.lower()

                # Formal indicators
                if any(
                    word in text
                    for word in ["please", "thank you", "would you", "could you"]
                ):
                    formal_indicators += 1

                # Casual indicators
                if any(word in text for word in ["hey", "cool", "lol", "haha", "yeah"]):
                    casual_indicators += 1

            return formal_indicators, casual_indicators

        early_formal, early_casual = analyze_formality(early_interactions)
        recent_formal, recent_casual = analyze_formality(recent_interactions)

        early_ratio = early_formal / max(early_casual, 1)
        recent_ratio = recent_formal / max(recent_casual, 1)

        if recent_ratio < early_ratio * 0.7:
            return CommunicationStyle.FORMAL_TO_CASUAL
        elif recent_ratio > early_ratio * 1.3:
            return CommunicationStyle.CASUAL_TO_FORMAL

        # Check for deepening
        early_msg_lengths = [
            len(i.conversation_context.message_text) for i in early_interactions
        ]
        recent_msg_lengths = [
            len(i.conversation_context.message_text) for i in recent_interactions
        ]

        if (
            statistics.mean(recent_msg_lengths)
            > statistics.mean(early_msg_lengths) * 1.3
        ):
            return CommunicationStyle.DEEPENING

        # Check topic patterns
        technical_topics = ["code", "programming", "technical", "algorithm", "data"]
        emotional_topics = ["feel", "emotion", "happy", "sad", "worried"]

        recent_text = " ".join(
            [i.conversation_context.message_text.lower() for i in recent_interactions]
        )

        technical_count = sum(1 for word in technical_topics if word in recent_text)
        emotional_count = sum(1 for word in emotional_topics if word in recent_text)

        if technical_count > emotional_count * 2:
            return CommunicationStyle.TECHNICAL_FOCUS
        elif emotional_count > technical_count * 2:
            return CommunicationStyle.EMOTIONAL_FOCUS

        return (
            CommunicationStyle.DEEPENING
        )  # Default to deepening for positive trajectory

    def _analyze_depth_progression(
        self, interaction_history: List[InteractionData]
    ) -> str:
        """
        Analyze whether conversation depth is increasing.

        Returns: "deepening", "stable", "superficial"
        """
        if len(interaction_history) < 6:
            return "stable"

        # Metrics for depth:
        # 1. Message length (longer = deeper)
        # 2. Topic diversity
        # 3. Emotional content
        # 4. Question complexity

        mid_point = len(interaction_history) // 2
        early = interaction_history[:mid_point]
        recent = interaction_history[mid_point:]

        # Message length analysis
        early_avg_length = statistics.mean(
            [len(i.conversation_context.message_text) for i in early]
        )
        recent_avg_length = statistics.mean(
            [len(i.conversation_context.message_text) for i in recent]
        )

        length_change = (recent_avg_length - early_avg_length) / max(
            early_avg_length, 1
        )

        # Topic diversity
        early_topics = set()
        recent_topics = set()
        for i in early:
            early_topics.update(i.topics_discussed)
        for i in recent:
            recent_topics.update(i.topics_discussed)

        topic_growth = len(recent_topics) - len(early_topics)

        # Combine metrics
        if length_change > 0.2 and topic_growth > 0:
            return "deepening"
        elif length_change < -0.2:
            return "superficial"
        else:
            return "stable"

    def _analyze_topic_evolution(
        self, interaction_history: List[InteractionData]
    ) -> Dict[str, float]:
        """
        Analyze how topic preferences have changed over time.

        Returns: Dict mapping topic categories to change scores (-1 to 1)
        """
        if len(interaction_history) < 6:
            return {}

        mid_point = len(interaction_history) // 2
        early = interaction_history[:mid_point]
        recent = interaction_history[mid_point:]

        # Categorize topics
        topic_categories = {
            "technical": ["code", "programming", "software", "algorithm", "debug"],
            "emotional": ["feel", "emotion", "happy", "sad", "worried"],
            "personal": ["life", "family", "friend", "work", "hobby"],
            "learning": ["learn", "study", "understand", "explain", "teach"],
            "creative": ["create", "design", "art", "write", "idea"],
        }

        def count_category_mentions(interactions):
            counts = {cat: 0 for cat in topic_categories}
            for interaction in interactions:
                text = interaction.conversation_context.message_text.lower()
                topics = " ".join(interaction.topics_discussed).lower()
                combined = text + " " + topics

                for category, keywords in topic_categories.items():
                    if any(kw in combined for kw in keywords):
                        counts[category] += 1

            return counts

        early_counts = count_category_mentions(early)
        recent_counts = count_category_mentions(recent)

        # Calculate change for each category
        evolution = {}
        for category in topic_categories:
            early_rate = early_counts[category] / max(len(early), 1)
            recent_rate = recent_counts[category] / max(len(recent), 1)

            # Normalize to -1 to 1 range
            change = recent_rate - early_rate
            evolution[category] = max(-1.0, min(1.0, change * 5))

        return evolution

    def _analyze_response_quality_trend(
        self, interaction_history: List[InteractionData]
    ) -> str:
        """
        Analyze trend in response quality based on user satisfaction.

        Returns: "improving", "stable", "declining"
        """
        if len(interaction_history) < 6:
            return "stable"

        # Get satisfaction scores
        satisfactions = [
            (i, i.user_satisfaction)
            for i in range(len(interaction_history))
            if interaction_history[i].user_satisfaction is not None
        ]

        if len(satisfactions) < 4:
            return "stable"

        # Linear regression approximation
        n = len(satisfactions)
        x_mean = sum(i for i, _ in satisfactions) / n
        y_mean = sum(s for _, s in satisfactions) / n

        numerator = sum((i - x_mean) * (s - y_mean) for i, s in satisfactions)
        denominator = sum((i - x_mean) ** 2 for i, _ in satisfactions)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Interpret slope
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def _assess_mutual_understanding(
        self, interaction_history: List[InteractionData]
    ) -> float:
        """
        Assess level of mutual understanding based on interaction patterns.
        """
        if len(interaction_history) < 5:
            return 0.5

        understanding_indicators = []

        for interaction in interaction_history:
            indicators = 0.5  # Base score

            # High satisfaction indicates good understanding
            if interaction.user_satisfaction:
                indicators = interaction.user_satisfaction

            # Topic follow-through indicates understanding
            if len(interaction.topics_discussed) > 0:
                indicators += 0.1

            # Longer sessions might indicate engagement
            if interaction.conversation_context.session_duration:
                duration_mins = (
                    interaction.conversation_context.session_duration.total_seconds()
                    / 60
                )
                if duration_mins > 5:
                    indicators += 0.1

            understanding_indicators.append(min(1.0, indicators))

        return statistics.mean(understanding_indicators)

    def _calculate_communication_efficiency(
        self, interaction_history: List[InteractionData], user_profile: CompanionProfile
    ) -> float:
        """
        Calculate communication efficiency score.

        Efficiency = achieving goals with minimal friction/confusion
        """
        if len(interaction_history) < 5:
            return 0.5

        # Factors:
        # 1. Satisfaction per message length ratio
        # 2. Topic resolution (staying on topic)
        # 3. Clarification requests (fewer = more efficient)

        efficiency_scores = []

        for interaction in interaction_history:
            score = 0.5

            if interaction.user_satisfaction:
                # High satisfaction with shorter messages = efficient
                msg_length = len(interaction.conversation_context.message_text)
                satisfaction = interaction.user_satisfaction

                if satisfaction > 0.7 and msg_length < 500:
                    score = 0.9
                elif satisfaction > 0.5:
                    score = 0.7
                elif satisfaction < 0.3:
                    score = 0.3

            efficiency_scores.append(score)

        # Factor in relationship maturity
        interaction_count = user_profile.interaction_count
        maturity_bonus = min(interaction_count / 50, 0.1)  # Max 0.1 bonus

        return min(1.0, statistics.mean(efficiency_scores) + maturity_bonus)

    def _identify_trajectory_factors(
        self, dynamics_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify key factors affecting relationship trajectory."""
        factors = []

        # Check health metrics
        health = dynamics_analysis.get("relationship_health", {})

        if health.get("health_score", 0) > 0.7:
            factors.append("strong_overall_health")
        elif health.get("health_score", 0) < 0.4:
            factors.append("health_needs_attention")

        # Check emotional dynamics
        emotional = dynamics_analysis.get("emotional_dynamics", {})
        if emotional.get("positive_ratio", 0.5) > 0.7:
            factors.append("positive_emotional_trend")
        elif emotional.get("positive_ratio", 0.5) < 0.3:
            factors.append("negative_emotional_trend")

        # Check interaction patterns
        patterns = dynamics_analysis.get("interaction_patterns", {})
        if patterns.get("frequency_trend") == "increasing":
            factors.append("increasing_engagement")
        elif patterns.get("frequency_trend") == "decreasing":
            factors.append("decreasing_engagement")

        # Check trust
        trust = dynamics_analysis.get("trust_dynamics", {})
        if trust.get("trust_trajectory") == "building":
            factors.append("trust_building")
        elif trust.get("trust_trajectory") == "declining":
            factors.append("trust_declining")

        return factors if factors else ["stable_baseline"]

    def _identify_intervention_opportunities(
        self, dynamics_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify opportunities for positive intervention."""
        opportunities = []

        health = dynamics_analysis.get("relationship_health", {})
        growth_areas = health.get("growth_areas", [])

        for area in growth_areas:
            if "trust" in area.lower():
                opportunities.append("build_trust_through_consistency")
            if "engagement" in area.lower():
                opportunities.append("increase_interaction_frequency")
            if "emotional" in area.lower():
                opportunities.append("provide_more_emotional_support")
            if "communication" in area.lower():
                opportunities.append("improve_clarity_and_understanding")

        # Check for stagnation
        if dynamics_analysis.get("overall_pattern") == "stagnant":
            opportunities.append("introduce_new_topics")
            opportunities.append("set_collaborative_goals")

        return opportunities if opportunities else ["maintain_current_approach"]

    def _predict_milestone_opportunities(
        self, user_profile: CompanionProfile, pattern: DynamicsPattern
    ) -> List[str]:
        """Predict upcoming milestone opportunities based on pattern."""
        opportunities = []

        achieved_types = {
            m.milestone_type.value for m in user_profile.relationship_milestones
        }

        # Based on current pattern, suggest likely milestones
        if pattern == DynamicsPattern.GROWING:
            if "first_deep_conversation" not in achieved_types:
                opportunities.append("first_deep_conversation")
            if "trust_milestone" not in achieved_types:
                opportunities.append("trust_milestone")
            if user_profile.interaction_count > 20:
                opportunities.append("regular_user_milestone")

        elif pattern == DynamicsPattern.STABLE:
            if "consistent_engagement" not in achieved_types:
                opportunities.append("consistent_engagement")
            opportunities.append("knowledge_sharing_milestone")

        elif pattern == DynamicsPattern.RECOVERING:
            opportunities.append("relationship_recovery_milestone")
            opportunities.append("renewed_trust_milestone")

        # Time-based milestones
        relationship_days = user_profile.get_relationship_age_days()
        if relationship_days > 30 and "month_anniversary" not in achieved_types:
            opportunities.append("month_anniversary")
        if relationship_days > 90 and "quarter_anniversary" not in achieved_types:
            opportunities.append("quarter_anniversary")

        return opportunities if opportunities else ["continue_building_relationship"]
