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

    # Placeholder implementations for remaining helper methods
    def _assess_emotional_responsiveness(
        self, emotional_history, interaction_history
    ) -> float:
        return 0.7  # Simplified implementation

    def _calculate_mood_correlation(self, emotional_history) -> float:
        return 0.6  # Simplified implementation

    def _determine_emotional_growth(self, emotional_history) -> str:
        return "stable"  # Simplified implementation

    def _determine_trust_trajectory(self, user_profile, interaction_history) -> str:
        return "building" if user_profile.trust_level > 0.5 else "stable"

    def _calculate_trust_velocity(self, user_profile, interaction_history) -> float:
        return 0.1  # Simplified implementation

    def _identify_trust_milestones(self, user_profile) -> List[str]:
        return [
            m.milestone_type.value
            for m in user_profile.relationship_milestones
            if "trust" in m.milestone_type.value
        ]

    def _count_vulnerability_indicators(self, interaction_history) -> int:
        return 2  # Simplified implementation

    def _assess_consistency_impact(self, user_profile, interaction_history) -> float:
        return 0.6  # Simplified implementation

    def _assess_trust_recovery_ability(self, user_profile) -> float:
        return 0.7  # Simplified implementation

    def _analyze_style_evolution(self, interaction_history) -> CommunicationStyle:
        return CommunicationStyle.DEEPENING  # Simplified implementation

    def _analyze_depth_progression(self, interaction_history) -> str:
        return "deepening"  # Simplified implementation

    def _analyze_topic_evolution(self, interaction_history) -> Dict[str, float]:
        return {}  # Simplified implementation

    def _analyze_response_quality_trend(self, interaction_history) -> str:
        return "improving"  # Simplified implementation

    def _assess_mutual_understanding(self, interaction_history) -> float:
        return 0.7  # Simplified implementation

    def _calculate_communication_efficiency(
        self, interaction_history, user_profile
    ) -> float:
        return 0.6  # Simplified implementation

    def _identify_trajectory_factors(self, dynamics_analysis) -> List[str]:
        return ["trust_level", "engagement_score"]  # Simplified implementation

    def _identify_intervention_opportunities(self, dynamics_analysis) -> List[str]:
        return ["increase_interaction_frequency"]  # Simplified implementation

    def _predict_milestone_opportunities(self, user_profile, pattern) -> List[str]:
        return ["trust_building", "learning_milestone"]  # Simplified implementation
