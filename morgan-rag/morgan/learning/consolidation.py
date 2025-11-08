"""
Knowledge Consolidation Engine for Morgan RAG.

Consolidates short-term patterns into long-term knowledge, synthesizes learning
from multiple sources, extracts meta-patterns, and manages periodic consolidation
to build a stable, evolving user understanding.
"""

import statistics
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .adaptation import AdaptationResult
from .feedback import FeedbackAnalysis
from .patterns import BehavioralPattern, CommunicationPattern, InteractionPatterns
from .preferences import PreferenceCategory, UserPreferenceProfile

logger = get_logger(__name__)


class ConsolidationType(Enum):
    """Types of consolidation operations."""

    PATTERN = "pattern"
    PREFERENCE = "preference"
    BEHAVIOR = "behavior"
    KNOWLEDGE = "knowledge"
    META_PATTERN = "meta_pattern"


class ConsolidationPeriod(Enum):
    """Periods for consolidation scheduling."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class KnowledgeStability(Enum):
    """Stability levels for consolidated knowledge."""

    VOLATILE = "volatile"  # Recent, may change (< 1 week)
    STABLE = "stable"  # Consistent over time (1-4 weeks)
    ESTABLISHED = "established"  # Well-established (1-3 months)
    PERMANENT = "permanent"  # Long-term, unlikely to change (> 3 months)


class ConsolidationStrategy(Enum):
    """Strategies for consolidating patterns."""

    MERGE = "merge"  # Merge similar patterns
    AGGREGATE = "aggregate"  # Aggregate multiple observations
    PROMOTE = "promote"  # Promote short-term to long-term
    VALIDATE = "validate"  # Validate existing patterns
    PRUNE = "prune"  # Remove outdated/invalid patterns


@dataclass
class ConsolidatedPattern:
    """
    Represents a consolidated pattern from multiple short-term observations.

    Combines multiple pattern observations into a stable, long-term pattern
    with higher confidence and broader evidence base.
    """

    pattern_id: str
    user_id: str
    pattern_type: str  # communication, topic, timing, behavioral
    pattern_data: Dict[str, Any]  # Consolidated pattern attributes
    confidence: float  # 0.0 to 1.0
    stability: KnowledgeStability
    evidence_count: int  # Number of observations supporting this pattern
    first_observed: datetime
    last_observed: datetime
    observation_span: timedelta  # Time span of observations
    source_pattern_ids: List[str]  # IDs of patterns consolidated into this one
    validation_score: float  # 0.0 to 1.0 - how well validated
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize pattern ID if not provided."""
        if not self.pattern_id:
            self.pattern_id = str(uuid.uuid4())


@dataclass
class ConsolidatedKnowledge:
    """
    Represents synthesized long-term knowledge about a user.

    Combines insights from patterns, preferences, feedback, and adaptations
    to form a coherent understanding of user characteristics.
    """

    knowledge_id: str
    user_id: str
    knowledge_domain: str  # e.g., "communication_style", "topic_interests"
    knowledge_items: Dict[str, Any]  # Consolidated knowledge facts
    confidence_scores: Dict[str, float]  # Confidence in each knowledge item
    supporting_evidence: Dict[str, List[str]]  # Evidence for each item
    stability: KnowledgeStability
    learning_sources: List[str]  # Sources: patterns, preferences, feedback, etc.
    synthesis_reasoning: str  # Why this knowledge was synthesized this way
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def __post_init__(self):
        """Initialize knowledge ID if not provided."""
        if not self.knowledge_id:
            self.knowledge_id = str(uuid.uuid4())


@dataclass
class MetaPattern:
    """
    Represents patterns about patterns - meta-level insights.

    Identifies higher-order patterns such as learning trajectories,
    behavior evolution, and characteristic changes over time.
    """

    meta_pattern_id: str
    user_id: str
    pattern_category: str  # e.g., "learning_trajectory", "behavior_evolution"
    description: str
    detected_trend: str  # e.g., "increasing_formality", "expanding_interests"
    trend_direction: float  # -1.0 (decreasing) to 1.0 (increasing)
    trend_strength: float  # 0.0 to 1.0
    time_span: timedelta
    affected_domains: List[str]  # Which domains this meta-pattern affects
    predictive_insights: List[str]  # Predictions based on this meta-pattern
    confidence: float  # 0.0 to 1.0
    supporting_patterns: List[str]  # IDs of patterns supporting this meta-pattern
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize meta-pattern ID if not provided."""
        if not self.meta_pattern_id:
            self.meta_pattern_id = str(uuid.uuid4())


@dataclass
class ConsolidationResult:
    """
    Result of a consolidation operation.

    Contains all consolidated patterns, knowledge, and meta-patterns
    produced during a consolidation cycle.
    """

    consolidation_id: str
    user_id: str
    consolidation_type: ConsolidationType
    consolidated_patterns: List[ConsolidatedPattern]
    consolidated_knowledge: List[ConsolidatedKnowledge]
    meta_patterns: List[MetaPattern]
    strategies_applied: List[ConsolidationStrategy]
    items_processed: int
    items_consolidated: int
    items_pruned: int
    confidence_improvement: float  # Average confidence increase
    consolidation_quality: float  # 0.0 to 1.0 - overall quality
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize consolidation ID if not provided."""
        if not self.consolidation_id:
            self.consolidation_id = str(uuid.uuid4())


@dataclass
class ConsolidationMetrics:
    """
    Metrics for tracking consolidation effectiveness.

    Measures the quality and impact of consolidation operations
    over time to optimize the consolidation process.
    """

    user_id: str
    total_consolidations: int
    successful_consolidations: int
    average_quality_score: float  # 0.0 to 1.0
    average_confidence_gain: float
    knowledge_stability_ratio: float  # Ratio of stable to volatile knowledge
    meta_patterns_discovered: int
    consolidation_efficiency: float  # Items consolidated / items processed
    last_consolidation: Optional[datetime] = None
    next_scheduled_consolidation: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PatternConsolidator:
    """
    Consolidates short-term patterns into long-term stable patterns.

    Analyzes multiple pattern observations over time, identifies consistent
    patterns, merges similar patterns, and promotes validated patterns to
    long-term knowledge.
    """

    # Consolidation thresholds
    MIN_OBSERVATIONS_FOR_STABLE = 3
    MIN_OBSERVATIONS_FOR_ESTABLISHED = 5
    MIN_OBSERVATIONS_FOR_PERMANENT = 10

    SIMILARITY_THRESHOLD = 0.7  # Pattern similarity threshold for merging

    def __init__(self):
        """Initialize pattern consolidator."""
        logger.debug("Pattern consolidator initialized")

    def consolidate_patterns(
        self, user_id: str, pattern_history: List[InteractionPatterns]
    ) -> List[ConsolidatedPattern]:
        """
        Consolidate multiple pattern observations into stable patterns.

        Args:
            user_id: User identifier
            pattern_history: List of interaction patterns over time

        Returns:
            List[ConsolidatedPattern]: Consolidated patterns
        """
        logger.info(f"Consolidating patterns for user {user_id}")

        if not pattern_history:
            return []

        consolidated = []

        # Consolidate communication patterns
        comm_patterns = self._consolidate_communication_patterns(
            user_id, pattern_history
        )
        consolidated.extend(comm_patterns)

        # Consolidate topic patterns
        topic_patterns = self._consolidate_topic_patterns(user_id, pattern_history)
        consolidated.extend(topic_patterns)

        # Consolidate timing patterns
        timing_patterns = self._consolidate_timing_patterns(user_id, pattern_history)
        consolidated.extend(timing_patterns)

        # Consolidate behavioral patterns
        behavioral_patterns = self._consolidate_behavioral_patterns(
            user_id, pattern_history
        )
        consolidated.extend(behavioral_patterns)

        logger.info(f"Consolidated {len(consolidated)} patterns for user {user_id}")
        return consolidated

    def _consolidate_communication_patterns(
        self, user_id: str, pattern_history: List[InteractionPatterns]
    ) -> List[ConsolidatedPattern]:
        """Consolidate communication patterns."""
        consolidated = []

        # Extract all communication patterns
        all_comm_patterns = []
        for patterns in pattern_history:
            all_comm_patterns.extend(patterns.communication_patterns)

        if not all_comm_patterns:
            return []

        # Aggregate communication style data
        style_counts = Counter(p.preferred_style for p in all_comm_patterns)
        formality_levels = [p.formality_level for p in all_comm_patterns]
        technical_depths = [p.technical_depth for p in all_comm_patterns]

        # Determine stability based on consistency
        stability = self._determine_stability(
            len(all_comm_patterns), self._calculate_consistency(formality_levels)
        )

        # Get dominant communication style
        dominant_style = style_counts.most_common(1)[0][0]
        style_confidence = style_counts[dominant_style] / len(all_comm_patterns)

        # Calculate validation score
        validation_score = self._calculate_validation_score(
            all_comm_patterns, len(pattern_history)
        )

        consolidated_pattern = ConsolidatedPattern(
            pattern_id=f"consolidated_comm_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_type="communication",
            pattern_data={
                "preferred_style": dominant_style.value,
                "formality_level": statistics.mean(formality_levels),
                "technical_depth": statistics.mean(technical_depths),
                "style_confidence": style_confidence,
                "vocabulary_level": self._aggregate_vocabulary_level(all_comm_patterns),
            },
            confidence=min(style_confidence * len(all_comm_patterns) / 10, 1.0),
            stability=stability,
            evidence_count=len(all_comm_patterns),
            first_observed=min(p.last_updated for p in all_comm_patterns),
            last_observed=max(p.last_updated for p in all_comm_patterns),
            observation_span=max(p.last_updated for p in all_comm_patterns)
            - min(p.last_updated for p in all_comm_patterns),
            source_pattern_ids=[p.pattern_id for p in all_comm_patterns],
            validation_score=validation_score,
        )

        consolidated.append(consolidated_pattern)
        return consolidated

    def _consolidate_topic_patterns(
        self, user_id: str, pattern_history: List[InteractionPatterns]
    ) -> List[ConsolidatedPattern]:
        """Consolidate topic patterns."""
        consolidated = []

        # Extract all topic patterns
        all_topic_patterns = []
        for patterns in pattern_history:
            all_topic_patterns.extend(patterns.topic_patterns)

        if not all_topic_patterns:
            return []

        # Aggregate topic frequencies across all patterns
        aggregated_topics = defaultdict(int)
        aggregated_expertise = defaultdict(list)
        all_learning_areas = []

        for pattern in all_topic_patterns:
            for topic, freq in pattern.topic_frequencies.items():
                aggregated_topics[topic] += freq

            for domain, expertise in pattern.domain_expertise.items():
                aggregated_expertise[domain].append(expertise)

            all_learning_areas.extend(pattern.learning_areas)

        # Calculate average expertise levels
        avg_expertise = {
            domain: statistics.mean(scores)
            for domain, scores in aggregated_expertise.items()
        }

        # Identify stable learning areas
        learning_area_counts = Counter(all_learning_areas)
        stable_learning_areas = [
            area
            for area, count in learning_area_counts.items()
            if count >= self.MIN_OBSERVATIONS_FOR_STABLE
        ]

        # Determine stability
        stability = self._determine_stability(
            len(all_topic_patterns),
            len(stable_learning_areas) / max(len(all_learning_areas), 1),
        )

        # Get top consistent topics
        top_topics = [
            topic
            for topic, _ in sorted(
                aggregated_topics.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        consolidated_pattern = ConsolidatedPattern(
            pattern_id=f"consolidated_topic_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_type="topic",
            pattern_data={
                "primary_topics": top_topics,
                "topic_frequencies": dict(aggregated_topics),
                "domain_expertise": avg_expertise,
                "stable_learning_areas": stable_learning_areas,
                "topic_diversity": len(aggregated_topics),
            },
            confidence=min(len(all_topic_patterns) / 10, 1.0),
            stability=stability,
            evidence_count=len(all_topic_patterns),
            first_observed=min(p.last_updated for p in all_topic_patterns),
            last_observed=max(p.last_updated for p in all_topic_patterns),
            observation_span=max(p.last_updated for p in all_topic_patterns)
            - min(p.last_updated for p in all_topic_patterns),
            source_pattern_ids=[p.pattern_id for p in all_topic_patterns],
            validation_score=self._calculate_validation_score(
                all_topic_patterns, len(pattern_history)
            ),
        )

        consolidated.append(consolidated_pattern)
        return consolidated

    def _consolidate_timing_patterns(
        self, user_id: str, pattern_history: List[InteractionPatterns]
    ) -> List[ConsolidatedPattern]:
        """Consolidate timing patterns."""
        consolidated = []

        all_timing_patterns = []
        for patterns in pattern_history:
            all_timing_patterns.extend(patterns.timing_patterns)

        if not all_timing_patterns:
            return []

        # Aggregate active hours
        all_active_hours = []
        for pattern in all_timing_patterns:
            all_active_hours.extend(pattern.active_hours)

        hour_counts = Counter(all_active_hours)
        stable_active_hours = [hour for hour, _ in hour_counts.most_common(5)]

        # Aggregate interaction frequency
        frequency_counts = Counter(p.interaction_frequency for p in all_timing_patterns)
        dominant_frequency = frequency_counts.most_common(1)[0][0]

        # Aggregate peak days
        all_peak_days = []
        for pattern in all_timing_patterns:
            all_peak_days.extend(pattern.peak_activity_days)

        day_counts = Counter(all_peak_days)
        stable_peak_days = [day for day, _ in day_counts.most_common(3)]

        stability = self._determine_stability(
            len(all_timing_patterns),
            (
                hour_counts[stable_active_hours[0]] / len(all_active_hours)
                if all_active_hours
                else 0
            ),
        )

        consolidated_pattern = ConsolidatedPattern(
            pattern_id=f"consolidated_timing_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_type="timing",
            pattern_data={
                "stable_active_hours": stable_active_hours,
                "interaction_frequency": dominant_frequency,
                "peak_activity_days": stable_peak_days,
                "consistency_score": (
                    hour_counts[stable_active_hours[0]] / len(all_active_hours)
                    if all_active_hours
                    else 0
                ),
            },
            confidence=min(len(all_timing_patterns) / 10, 1.0),
            stability=stability,
            evidence_count=len(all_timing_patterns),
            first_observed=min(p.last_updated for p in all_timing_patterns),
            last_observed=max(p.last_updated for p in all_timing_patterns),
            observation_span=max(p.last_updated for p in all_timing_patterns)
            - min(p.last_updated for p in all_timing_patterns),
            source_pattern_ids=[p.pattern_id for p in all_timing_patterns],
            validation_score=self._calculate_validation_score(
                all_timing_patterns, len(pattern_history)
            ),
        )

        consolidated.append(consolidated_pattern)
        return consolidated

    def _consolidate_behavioral_patterns(
        self, user_id: str, pattern_history: List[InteractionPatterns]
    ) -> List[ConsolidatedPattern]:
        """Consolidate behavioral patterns."""
        consolidated = []

        all_behavioral_patterns = []
        for patterns in pattern_history:
            all_behavioral_patterns.extend(patterns.behavioral_patterns)

        if not all_behavioral_patterns:
            return []

        # Aggregate interaction styles
        style_counts = Counter(p.interaction_style for p in all_behavioral_patterns)
        dominant_style = style_counts.most_common(1)[0][0]

        # Aggregate feedback frequency
        feedback_frequencies = [p.feedback_frequency for p in all_behavioral_patterns]
        avg_feedback_frequency = statistics.mean(feedback_frequencies)

        # Aggregate error tolerance
        error_tolerances = [p.error_tolerance for p in all_behavioral_patterns]
        avg_error_tolerance = statistics.mean(error_tolerances)

        stability = self._determine_stability(
            len(all_behavioral_patterns),
            self._calculate_consistency(feedback_frequencies),
        )

        consolidated_pattern = ConsolidatedPattern(
            pattern_id=f"consolidated_behavior_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_type="behavioral",
            pattern_data={
                "interaction_style": dominant_style,
                "feedback_frequency": avg_feedback_frequency,
                "error_tolerance": avg_error_tolerance,
                "attention_span": self._aggregate_attention_span(
                    all_behavioral_patterns
                ),
            },
            confidence=min(len(all_behavioral_patterns) / 10, 1.0),
            stability=stability,
            evidence_count=len(all_behavioral_patterns),
            first_observed=min(p.last_updated for p in all_behavioral_patterns),
            last_observed=max(p.last_updated for p in all_behavioral_patterns),
            observation_span=max(p.last_updated for p in all_behavioral_patterns)
            - min(p.last_updated for p in all_behavioral_patterns),
            source_pattern_ids=[p.pattern_id for p in all_behavioral_patterns],
            validation_score=self._calculate_validation_score(
                all_behavioral_patterns, len(pattern_history)
            ),
        )

        consolidated.append(consolidated_pattern)
        return consolidated

    def _determine_stability(
        self, observation_count: int, consistency_score: float
    ) -> KnowledgeStability:
        """Determine stability level based on observations and consistency."""
        if observation_count < self.MIN_OBSERVATIONS_FOR_STABLE:
            return KnowledgeStability.VOLATILE
        elif observation_count < self.MIN_OBSERVATIONS_FOR_ESTABLISHED:
            return (
                KnowledgeStability.STABLE
                if consistency_score > 0.7
                else KnowledgeStability.VOLATILE
            )
        elif observation_count < self.MIN_OBSERVATIONS_FOR_PERMANENT:
            return (
                KnowledgeStability.ESTABLISHED
                if consistency_score > 0.6
                else KnowledgeStability.STABLE
            )
        else:
            return (
                KnowledgeStability.PERMANENT
                if consistency_score > 0.5
                else KnowledgeStability.ESTABLISHED
            )

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score (inverse of coefficient of variation)."""
        if not values or len(values) < 2:
            return 0.0

        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0

        stdev = statistics.stdev(values)
        cv = stdev / mean_val  # Coefficient of variation

        # Convert to consistency score (lower CV = higher consistency)
        consistency = 1.0 / (1.0 + cv)
        return min(consistency, 1.0)

    def _calculate_validation_score(
        self, patterns: List[Any], time_periods: int
    ) -> float:
        """Calculate validation score based on pattern consistency over time."""
        if not patterns or time_periods == 0:
            return 0.0

        # Score based on number of observations and time periods covered
        observation_score = min(len(patterns) / 10, 1.0)
        coverage_score = min(time_periods / 5, 1.0)

        return (observation_score + coverage_score) / 2

    def _aggregate_vocabulary_level(self, patterns: List[CommunicationPattern]) -> str:
        """Aggregate vocabulary level from multiple patterns."""
        level_counts = Counter(p.vocabulary_level for p in patterns)
        return level_counts.most_common(1)[0][0] if level_counts else "intermediate"

    def _aggregate_attention_span(self, patterns: List[BehavioralPattern]) -> str:
        """Aggregate attention span from multiple patterns."""
        span_counts = Counter(p.attention_span for p in patterns)
        return span_counts.most_common(1)[0][0] if span_counts else "medium"


class KnowledgeSynthesizer:
    """
    Synthesizes knowledge from multiple learning sources.

    Combines insights from patterns, preferences, feedback, and adaptations
    to form coherent, validated long-term knowledge about users.
    """

    def __init__(self):
        """Initialize knowledge synthesizer."""
        logger.debug("Knowledge synthesizer initialized")

    def synthesize_knowledge(
        self,
        user_id: str,
        consolidated_patterns: List[ConsolidatedPattern],
        preference_profile: UserPreferenceProfile,
        feedback_history: List[FeedbackAnalysis],
        adaptation_history: List[AdaptationResult],
    ) -> List[ConsolidatedKnowledge]:
        """
        Synthesize knowledge from multiple learning sources.

        Args:
            user_id: User identifier
            consolidated_patterns: Consolidated patterns
            preference_profile: User preference profile
            feedback_history: Feedback analysis history
            adaptation_history: Adaptation history

        Returns:
            List[ConsolidatedKnowledge]: Synthesized knowledge items
        """
        logger.info(f"Synthesizing knowledge for user {user_id}")

        knowledge_items = []

        # Synthesize communication knowledge
        comm_knowledge = self._synthesize_communication_knowledge(
            user_id, consolidated_patterns, preference_profile, feedback_history
        )
        if comm_knowledge:
            knowledge_items.append(comm_knowledge)

        # Synthesize topic interest knowledge
        topic_knowledge = self._synthesize_topic_knowledge(
            user_id, consolidated_patterns, preference_profile
        )
        if topic_knowledge:
            knowledge_items.append(topic_knowledge)

        # Synthesize behavioral knowledge
        behavioral_knowledge = self._synthesize_behavioral_knowledge(
            user_id, consolidated_patterns, feedback_history, adaptation_history
        )
        if behavioral_knowledge:
            knowledge_items.append(behavioral_knowledge)

        # Synthesize learning style knowledge
        learning_knowledge = self._synthesize_learning_knowledge(
            user_id, preference_profile, feedback_history
        )
        if learning_knowledge:
            knowledge_items.append(learning_knowledge)

        logger.info(
            f"Synthesized {len(knowledge_items)} knowledge items for user {user_id}"
        )
        return knowledge_items

    def _synthesize_communication_knowledge(
        self,
        user_id: str,
        patterns: List[ConsolidatedPattern],
        preferences: UserPreferenceProfile,
        feedback: List[FeedbackAnalysis],
    ) -> Optional[ConsolidatedKnowledge]:
        """Synthesize communication style knowledge."""
        # Find communication patterns
        comm_patterns = [p for p in patterns if p.pattern_type == "communication"]
        if not comm_patterns:
            return None

        knowledge_items = {}
        confidence_scores = {}
        supporting_evidence = {}

        # Extract from patterns
        for pattern in comm_patterns:
            data = pattern.pattern_data
            knowledge_items["preferred_style"] = data.get("preferred_style", "casual")
            knowledge_items["formality_level"] = data.get("formality_level", 0.5)
            knowledge_items["technical_depth"] = data.get("technical_depth", 0.5)
            knowledge_items["vocabulary_level"] = data.get(
                "vocabulary_level", "intermediate"
            )

            confidence_scores["style"] = pattern.confidence
            supporting_evidence["style"] = [
                f"Pattern {pattern.pattern_id}: {pattern.evidence_count} observations"
            ]

        # Cross-reference with preferences
        comm_prefs = preferences.preferences.get(
            PreferenceCategory.COMMUNICATION.value, {}
        )
        if comm_prefs:
            for key, value in comm_prefs.items():
                knowledge_items[f"preference_{key}"] = value
                confidence_scores[f"preference_{key}"] = preferences.get_confidence(
                    PreferenceCategory.COMMUNICATION, key
                )
                supporting_evidence[f"preference_{key}"] = [
                    "Extracted from user preferences"
                ]

        # Validate with feedback
        positive_feedback = sum(
            1
            for f in feedback
            if any(
                "clarity" in aspect or "tone" in aspect for aspect in f.positive_aspects
            )
        )

        if positive_feedback > 0:
            knowledge_items["communication_effectiveness"] = min(
                positive_feedback / len(feedback), 1.0
            )
            confidence_scores["effectiveness"] = 0.7
            supporting_evidence["effectiveness"] = [
                f"{positive_feedback} positive feedback instances"
            ]

        # Determine stability
        stability = (
            comm_patterns[0].stability if comm_patterns else KnowledgeStability.VOLATILE
        )

        return ConsolidatedKnowledge(
            knowledge_id=f"knowledge_comm_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            knowledge_domain="communication_style",
            knowledge_items=knowledge_items,
            confidence_scores=confidence_scores,
            supporting_evidence=supporting_evidence,
            stability=stability,
            learning_sources=["patterns", "preferences", "feedback"],
            synthesis_reasoning="Synthesized from communication patterns, user preferences, and feedback validation",
        )

    def _synthesize_topic_knowledge(
        self,
        user_id: str,
        patterns: List[ConsolidatedPattern],
        preferences: UserPreferenceProfile,
    ) -> Optional[ConsolidatedKnowledge]:
        """Synthesize topic interest knowledge."""
        topic_patterns = [p for p in patterns if p.pattern_type == "topic"]
        if not topic_patterns:
            return None

        knowledge_items = {}
        confidence_scores = {}
        supporting_evidence = {}

        # Extract from patterns
        for pattern in topic_patterns:
            data = pattern.pattern_data
            knowledge_items["primary_interests"] = data.get("primary_topics", [])
            knowledge_items["domain_expertise"] = data.get("domain_expertise", {})
            knowledge_items["learning_areas"] = data.get("stable_learning_areas", [])
            knowledge_items["topic_diversity"] = data.get("topic_diversity", 0)

            confidence_scores["interests"] = pattern.confidence
            supporting_evidence["interests"] = [
                f"Pattern {pattern.pattern_id}: {pattern.evidence_count} observations"
            ]

        # Cross-reference with topic preferences
        topic_prefs = preferences.preferences.get(PreferenceCategory.TOPICS.value, {})
        if topic_prefs:
            for key, value in topic_prefs.items():
                if key.startswith("interest_"):
                    topic = key.replace("interest_", "")
                    knowledge_items[f"validated_interest_{topic}"] = value
                    confidence_scores[f"interest_{topic}"] = preferences.get_confidence(
                        PreferenceCategory.TOPICS, key
                    )

        stability = (
            topic_patterns[0].stability
            if topic_patterns
            else KnowledgeStability.VOLATILE
        )

        return ConsolidatedKnowledge(
            knowledge_id=f"knowledge_topic_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            knowledge_domain="topic_interests",
            knowledge_items=knowledge_items,
            confidence_scores=confidence_scores,
            supporting_evidence=supporting_evidence,
            stability=stability,
            learning_sources=["patterns", "preferences"],
            synthesis_reasoning="Synthesized from topic patterns and validated with user preferences",
        )

    def _synthesize_behavioral_knowledge(
        self,
        user_id: str,
        patterns: List[ConsolidatedPattern],
        feedback: List[FeedbackAnalysis],
        adaptations: List[AdaptationResult],
    ) -> Optional[ConsolidatedKnowledge]:
        """Synthesize behavioral knowledge."""
        behavioral_patterns = [p for p in patterns if p.pattern_type == "behavioral"]
        if not behavioral_patterns:
            return None

        knowledge_items = {}
        confidence_scores = {}
        supporting_evidence = {}

        for pattern in behavioral_patterns:
            data = pattern.pattern_data
            knowledge_items["interaction_style"] = data.get(
                "interaction_style", "conversational"
            )
            knowledge_items["feedback_frequency"] = data.get("feedback_frequency", 0.0)
            knowledge_items["error_tolerance"] = data.get("error_tolerance", 0.5)
            knowledge_items["attention_span"] = data.get("attention_span", "medium")

            confidence_scores["behavior"] = pattern.confidence
            supporting_evidence["behavior"] = [
                f"Pattern {pattern.pattern_id}: {pattern.evidence_count} observations"
            ]

        # Calculate adaptation success rate
        if adaptations:
            successful_adaptations = sum(
                1 for a in adaptations if a.confidence_score > 0.7
            )
            knowledge_items["adaptation_receptiveness"] = successful_adaptations / len(
                adaptations
            )
            confidence_scores["adaptation"] = 0.6
            supporting_evidence["adaptation"] = [
                f"{successful_adaptations}/{len(adaptations)} successful adaptations"
            ]

        stability = (
            behavioral_patterns[0].stability
            if behavioral_patterns
            else KnowledgeStability.VOLATILE
        )

        return ConsolidatedKnowledge(
            knowledge_id=f"knowledge_behavior_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            knowledge_domain="behavioral_characteristics",
            knowledge_items=knowledge_items,
            confidence_scores=confidence_scores,
            supporting_evidence=supporting_evidence,
            stability=stability,
            learning_sources=["patterns", "feedback", "adaptations"],
            synthesis_reasoning="Synthesized from behavioral patterns, feedback, and adaptation success",
        )

    def _synthesize_learning_knowledge(
        self,
        user_id: str,
        preferences: UserPreferenceProfile,
        feedback: List[FeedbackAnalysis],
    ) -> Optional[ConsolidatedKnowledge]:
        """Synthesize learning style knowledge."""
        knowledge_items = {}
        confidence_scores = {}
        supporting_evidence = {}

        # Extract learning preferences
        learning_prefs = preferences.preferences.get(
            PreferenceCategory.LEARNING.value, {}
        )
        if learning_prefs:
            for key, value in learning_prefs.items():
                knowledge_items[key] = value
                confidence_scores[key] = preferences.get_confidence(
                    PreferenceCategory.LEARNING, key
                )
                supporting_evidence[key] = ["Extracted from learning preferences"]

        # Analyze feedback for learning effectiveness
        if feedback:
            improvement_mentions = sum(
                1
                for f in feedback
                if any(
                    "learning" in insight.lower() for insight in f.actionable_insights
                )
            )

            if improvement_mentions > 0:
                knowledge_items["learning_engagement"] = min(
                    improvement_mentions / len(feedback), 1.0
                )
                confidence_scores["engagement"] = 0.5
                supporting_evidence["engagement"] = [
                    f"{improvement_mentions} feedback instances mention learning"
                ]

        if not knowledge_items:
            return None

        return ConsolidatedKnowledge(
            knowledge_id=f"knowledge_learning_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            knowledge_domain="learning_style",
            knowledge_items=knowledge_items,
            confidence_scores=confidence_scores,
            supporting_evidence=supporting_evidence,
            stability=KnowledgeStability.STABLE,
            learning_sources=["preferences", "feedback"],
            synthesis_reasoning="Synthesized from learning preferences and feedback analysis",
        )


class MetaPatternExtractor:
    """
    Extracts meta-patterns from consolidated data.

    Identifies higher-order patterns such as learning trajectories,
    behavior evolution, and characteristic changes over time.
    """

    def __init__(self):
        """Initialize meta-pattern extractor."""
        logger.debug("Meta-pattern extractor initialized")

    def extract_meta_patterns(
        self,
        user_id: str,
        historical_patterns: List[ConsolidatedPattern],
        historical_knowledge: List[ConsolidatedKnowledge],
    ) -> List[MetaPattern]:
        """
        Extract meta-patterns from historical data.

        Args:
            user_id: User identifier
            historical_patterns: Historical consolidated patterns
            historical_knowledge: Historical consolidated knowledge

        Returns:
            List[MetaPattern]: Extracted meta-patterns
        """
        logger.info(f"Extracting meta-patterns for user {user_id}")

        meta_patterns = []

        # Detect learning trajectory
        learning_trajectory = self._detect_learning_trajectory(
            user_id, historical_patterns
        )
        if learning_trajectory:
            meta_patterns.append(learning_trajectory)

        # Detect behavior evolution
        behavior_evolution = self._detect_behavior_evolution(
            user_id, historical_patterns
        )
        if behavior_evolution:
            meta_patterns.append(behavior_evolution)

        # Detect interest evolution
        interest_evolution = self._detect_interest_evolution(
            user_id, historical_patterns
        )
        if interest_evolution:
            meta_patterns.append(interest_evolution)

        # Detect communication style drift
        style_drift = self._detect_communication_drift(user_id, historical_patterns)
        if style_drift:
            meta_patterns.append(style_drift)

        logger.info(f"Extracted {len(meta_patterns)} meta-patterns for user {user_id}")
        return meta_patterns

    def _detect_learning_trajectory(
        self, user_id: str, patterns: List[ConsolidatedPattern]
    ) -> Optional[MetaPattern]:
        """Detect learning trajectory patterns."""
        topic_patterns = [p for p in patterns if p.pattern_type == "topic"]
        if len(topic_patterns) < 2:
            return None

        # Sort by time
        sorted_patterns = sorted(topic_patterns, key=lambda x: x.timestamp)

        # Analyze learning area expansion
        learning_areas_over_time = [
            len(p.pattern_data.get("stable_learning_areas", []))
            for p in sorted_patterns
        ]

        if len(learning_areas_over_time) < 2:
            return None

        # Calculate trend
        trend_direction = (
            learning_areas_over_time[-1] - learning_areas_over_time[0]
        ) / max(learning_areas_over_time[0], 1)
        trend_strength = abs(trend_direction) / (1 + abs(trend_direction))  # Normalize

        if abs(trend_direction) < 0.1:  # Not significant
            return None

        description = (
            f"User shows {'expanding' if trend_direction > 0 else 'focusing'} learning trajectory "
            f"with {abs(trend_direction):.1%} change in learning area breadth"
        )

        predictive_insights = []
        if trend_direction > 0:
            predictive_insights.append("User is likely to explore more diverse topics")
            predictive_insights.append("Consider suggesting related learning areas")
        else:
            predictive_insights.append("User is deepening focus on specific areas")
            predictive_insights.append("Provide more advanced content in focused areas")

        return MetaPattern(
            meta_pattern_id=f"meta_learning_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_category="learning_trajectory",
            description=description,
            detected_trend=(
                "expanding_interests" if trend_direction > 0 else "deepening_focus"
            ),
            trend_direction=min(max(trend_direction, -1.0), 1.0),
            trend_strength=trend_strength,
            time_span=sorted_patterns[-1].timestamp - sorted_patterns[0].timestamp,
            affected_domains=["topic_interests", "learning_style"],
            predictive_insights=predictive_insights,
            confidence=0.7,
            supporting_patterns=[p.pattern_id for p in sorted_patterns],
        )

    def _detect_behavior_evolution(
        self, user_id: str, patterns: List[ConsolidatedPattern]
    ) -> Optional[MetaPattern]:
        """Detect behavior evolution patterns."""
        behavioral_patterns = [p for p in patterns if p.pattern_type == "behavioral"]
        if len(behavioral_patterns) < 2:
            return None

        sorted_patterns = sorted(behavioral_patterns, key=lambda x: x.timestamp)

        # Analyze feedback frequency change
        feedback_frequencies = [
            p.pattern_data.get("feedback_frequency", 0.0) for p in sorted_patterns
        ]

        if len(feedback_frequencies) < 2:
            return None

        trend_direction = (feedback_frequencies[-1] - feedback_frequencies[0]) / max(
            feedback_frequencies[0], 0.1
        )
        trend_strength = abs(trend_direction) / (1 + abs(trend_direction))

        if abs(trend_direction) < 0.2:
            return None

        description = (
            f"User engagement is {'increasing' if trend_direction > 0 else 'decreasing'} "
            f"with {abs(trend_direction):.1%} change in feedback frequency"
        )

        predictive_insights = []
        if trend_direction > 0:
            predictive_insights.append(
                "User becoming more engaged, seek more feedback opportunities"
            )
        else:
            predictive_insights.append(
                "User engagement decreasing, may need re-engagement strategies"
            )

        return MetaPattern(
            meta_pattern_id=f"meta_behavior_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_category="behavior_evolution",
            description=description,
            detected_trend=(
                "increasing_engagement"
                if trend_direction > 0
                else "decreasing_engagement"
            ),
            trend_direction=min(max(trend_direction, -1.0), 1.0),
            trend_strength=trend_strength,
            time_span=sorted_patterns[-1].timestamp - sorted_patterns[0].timestamp,
            affected_domains=["behavioral_characteristics"],
            predictive_insights=predictive_insights,
            confidence=0.6,
            supporting_patterns=[p.pattern_id for p in sorted_patterns],
        )

    def _detect_interest_evolution(
        self, user_id: str, patterns: List[ConsolidatedPattern]
    ) -> Optional[MetaPattern]:
        """Detect interest evolution patterns."""
        topic_patterns = [p for p in patterns if p.pattern_type == "topic"]
        if len(topic_patterns) < 2:
            return None

        sorted_patterns = sorted(topic_patterns, key=lambda x: x.timestamp)

        # Analyze topic diversity change
        topic_diversities = [
            p.pattern_data.get("topic_diversity", 0) for p in sorted_patterns
        ]

        if len(topic_diversities) < 2:
            return None

        trend_direction = (topic_diversities[-1] - topic_diversities[0]) / max(
            topic_diversities[0], 1
        )
        trend_strength = abs(trend_direction) / (1 + abs(trend_direction))

        if abs(trend_direction) < 0.15:
            return None

        description = (
            f"User interests are {'diversifying' if trend_direction > 0 else 'specializing'} "
            f"with {abs(trend_direction):.1%} change in topic diversity"
        )

        return MetaPattern(
            meta_pattern_id=f"meta_interest_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_category="interest_evolution",
            description=description,
            detected_trend=(
                "diversifying_interests"
                if trend_direction > 0
                else "specializing_interests"
            ),
            trend_direction=min(max(trend_direction, -1.0), 1.0),
            trend_strength=trend_strength,
            time_span=sorted_patterns[-1].timestamp - sorted_patterns[0].timestamp,
            affected_domains=["topic_interests"],
            predictive_insights=[
                "User interest patterns are evolving, adjust content recommendations accordingly"
            ],
            confidence=0.65,
            supporting_patterns=[p.pattern_id for p in sorted_patterns],
        )

    def _detect_communication_drift(
        self, user_id: str, patterns: List[ConsolidatedPattern]
    ) -> Optional[MetaPattern]:
        """Detect communication style drift."""
        comm_patterns = [p for p in patterns if p.pattern_type == "communication"]
        if len(comm_patterns) < 2:
            return None

        sorted_patterns = sorted(comm_patterns, key=lambda x: x.timestamp)

        # Analyze formality level change
        formality_levels = [
            p.pattern_data.get("formality_level", 0.5) for p in sorted_patterns
        ]

        if len(formality_levels) < 2:
            return None

        trend_direction = formality_levels[-1] - formality_levels[0]
        trend_strength = abs(trend_direction)

        if abs(trend_direction) < 0.1:
            return None

        description = f"User communication style is shifting {'more formal' if trend_direction > 0 else 'more casual'}"

        return MetaPattern(
            meta_pattern_id=f"meta_comm_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            pattern_category="communication_drift",
            description=description,
            detected_trend=(
                "increasing_formality"
                if trend_direction > 0
                else "decreasing_formality"
            ),
            trend_direction=min(max(trend_direction, -1.0), 1.0),
            trend_strength=trend_strength,
            time_span=sorted_patterns[-1].timestamp - sorted_patterns[0].timestamp,
            affected_domains=["communication_style"],
            predictive_insights=[
                "Adjust response formality to match evolving user communication style"
            ],
            confidence=0.6,
            supporting_patterns=[p.pattern_id for p in sorted_patterns],
        )


class ConsolidationScheduler:
    """
    Manages periodic consolidation scheduling.

    Determines when consolidation should occur based on user activity,
    data volume, and time-based triggers.
    """

    # Default consolidation intervals
    DEFAULT_INTERVALS = {
        ConsolidationPeriod.HOURLY: timedelta(hours=1),
        ConsolidationPeriod.DAILY: timedelta(days=1),
        ConsolidationPeriod.WEEKLY: timedelta(weeks=1),
        ConsolidationPeriod.MONTHLY: timedelta(days=30),
        ConsolidationPeriod.QUARTERLY: timedelta(days=90),
    }

    def __init__(self):
        """Initialize consolidation scheduler."""
        self.last_consolidation: Dict[str, datetime] = {}
        self.consolidation_periods: Dict[str, ConsolidationPeriod] = {}
        logger.debug("Consolidation scheduler initialized")

    def should_consolidate(
        self, user_id: str, current_time: Optional[datetime] = None
    ) -> bool:
        """
        Determine if consolidation should occur for a user.

        Args:
            user_id: User identifier
            current_time: Current time (defaults to now)

        Returns:
            bool: True if consolidation should occur
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Get user's consolidation period
        period = self.consolidation_periods.get(user_id, ConsolidationPeriod.WEEKLY)
        interval = self.DEFAULT_INTERVALS[period]

        # Check last consolidation time
        last_time = self.last_consolidation.get(user_id)
        if last_time is None:
            # Never consolidated before
            return True

        # Check if interval has elapsed
        return (current_time - last_time) >= interval

    def schedule_consolidation(
        self, user_id: str, period: ConsolidationPeriod = ConsolidationPeriod.WEEKLY
    ):
        """
        Schedule consolidation for a user.

        Args:
            user_id: User identifier
            period: Consolidation period
        """
        self.consolidation_periods[user_id] = period
        logger.debug(f"Scheduled {period.value} consolidation for user {user_id}")

    def mark_consolidation_complete(
        self, user_id: str, completion_time: Optional[datetime] = None
    ):
        """
        Mark consolidation as complete.

        Args:
            user_id: User identifier
            completion_time: Completion time (defaults to now)
        """
        if completion_time is None:
            completion_time = datetime.utcnow()

        self.last_consolidation[user_id] = completion_time
        logger.debug(f"Marked consolidation complete for user {user_id}")

    def get_next_consolidation_time(self, user_id: str) -> Optional[datetime]:
        """
        Get next scheduled consolidation time.

        Args:
            user_id: User identifier

        Returns:
            Optional[datetime]: Next consolidation time
        """
        last_time = self.last_consolidation.get(user_id)
        if last_time is None:
            return datetime.utcnow()  # Now if never consolidated

        period = self.consolidation_periods.get(user_id, ConsolidationPeriod.WEEKLY)
        interval = self.DEFAULT_INTERVALS[period]

        return last_time + interval


class ConsolidationEngine:
    """
    Main consolidation engine.

    Coordinates all consolidation activities including pattern consolidation,
    knowledge synthesis, meta-pattern extraction, and consolidation scheduling.
    """

    def __init__(self):
        """Initialize consolidation engine."""
        self.pattern_consolidator = PatternConsolidator()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.meta_pattern_extractor = MetaPatternExtractor()
        self.scheduler = ConsolidationScheduler()

        # Storage for consolidated data
        self.consolidated_patterns: Dict[str, List[ConsolidatedPattern]] = {}
        self.consolidated_knowledge: Dict[str, List[ConsolidatedKnowledge]] = {}
        self.meta_patterns: Dict[str, List[MetaPattern]] = {}
        self.consolidation_history: Dict[str, List[ConsolidationResult]] = {}
        self.metrics: Dict[str, ConsolidationMetrics] = {}

        logger.info("Consolidation engine initialized")

    def consolidate(
        self,
        user_id: str,
        pattern_history: List[InteractionPatterns],
        preference_profile: UserPreferenceProfile,
        feedback_history: List[FeedbackAnalysis],
        adaptation_history: List[AdaptationResult],
    ) -> ConsolidationResult:
        """
        Perform complete consolidation for a user.

        Args:
            user_id: User identifier
            pattern_history: Historical interaction patterns
            preference_profile: User preference profile
            feedback_history: Feedback analysis history
            adaptation_history: Adaptation history

        Returns:
            ConsolidationResult: Consolidation results
        """
        logger.info(f"Starting consolidation for user {user_id}")

        start_time = datetime.utcnow()
        strategies_applied = []

        # 1. Consolidate patterns
        consolidated_patterns = self.pattern_consolidator.consolidate_patterns(
            user_id, pattern_history
        )
        strategies_applied.append(ConsolidationStrategy.AGGREGATE)

        # Store consolidated patterns
        if user_id not in self.consolidated_patterns:
            self.consolidated_patterns[user_id] = []
        self.consolidated_patterns[user_id].extend(consolidated_patterns)

        # 2. Synthesize knowledge
        consolidated_knowledge = self.knowledge_synthesizer.synthesize_knowledge(
            user_id,
            consolidated_patterns,
            preference_profile,
            feedback_history,
            adaptation_history,
        )
        strategies_applied.append(ConsolidationStrategy.MERGE)

        # Store consolidated knowledge
        if user_id not in self.consolidated_knowledge:
            self.consolidated_knowledge[user_id] = []
        self.consolidated_knowledge[user_id].extend(consolidated_knowledge)

        # 3. Extract meta-patterns
        historical_patterns = self.consolidated_patterns.get(user_id, [])
        historical_knowledge = self.consolidated_knowledge.get(user_id, [])

        meta_patterns = self.meta_pattern_extractor.extract_meta_patterns(
            user_id, historical_patterns, historical_knowledge
        )
        strategies_applied.append(ConsolidationStrategy.PROMOTE)

        # Store meta-patterns
        if user_id not in self.meta_patterns:
            self.meta_patterns[user_id] = []
        self.meta_patterns[user_id].extend(meta_patterns)

        # 4. Calculate consolidation quality
        consolidation_quality = self._calculate_consolidation_quality(
            consolidated_patterns, consolidated_knowledge, meta_patterns
        )

        # 5. Calculate confidence improvement
        confidence_improvement = self._calculate_confidence_improvement(
            pattern_history, consolidated_patterns
        )

        # 6. Create consolidation result
        items_processed = len(pattern_history) + len(feedback_history)
        items_consolidated = len(consolidated_patterns) + len(consolidated_knowledge)

        result = ConsolidationResult(
            consolidation_id=str(uuid.uuid4()),
            user_id=user_id,
            consolidation_type=ConsolidationType.KNOWLEDGE,
            consolidated_patterns=consolidated_patterns,
            consolidated_knowledge=consolidated_knowledge,
            meta_patterns=meta_patterns,
            strategies_applied=strategies_applied,
            items_processed=items_processed,
            items_consolidated=items_consolidated,
            items_pruned=0,  # Would be calculated if pruning implemented
            confidence_improvement=confidence_improvement,
            consolidation_quality=consolidation_quality,
            reasoning=self._generate_consolidation_reasoning(
                consolidated_patterns, consolidated_knowledge, meta_patterns
            ),
        )

        # 7. Update metrics
        self._update_metrics(user_id, result)

        # 8. Mark consolidation complete
        self.scheduler.mark_consolidation_complete(user_id)

        # 9. Store in history
        if user_id not in self.consolidation_history:
            self.consolidation_history[user_id] = []
        self.consolidation_history[user_id].append(result)

        duration = datetime.utcnow() - start_time
        logger.info(
            f"Consolidation complete for user {user_id} in {duration.total_seconds():.2f}s: "
            f"{items_consolidated} items consolidated, quality score {consolidation_quality:.2f}"
        )

        return result

    def get_consolidated_patterns(self, user_id: str) -> List[ConsolidatedPattern]:
        """
        Get consolidated patterns for a user.

        Args:
            user_id: User identifier

        Returns:
            List[ConsolidatedPattern]: Consolidated patterns
        """
        return self.consolidated_patterns.get(user_id, [])

    def get_consolidated_knowledge(self, user_id: str) -> List[ConsolidatedKnowledge]:
        """
        Get consolidated knowledge for a user.

        Args:
            user_id: User identifier

        Returns:
            List[ConsolidatedKnowledge]: Consolidated knowledge
        """
        return self.consolidated_knowledge.get(user_id, [])

    def get_meta_patterns(self, user_id: str) -> List[MetaPattern]:
        """
        Get meta-patterns for a user.

        Args:
            user_id: User identifier

        Returns:
            List[MetaPattern]: Meta-patterns
        """
        return self.meta_patterns.get(user_id, [])

    def get_consolidation_metrics(self, user_id: str) -> Optional[ConsolidationMetrics]:
        """
        Get consolidation metrics for a user.

        Args:
            user_id: User identifier

        Returns:
            Optional[ConsolidationMetrics]: Consolidation metrics
        """
        return self.metrics.get(user_id)

    def _calculate_consolidation_quality(
        self,
        patterns: List[ConsolidatedPattern],
        knowledge: List[ConsolidatedKnowledge],
        meta_patterns: List[MetaPattern],
    ) -> float:
        """Calculate overall consolidation quality score."""
        quality_factors = []

        # Pattern quality
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            avg_validation = sum(p.validation_score for p in patterns) / len(patterns)
            pattern_quality = (avg_pattern_confidence + avg_validation) / 2
            quality_factors.append(pattern_quality)

        # Knowledge quality
        if knowledge:
            avg_knowledge_confidence = sum(
                sum(k.confidence_scores.values()) / max(len(k.confidence_scores), 1)
                for k in knowledge
            ) / len(knowledge)
            quality_factors.append(avg_knowledge_confidence)

        # Meta-pattern quality
        if meta_patterns:
            avg_meta_confidence = sum(m.confidence for m in meta_patterns) / len(
                meta_patterns
            )
            quality_factors.append(avg_meta_confidence)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

    def _calculate_confidence_improvement(
        self,
        original_patterns: List[InteractionPatterns],
        consolidated_patterns: List[ConsolidatedPattern],
    ) -> float:
        """Calculate confidence improvement from consolidation."""
        if not original_patterns or not consolidated_patterns:
            return 0.0

        # Calculate average confidence of original patterns
        original_confidence_values = []
        for patterns in original_patterns:
            original_confidence_values.append(patterns.overall_confidence)

        if not original_confidence_values:
            return 0.0

        avg_original_confidence = sum(original_confidence_values) / len(
            original_confidence_values
        )

        # Calculate average confidence of consolidated patterns
        avg_consolidated_confidence = sum(
            p.confidence for p in consolidated_patterns
        ) / len(consolidated_patterns)

        return avg_consolidated_confidence - avg_original_confidence

    def _generate_consolidation_reasoning(
        self,
        patterns: List[ConsolidatedPattern],
        knowledge: List[ConsolidatedKnowledge],
        meta_patterns: List[MetaPattern],
    ) -> str:
        """Generate reasoning for consolidation results."""
        reasoning_parts = []

        if patterns:
            pattern_types = Counter(p.pattern_type for p in patterns)
            reasoning_parts.append(
                f"Consolidated {len(patterns)} patterns: "
                + ", ".join(
                    f"{count} {ptype}" for ptype, count in pattern_types.items()
                )
            )

        if knowledge:
            domains = [k.knowledge_domain for k in knowledge]
            reasoning_parts.append(
                f"Synthesized knowledge in {len(domains)} domains: "
                + ", ".join(domains)
            )

        if meta_patterns:
            categories = Counter(m.pattern_category for m in meta_patterns)
            reasoning_parts.append(
                f"Discovered {len(meta_patterns)} meta-patterns: "
                + ", ".join(f"{count} {cat}" for cat, count in categories.items())
            )

        return (
            "; ".join(reasoning_parts)
            if reasoning_parts
            else "No consolidation performed"
        )

    def _update_metrics(self, user_id: str, result: ConsolidationResult):
        """Update consolidation metrics."""
        if user_id not in self.metrics:
            self.metrics[user_id] = ConsolidationMetrics(
                user_id=user_id,
                total_consolidations=0,
                successful_consolidations=0,
                average_quality_score=0.0,
                average_confidence_gain=0.0,
                knowledge_stability_ratio=0.0,
                meta_patterns_discovered=0,
                consolidation_efficiency=0.0,
            )

        metrics = self.metrics[user_id]
        metrics.total_consolidations += 1

        if result.consolidation_quality > 0.5:
            metrics.successful_consolidations += 1

        # Update averages
        metrics.average_quality_score = (
            metrics.average_quality_score * (metrics.total_consolidations - 1)
            + result.consolidation_quality
        ) / metrics.total_consolidations

        metrics.average_confidence_gain = (
            metrics.average_confidence_gain * (metrics.total_consolidations - 1)
            + result.confidence_improvement
        ) / metrics.total_consolidations

        metrics.meta_patterns_discovered += len(result.meta_patterns)

        if result.items_processed > 0:
            metrics.consolidation_efficiency = (
                result.items_consolidated / result.items_processed
            )

        # Calculate knowledge stability ratio
        all_knowledge = self.consolidated_knowledge.get(user_id, [])
        if all_knowledge:
            stable_count = sum(
                1
                for k in all_knowledge
                if k.stability
                in [
                    KnowledgeStability.STABLE,
                    KnowledgeStability.ESTABLISHED,
                    KnowledgeStability.PERMANENT,
                ]
            )
            metrics.knowledge_stability_ratio = stable_count / len(all_knowledge)

        metrics.last_consolidation = datetime.utcnow()
        metrics.next_scheduled_consolidation = (
            self.scheduler.get_next_consolidation_time(user_id)
        )
        metrics.timestamp = datetime.utcnow()


# Global consolidation engine instance
_consolidation_engine: Optional[ConsolidationEngine] = None


def get_consolidation_engine() -> ConsolidationEngine:
    """
    Get the global consolidation engine instance.

    Returns:
        ConsolidationEngine: Global consolidation engine instance
    """
    global _consolidation_engine
    if _consolidation_engine is None:
        _consolidation_engine = ConsolidationEngine()
    return _consolidation_engine
