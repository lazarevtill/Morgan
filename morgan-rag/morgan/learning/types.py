"""
Core learning types and data structures.

Defines the fundamental types used throughout the learning system,
following immutable dataclass patterns for thread safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PatternType(str, Enum):
    """Types of behavioral patterns that can be detected."""

    RECURRING = "recurring"  # Patterns that repeat regularly
    ESCALATING = "escalating"  # Patterns that intensify over time
    ALTERNATING = "alternating"  # Patterns that switch between states
    TEMPORAL = "temporal"  # Time-based patterns
    CONTEXTUAL = "contextual"  # Context-dependent patterns
    SEQUENTIAL = "sequential"  # Sequential action patterns
    HABITUAL = "habitual"  # Habit formation patterns


class FeedbackType(str, Enum):
    """Types of user feedback."""

    EXPLICIT_POSITIVE = "explicit_positive"  # Direct positive feedback
    EXPLICIT_NEGATIVE = "explicit_negative"  # Direct negative feedback
    IMPLICIT_POSITIVE = "implicit_positive"  # Inferred positive signals
    IMPLICIT_NEGATIVE = "implicit_negative"  # Inferred negative signals
    CORRECTION = "correction"  # User corrections
    RATING = "rating"  # Numeric rating
    PREFERENCE = "preference"  # Stated preference


class AdaptationStrategy(str, Enum):
    """Strategies for response adaptation."""

    IMMEDIATE = "immediate"  # Apply immediately
    GRADUAL = "gradual"  # Apply gradually over time
    CONTEXTUAL = "contextual"  # Apply based on context
    EXPERIMENTAL = "experimental"  # A/B test before applying
    CONSERVATIVE = "conservative"  # Apply only when confident


class PreferenceDimension(str, Enum):
    """Dimensions of user preferences."""

    COMMUNICATION_STYLE = "communication_style"  # Formal vs casual, etc.
    DETAIL_LEVEL = "detail_level"  # Brief vs detailed
    RESPONSE_LENGTH = "response_length"  # Short vs long
    TECHNICAL_DEPTH = "technical_depth"  # Simple vs technical
    EXAMPLES = "examples"  # Preference for examples
    EXPLANATIONS = "explanations"  # Preference for explanations
    TONE = "tone"  # Professional, friendly, etc.
    FORMATTING = "formatting"  # Markdown, lists, etc.


@dataclass(frozen=True)
class LearningPattern:
    """A detected behavioral pattern."""

    pattern_id: str
    pattern_type: PatternType
    description: str
    confidence: float  # 0-1, confidence in pattern detection
    frequency: int  # How many times pattern observed
    first_observed: datetime
    last_observed: datetime

    # Pattern-specific data
    trigger_contexts: List[str] = field(default_factory=list)
    associated_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistical metrics
    regularity_score: float = 0.0  # 0-1, how regular the pattern is
    strength: float = 0.0  # 0-1, pattern strength

    def __post_init__(self) -> None:
        """Validate pattern data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if not 0.0 <= self.regularity_score <= 1.0:
            raise ValueError(f"Regularity must be 0-1, got {self.regularity_score}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength must be 0-1, got {self.strength}")
        if self.frequency < 1:
            raise ValueError(f"Frequency must be >= 1, got {self.frequency}")
        if self.last_observed < self.first_observed:
            raise ValueError("last_observed cannot be before first_observed")

    @property
    def is_significant(self) -> bool:
        """Check if pattern is significant enough to act on."""
        return (
            self.confidence >= 0.6
            and self.frequency >= 3
            and self.strength >= 0.5
        )

    @property
    def age_days(self) -> float:
        """Get pattern age in days."""
        return (datetime.utcnow() - self.first_observed).total_seconds() / 86400


@dataclass(frozen=True)
class FeedbackSignal:
    """A user feedback signal."""

    feedback_id: str
    user_id: str
    feedback_type: FeedbackType
    timestamp: datetime

    # Feedback content
    message_id: Optional[str] = None  # Associated message
    rating: Optional[float] = None  # 0-1 for numeric ratings
    text: Optional[str] = None  # Text feedback
    correction: Optional[str] = None  # Corrected text

    # Sentiment analysis
    sentiment: float = 0.0  # -1 (negative) to +1 (positive)
    sentiment_confidence: float = 0.0  # 0-1

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate feedback data."""
        if self.rating is not None and not 0.0 <= self.rating <= 1.0:
            raise ValueError(f"Rating must be 0-1, got {self.rating}")
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(f"Sentiment must be -1 to +1, got {self.sentiment}")
        if not 0.0 <= self.sentiment_confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.sentiment_confidence}")

    @property
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        if self.rating is not None:
            return self.rating >= 0.6
        return self.sentiment >= 0.3

    @property
    def is_negative(self) -> bool:
        """Check if feedback is negative."""
        if self.rating is not None:
            return self.rating <= 0.4
        return self.sentiment <= -0.3

    @property
    def is_actionable(self) -> bool:
        """Check if feedback is actionable."""
        return (
            self.correction is not None
            or (self.rating is not None and (self.is_positive or self.is_negative))
            or self.sentiment_confidence >= 0.6
        )


@dataclass(frozen=True)
class UserPreference:
    """A learned user preference."""

    preference_id: str
    user_id: str
    dimension: PreferenceDimension
    value: Any  # Preference value (string, number, etc.)
    confidence: float  # 0-1, confidence in this preference

    # Learning history
    first_learned: datetime
    last_updated: datetime
    update_count: int

    # Supporting evidence
    supporting_signals: List[str] = field(default_factory=list)  # Signal IDs
    conflicting_signals: List[str] = field(default_factory=list)  # Signal IDs

    # Context
    context_tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate preference data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.update_count < 1:
            raise ValueError(f"Update count must be >= 1, got {self.update_count}")
        if self.last_updated < self.first_learned:
            raise ValueError("last_updated cannot be before first_learned")

    @property
    def is_stable(self) -> bool:
        """Check if preference is stable (few conflicts)."""
        if not self.supporting_signals:
            return False
        conflict_ratio = len(self.conflicting_signals) / len(self.supporting_signals)
        return conflict_ratio < 0.2 and self.confidence >= 0.7

    @property
    def strength(self) -> float:
        """Get preference strength based on evidence."""
        if not self.supporting_signals:
            return 0.0
        evidence_ratio = len(self.supporting_signals) / (
            len(self.supporting_signals) + len(self.conflicting_signals)
        )
        return evidence_ratio * self.confidence


@dataclass(frozen=True)
class AdaptationResult:
    """Result of an adaptation operation."""

    adaptation_id: str
    strategy: AdaptationStrategy
    timestamp: datetime

    # What was adapted
    target: str  # What was being adapted
    changes: Dict[str, Any]  # Changes made

    # Adaptation metrics
    confidence: float  # 0-1, confidence in adaptation
    expected_improvement: float  # 0-1, expected improvement
    actual_improvement: Optional[float] = None  # 0-1, measured after application

    # Context
    triggering_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    triggering_feedback: List[str] = field(default_factory=list)  # Feedback IDs

    # Rollback capability
    rollback_data: Optional[Dict[str, Any]] = None
    can_rollback: bool = True
    rolled_back: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate adaptation result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if not 0.0 <= self.expected_improvement <= 1.0:
            raise ValueError(f"Expected improvement must be 0-1, got {self.expected_improvement}")
        if self.actual_improvement is not None:
            if not 0.0 <= self.actual_improvement <= 1.0:
                raise ValueError(f"Actual improvement must be 0-1, got {self.actual_improvement}")

    @property
    def was_successful(self) -> bool:
        """Check if adaptation was successful."""
        if self.actual_improvement is None:
            return False  # Can't determine yet
        return self.actual_improvement >= 0.5

    @property
    def needs_rollback(self) -> bool:
        """Check if adaptation should be rolled back."""
        if self.rolled_back or not self.can_rollback:
            return False
        if self.actual_improvement is None:
            return False
        return self.actual_improvement < 0.3  # Clear negative result


@dataclass
class ConsolidationResult:
    """Result of a knowledge consolidation operation."""

    consolidation_id: str
    timestamp: datetime
    duration_ms: float

    # What was consolidated
    patterns_processed: int
    patterns_merged: int
    patterns_promoted: int
    patterns_archived: int

    feedback_processed: int
    preferences_updated: int
    preferences_created: int
    preferences_removed: int

    # Knowledge graph updates
    knowledge_nodes_added: int = 0
    knowledge_edges_added: int = 0
    knowledge_updates: int = 0

    # Meta-learning
    meta_insights: List[str] = field(default_factory=list)
    learning_rate_adjustments: Dict[str, float] = field(default_factory=dict)

    # Performance
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_changes(self) -> int:
        """Get total number of changes made."""
        return (
            self.patterns_merged
            + self.patterns_promoted
            + self.patterns_archived
            + self.preferences_updated
            + self.preferences_created
            + self.preferences_removed
            + self.knowledge_nodes_added
            + self.knowledge_edges_added
        )

    @property
    def efficiency_score(self) -> float:
        """Calculate consolidation efficiency (changes per second)."""
        if self.duration_ms == 0:
            return 0.0
        return (self.total_changes / self.duration_ms) * 1000


@dataclass
class LearningContext:
    """Context for learning operations."""

    user_id: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

    # Current state
    message_index: int = 0
    previous_patterns: List[LearningPattern] = field(default_factory=list)
    previous_feedback: List[FeedbackSignal] = field(default_factory=list)
    active_preferences: Dict[PreferenceDimension, UserPreference] = field(default_factory=dict)

    # Temporal info
    time_since_last_interaction: Optional[float] = None  # seconds
    time_of_day: Optional[str] = None  # morning, afternoon, evening, night
    day_of_week: Optional[str] = None

    # Context tags
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_history(self) -> bool:
        """Check if we have learning history."""
        return (
            len(self.previous_patterns) > 0
            or len(self.previous_feedback) > 0
            or len(self.active_preferences) > 0
        )


@dataclass
class LearningMetrics:
    """Metrics for learning system performance."""

    # Pattern detection metrics
    patterns_detected: int = 0
    patterns_active: int = 0
    avg_pattern_confidence: float = 0.0

    # Feedback metrics
    feedback_signals: int = 0
    positive_feedback_ratio: float = 0.0
    feedback_actionability: float = 0.0

    # Preference metrics
    preferences_learned: int = 0
    preferences_stable: int = 0
    avg_preference_confidence: float = 0.0

    # Adaptation metrics
    adaptations_applied: int = 0
    adaptations_successful: int = 0
    adaptations_rolled_back: int = 0
    avg_adaptation_improvement: float = 0.0

    # Consolidation metrics
    consolidations_run: int = 0
    avg_consolidation_time_ms: float = 0.0
    last_consolidation: Optional[datetime] = None

    # Overall metrics
    learning_rate: float = 1.0  # Multiplier for learning speed
    exploration_rate: float = 0.1  # Exploration vs exploitation

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.adaptations_applied == 0:
            return 0.0
        return self.adaptations_successful / self.adaptations_applied

    @property
    def stability_score(self) -> float:
        """Calculate system stability score."""
        if self.preferences_learned == 0:
            return 0.0
        return self.preferences_stable / self.preferences_learned
