"""
Learning system utilities.

Helper functions for common learning tasks.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from morgan.learning.types import (
    AdaptationResult,
    ConsolidationResult,
    FeedbackSignal,
    LearningMetrics,
    LearningPattern,
    PreferenceDimension,
    UserPreference,
)


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracing."""
    return f"corr_{uuid.uuid4().hex[:16]}"


def hash_text(text: str) -> str:
    """
    Generate a hash of text for cache keys.

    Args:
        text: Text to hash

    Returns:
        Hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Convert to lowercase
    normalized = text.lower().strip()
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    return normalized


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple token overlap.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score 0-1
    """
    if not text1 or not text2:
        return 0.0

    # Normalize
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    # Tokenize
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0


def merge_patterns(patterns: List[LearningPattern]) -> List[LearningPattern]:
    """
    Merge similar patterns.

    Args:
        patterns: List of patterns to merge

    Returns:
        Merged pattern list
    """
    if not patterns:
        return []

    # Group by pattern type
    grouped: Dict[str, List[LearningPattern]] = {}
    for pattern in patterns:
        key = pattern.pattern_type.value
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pattern)

    merged = []
    for group in grouped.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Sort by confidence and take highest
            sorted_group = sorted(group, key=lambda p: p.confidence, reverse=True)
            merged.append(sorted_group[0])

    return merged


def aggregate_feedback(feedback_list: List[FeedbackSignal]) -> Dict[str, Any]:
    """
    Aggregate feedback signals into summary statistics.

    Args:
        feedback_list: List of feedback signals

    Returns:
        Aggregated statistics
    """
    if not feedback_list:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "avg_sentiment": 0.0,
            "actionable_count": 0,
        }

    positive = sum(1 for f in feedback_list if f.is_positive)
    negative = sum(1 for f in feedback_list if f.is_negative)
    neutral = len(feedback_list) - positive - negative

    sentiments = [f.sentiment for f in feedback_list]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    actionable = sum(1 for f in feedback_list if f.is_actionable)

    return {
        "total": len(feedback_list),
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "avg_sentiment": avg_sentiment,
        "actionable_count": actionable,
        "positive_ratio": positive / len(feedback_list),
        "negative_ratio": negative / len(feedback_list),
        "actionable_ratio": actionable / len(feedback_list),
    }


def resolve_preference_conflicts(
    preferences: List[UserPreference],
) -> UserPreference:
    """
    Resolve conflicting preferences for the same dimension.

    Args:
        preferences: List of conflicting preferences

    Returns:
        Resolved preference (highest confidence)
    """
    if not preferences:
        raise ValueError("Cannot resolve conflicts from empty list")

    if len(preferences) == 1:
        return preferences[0]

    # Sort by confidence and recency
    sorted_prefs = sorted(
        preferences,
        key=lambda p: (p.confidence, p.last_updated),
        reverse=True,
    )

    return sorted_prefs[0]


def calculate_learning_rate_adjustment(
    success_rate: float,
    current_rate: float,
    min_rate: float = 0.1,
    max_rate: float = 2.0,
) -> float:
    """
    Calculate learning rate adjustment based on success rate.

    Args:
        success_rate: Current success rate (0-1)
        current_rate: Current learning rate
        min_rate: Minimum learning rate
        max_rate: Maximum learning rate

    Returns:
        Adjusted learning rate
    """
    if success_rate >= 0.8:
        # High success, increase rate
        adjusted = current_rate * 1.1
    elif success_rate >= 0.6:
        # Good success, small increase
        adjusted = current_rate * 1.05
    elif success_rate >= 0.4:
        # Moderate success, keep stable
        adjusted = current_rate
    elif success_rate >= 0.2:
        # Low success, decrease rate
        adjusted = current_rate * 0.9
    else:
        # Very low success, decrease significantly
        adjusted = current_rate * 0.8

    # Clamp to valid range
    return max(min_rate, min(max_rate, adjusted))


def calculate_exploration_rate(
    total_interactions: int,
    success_rate: float,
    initial_rate: float = 0.2,
    min_rate: float = 0.05,
) -> float:
    """
    Calculate exploration rate using epsilon-greedy decay.

    Args:
        total_interactions: Total number of interactions
        success_rate: Current success rate (0-1)
        initial_rate: Initial exploration rate
        min_rate: Minimum exploration rate

    Returns:
        Exploration rate
    """
    # Decay based on interaction count
    decay_factor = 0.995
    rate = initial_rate * (decay_factor ** total_interactions)

    # Adjust based on success rate
    if success_rate < 0.5:
        # Low success, explore more
        rate *= 1.5

    # Clamp to minimum
    return max(min_rate, rate)


def format_pattern_summary(pattern: LearningPattern) -> str:
    """
    Format pattern as human-readable summary.

    Args:
        pattern: Learning pattern

    Returns:
        Formatted summary string
    """
    return (
        f"{pattern.pattern_type.value}: {pattern.description} "
        f"(confidence: {pattern.confidence:.2f}, "
        f"frequency: {pattern.frequency}, "
        f"age: {pattern.age_days:.1f} days)"
    )


def format_preference_summary(preference: UserPreference) -> str:
    """
    Format preference as human-readable summary.

    Args:
        preference: User preference

    Returns:
        Formatted summary string
    """
    return (
        f"{preference.dimension.value}: {preference.value} "
        f"(confidence: {preference.confidence:.2f}, "
        f"stability: {'stable' if preference.is_stable else 'unstable'})"
    )


def format_adaptation_summary(adaptation: AdaptationResult) -> str:
    """
    Format adaptation as human-readable summary.

    Args:
        adaptation: Adaptation result

    Returns:
        Formatted summary string
    """
    success = "✓" if adaptation.was_successful else "?"
    return (
        f"{success} {adaptation.target}: {adaptation.strategy.value} "
        f"(expected: {adaptation.expected_improvement:.2f}, "
        f"actual: {adaptation.actual_improvement or 'pending'})"
    )


def format_metrics_summary(metrics: LearningMetrics) -> str:
    """
    Format metrics as human-readable summary.

    Args:
        metrics: Learning metrics

    Returns:
        Formatted summary string
    """
    lines = [
        f"Patterns: {metrics.patterns_active}/{metrics.patterns_detected} active "
        f"(avg conf: {metrics.avg_pattern_confidence:.2f})",
        f"Feedback: {metrics.feedback_signals} signals "
        f"(positive: {metrics.positive_feedback_ratio:.1%})",
        f"Preferences: {metrics.preferences_stable}/{metrics.preferences_learned} stable "
        f"(avg conf: {metrics.avg_preference_confidence:.2f})",
        f"Adaptations: {metrics.adaptations_successful}/{metrics.adaptations_applied} successful "
        f"({metrics.success_rate:.1%})",
        f"Learning rate: {metrics.learning_rate:.2f}, "
        f"Exploration: {metrics.exploration_rate:.1%}",
    ]
    return "\n".join(lines)


def validate_feedback_signal(signal: FeedbackSignal) -> List[str]:
    """
    Validate feedback signal and return list of issues.

    Args:
        signal: Feedback signal to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    if not signal.user_id:
        issues.append("Missing user_id")

    if signal.rating is not None:
        if signal.rating < 0 or signal.rating > 1:
            issues.append(f"Invalid rating: {signal.rating}")

    if signal.sentiment < -1 or signal.sentiment > 1:
        issues.append(f"Invalid sentiment: {signal.sentiment}")

    if signal.sentiment_confidence < 0 or signal.sentiment_confidence > 1:
        issues.append(f"Invalid sentiment confidence: {signal.sentiment_confidence}")

    if not signal.is_actionable and signal.feedback_type.value.startswith("explicit"):
        issues.append("Explicit feedback should be actionable")

    return issues


def validate_pattern(pattern: LearningPattern) -> List[str]:
    """
    Validate learning pattern and return list of issues.

    Args:
        pattern: Learning pattern to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    if pattern.confidence < 0 or pattern.confidence > 1:
        issues.append(f"Invalid confidence: {pattern.confidence}")

    if pattern.frequency < 1:
        issues.append(f"Invalid frequency: {pattern.frequency}")

    if pattern.last_observed < pattern.first_observed:
        issues.append("last_observed before first_observed")

    if pattern.regularity_score < 0 or pattern.regularity_score > 1:
        issues.append(f"Invalid regularity_score: {pattern.regularity_score}")

    if pattern.strength < 0 or pattern.strength > 1:
        issues.append(f"Invalid strength: {pattern.strength}")

    return issues


def create_cache_key(*parts: str) -> str:
    """
    Create a cache key from multiple parts.

    Args:
        *parts: Key parts

    Returns:
        Cache key string
    """
    return ":".join(str(p) for p in parts if p)


def get_time_bucket(timestamp: datetime) -> str:
    """
    Get time bucket for temporal analysis.

    Args:
        timestamp: Timestamp to bucket

    Returns:
        Time bucket string (e.g., "2024-01-15-14" for hourly)
    """
    return timestamp.strftime("%Y-%m-%d-%H")


def get_day_of_week(timestamp: datetime) -> str:
    """
    Get day of week from timestamp.

    Args:
        timestamp: Timestamp

    Returns:
        Day name (Monday, Tuesday, etc.)
    """
    return timestamp.strftime("%A")


def get_time_of_day(timestamp: datetime) -> str:
    """
    Get time of day category.

    Args:
        timestamp: Timestamp

    Returns:
        Time category: morning, afternoon, evening, night
    """
    hour = timestamp.hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"
