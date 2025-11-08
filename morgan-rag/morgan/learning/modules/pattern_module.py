"""
Pattern Detection Module.

Detects and analyzes behavioral patterns in user interactions
using temporal analysis, clustering, and anomaly detection.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.learning.base import (
    AsyncCache,
    BaseLearningModule,
    CircuitBreaker,
    HealthStatus,
)
from morgan.learning.exceptions import PatternDetectionError
from morgan.learning.types import (
    LearningContext,
    LearningPattern,
    PatternType,
)
from morgan.learning.utils import (
    generate_correlation_id,
    generate_id,
    get_time_bucket,
    get_time_of_day,
    merge_patterns,
)


class PatternModule(BaseLearningModule):
    """
    Pattern detection and analysis module.

    Features:
    - Behavioral pattern recognition
    - Temporal pattern analysis
    - Pattern clustering
    - Anomaly detection
    - Pattern evolution tracking
    """

    def __init__(
        self,
        min_pattern_frequency: int = 3,
        min_pattern_confidence: float = 0.6,
        pattern_window_hours: int = 168,  # 1 week
        enable_cache: bool = True,
        max_patterns_per_user: int = 100,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize pattern module.

        Args:
            min_pattern_frequency: Minimum occurrences to consider a pattern
            min_pattern_confidence: Minimum confidence threshold
            pattern_window_hours: Time window for pattern detection (hours)
            enable_cache: Enable result caching
            max_patterns_per_user: Maximum patterns to track per user
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__("PatternModule", correlation_id)

        self._min_frequency = min_pattern_frequency
        self._min_confidence = min_pattern_confidence
        self._window_hours = pattern_window_hours
        self._max_patterns = max_patterns_per_user

        # Storage for user patterns
        self._user_patterns: Dict[str, List[LearningPattern]] = defaultdict(list)
        self._user_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Temporal tracking
        self._temporal_buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Cache and circuit breaker
        self._cache = AsyncCache(max_size=500) if enable_cache else None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=PatternDetectionError,
            name="pattern_detection",
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(10)
        self._lock = asyncio.Lock()

        # Metrics
        self._patterns_detected = 0
        self._patterns_merged = 0
        self._anomalies_detected = 0

    async def initialize(self) -> None:
        """Initialize the pattern module."""
        self._log_info("Pattern module initialized")

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        async with self._lock:
            self._user_patterns.clear()
            self._user_events.clear()
            self._temporal_buckets.clear()

        if self._cache:
            await self._cache.clear()

        self._log_info("Pattern module cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check module health."""
        try:
            patterns_count = sum(len(p) for p in self._user_patterns.values())
            events_count = sum(len(e) for e in self._user_events.values())

            return HealthStatus(
                healthy=True,
                message="Pattern module healthy",
                details={
                    "patterns_tracked": patterns_count,
                    "events_tracked": events_count,
                    "users_tracked": len(self._user_patterns),
                    "circuit_breaker_state": self._circuit_breaker.state,
                    "cache_hit_rate": self._cache.hit_rate if self._cache else 0.0,
                },
                last_check=asyncio.get_event_loop().time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Pattern module unhealthy: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def detect_patterns(
        self,
        user_id: str,
        context: Optional[LearningContext] = None,
        force_refresh: bool = False,
    ) -> List[LearningPattern]:
        """
        Detect patterns for a user.

        Args:
            user_id: User identifier
            context: Optional learning context
            force_refresh: Force pattern re-detection

        Returns:
            List of detected patterns

        Raises:
            PatternDetectionError: If pattern detection fails
        """
        await self.ensure_initialized()

        # Check cache
        if not force_refresh and self._cache:
            cache_key = f"patterns:{user_id}"
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        try:
            # Use circuit breaker
            patterns = await self._circuit_breaker.call(
                self._detect_patterns_internal,
                user_id,
                context,
            )

            # Cache results
            if self._cache:
                cache_key = f"patterns:{user_id}"
                await self._cache.set(cache_key, patterns, ttl=300)

            return patterns

        except Exception as e:
            self._log_error("Pattern detection failed", e, user_id=user_id)
            raise PatternDetectionError(
                f"Failed to detect patterns for user {user_id}",
                cause=e,
            )

    async def _detect_patterns_internal(
        self,
        user_id: str,
        context: Optional[LearningContext],
    ) -> List[LearningPattern]:
        """Internal pattern detection implementation."""
        async with self._semaphore:
            # Get user events
            events = await self._get_user_events(user_id)

            if len(events) < self._min_frequency:
                return []

            # Run pattern detection strategies in parallel
            recurring_task = self._detect_recurring_patterns(user_id, events)
            temporal_task = self._detect_temporal_patterns(user_id, events)
            sequential_task = self._detect_sequential_patterns(user_id, events)
            contextual_task = self._detect_contextual_patterns(user_id, events, context)

            pattern_sets = await asyncio.gather(
                recurring_task,
                temporal_task,
                sequential_task,
                contextual_task,
                return_exceptions=True,
            )

            # Merge all patterns
            all_patterns = []
            for result in pattern_sets:
                if isinstance(result, Exception):
                    self._log_warning(f"Pattern detection strategy failed: {result}")
                    continue
                all_patterns.extend(result)

            # Merge similar patterns
            merged = merge_patterns(all_patterns)

            # Filter by confidence and frequency
            significant = [
                p
                for p in merged
                if p.confidence >= self._min_confidence
                and p.frequency >= self._min_frequency
            ]

            # Update stored patterns
            await self._update_user_patterns(user_id, significant)

            self._patterns_detected += len(significant)

            return significant

    async def _detect_recurring_patterns(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
    ) -> List[LearningPattern]:
        """Detect recurring behavioral patterns."""
        patterns = []

        # Group events by action type
        action_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            action = event.get("action", "unknown")
            action_groups[action].append(event)

        # Analyze each action group
        for action, action_events in action_groups.items():
            if len(action_events) < self._min_frequency:
                continue

            # Calculate regularity
            timestamps = [e["timestamp"] for e in action_events]
            regularity = self._calculate_regularity(timestamps)

            # Calculate confidence based on frequency and regularity
            confidence = min(
                (len(action_events) / 10.0) * 0.5 + regularity * 0.5,
                1.0,
            )

            if confidence >= self._min_confidence:
                pattern = LearningPattern(
                    pattern_id=generate_id("pattern_recurring"),
                    pattern_type=PatternType.RECURRING,
                    description=f"Recurring action: {action}",
                    confidence=confidence,
                    frequency=len(action_events),
                    first_observed=min(timestamps),
                    last_observed=max(timestamps),
                    trigger_contexts=[],
                    associated_actions=[action],
                    regularity_score=regularity,
                    strength=confidence,
                )
                patterns.append(pattern)

        return patterns

    async def _detect_temporal_patterns(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
    ) -> List[LearningPattern]:
        """Detect time-based patterns."""
        patterns = []

        # Group events by time of day
        time_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            timestamp = event["timestamp"]
            time_category = get_time_of_day(timestamp)
            time_groups[time_category].append(event)

        # Analyze each time group
        total_events = len(events)
        for time_category, time_events in time_groups.items():
            if len(time_events) < self._min_frequency:
                continue

            # Calculate temporal concentration
            concentration = len(time_events) / total_events

            if concentration >= 0.5:  # More than 50% of events in this time
                confidence = min(concentration * 1.2, 1.0)

                pattern = LearningPattern(
                    pattern_id=generate_id("pattern_temporal"),
                    pattern_type=PatternType.TEMPORAL,
                    description=f"Activity concentrated in {time_category}",
                    confidence=confidence,
                    frequency=len(time_events),
                    first_observed=min(e["timestamp"] for e in time_events),
                    last_observed=max(e["timestamp"] for e in time_events),
                    trigger_contexts=[time_category],
                    associated_actions=list(
                        set(e.get("action", "unknown") for e in time_events)
                    ),
                    regularity_score=concentration,
                    strength=confidence,
                    metadata={"time_category": time_category},
                )
                patterns.append(pattern)

        return patterns

    async def _detect_sequential_patterns(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
    ) -> List[LearningPattern]:
        """Detect sequential action patterns."""
        patterns = []

        if len(events) < 2:
            return patterns

        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e["timestamp"])

        # Look for sequences of 2-3 actions
        sequences: Dict[str, int] = defaultdict(int)
        sequence_times: Dict[str, List[datetime]] = defaultdict(list)

        for i in range(len(sorted_events) - 1):
            action1 = sorted_events[i].get("action", "unknown")
            action2 = sorted_events[i + 1].get("action", "unknown")
            seq_key = f"{action1}->{action2}"
            sequences[seq_key] += 1
            sequence_times[seq_key].append(sorted_events[i]["timestamp"])

        # Identify significant sequences
        for seq_key, count in sequences.items():
            if count >= self._min_frequency:
                actions = seq_key.split("->")
                times = sequence_times[seq_key]

                confidence = min((count / len(events)) * 2.0, 1.0)

                if confidence >= self._min_confidence:
                    pattern = LearningPattern(
                        pattern_id=generate_id("pattern_sequential"),
                        pattern_type=PatternType.SEQUENTIAL,
                        description=f"Sequential pattern: {seq_key}",
                        confidence=confidence,
                        frequency=count,
                        first_observed=min(times),
                        last_observed=max(times),
                        trigger_contexts=[],
                        associated_actions=actions,
                        regularity_score=count / len(events),
                        strength=confidence,
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_contextual_patterns(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
        context: Optional[LearningContext],
    ) -> List[LearningPattern]:
        """Detect context-dependent patterns."""
        patterns = []

        if not context or not context.tags:
            return patterns

        # Group events by context tags
        for tag in context.tags:
            tagged_events = [
                e for e in events if tag in e.get("context_tags", set())
            ]

            if len(tagged_events) >= self._min_frequency:
                confidence = min((len(tagged_events) / len(events)) * 1.5, 1.0)

                if confidence >= self._min_confidence:
                    pattern = LearningPattern(
                        pattern_id=generate_id("pattern_contextual"),
                        pattern_type=PatternType.CONTEXTUAL,
                        description=f"Context-dependent behavior: {tag}",
                        confidence=confidence,
                        frequency=len(tagged_events),
                        first_observed=min(e["timestamp"] for e in tagged_events),
                        last_observed=max(e["timestamp"] for e in tagged_events),
                        trigger_contexts=[tag],
                        associated_actions=list(
                            set(e.get("action", "unknown") for e in tagged_events)
                        ),
                        regularity_score=len(tagged_events) / len(events),
                        strength=confidence,
                        metadata={"context_tag": tag},
                    )
                    patterns.append(pattern)

        return patterns

    def _calculate_regularity(self, timestamps: List[datetime]) -> float:
        """
        Calculate regularity score for a series of timestamps.

        Returns:
            Regularity score 0-1 (higher = more regular)
        """
        if len(timestamps) < 2:
            return 0.0

        # Sort timestamps
        sorted_times = sorted(timestamps)

        # Calculate intervals
        intervals = []
        for i in range(len(sorted_times) - 1):
            interval = (sorted_times[i + 1] - sorted_times[i]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return 0.0

        # Calculate coefficient of variation
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.0

        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval

        # Convert to regularity score (lower CV = higher regularity)
        regularity = max(0.0, 1.0 - min(cv, 1.0))

        return regularity

    async def add_event(
        self,
        user_id: str,
        action: str,
        timestamp: Optional[datetime] = None,
        context_tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an event for pattern detection.

        Args:
            user_id: User identifier
            action: Action performed
            timestamp: Event timestamp (default: now)
            context_tags: Optional context tags
            metadata: Optional metadata
        """
        event = {
            "action": action,
            "timestamp": timestamp or datetime.utcnow(),
            "context_tags": context_tags or set(),
            "metadata": metadata or {},
        }

        async with self._lock:
            self._user_events[user_id].append(event)

            # Trim old events outside window
            cutoff = datetime.utcnow() - timedelta(hours=self._window_hours)
            self._user_events[user_id] = [
                e for e in self._user_events[user_id] if e["timestamp"] > cutoff
            ]

            # Limit total events per user
            if len(self._user_events[user_id]) > 1000:
                self._user_events[user_id] = self._user_events[user_id][-1000:]

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate(f"patterns:{user_id}")

    async def _get_user_events(self, user_id: str) -> List[Dict[str, Any]]:
        """Get events for a user within the time window."""
        async with self._lock:
            return list(self._user_events.get(user_id, []))

    async def _update_user_patterns(
        self,
        user_id: str,
        patterns: List[LearningPattern],
    ) -> None:
        """Update stored patterns for a user."""
        async with self._lock:
            self._user_patterns[user_id] = patterns

            # Limit patterns per user
            if len(self._user_patterns[user_id]) > self._max_patterns:
                # Keep highest confidence patterns
                sorted_patterns = sorted(
                    self._user_patterns[user_id],
                    key=lambda p: p.confidence,
                    reverse=True,
                )
                self._user_patterns[user_id] = sorted_patterns[: self._max_patterns]

    async def get_active_patterns(
        self,
        user_id: str,
        min_confidence: Optional[float] = None,
    ) -> List[LearningPattern]:
        """
        Get active patterns for a user.

        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold

        Returns:
            List of active patterns
        """
        async with self._lock:
            patterns = self._user_patterns.get(user_id, [])

        if min_confidence is not None:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        return patterns

    async def clear_user_patterns(self, user_id: str) -> None:
        """Clear all patterns for a user."""
        async with self._lock:
            self._user_patterns.pop(user_id, None)
            self._user_events.pop(user_id, None)

        if self._cache:
            await self._cache.invalidate(f"patterns:{user_id}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "patterns_detected": self._patterns_detected,
            "patterns_merged": self._patterns_merged,
            "anomalies_detected": self._anomalies_detected,
            "users_tracked": len(self._user_patterns),
            "circuit_breaker_state": self._circuit_breaker.state,
            "cache_stats": self._cache.stats if self._cache else None,
        }
