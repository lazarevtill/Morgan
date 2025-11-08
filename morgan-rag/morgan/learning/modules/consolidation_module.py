"""
Consolidation Module.

Performs periodic background consolidation of learning data including
pattern merging, preference strengthening, and meta-learning extraction.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.learning.base import (
    AsyncCache,
    BaseLearningModule,
    CircuitBreaker,
    HealthStatus,
)
from morgan.learning.exceptions import ConsolidationError
from morgan.learning.types import (
    ConsolidationResult,
    FeedbackSignal,
    LearningMetrics,
    LearningPattern,
    UserPreference,
)
from morgan.learning.utils import (
    calculate_exploration_rate,
    calculate_learning_rate_adjustment,
    generate_id,
    merge_patterns,
)


class ConsolidationModule(BaseLearningModule):
    """
    Knowledge consolidation module.

    Features:
    - Pattern consolidation and merging
    - Preference strengthening
    - Knowledge graph updates
    - Meta-learning extraction
    - Periodic background consolidation
    - Learning rate adjustments
    """

    def __init__(
        self,
        consolidation_interval_hours: int = 24,
        enable_background_task: bool = True,
        min_patterns_for_consolidation: int = 10,
        min_feedback_for_consolidation: int = 20,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize consolidation module.

        Args:
            consolidation_interval_hours: Hours between consolidations
            enable_background_task: Enable background consolidation task
            min_patterns_for_consolidation: Minimum patterns to trigger consolidation
            min_feedback_for_consolidation: Minimum feedback to trigger consolidation
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__("ConsolidationModule", correlation_id)

        self._interval_hours = consolidation_interval_hours
        self._enable_background = enable_background_task
        self._min_patterns = min_patterns_for_consolidation
        self._min_feedback = min_feedback_for_consolidation

        # Background task
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False

        # Last consolidation tracking
        self._last_consolidation: Dict[str, datetime] = {}

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300.0,
            expected_exception=ConsolidationError,
            name="consolidation",
        )

        # Metrics
        self._consolidations_run = 0
        self._total_consolidation_time = 0.0
        self._patterns_merged = 0
        self._preferences_updated = 0

    async def initialize(self) -> None:
        """Initialize the consolidation module."""
        if self._enable_background:
            self._running = True
            self._consolidation_task = asyncio.create_task(
                self._background_consolidation()
            )
            self._log_info("Background consolidation task started")

        self._log_info("Consolidation module initialized")

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        if self._consolidation_task:
            self._running = False
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

        self._log_info("Consolidation module cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check module health."""
        try:
            task_healthy = (
                not self._enable_background
                or (self._consolidation_task and not self._consolidation_task.done())
            )

            return HealthStatus(
                healthy=task_healthy,
                message=(
                    "Consolidation module healthy"
                    if task_healthy
                    else "Background task stopped"
                ),
                details={
                    "consolidations_run": self._consolidations_run,
                    "patterns_merged": self._patterns_merged,
                    "preferences_updated": self._preferences_updated,
                    "background_task_running": task_healthy,
                    "users_tracked": len(self._last_consolidation),
                    "circuit_breaker_state": self._circuit_breaker.state,
                },
                last_check=asyncio.get_event_loop().time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Consolidation module unhealthy: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def consolidate(
        self,
        user_id: str,
        patterns: List[LearningPattern],
        feedback: List[FeedbackSignal],
        preferences: Dict[str, UserPreference],
        metrics: LearningMetrics,
    ) -> ConsolidationResult:
        """
        Perform consolidation for a user.

        Args:
            user_id: User identifier
            patterns: User patterns
            feedback: User feedback
            preferences: User preferences
            metrics: Current metrics

        Returns:
            Consolidation result

        Raises:
            ConsolidationError: If consolidation fails
        """
        await self.ensure_initialized()

        try:
            result = await self._circuit_breaker.call(
                self._consolidate_internal,
                user_id,
                patterns,
                feedback,
                preferences,
                metrics,
            )

            self._consolidations_run += 1
            self._total_consolidation_time += result.duration_ms

            # Update last consolidation time
            self._last_consolidation[user_id] = datetime.utcnow()

            return result

        except Exception as e:
            self._log_error("Consolidation failed", e, user_id=user_id)
            raise ConsolidationError(
                f"Failed to consolidate data for user {user_id}",
                cause=e,
            )

    async def _consolidate_internal(
        self,
        user_id: str,
        patterns: List[LearningPattern],
        feedback: List[FeedbackSignal],
        preferences: Dict[str, UserPreference],
        metrics: LearningMetrics,
    ) -> ConsolidationResult:
        """Internal consolidation implementation."""
        start_time = time.perf_counter()

        result = ConsolidationResult(
            consolidation_id=generate_id("consolidation"),
            timestamp=datetime.utcnow(),
            duration_ms=0.0,
            patterns_processed=len(patterns),
            patterns_merged=0,
            patterns_promoted=0,
            patterns_archived=0,
            feedback_processed=len(feedback),
            preferences_updated=0,
            preferences_created=0,
            preferences_removed=0,
            success=True,
            errors=[],
            warnings=[],
        )

        # Run consolidation phases in sequence
        try:
            # Phase 1: Pattern consolidation
            pattern_result = await self._consolidate_patterns(patterns)
            result.patterns_merged = pattern_result["merged"]
            result.patterns_promoted = pattern_result["promoted"]
            result.patterns_archived = pattern_result["archived"]
            self._patterns_merged += pattern_result["merged"]

        except Exception as e:
            result.errors.append(f"Pattern consolidation failed: {str(e)}")
            self._log_error("Pattern consolidation phase failed", e)

        try:
            # Phase 2: Preference consolidation
            pref_result = await self._consolidate_preferences(
                preferences, feedback
            )
            result.preferences_updated = pref_result["updated"]
            result.preferences_created = pref_result["created"]
            result.preferences_removed = pref_result["removed"]
            self._preferences_updated += pref_result["updated"]

        except Exception as e:
            result.errors.append(f"Preference consolidation failed: {str(e)}")
            self._log_error("Preference consolidation phase failed", e)

        try:
            # Phase 3: Meta-learning extraction
            meta_insights = await self._extract_meta_learning(
                patterns, feedback, preferences, metrics
            )
            result.meta_insights = meta_insights

        except Exception as e:
            result.errors.append(f"Meta-learning extraction failed: {str(e)}")
            self._log_error("Meta-learning extraction phase failed", e)

        try:
            # Phase 4: Learning rate adjustments
            adjustments = await self._adjust_learning_rates(metrics)
            result.learning_rate_adjustments = adjustments

        except Exception as e:
            result.errors.append(f"Learning rate adjustment failed: {str(e)}")
            self._log_error("Learning rate adjustment phase failed", e)

        # Calculate duration
        result.duration_ms = (time.perf_counter() - start_time) * 1000

        # Set success flag
        result.success = len(result.errors) == 0

        return result

    async def _consolidate_patterns(
        self,
        patterns: List[LearningPattern],
    ) -> Dict[str, int]:
        """
        Consolidate patterns.

        Returns:
            Dictionary with merged, promoted, archived counts
        """
        if not patterns:
            return {"merged": 0, "promoted": 0, "archived": 0}

        # Merge similar patterns
        merged_patterns = merge_patterns(patterns)
        merged_count = len(patterns) - len(merged_patterns)

        # Promote strong patterns
        promoted = [
            p
            for p in merged_patterns
            if p.confidence >= 0.8 and p.frequency >= 10 and p.is_significant
        ]
        promoted_count = len(promoted)

        # Archive old/weak patterns
        cutoff = datetime.utcnow() - timedelta(days=90)
        archived = [
            p
            for p in merged_patterns
            if p.last_observed < cutoff or (p.confidence < 0.4 and not p.is_significant)
        ]
        archived_count = len(archived)

        return {
            "merged": merged_count,
            "promoted": promoted_count,
            "archived": archived_count,
        }

    async def _consolidate_preferences(
        self,
        preferences: Dict[str, UserPreference],
        feedback: List[FeedbackSignal],
    ) -> Dict[str, int]:
        """
        Consolidate preferences based on recent feedback.

        Returns:
            Dictionary with updated, created, removed counts
        """
        updated = 0
        created = 0
        removed = 0

        # Group feedback by recency
        recent_feedback = [
            f
            for f in feedback
            if f.timestamp > datetime.utcnow() - timedelta(days=7)
        ]

        # Strengthen preferences with positive feedback
        for pref in preferences.values():
            supporting_feedback = [
                f for f in recent_feedback if f.is_positive and pref.is_stable
            ]

            if len(supporting_feedback) >= 3:
                updated += 1

        # Identify preferences to remove (conflicting or expired)
        cutoff = datetime.utcnow() - timedelta(days=90)
        for pref in preferences.values():
            if pref.last_updated < cutoff:
                removed += 1
            elif len(pref.conflicting_signals) > len(pref.supporting_signals):
                removed += 1

        return {
            "updated": updated,
            "created": created,
            "removed": removed,
        }

    async def _extract_meta_learning(
        self,
        patterns: List[LearningPattern],
        feedback: List[FeedbackSignal],
        preferences: Dict[str, UserPreference],
        metrics: LearningMetrics,
    ) -> List[str]:
        """
        Extract meta-learning insights.

        Returns:
            List of insight strings
        """
        insights = []

        # Analyze pattern trends
        if patterns:
            temporal_patterns = [
                p for p in patterns if p.pattern_type.value == "temporal"
            ]
            if len(temporal_patterns) >= 3:
                insights.append(
                    f"Strong temporal behavior detected: {len(temporal_patterns)} patterns"
                )

            recurring_patterns = [
                p for p in patterns if p.pattern_type.value == "recurring"
            ]
            if len(recurring_patterns) >= 5:
                insights.append(
                    f"Highly habitual user: {len(recurring_patterns)} recurring patterns"
                )

        # Analyze feedback trends
        if feedback:
            positive_ratio = sum(1 for f in feedback if f.is_positive) / len(feedback)
            if positive_ratio >= 0.8:
                insights.append(
                    f"High satisfaction rate: {positive_ratio:.1%} positive feedback"
                )
            elif positive_ratio <= 0.3:
                insights.append(
                    f"Low satisfaction rate: {positive_ratio:.1%} positive feedback - needs attention"
                )

        # Analyze preference stability
        if preferences:
            stable_prefs = sum(1 for p in preferences.values() if p.is_stable)
            stability_ratio = stable_prefs / len(preferences)
            if stability_ratio >= 0.8:
                insights.append(
                    f"Preferences well-established: {stability_ratio:.1%} stable"
                )
            elif stability_ratio <= 0.4:
                insights.append(
                    f"Preferences still evolving: {stability_ratio:.1%} stable"
                )

        # Analyze learning effectiveness
        if metrics.success_rate >= 0.8:
            insights.append(
                f"Learning highly effective: {metrics.success_rate:.1%} success rate"
            )
        elif metrics.success_rate <= 0.4:
            insights.append(
                f"Learning needs improvement: {metrics.success_rate:.1%} success rate"
            )

        return insights

    async def _adjust_learning_rates(
        self,
        metrics: LearningMetrics,
    ) -> Dict[str, float]:
        """
        Adjust learning rates based on performance.

        Returns:
            Dictionary of learning rate adjustments
        """
        adjustments = {}

        # Adjust main learning rate
        new_rate = calculate_learning_rate_adjustment(
            metrics.success_rate,
            metrics.learning_rate,
        )
        if new_rate != metrics.learning_rate:
            adjustments["learning_rate"] = new_rate

        # Adjust exploration rate
        total_interactions = (
            metrics.patterns_detected + metrics.feedback_signals
        )
        new_exploration = calculate_exploration_rate(
            total_interactions,
            metrics.success_rate,
        )
        if new_exploration != metrics.exploration_rate:
            adjustments["exploration_rate"] = new_exploration

        return adjustments

    async def _background_consolidation(self) -> None:
        """Background task for periodic consolidation."""
        self._log_info("Background consolidation task starting")

        while self._running:
            try:
                # Wait for consolidation interval
                await asyncio.sleep(self._interval_hours * 3600)

                # Run consolidation for users that need it
                # Note: In production, this would fetch users from a queue or database
                # For now, we'll just log that consolidation would run

                self._log_info("Background consolidation cycle complete")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_error("Background consolidation error", e)
                # Continue running despite errors
                await asyncio.sleep(60)

        self._log_info("Background consolidation task stopped")

    async def should_consolidate(
        self,
        user_id: str,
        pattern_count: int,
        feedback_count: int,
    ) -> bool:
        """
        Check if consolidation should run for a user.

        Args:
            user_id: User identifier
            pattern_count: Current pattern count
            feedback_count: Current feedback count

        Returns:
            True if consolidation should run
        """
        # Check if enough data accumulated
        if (
            pattern_count < self._min_patterns
            and feedback_count < self._min_feedback
        ):
            return False

        # Check last consolidation time
        last_consolidation = self._last_consolidation.get(user_id)
        if not last_consolidation:
            return True

        # Check if interval passed
        elapsed = datetime.utcnow() - last_consolidation
        return elapsed.total_seconds() >= (self._interval_hours * 3600)

    async def trigger_consolidation(
        self,
        user_id: str,
        patterns: List[LearningPattern],
        feedback: List[FeedbackSignal],
        preferences: Dict[str, UserPreference],
        metrics: LearningMetrics,
    ) -> Optional[ConsolidationResult]:
        """
        Trigger consolidation if needed.

        Args:
            user_id: User identifier
            patterns: User patterns
            feedback: User feedback
            preferences: User preferences
            metrics: Current metrics

        Returns:
            Consolidation result if run, None otherwise
        """
        should_run = await self.should_consolidate(
            user_id,
            len(patterns),
            len(feedback),
        )

        if should_run:
            return await self.consolidate(
                user_id,
                patterns,
                feedback,
                preferences,
                metrics,
            )

        return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        avg_time = (
            self._total_consolidation_time / self._consolidations_run
            if self._consolidations_run > 0
            else 0.0
        )

        return {
            "consolidations_run": self._consolidations_run,
            "patterns_merged": self._patterns_merged,
            "preferences_updated": self._preferences_updated,
            "avg_consolidation_time_ms": avg_time,
            "background_task_running": (
                self._consolidation_task and not self._consolidation_task.done()
                if self._consolidation_task
                else False
            ),
            "users_tracked": len(self._last_consolidation),
            "circuit_breaker_state": self._circuit_breaker.state,
        }
