"""
Adaptation Module.

Applies real-time response adaptations based on learned patterns,
feedback, and preferences with A/B testing and rollback capabilities.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.learning.base import (
    AsyncCache,
    BaseLearningModule,
    CircuitBreaker,
    HealthStatus,
)
from morgan.learning.exceptions import AdaptationError
from morgan.learning.types import (
    AdaptationResult,
    AdaptationStrategy,
    FeedbackSignal,
    LearningContext,
    LearningPattern,
    UserPreference,
)
from morgan.learning.utils import generate_id


class AdaptationModule(BaseLearningModule):
    """
    Response adaptation module.

    Features:
    - Real-time response adaptation
    - Context-aware adjustments
    - A/B testing for adaptation strategies
    - Rollback capability for failed adaptations
    - Gradual adaptation rollout
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_ab_testing: bool = True,
        ab_test_ratio: float = 0.2,
        adaptation_threshold: float = 0.7,
        max_adaptations_per_user: int = 50,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize adaptation module.

        Args:
            enable_cache: Enable result caching
            enable_ab_testing: Enable A/B testing
            ab_test_ratio: Ratio for A/B test group
            adaptation_threshold: Confidence threshold for applying adaptations
            max_adaptations_per_user: Maximum adaptations to track per user
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__("AdaptationModule", correlation_id)

        self._adaptation_threshold = adaptation_threshold
        self._enable_ab_testing = enable_ab_testing
        self._ab_test_ratio = ab_test_ratio
        self._max_adaptations = max_adaptations_per_user

        # Adaptation storage
        self._user_adaptations: Dict[str, List[AdaptationResult]] = defaultdict(list)
        self._ab_tests: Dict[str, Dict[str, Any]] = {}  # Test ID -> test data

        # Rollback storage
        self._rollback_queue: List[AdaptationResult] = []

        # Cache and circuit breaker
        self._cache = AsyncCache(max_size=500) if enable_cache else None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=AdaptationError,
            name="adaptation",
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(10)
        self._lock = asyncio.Lock()

        # Metrics
        self._adaptations_applied = 0
        self._adaptations_successful = 0
        self._adaptations_rolled_back = 0
        self._ab_tests_run = 0

    async def initialize(self) -> None:
        """Initialize the adaptation module."""
        self._log_info("Adaptation module initialized")

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        async with self._lock:
            self._user_adaptations.clear()
            self._ab_tests.clear()
            self._rollback_queue.clear()

        if self._cache:
            await self._cache.clear()

        self._log_info("Adaptation module cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check module health."""
        try:
            total_adaptations = sum(len(a) for a in self._user_adaptations.values())
            success_rate = (
                self._adaptations_successful / self._adaptations_applied
                if self._adaptations_applied > 0
                else 0.0
            )

            return HealthStatus(
                healthy=True,
                message="Adaptation module healthy",
                details={
                    "adaptations_tracked": total_adaptations,
                    "users_tracked": len(self._user_adaptations),
                    "adaptations_applied": self._adaptations_applied,
                    "success_rate": success_rate,
                    "rollbacks_pending": len(self._rollback_queue),
                    "ab_tests_active": len(self._ab_tests),
                    "circuit_breaker_state": self._circuit_breaker.state,
                },
                last_check=asyncio.get_event_loop().time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Adaptation module unhealthy: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def adapt_response(
        self,
        user_id: str,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext] = None,
        strategy: AdaptationStrategy = AdaptationStrategy.CONTEXTUAL,
    ) -> Tuple[Dict[str, Any], Optional[AdaptationResult]]:
        """
        Adapt response based on learned data.

        Args:
            user_id: User identifier
            base_response: Base response to adapt
            patterns: Learned patterns
            preferences: User preferences
            context: Optional learning context
            strategy: Adaptation strategy

        Returns:
            Tuple of (adapted response, adaptation result)

        Raises:
            AdaptationError: If adaptation fails
        """
        await self.ensure_initialized()

        try:
            result = await self._circuit_breaker.call(
                self._adapt_response_internal,
                user_id,
                base_response,
                patterns,
                preferences,
                context,
                strategy,
            )

            return result

        except Exception as e:
            self._log_error("Response adaptation failed", e, user_id=user_id)
            raise AdaptationError(
                f"Failed to adapt response for user {user_id}",
                cause=e,
            )

    async def _adapt_response_internal(
        self,
        user_id: str,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
        strategy: AdaptationStrategy,
    ) -> Tuple[Dict[str, Any], Optional[AdaptationResult]]:
        """Internal adaptation implementation."""
        async with self._semaphore:
            # Check if user should be in A/B test
            if self._enable_ab_testing and random.random() < self._ab_test_ratio:
                return base_response, None

            # Apply adaptations based on strategy
            if strategy == AdaptationStrategy.IMMEDIATE:
                adapted, changes = await self._apply_immediate_adaptations(
                    base_response, patterns, preferences, context
                )
                confidence = 0.8
                expected_improvement = 0.6

            elif strategy == AdaptationStrategy.GRADUAL:
                adapted, changes = await self._apply_gradual_adaptations(
                    user_id, base_response, patterns, preferences, context
                )
                confidence = 0.7
                expected_improvement = 0.5

            elif strategy == AdaptationStrategy.CONTEXTUAL:
                adapted, changes = await self._apply_contextual_adaptations(
                    base_response, patterns, preferences, context
                )
                confidence = 0.75
                expected_improvement = 0.55

            elif strategy == AdaptationStrategy.EXPERIMENTAL:
                adapted, changes = await self._apply_experimental_adaptations(
                    user_id, base_response, patterns, preferences, context
                )
                confidence = 0.6
                expected_improvement = 0.4

            else:  # CONSERVATIVE
                adapted, changes = await self._apply_conservative_adaptations(
                    base_response, patterns, preferences, context
                )
                confidence = 0.85
                expected_improvement = 0.7

            # Create adaptation result
            if changes:
                result = AdaptationResult(
                    adaptation_id=generate_id("adaptation"),
                    strategy=strategy,
                    timestamp=datetime.utcnow(),
                    target="response",
                    changes=changes,
                    confidence=confidence,
                    expected_improvement=expected_improvement,
                    triggering_patterns=[p.pattern_id for p in patterns],
                    triggering_feedback=[],
                    rollback_data=base_response,
                    can_rollback=True,
                    rolled_back=False,
                )

                # Store adaptation
                await self._store_adaptation(user_id, result)

                self._adaptations_applied += 1

                return adapted, result
            else:
                return base_response, None

    async def _apply_immediate_adaptations(
        self,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply immediate adaptations."""
        adapted = dict(base_response)
        changes = {}

        # Apply preference-based adaptations
        for pref_dim, pref in preferences.items():
            if pref.confidence >= self._adaptation_threshold:
                key = pref_dim.value
                if pref.value != adapted.get(key):
                    changes[key] = {
                        "old": adapted.get(key),
                        "new": pref.value,
                        "reason": "user_preference",
                    }
                    adapted[key] = pref.value

        return adapted, changes

    async def _apply_gradual_adaptations(
        self,
        user_id: str,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply gradual adaptations over time."""
        adapted = dict(base_response)
        changes = {}

        # Get adaptation history
        history = await self._get_user_adaptations(user_id)

        # Only apply if we have established patterns
        if len(history) >= 5:
            # Calculate gradual adjustment factor
            factor = min(len(history) / 20.0, 1.0)

            for pref_dim, pref in preferences.items():
                if pref.confidence >= self._adaptation_threshold:
                    key = pref_dim.value
                    current = adapted.get(key)
                    target = pref.value

                    # Gradually move toward preference
                    if isinstance(target, (int, float)) and isinstance(
                        current, (int, float)
                    ):
                        new_value = current + (target - current) * factor
                        changes[key] = {
                            "old": current,
                            "new": new_value,
                            "reason": "gradual_adjustment",
                        }
                        adapted[key] = new_value

        return adapted, changes

    async def _apply_contextual_adaptations(
        self,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply context-aware adaptations."""
        adapted = dict(base_response)
        changes = {}

        # Filter preferences by context
        if context and context.tags:
            contextual_prefs = {
                dim: pref
                for dim, pref in preferences.items()
                if not pref.context_tags or pref.context_tags.intersection(context.tags)
            }
        else:
            contextual_prefs = preferences

        # Apply contextual preferences
        for pref_dim, pref in contextual_prefs.items():
            if pref.confidence >= self._adaptation_threshold:
                key = pref_dim.value
                if pref.value != adapted.get(key):
                    changes[key] = {
                        "old": adapted.get(key),
                        "new": pref.value,
                        "reason": "contextual_preference",
                        "context": list(context.tags) if context else [],
                    }
                    adapted[key] = pref.value

        # Apply pattern-based adaptations
        for pattern in patterns:
            if pattern.is_significant:
                # Adapt based on pattern type
                if pattern.pattern_type.value == "temporal":
                    time_category = pattern.metadata.get("time_category")
                    if time_category:
                        changes["temporal_context"] = {
                            "old": adapted.get("temporal_context"),
                            "new": time_category,
                            "reason": "temporal_pattern",
                        }
                        adapted["temporal_context"] = time_category

        return adapted, changes

    async def _apply_experimental_adaptations(
        self,
        user_id: str,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply experimental adaptations for A/B testing."""
        adapted = dict(base_response)
        changes = {}

        # Create experimental variation
        test_id = generate_id("ab_test")

        # Store test data
        self._ab_tests[test_id] = {
            "user_id": user_id,
            "created": datetime.utcnow(),
            "control": base_response,
            "preferences": preferences,
            "patterns": patterns,
        }

        # Apply one random preference as experiment
        if preferences:
            test_pref = random.choice(list(preferences.values()))
            key = test_pref.dimension.value
            changes[key] = {
                "old": adapted.get(key),
                "new": test_pref.value,
                "reason": "experimental",
                "test_id": test_id,
            }
            adapted[key] = test_pref.value

        self._ab_tests_run += 1

        return adapted, changes

    async def _apply_conservative_adaptations(
        self,
        base_response: Dict[str, Any],
        patterns: List[LearningPattern],
        preferences: Dict[str, UserPreference],
        context: Optional[LearningContext],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply only highly confident adaptations."""
        adapted = dict(base_response)
        changes = {}

        # Only apply very stable preferences
        high_confidence_prefs = {
            dim: pref
            for dim, pref in preferences.items()
            if pref.confidence >= 0.85 and pref.is_stable
        }

        for pref_dim, pref in high_confidence_prefs.items():
            key = pref_dim.value
            if pref.value != adapted.get(key):
                changes[key] = {
                    "old": adapted.get(key),
                    "new": pref.value,
                    "reason": "high_confidence_preference",
                }
                adapted[key] = pref.value

        return adapted, changes

    async def record_adaptation_result(
        self,
        adaptation_id: str,
        success: bool,
        actual_improvement: Optional[float] = None,
        feedback: Optional[FeedbackSignal] = None,
    ) -> None:
        """
        Record the result of an adaptation.

        Args:
            adaptation_id: Adaptation identifier
            success: Whether adaptation was successful
            actual_improvement: Measured improvement (0-1)
            feedback: Optional feedback signal
        """
        # Find adaptation
        adaptation = await self._find_adaptation(adaptation_id)

        if not adaptation:
            self._log_warning(f"Adaptation {adaptation_id} not found")
            return

        # Update adaptation result
        updated = AdaptationResult(
            adaptation_id=adaptation.adaptation_id,
            strategy=adaptation.strategy,
            timestamp=adaptation.timestamp,
            target=adaptation.target,
            changes=adaptation.changes,
            confidence=adaptation.confidence,
            expected_improvement=adaptation.expected_improvement,
            actual_improvement=actual_improvement,
            triggering_patterns=adaptation.triggering_patterns,
            triggering_feedback=(
                adaptation.triggering_feedback + [feedback.feedback_id]
                if feedback
                else adaptation.triggering_feedback
            ),
            rollback_data=adaptation.rollback_data,
            can_rollback=adaptation.can_rollback,
            rolled_back=adaptation.rolled_back,
        )

        # Update storage
        await self._update_adaptation(updated)

        if success:
            self._adaptations_successful += 1
        elif updated.needs_rollback:
            await self._queue_rollback(updated)

    async def _find_adaptation(
        self, adaptation_id: str
    ) -> Optional[AdaptationResult]:
        """Find adaptation by ID."""
        async with self._lock:
            for adaptations in self._user_adaptations.values():
                for adaptation in adaptations:
                    if adaptation.adaptation_id == adaptation_id:
                        return adaptation
        return None

    async def _update_adaptation(self, adaptation: AdaptationResult) -> None:
        """Update stored adaptation."""
        async with self._lock:
            for user_id, adaptations in self._user_adaptations.items():
                for i, existing in enumerate(adaptations):
                    if existing.adaptation_id == adaptation.adaptation_id:
                        self._user_adaptations[user_id][i] = adaptation
                        return

    async def _queue_rollback(self, adaptation: AdaptationResult) -> None:
        """Queue adaptation for rollback."""
        async with self._lock:
            self._rollback_queue.append(adaptation)

    async def process_rollbacks(self) -> int:
        """
        Process queued rollbacks.

        Returns:
            Number of rollbacks processed
        """
        async with self._lock:
            rollbacks = list(self._rollback_queue)
            self._rollback_queue.clear()

        count = 0
        for adaptation in rollbacks:
            try:
                # Mark as rolled back
                updated = AdaptationResult(
                    adaptation_id=adaptation.adaptation_id,
                    strategy=adaptation.strategy,
                    timestamp=adaptation.timestamp,
                    target=adaptation.target,
                    changes=adaptation.changes,
                    confidence=adaptation.confidence,
                    expected_improvement=adaptation.expected_improvement,
                    actual_improvement=adaptation.actual_improvement,
                    triggering_patterns=adaptation.triggering_patterns,
                    triggering_feedback=adaptation.triggering_feedback,
                    rollback_data=adaptation.rollback_data,
                    can_rollback=False,
                    rolled_back=True,
                )

                await self._update_adaptation(updated)
                count += 1
                self._adaptations_rolled_back += 1

            except Exception as e:
                self._log_error(f"Rollback failed for {adaptation.adaptation_id}", e)

        return count

    async def _store_adaptation(
        self, user_id: str, adaptation: AdaptationResult
    ) -> None:
        """Store adaptation result."""
        async with self._lock:
            self._user_adaptations[user_id].append(adaptation)

            # Limit adaptations per user
            if len(self._user_adaptations[user_id]) > self._max_adaptations:
                self._user_adaptations[user_id] = self._user_adaptations[user_id][
                    -self._max_adaptations :
                ]

    async def _get_user_adaptations(self, user_id: str) -> List[AdaptationResult]:
        """Get adaptations for a user."""
        async with self._lock:
            return list(self._user_adaptations.get(user_id, []))

    async def get_adaptation_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get adaptation statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Statistics dictionary
        """
        adaptations = await self._get_user_adaptations(user_id)

        if not adaptations:
            return {
                "total": 0,
                "successful": 0,
                "rolled_back": 0,
                "success_rate": 0.0,
            }

        successful = sum(1 for a in adaptations if a.was_successful)
        rolled_back = sum(1 for a in adaptations if a.rolled_back)

        return {
            "total": len(adaptations),
            "successful": successful,
            "rolled_back": rolled_back,
            "success_rate": successful / len(adaptations),
            "avg_improvement": sum(
                a.actual_improvement for a in adaptations if a.actual_improvement
            )
            / len(adaptations),
        }

    async def clear_user_adaptations(self, user_id: str) -> None:
        """Clear all adaptations for a user."""
        async with self._lock:
            self._user_adaptations.pop(user_id, None)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "adaptations_applied": self._adaptations_applied,
            "adaptations_successful": self._adaptations_successful,
            "adaptations_rolled_back": self._adaptations_rolled_back,
            "ab_tests_run": self._ab_tests_run,
            "rollbacks_pending": len(self._rollback_queue),
            "users_tracked": len(self._user_adaptations),
            "circuit_breaker_state": self._circuit_breaker.state,
            "cache_stats": self._cache.stats if self._cache else None,
        }
