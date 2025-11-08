"""
Learning Engine.

Main orchestrator for the learning system that coordinates all learning modules
to provide comprehensive user learning and adaptation capabilities.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from morgan.learning.base import BaseLearningModule, HealthStatus
from morgan.learning.exceptions import LearningError, LearningResourceError
from morgan.learning.modules import (
    AdaptationModule,
    ConsolidationModule,
    FeedbackModule,
    PatternModule,
    PreferenceModule,
)
from morgan.learning.types import (
    AdaptationResult,
    AdaptationStrategy,
    ConsolidationResult,
    FeedbackSignal,
    FeedbackType,
    LearningContext,
    LearningMetrics,
    LearningPattern,
    PreferenceDimension,
    UserPreference,
)
from morgan.learning.utils import (
    format_metrics_summary,
    generate_correlation_id,
)


class LearningEngine(BaseLearningModule):
    """
    Main learning engine that orchestrates all learning modules.

    Coordinates:
    1. PatternModule - Behavioral pattern detection
    2. FeedbackModule - Feedback processing and sentiment analysis
    3. PreferenceModule - Preference learning and management
    4. AdaptationModule - Response adaptation
    5. ConsolidationModule - Knowledge consolidation

    Provides unified interface for:
    - Learning from user interactions
    - Detecting patterns and preferences
    - Adapting responses
    - Consolidating knowledge
    - Monitoring learning effectiveness
    """

    def __init__(
        self,
        enable_pattern_detection: bool = True,
        enable_feedback_processing: bool = True,
        enable_preference_learning: bool = True,
        enable_adaptation: bool = True,
        enable_consolidation: bool = True,
        consolidation_interval_hours: int = 24,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize learning engine.

        Args:
            enable_pattern_detection: Enable pattern detection module
            enable_feedback_processing: Enable feedback processing module
            enable_preference_learning: Enable preference learning module
            enable_adaptation: Enable adaptation module
            enable_consolidation: Enable consolidation module
            consolidation_interval_hours: Hours between consolidations
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__(
            "LearningEngine",
            correlation_id or generate_correlation_id(),
        )

        # Initialize modules
        self._pattern_module = (
            PatternModule(correlation_id=self.correlation_id)
            if enable_pattern_detection
            else None
        )
        self._feedback_module = (
            FeedbackModule(correlation_id=self.correlation_id)
            if enable_feedback_processing
            else None
        )
        self._preference_module = (
            PreferenceModule(correlation_id=self.correlation_id)
            if enable_preference_learning
            else None
        )
        self._adaptation_module = (
            AdaptationModule(correlation_id=self.correlation_id)
            if enable_adaptation
            else None
        )
        self._consolidation_module = (
            ConsolidationModule(
                consolidation_interval_hours=consolidation_interval_hours,
                correlation_id=self.correlation_id,
            )
            if enable_consolidation
            else None
        )

        # Metrics tracking
        self._metrics: Dict[str, LearningMetrics] = {}

        # Concurrency control
        self._semaphore = asyncio.Semaphore(5)

    async def initialize(self) -> None:
        """Initialize all learning modules."""
        modules = self._get_active_modules()

        try:
            # Initialize all modules in parallel
            await asyncio.gather(*[module.initialize() for module in modules])
            self._log_info("All learning modules initialized successfully")

        except Exception as e:
            self._log_error("Failed to initialize learning modules", e)
            raise LearningResourceError(
                f"Failed to initialize learning engine: {str(e)}",
                resource_type="modules",
            )

    async def cleanup(self) -> None:
        """Cleanup all learning modules."""
        modules = self._get_active_modules()
        await asyncio.gather(*[module.cleanup() for module in modules])
        self._log_info("All learning modules cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check health of all modules."""
        modules = self._get_active_modules()

        try:
            # Check all modules in parallel
            health_results = await asyncio.gather(
                *[module.health_check() for module in modules],
                return_exceptions=True,
            )

            # Aggregate health status
            all_healthy = all(
                isinstance(result, HealthStatus) and result.healthy
                for result in health_results
            )

            module_statuses = {}
            for module, result in zip(modules, health_results):
                if isinstance(result, HealthStatus):
                    module_statuses[module.name] = {
                        "healthy": result.healthy,
                        "message": result.message,
                    }
                else:
                    module_statuses[module.name] = {
                        "healthy": False,
                        "message": str(result),
                    }

            return HealthStatus(
                healthy=all_healthy,
                message=(
                    "Learning engine healthy"
                    if all_healthy
                    else "Some modules unhealthy"
                ),
                details={
                    "modules": module_statuses,
                    "users_tracked": len(self._metrics),
                },
                last_check=asyncio.get_event_loop().time(),
            )

        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Learning engine health check failed: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def learn_from_interaction(
        self,
        user_id: str,
        action: str,
        context: Optional[LearningContext] = None,
        feedback: Optional[FeedbackSignal] = None,
    ) -> None:
        """
        Learn from a user interaction.

        Args:
            user_id: User identifier
            action: Action performed
            context: Optional learning context
            feedback: Optional feedback signal
        """
        await self.ensure_initialized()

        async with self._semaphore:
            # Add event to pattern module
            if self._pattern_module:
                await self._pattern_module.add_event(
                    user_id=user_id,
                    action=action,
                    context_tags=context.tags if context else None,
                )

            # Process feedback if provided
            if feedback and self._feedback_module:
                await self._feedback_module._store_feedback(feedback)

                # Infer preferences from feedback
                if self._preference_module:
                    await self._preference_module.infer_preference_from_feedback(
                        user_id=user_id,
                        feedback=feedback,
                        context=context,
                    )

            # Update metrics
            await self._update_metrics(user_id)

    async def process_feedback(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        message_id: Optional[str] = None,
        rating: Optional[float] = None,
        text: Optional[str] = None,
        correction: Optional[str] = None,
        context: Optional[LearningContext] = None,
    ) -> FeedbackSignal:
        """
        Process user feedback.

        Args:
            user_id: User identifier
            feedback_type: Type of feedback
            message_id: Optional message ID
            rating: Optional numeric rating
            text: Optional text feedback
            correction: Optional correction
            context: Optional learning context

        Returns:
            Processed feedback signal

        Raises:
            LearningError: If feedback processing fails
        """
        await self.ensure_initialized()

        if not self._feedback_module:
            raise LearningError("Feedback module not enabled")

        feedback = await self._feedback_module.process_feedback(
            user_id=user_id,
            feedback_type=feedback_type,
            message_id=message_id,
            rating=rating,
            text=text,
            correction=correction,
            context=context,
        )

        # Infer preferences from feedback
        if self._preference_module:
            await self._preference_module.infer_preference_from_feedback(
                user_id=user_id,
                feedback=feedback,
                context=context,
            )

        return feedback

    async def detect_patterns(
        self,
        user_id: str,
        context: Optional[LearningContext] = None,
    ) -> List[LearningPattern]:
        """
        Detect patterns for a user.

        Args:
            user_id: User identifier
            context: Optional learning context

        Returns:
            List of detected patterns

        Raises:
            LearningError: If pattern detection fails
        """
        await self.ensure_initialized()

        if not self._pattern_module:
            raise LearningError("Pattern module not enabled")

        return await self._pattern_module.detect_patterns(
            user_id=user_id,
            context=context,
        )

    async def get_user_preferences(
        self,
        user_id: str,
        context: Optional[LearningContext] = None,
    ) -> Dict[PreferenceDimension, UserPreference]:
        """
        Get user preferences.

        Args:
            user_id: User identifier
            context: Optional learning context

        Returns:
            Dictionary of preferences by dimension

        Raises:
            LearningError: If preference retrieval fails
        """
        await self.ensure_initialized()

        if not self._preference_module:
            raise LearningError("Preference module not enabled")

        return await self._preference_module.get_user_preferences(
            user_id=user_id,
            context=context,
        )

    async def adapt_response(
        self,
        user_id: str,
        base_response: Dict[str, Any],
        context: Optional[LearningContext] = None,
        strategy: AdaptationStrategy = AdaptationStrategy.CONTEXTUAL,
    ) -> Tuple[Dict[str, Any], Optional[AdaptationResult]]:
        """
        Adapt a response based on learned data.

        Args:
            user_id: User identifier
            base_response: Base response to adapt
            context: Optional learning context
            strategy: Adaptation strategy

        Returns:
            Tuple of (adapted response, adaptation result)

        Raises:
            LearningError: If adaptation fails
        """
        await self.ensure_initialized()

        if not self._adaptation_module:
            return base_response, None

        # Get patterns and preferences
        patterns_task = (
            self._pattern_module.detect_patterns(user_id, context)
            if self._pattern_module
            else asyncio.sleep(0, [])
        )
        preferences_task = (
            self._preference_module.get_user_preferences(user_id, context)
            if self._preference_module
            else asyncio.sleep(0, {})
        )

        patterns, preferences = await asyncio.gather(
            patterns_task,
            preferences_task,
        )

        # Adapt response
        adapted_response, adaptation_result = await self._adaptation_module.adapt_response(
            user_id=user_id,
            base_response=base_response,
            patterns=patterns,
            preferences=preferences,
            context=context,
            strategy=strategy,
        )

        return adapted_response, adaptation_result

    async def consolidate_knowledge(
        self,
        user_id: str,
    ) -> Optional[ConsolidationResult]:
        """
        Trigger knowledge consolidation for a user.

        Args:
            user_id: User identifier

        Returns:
            Consolidation result if run, None if skipped

        Raises:
            LearningError: If consolidation fails
        """
        await self.ensure_initialized()

        if not self._consolidation_module:
            raise LearningError("Consolidation module not enabled")

        # Gather all data for consolidation
        patterns = (
            await self._pattern_module.get_active_patterns(user_id)
            if self._pattern_module
            else []
        )
        feedback = (
            await self._feedback_module.get_user_feedback(user_id)
            if self._feedback_module
            else []
        )
        preferences = (
            await self._preference_module.get_user_preferences(user_id)
            if self._preference_module
            else {}
        )
        metrics = self._get_user_metrics(user_id)

        # Trigger consolidation if needed
        result = await self._consolidation_module.trigger_consolidation(
            user_id=user_id,
            patterns=patterns,
            feedback=feedback,
            preferences=preferences,
            metrics=metrics,
        )

        return result

    async def get_learning_summary(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive learning summary for a user.

        Args:
            user_id: User identifier

        Returns:
            Learning summary dictionary
        """
        await self.ensure_initialized()

        summary = {
            "user_id": user_id,
            "metrics": self._get_user_metrics(user_id).__dict__,
        }

        # Get patterns
        if self._pattern_module:
            patterns = await self._pattern_module.get_active_patterns(user_id)
            summary["patterns"] = {
                "count": len(patterns),
                "significant": sum(1 for p in patterns if p.is_significant),
            }

        # Get feedback summary
        if self._feedback_module:
            feedback_summary = await self._feedback_module.get_feedback_summary(
                user_id
            )
            summary["feedback"] = feedback_summary

        # Get preferences
        if self._preference_module:
            preferences = await self._preference_module.get_user_preferences(user_id)
            summary["preferences"] = {
                "count": len(preferences),
                "stable": sum(1 for p in preferences.values() if p.is_stable),
            }

        # Get adaptation stats
        if self._adaptation_module:
            adaptation_stats = await self._adaptation_module.get_adaptation_stats(
                user_id
            )
            summary["adaptations"] = adaptation_stats

        return summary

    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all learning data for a user.

        Args:
            user_id: User identifier
        """
        tasks = []

        if self._pattern_module:
            tasks.append(self._pattern_module.clear_user_patterns(user_id))

        if self._feedback_module:
            tasks.append(self._feedback_module.clear_user_feedback(user_id))

        if self._preference_module:
            tasks.append(self._preference_module.clear_user_preferences(user_id))

        if self._adaptation_module:
            tasks.append(self._adaptation_module.clear_user_adaptations(user_id))

        if tasks:
            await asyncio.gather(*tasks)

        # Clear metrics
        self._metrics.pop(user_id, None)

    def _get_active_modules(self) -> List[BaseLearningModule]:
        """Get list of active modules."""
        modules = []

        if self._pattern_module:
            modules.append(self._pattern_module)
        if self._feedback_module:
            modules.append(self._feedback_module)
        if self._preference_module:
            modules.append(self._preference_module)
        if self._adaptation_module:
            modules.append(self._adaptation_module)
        if self._consolidation_module:
            modules.append(self._consolidation_module)

        return modules

    def _get_user_metrics(self, user_id: str) -> LearningMetrics:
        """Get metrics for a user."""
        if user_id not in self._metrics:
            self._metrics[user_id] = LearningMetrics()

        return self._metrics[user_id]

    async def _update_metrics(self, user_id: str) -> None:
        """Update metrics for a user."""
        metrics = self._get_user_metrics(user_id)

        # Update from modules
        if self._pattern_module:
            patterns = await self._pattern_module.get_active_patterns(user_id)
            metrics.patterns_detected = len(patterns)
            metrics.patterns_active = sum(1 for p in patterns if p.is_significant)
            if patterns:
                metrics.avg_pattern_confidence = sum(p.confidence for p in patterns) / len(
                    patterns
                )

        if self._feedback_module:
            feedback = await self._feedback_module.get_user_feedback(user_id)
            metrics.feedback_signals = len(feedback)
            if feedback:
                metrics.positive_feedback_ratio = sum(
                    1 for f in feedback if f.is_positive
                ) / len(feedback)
                metrics.feedback_actionability = sum(
                    1 for f in feedback if f.is_actionable
                ) / len(feedback)

        if self._preference_module:
            preferences = await self._preference_module.get_user_preferences(user_id)
            metrics.preferences_learned = len(preferences)
            metrics.preferences_stable = sum(
                1 for p in preferences.values() if p.is_stable
            )
            if preferences:
                metrics.avg_preference_confidence = sum(
                    p.confidence for p in preferences.values()
                ) / len(preferences)

        if self._adaptation_module:
            adaptation_stats = await self._adaptation_module.get_adaptation_stats(
                user_id
            )
            metrics.adaptations_applied = adaptation_stats.get("total", 0)
            metrics.adaptations_successful = adaptation_stats.get("successful", 0)
            metrics.adaptations_rolled_back = adaptation_stats.get("rolled_back", 0)
            if metrics.adaptations_applied > 0:
                metrics.avg_adaptation_improvement = adaptation_stats.get(
                    "avg_improvement", 0.0
                )

        self._metrics[user_id] = metrics

    @property
    def stats(self) -> Dict[str, Any]:
        """Get learning engine statistics."""
        module_stats = {}

        if self._pattern_module:
            module_stats["pattern_module"] = self._pattern_module.stats

        if self._feedback_module:
            module_stats["feedback_module"] = self._feedback_module.stats

        if self._preference_module:
            module_stats["preference_module"] = self._preference_module.stats

        if self._adaptation_module:
            module_stats["adaptation_module"] = self._adaptation_module.stats

        if self._consolidation_module:
            module_stats["consolidation_module"] = self._consolidation_module.stats

        return {
            "users_tracked": len(self._metrics),
            "modules": module_stats,
        }

    def get_metrics_summary(self, user_id: str) -> str:
        """
        Get formatted metrics summary for a user.

        Args:
            user_id: User identifier

        Returns:
            Formatted summary string
        """
        metrics = self._get_user_metrics(user_id)
        return format_metrics_summary(metrics)
