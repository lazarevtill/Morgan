"""
Preference Learning Module.

Learns and manages user preferences across multiple dimensions
with conflict resolution and context-aware preference application.
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
from morgan.learning.exceptions import PreferenceLearningError
from morgan.learning.types import (
    FeedbackSignal,
    LearningContext,
    PreferenceDimension,
    UserPreference,
)
from morgan.learning.utils import (
    generate_id,
    resolve_preference_conflicts,
)


class PreferenceModule(BaseLearningModule):
    """
    Preference learning and management module.

    Features:
    - Multi-dimensional preference learning
    - Conflict resolution for competing preferences
    - Context-aware preference application
    - Preference evolution tracking
    - Preference transfer across contexts
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_supporting_signals: int = 3,
        enable_cache: bool = True,
        preference_ttl_days: int = 90,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize preference module.

        Args:
            min_confidence: Minimum confidence for preference application
            min_supporting_signals: Minimum signals to establish preference
            enable_cache: Enable result caching
            preference_ttl_days: Days before preference expires
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__("PreferenceModule", correlation_id)

        self._min_confidence = min_confidence
        self._min_supporting_signals = min_supporting_signals
        self._ttl_days = preference_ttl_days

        # Preference storage
        self._user_preferences: Dict[str, Dict[PreferenceDimension, UserPreference]] = (
            defaultdict(dict)
        )
        self._preference_signals: Dict[str, List[FeedbackSignal]] = defaultdict(list)

        # Cache and circuit breaker
        self._cache = AsyncCache(max_size=500) if enable_cache else None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=PreferenceLearningError,
            name="preference_learning",
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(10)
        self._lock = asyncio.Lock()

        # Metrics
        self._preferences_learned = 0
        self._preferences_updated = 0
        self._conflicts_resolved = 0

    async def initialize(self) -> None:
        """Initialize the preference module."""
        self._log_info("Preference module initialized")

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        async with self._lock:
            self._user_preferences.clear()
            self._preference_signals.clear()

        if self._cache:
            await self._cache.clear()

        self._log_info("Preference module cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check module health."""
        try:
            total_prefs = sum(len(p) for p in self._user_preferences.values())
            stable_prefs = sum(
                1
                for user_prefs in self._user_preferences.values()
                for pref in user_prefs.values()
                if pref.is_stable
            )

            return HealthStatus(
                healthy=True,
                message="Preference module healthy",
                details={
                    "preferences_tracked": total_prefs,
                    "stable_preferences": stable_prefs,
                    "users_tracked": len(self._user_preferences),
                    "preferences_learned": self._preferences_learned,
                    "conflicts_resolved": self._conflicts_resolved,
                    "circuit_breaker_state": self._circuit_breaker.state,
                },
                last_check=asyncio.get_event_loop().time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Preference module unhealthy: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def learn_preference(
        self,
        user_id: str,
        dimension: PreferenceDimension,
        value: Any,
        signal: Optional[FeedbackSignal] = None,
        confidence: Optional[float] = None,
        context: Optional[LearningContext] = None,
    ) -> UserPreference:
        """
        Learn or update a user preference.

        Args:
            user_id: User identifier
            dimension: Preference dimension
            value: Preference value
            signal: Optional supporting feedback signal
            confidence: Optional confidence override
            context: Optional learning context

        Returns:
            Updated preference

        Raises:
            PreferenceLearningError: If learning fails
        """
        await self.ensure_initialized()

        try:
            preference = await self._circuit_breaker.call(
                self._learn_preference_internal,
                user_id,
                dimension,
                value,
                signal,
                confidence,
                context,
            )

            # Invalidate cache
            if self._cache:
                await self._cache.invalidate(f"preferences:{user_id}")

            return preference

        except Exception as e:
            self._log_error("Preference learning failed", e, user_id=user_id)
            raise PreferenceLearningError(
                f"Failed to learn preference for user {user_id}",
                dimension=dimension.value,
                cause=e,
            )

    async def _learn_preference_internal(
        self,
        user_id: str,
        dimension: PreferenceDimension,
        value: Any,
        signal: Optional[FeedbackSignal],
        confidence: Optional[float],
        context: Optional[LearningContext],
    ) -> UserPreference:
        """Internal preference learning implementation."""
        async with self._semaphore:
            # Get existing preference if any
            existing = await self._get_preference(user_id, dimension)

            if existing:
                # Update existing preference
                preference = await self._update_preference(
                    existing,
                    value,
                    signal,
                    confidence,
                    context,
                )
                self._preferences_updated += 1
            else:
                # Create new preference
                preference = await self._create_preference(
                    user_id,
                    dimension,
                    value,
                    signal,
                    confidence,
                    context,
                )
                self._preferences_learned += 1

            # Store preference
            await self._store_preference(preference)

            return preference

    async def _create_preference(
        self,
        user_id: str,
        dimension: PreferenceDimension,
        value: Any,
        signal: Optional[FeedbackSignal],
        confidence: Optional[float],
        context: Optional[LearningContext],
    ) -> UserPreference:
        """Create a new preference."""
        supporting_signals = []
        if signal:
            supporting_signals.append(signal.feedback_id)

        # Calculate confidence
        if confidence is None:
            confidence = min(len(supporting_signals) / self._min_supporting_signals, 1.0)

        # Extract context tags
        context_tags = context.tags if context else set()

        preference = UserPreference(
            preference_id=generate_id("preference"),
            user_id=user_id,
            dimension=dimension,
            value=value,
            confidence=max(confidence, 0.3),  # Minimum initial confidence
            first_learned=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            update_count=1,
            supporting_signals=supporting_signals,
            conflicting_signals=[],
            context_tags=context_tags,
            metadata={},
        )

        return preference

    async def _update_preference(
        self,
        existing: UserPreference,
        new_value: Any,
        signal: Optional[FeedbackSignal],
        confidence: Optional[float],
        context: Optional[LearningContext],
    ) -> UserPreference:
        """Update an existing preference."""
        # Check if value conflicts
        if new_value != existing.value:
            # Value changed - could be conflict or evolution
            conflicting_signals = list(existing.conflicting_signals)
            if signal:
                conflicting_signals.append(signal.feedback_id)

            # If strong evidence for new value, update
            new_confidence = confidence or 0.5
            if new_confidence > existing.confidence or len(conflicting_signals) > len(
                existing.supporting_signals
            ):
                # Switch to new value
                supporting_signals = conflicting_signals
                conflicting_signals = list(existing.supporting_signals)
                value = new_value
                confidence_value = new_confidence
                self._conflicts_resolved += 1
            else:
                # Keep existing value, add to conflicts
                supporting_signals = list(existing.supporting_signals)
                value = existing.value
                confidence_value = existing.confidence * 0.9  # Reduce confidence
        else:
            # Value matches - reinforce preference
            supporting_signals = list(existing.supporting_signals)
            if signal:
                supporting_signals.append(signal.feedback_id)

            conflicting_signals = list(existing.conflicting_signals)
            value = existing.value

            # Increase confidence
            evidence_ratio = len(supporting_signals) / (
                len(supporting_signals) + len(conflicting_signals)
            )
            confidence_value = min(existing.confidence * 1.1, evidence_ratio)

        # Merge context tags
        context_tags = set(existing.context_tags)
        if context:
            context_tags.update(context.tags)

        # Create updated preference (immutable)
        updated = UserPreference(
            preference_id=existing.preference_id,
            user_id=existing.user_id,
            dimension=existing.dimension,
            value=value,
            confidence=confidence_value,
            first_learned=existing.first_learned,
            last_updated=datetime.utcnow(),
            update_count=existing.update_count + 1,
            supporting_signals=supporting_signals,
            conflicting_signals=conflicting_signals,
            context_tags=context_tags,
            metadata=existing.metadata,
        )

        return updated

    async def _get_preference(
        self,
        user_id: str,
        dimension: PreferenceDimension,
    ) -> Optional[UserPreference]:
        """Get existing preference."""
        async with self._lock:
            return self._user_preferences.get(user_id, {}).get(dimension)

    async def _store_preference(self, preference: UserPreference) -> None:
        """Store preference."""
        async with self._lock:
            if preference.user_id not in self._user_preferences:
                self._user_preferences[preference.user_id] = {}

            self._user_preferences[preference.user_id][
                preference.dimension
            ] = preference

    async def get_user_preferences(
        self,
        user_id: str,
        context: Optional[LearningContext] = None,
        min_confidence: Optional[float] = None,
        use_cache: bool = True,
    ) -> Dict[PreferenceDimension, UserPreference]:
        """
        Get all preferences for a user.

        Args:
            user_id: User identifier
            context: Optional context for filtering
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use cache

        Returns:
            Dictionary of preferences by dimension
        """
        # Check cache
        if use_cache and self._cache:
            cache_key = f"preferences:{user_id}"
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        async with self._lock:
            preferences = dict(self._user_preferences.get(user_id, {}))

        # Filter by confidence
        threshold = min_confidence or self._min_confidence
        preferences = {
            dim: pref
            for dim, pref in preferences.items()
            if pref.confidence >= threshold
        }

        # Filter by context if provided
        if context and context.tags:
            preferences = {
                dim: pref
                for dim, pref in preferences.items()
                if not pref.context_tags or pref.context_tags.intersection(context.tags)
            }

        # Remove expired preferences
        cutoff = datetime.utcnow() - timedelta(days=self._ttl_days)
        preferences = {
            dim: pref
            for dim, pref in preferences.items()
            if pref.last_updated > cutoff
        }

        # Cache result
        if self._cache:
            cache_key = f"preferences:{user_id}"
            await self._cache.set(cache_key, preferences, ttl=300)

        return preferences

    async def get_preference_value(
        self,
        user_id: str,
        dimension: PreferenceDimension,
        default: Optional[Any] = None,
        context: Optional[LearningContext] = None,
    ) -> Optional[Any]:
        """
        Get preference value for a dimension.

        Args:
            user_id: User identifier
            dimension: Preference dimension
            default: Default value if not found
            context: Optional context

        Returns:
            Preference value or default
        """
        preferences = await self.get_user_preferences(user_id, context=context)
        preference = preferences.get(dimension)

        if preference:
            return preference.value
        return default

    async def apply_preferences(
        self,
        user_id: str,
        base_config: Dict[str, Any],
        context: Optional[LearningContext] = None,
    ) -> Dict[str, Any]:
        """
        Apply user preferences to a base configuration.

        Args:
            user_id: User identifier
            base_config: Base configuration dictionary
            context: Optional context

        Returns:
            Configuration with preferences applied
        """
        preferences = await self.get_user_preferences(user_id, context=context)

        config = dict(base_config)

        for dimension, preference in preferences.items():
            # Map dimension to config key
            config_key = self._dimension_to_config_key(dimension)
            if config_key:
                config[config_key] = preference.value

        return config

    def _dimension_to_config_key(self, dimension: PreferenceDimension) -> Optional[str]:
        """Map preference dimension to config key."""
        mapping = {
            PreferenceDimension.COMMUNICATION_STYLE: "communication_style",
            PreferenceDimension.DETAIL_LEVEL: "detail_level",
            PreferenceDimension.RESPONSE_LENGTH: "response_length",
            PreferenceDimension.TECHNICAL_DEPTH: "technical_depth",
            PreferenceDimension.EXAMPLES: "include_examples",
            PreferenceDimension.EXPLANATIONS: "include_explanations",
            PreferenceDimension.TONE: "tone",
            PreferenceDimension.FORMATTING: "formatting_style",
        }
        return mapping.get(dimension)

    async def infer_preference_from_feedback(
        self,
        user_id: str,
        feedback: FeedbackSignal,
        context: Optional[LearningContext] = None,
    ) -> List[UserPreference]:
        """
        Infer preferences from feedback signal.

        Args:
            user_id: User identifier
            feedback: Feedback signal
            context: Optional context

        Returns:
            List of inferred preferences
        """
        inferred_preferences = []

        # Analyze feedback for preference signals
        if feedback.text:
            # Simple keyword-based inference
            text_lower = feedback.text.lower()

            # Detail level preferences
            if "more detail" in text_lower or "elaborate" in text_lower:
                pref = await self.learn_preference(
                    user_id,
                    PreferenceDimension.DETAIL_LEVEL,
                    "detailed",
                    signal=feedback,
                    confidence=0.7,
                    context=context,
                )
                inferred_preferences.append(pref)
            elif "brief" in text_lower or "shorter" in text_lower:
                pref = await self.learn_preference(
                    user_id,
                    PreferenceDimension.DETAIL_LEVEL,
                    "brief",
                    signal=feedback,
                    confidence=0.7,
                    context=context,
                )
                inferred_preferences.append(pref)

            # Example preferences
            if "example" in text_lower:
                pref = await self.learn_preference(
                    user_id,
                    PreferenceDimension.EXAMPLES,
                    True,
                    signal=feedback,
                    confidence=0.7,
                    context=context,
                )
                inferred_preferences.append(pref)

            # Technical depth
            if "simpler" in text_lower or "eli5" in text_lower:
                pref = await self.learn_preference(
                    user_id,
                    PreferenceDimension.TECHNICAL_DEPTH,
                    "simple",
                    signal=feedback,
                    confidence=0.7,
                    context=context,
                )
                inferred_preferences.append(pref)
            elif "technical" in text_lower or "details" in text_lower:
                pref = await self.learn_preference(
                    user_id,
                    PreferenceDimension.TECHNICAL_DEPTH,
                    "technical",
                    signal=feedback,
                    confidence=0.7,
                    context=context,
                )
                inferred_preferences.append(pref)

        return inferred_preferences

    async def clear_user_preferences(self, user_id: str) -> None:
        """Clear all preferences for a user."""
        async with self._lock:
            self._user_preferences.pop(user_id, None)
            self._preference_signals.pop(user_id, None)

        if self._cache:
            await self._cache.invalidate(f"preferences:{user_id}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        total_prefs = sum(len(p) for p in self._user_preferences.values())
        stable_prefs = sum(
            1
            for user_prefs in self._user_preferences.values()
            for pref in user_prefs.values()
            if pref.is_stable
        )

        return {
            "preferences_learned": self._preferences_learned,
            "preferences_updated": self._preferences_updated,
            "conflicts_resolved": self._conflicts_resolved,
            "total_preferences": total_prefs,
            "stable_preferences": stable_prefs,
            "users_tracked": len(self._user_preferences),
            "circuit_breaker_state": self._circuit_breaker.state,
            "cache_stats": self._cache.stats if self._cache else None,
        }
