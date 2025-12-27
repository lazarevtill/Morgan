"""
Context Monitor for Proactive Assistance.

Monitors user context and activity to enable proactive suggestions:
- Tracks conversation patterns
- Monitors time-based patterns
- Detects context changes
- Triggers proactive actions
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ContextEventType(str, Enum):
    """Types of context events."""

    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    TOPIC_CHANGE = "topic_change"
    EMOTIONAL_SHIFT = "emotional_shift"
    TIME_TRIGGER = "time_trigger"
    PATTERN_DETECTED = "pattern_detected"
    GOAL_PROGRESS = "goal_progress"
    INACTIVITY = "inactivity"
    RETURN_VISIT = "return_visit"


@dataclass
class ContextEvent:
    """A monitored context event."""

    event_id: str
    event_type: ContextEventType
    timestamp: datetime
    user_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class UserContext:
    """Current context for a user."""

    user_id: str
    current_topic: Optional[str] = None
    emotional_state: Optional[str] = None
    active_goals: List[str] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    last_activity: Optional[datetime] = None
    session_start: Optional[datetime] = None
    interaction_count: int = 0
    patterns: Dict[str, Any] = field(default_factory=dict)


class ContextMonitor:
    """
    Monitors user context for proactive assistance.

    Tracks:
    - Conversation flow and topic changes
    - Emotional state transitions
    - Time patterns (daily routines, etc.)
    - Goal progress
    - User behavior patterns

    Example:
        >>> monitor = ContextMonitor()
        >>>
        >>> # Register event handlers
        >>> monitor.on_event(ContextEventType.INACTIVITY, handle_inactivity)
        >>>
        >>> # Update context
        >>> await monitor.update_context(
        ...     user_id="user123",
        ...     topic="machine learning",
        ...     emotional_state="curious"
        ... )
        >>>
        >>> # Start monitoring
        >>> await monitor.start()
    """

    def __init__(
        self,
        inactivity_threshold_minutes: int = 30,
        pattern_detection_enabled: bool = True,
    ):
        """
        Initialize context monitor.

        Args:
            inactivity_threshold_minutes: Minutes before inactivity event
            pattern_detection_enabled: Enable pattern detection
        """
        self.settings = get_settings()
        self.inactivity_threshold = timedelta(minutes=inactivity_threshold_minutes)
        self.pattern_detection_enabled = pattern_detection_enabled

        # User contexts
        self._contexts: Dict[str, UserContext] = {}

        # Event handlers
        self._handlers: Dict[ContextEventType, List[Callable]] = {
            event_type: [] for event_type in ContextEventType
        }

        # Event queue
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._processor_task: Optional[asyncio.Task] = None

        logger.info(
            f"ContextMonitor initialized: "
            f"inactivity_threshold={inactivity_threshold_minutes}min"
        )

    def on_event(
        self,
        event_type: ContextEventType,
        handler: Callable[[ContextEvent], Any],
    ):
        """
        Register a handler for an event type.

        Args:
            event_type: Type of event to handle
            handler: Async or sync function to call
        """
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.value}")

    async def update_context(
        self,
        user_id: str,
        topic: Optional[str] = None,
        emotional_state: Optional[str] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
    ):
        """
        Update user context and check for events.

        Args:
            user_id: User identifier
            topic: Current conversation topic
            emotional_state: Current emotional state
            query: User's query
            goal: Active goal
        """
        now = datetime.utcnow()

        # Get or create context
        if user_id not in self._contexts:
            self._contexts[user_id] = UserContext(
                user_id=user_id,
                session_start=now,
            )
            # New session event
            await self._emit_event(
                ContextEventType.CONVERSATION_START,
                user_id,
                {"session_start": now.isoformat()},
            )

        context = self._contexts[user_id]

        # Check for return visit
        if context.last_activity:
            gap = now - context.last_activity
            if gap > timedelta(hours=1):
                await self._emit_event(
                    ContextEventType.RETURN_VISIT,
                    user_id,
                    {
                        "gap_hours": gap.total_seconds() / 3600,
                        "last_topic": context.current_topic,
                    },
                )

        # Check for topic change
        if topic and topic != context.current_topic:
            old_topic = context.current_topic
            context.current_topic = topic
            if old_topic:
                await self._emit_event(
                    ContextEventType.TOPIC_CHANGE,
                    user_id,
                    {"old_topic": old_topic, "new_topic": topic},
                )

        # Check for emotional shift
        if emotional_state and emotional_state != context.emotional_state:
            old_state = context.emotional_state
            context.emotional_state = emotional_state
            if old_state:
                await self._emit_event(
                    ContextEventType.EMOTIONAL_SHIFT,
                    user_id,
                    {"old_state": old_state, "new_state": emotional_state},
                )

        # Track query
        if query:
            context.recent_queries.append(query)
            context.recent_queries = context.recent_queries[-20:]  # Keep last 20

        # Track goal
        if goal and goal not in context.active_goals:
            context.active_goals.append(goal)
            await self._emit_event(
                ContextEventType.GOAL_PROGRESS,
                user_id,
                {"goal": goal, "action": "started"},
            )

        # Update activity
        context.last_activity = now
        context.interaction_count += 1

        # Pattern detection
        if self.pattern_detection_enabled:
            await self._detect_patterns(context)

    async def get_context(self, user_id: str) -> Optional[UserContext]:
        """Get current context for a user."""
        return self._contexts.get(user_id)

    async def get_all_active_users(self) -> List[str]:
        """Get all users with active contexts."""
        now = datetime.utcnow()
        active = []
        for user_id, context in self._contexts.items():
            if context.last_activity:
                if now - context.last_activity < timedelta(hours=24):
                    active.append(user_id)
        return active

    async def start(self):
        """Start context monitoring."""
        if self._running:
            return

        self._running = True

        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._processor_task = asyncio.create_task(self._event_processor())

        logger.info("Context monitoring started")

    async def stop(self):
        """Stop context monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        if self._processor_task:
            self._processor_task.cancel()
            self._processor_task = None

        logger.info("Context monitoring stopped")

    async def _emit_event(
        self,
        event_type: ContextEventType,
        user_id: str,
        data: Dict[str, Any],
    ):
        """Emit a context event."""
        import uuid

        event = ContextEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            data=data,
        )

        await self._event_queue.put(event)
        logger.debug(f"Emitted event: {event_type.value} for {user_id}")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()

                for user_id, context in self._contexts.items():
                    if not context.last_activity:
                        continue

                    # Check for inactivity
                    inactive_time = now - context.last_activity
                    if inactive_time > self.inactivity_threshold:
                        await self._emit_event(
                            ContextEventType.INACTIVITY,
                            user_id,
                            {"inactive_minutes": inactive_time.total_seconds() / 60},
                        )

                    # Check time-based triggers
                    await self._check_time_triggers(user_id, context)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _event_processor(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )

                # Call registered handlers
                handlers = self._handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")

                event.processed = True

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

    async def _check_time_triggers(self, user_id: str, context: UserContext):
        """Check for time-based trigger conditions."""
        now = datetime.utcnow()

        # Morning check-in (9 AM local time approximation)
        if now.hour == 9 and now.minute == 0:
            await self._emit_event(
                ContextEventType.TIME_TRIGGER,
                user_id,
                {"trigger": "morning_checkin", "time": now.isoformat()},
            )

        # End of day summary (6 PM)
        if now.hour == 18 and now.minute == 0:
            await self._emit_event(
                ContextEventType.TIME_TRIGGER,
                user_id,
                {"trigger": "evening_summary", "time": now.isoformat()},
            )

    async def _detect_patterns(self, context: UserContext):
        """Detect patterns in user behavior."""
        # Query frequency pattern
        if len(context.recent_queries) >= 5:
            # Check for repeated topic patterns
            topics = context.recent_queries[-5:]
            # Simple pattern: same topic keywords appearing
            words = []
            for q in topics:
                words.extend(q.lower().split())

            # Count word frequency
            from collections import Counter

            word_counts = Counter(words)
            common = word_counts.most_common(3)

            if common and common[0][1] >= 3:
                context.patterns["common_topic"] = common[0][0]
                await self._emit_event(
                    ContextEventType.PATTERN_DETECTED,
                    context.user_id,
                    {"pattern": "repeated_topic", "topic": common[0][0]},
                )


# Singleton instance
_monitor: Optional[ContextMonitor] = None


def get_context_monitor() -> ContextMonitor:
    """Get singleton context monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ContextMonitor()
    return _monitor
