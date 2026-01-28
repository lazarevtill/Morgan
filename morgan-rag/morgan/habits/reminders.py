"""
Intelligent Reminder System for Morgan RAG.

Provides context-aware, personalized reminders based on user habits,
preferences, and current situation to support user goals and routines.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from .detector import HabitPattern, HabitType, HabitAnalysis
from ..intelligence.core.models import InteractionData, EmotionalState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReminderType(Enum):
    """Types of reminders that can be created."""
    
    HABIT_REMINDER = "habit_reminder"  # Reminders for specific habits
    GOAL_REMINDER = "goal_reminder"  # Reminders for goals and objectives
    WELLNESS_REMINDER = "wellness_reminder"  # Health and wellness reminders
    LEARNING_REMINDER = "learning_reminder"  # Learning and education reminders
    SOCIAL_REMINDER = "social_reminder"  # Social interaction reminders
    WORK_REMINDER = "work_reminder"  # Work-related reminders
    CUSTOM_REMINDER = "custom_reminder"  # User-defined custom reminders


class ReminderPriority(Enum):
    """Priority levels for reminders."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReminderStatus(Enum):
    """Status of reminders."""
    
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class ReminderDeliveryMethod(Enum):
    """Methods for delivering reminders."""
    
    PROACTIVE_MESSAGE = "proactive_message"  # Proactive conversation starter
    CONTEXT_INJECTION = "context_injection"  # Inject into ongoing conversation
    GENTLE_NUDGE = "gentle_nudge"  # Subtle reminder in response
    DIRECT_NOTIFICATION = "direct_notification"  # Direct notification


@dataclass
class Reminder:
    """Represents an intelligent reminder."""
    
    reminder_id: str
    user_id: str
    reminder_type: ReminderType
    priority: ReminderPriority
    
    # Content and context
    title: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling information
    scheduled_time: datetime = field(default_factory=datetime.utcnow)
    delivery_method: ReminderDeliveryMethod = ReminderDeliveryMethod.PROACTIVE_MESSAGE
    snooze_duration: timedelta = timedelta(minutes=15)
    max_snoozes: int = 3
    
    # Personalization
    tone: str = "friendly"  # friendly, professional, casual, urgent
    personalization_context: Dict[str, Any] = field(default_factory=dict)
    
    # Habit association
    related_habit_id: Optional[str] = None
    habit_context: Dict[str, Any] = field(default_factory=dict)
    
    # Status and tracking
    status: ReminderStatus = ReminderStatus.SCHEDULED
    snooze_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Effectiveness tracking
    user_response_time: Optional[timedelta] = None
    effectiveness_score: float = 0.5  # 0.0 to 1.0
    user_satisfaction: Optional[float] = None  # User feedback on reminder


@dataclass
class ReminderPreferences:
    """User preferences for reminders."""
    
    user_id: str
    
    # General preferences
    enabled: bool = True
    quiet_hours_start: time = time(22, 0)  # 10 PM
    quiet_hours_end: time = time(7, 0)  # 7 AM
    max_reminders_per_day: int = 10
    
    # Delivery preferences
    preferred_delivery_methods: List[ReminderDeliveryMethod] = field(
        default_factory=lambda: [ReminderDeliveryMethod.PROACTIVE_MESSAGE]
    )
    preferred_tone: str = "friendly"
    
    # Type-specific preferences
    habit_reminders_enabled: bool = True
    wellness_reminders_enabled: bool = True
    learning_reminders_enabled: bool = True
    work_reminders_enabled: bool = True
    
    # Timing preferences
    morning_reminders: bool = True
    afternoon_reminders: bool = True
    evening_reminders: bool = True
    weekend_reminders: bool = False
    
    # Personalization
    use_emotional_context: bool = True
    adapt_to_mood: bool = True
    respect_busy_periods: bool = True


class IntelligentReminderSystem:
    """
    Intelligent reminder system that creates context-aware, personalized reminders
    based on user habits, preferences, and current situation.
    """
    
    # Default reminder templates
    REMINDER_TEMPLATES = {
        ReminderType.HABIT_REMINDER: {
            "friendly": "Hey! Just a gentle reminder about {habit_name}. You've got this! 🌟",
            "professional": "Reminder: It's time for {habit_name} as part of your routine.",
            "casual": "Time for {habit_name}! 😊",
            "urgent": "Important reminder: {habit_name} - don't forget!"
        },
        ReminderType.WELLNESS_REMINDER: {
            "friendly": "Time to take care of yourself! How about {activity}? 💚",
            "professional": "Wellness reminder: Consider {activity} for your health.",
            "casual": "Self-care time! {activity}? 🧘‍♀️",
            "urgent": "Important: Don't forget about {activity} for your wellbeing!"
        },
        ReminderType.LEARNING_REMINDER: {
            "friendly": "Ready to learn something new? Time for {activity}! 📚",
            "professional": "Learning reminder: {activity} is scheduled now.",
            "casual": "Study time! {activity} 🤓",
            "urgent": "Don't miss your learning session: {activity}!"
        },
        ReminderType.WORK_REMINDER: {
            "friendly": "Work time! Ready to tackle {task}? You've got this! 💪",
            "professional": "Work reminder: {task} is scheduled for now.",
            "casual": "Time to get stuff done! {task} 💼",
            "urgent": "Important work reminder: {task} needs attention!"
        }
    }
    
    def __init__(self):
        """Initialize the intelligent reminder system."""
        self.user_reminders: Dict[str, List[Reminder]] = defaultdict(list)
        self.user_preferences: Dict[str, ReminderPreferences] = {}
        self.active_reminders: Dict[str, List[Reminder]] = defaultdict(list)
        self.reminder_callbacks: Dict[str, Callable] = {}
        logger.info("Intelligent reminder system initialized")
    
    def create_habit_reminders(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Optional[ReminderPreferences] = None
    ) -> List[Reminder]:
        """
        Create intelligent reminders based on user habits.
        
        Args:
            user_id: User identifier
            habit_analysis: User's habit analysis
            preferences: User's reminder preferences
            
        Returns:
            List[Reminder]: Created reminders
        """
        logger.info(f"Creating habit reminders for user {user_id}")
        
        preferences = preferences or self.get_user_preferences(user_id)
        if not preferences.habit_reminders_enabled:
            return []
        
        reminders = []
        
        # Create reminders for habits that need reinforcement
        for habit in habit_analysis.detected_habits:
            if self._should_create_reminder_for_habit(habit, preferences):
                reminder = self._create_habit_reminder(user_id, habit, preferences)
                if reminder:
                    reminders.append(reminder)
        
        # Store reminders
        self.user_reminders[user_id].extend(reminders)
        
        logger.info(f"Created {len(reminders)} habit reminders for user {user_id}")
        return reminders
    
    def create_wellness_reminder(
        self,
        user_id: str,
        activity: str,
        scheduled_time: datetime,
        context: Optional[Dict[str, Any]] = None,
        preferences: Optional[ReminderPreferences] = None
    ) -> Reminder:
        """Create a wellness reminder."""
        preferences = preferences or self.get_user_preferences(user_id)
        
        reminder_id = f"wellness_{user_id}_{scheduled_time.timestamp()}"
        
        # Personalize message based on context and preferences
        message = self._personalize_reminder_message(
            ReminderType.WELLNESS_REMINDER,
            {"activity": activity},
            preferences.preferred_tone,
            context or {}
        )
        
        reminder = Reminder(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_type=ReminderType.WELLNESS_REMINDER,
            priority=ReminderPriority.MEDIUM,
            title="Wellness Reminder",
            message=message,
            scheduled_time=scheduled_time,
            context=context or {},
            tone=preferences.preferred_tone,
            delivery_method=preferences.preferred_delivery_methods[0]
        )
        
        self.user_reminders[user_id].append(reminder)
        return reminder
    
    def create_learning_reminder(
        self,
        user_id: str,
        activity: str,
        scheduled_time: datetime,
        context: Optional[Dict[str, Any]] = None,
        preferences: Optional[ReminderPreferences] = None
    ) -> Reminder:
        """Create a learning reminder."""
        preferences = preferences or self.get_user_preferences(user_id)
        
        reminder_id = f"learning_{user_id}_{scheduled_time.timestamp()}"
        
        message = self._personalize_reminder_message(
            ReminderType.LEARNING_REMINDER,
            {"activity": activity},
            preferences.preferred_tone,
            context or {}
        )
        
        reminder = Reminder(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_type=ReminderType.LEARNING_REMINDER,
            priority=ReminderPriority.MEDIUM,
            title="Learning Reminder",
            message=message,
            scheduled_time=scheduled_time,
            context=context or {},
            tone=preferences.preferred_tone,
            delivery_method=preferences.preferred_delivery_methods[0]
        )
        
        self.user_reminders[user_id].append(reminder)
        return reminder
    
    def create_custom_reminder(
        self,
        user_id: str,
        title: str,
        message: str,
        scheduled_time: datetime,
        priority: ReminderPriority = ReminderPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> Reminder:
        """Create a custom user-defined reminder."""
        reminder_id = f"custom_{user_id}_{scheduled_time.timestamp()}"
        
        reminder = Reminder(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_type=ReminderType.CUSTOM_REMINDER,
            priority=priority,
            title=title,
            message=message,
            scheduled_time=scheduled_time,
            context=context or {}
        )
        
        self.user_reminders[user_id].append(reminder)
        return reminder
    
    def _should_create_reminder_for_habit(
        self, habit: HabitPattern, preferences: ReminderPreferences
    ) -> bool:
        """Determine if a reminder should be created for a habit."""
        # Don't remind for very consistent habits
        if habit.consistency_score > 0.8:
            return False
        
        # Don't remind for low-confidence habits
        if habit.confidence.value == "low":
            return False
        
        # Check type-specific preferences
        type_enabled_map = {
            HabitType.WELLNESS: preferences.wellness_reminders_enabled,
            HabitType.LEARNING: preferences.learning_reminders_enabled,
            HabitType.WORK: preferences.work_reminders_enabled,
        }
        
        return type_enabled_map.get(habit.habit_type, True)
    
    def _create_habit_reminder(
        self,
        user_id: str,
        habit: HabitPattern,
        preferences: ReminderPreferences
    ) -> Optional[Reminder]:
        """Create a reminder for a specific habit."""
        if not habit.typical_times:
            return None
        
        # Schedule reminder 15 minutes before typical habit time
        typical_time = habit.typical_times[0]
        reminder_time = self._subtract_time_from_time(typical_time, timedelta(minutes=15))
        scheduled_time = self._get_next_occurrence(reminder_time)
        
        # Skip if in quiet hours
        if self._is_in_quiet_hours(scheduled_time.time(), preferences):
            return None
        
        reminder_id = f"habit_{habit.habit_id}_{scheduled_time.timestamp()}"
        
        # Map habit type to reminder type
        reminder_type_map = {
            HabitType.WELLNESS: ReminderType.WELLNESS_REMINDER,
            HabitType.LEARNING: ReminderType.LEARNING_REMINDER,
            HabitType.WORK: ReminderType.WORK_REMINDER,
        }
        reminder_type = reminder_type_map.get(habit.habit_type, ReminderType.HABIT_REMINDER)
        
        # Personalize message
        message = self._personalize_reminder_message(
            reminder_type,
            {"habit_name": habit.name, "activity": habit.name},
            preferences.preferred_tone,
            habit.context
        )
        
        reminder = Reminder(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_type=reminder_type,
            priority=ReminderPriority.MEDIUM,
            title=f"{habit.name} Reminder",
            message=message,
            scheduled_time=scheduled_time,
            related_habit_id=habit.habit_id,
            habit_context={
                "habit_type": habit.habit_type.value,
                "consistency_score": habit.consistency_score,
                "triggers": habit.triggers
            },
            tone=preferences.preferred_tone,
            delivery_method=preferences.preferred_delivery_methods[0]
        )
        
        return reminder
    
    def _personalize_reminder_message(
        self,
        reminder_type: ReminderType,
        template_vars: Dict[str, str],
        tone: str,
        context: Dict[str, Any]
    ) -> str:
        """Personalize reminder message based on context and tone."""
        templates = self.REMINDER_TEMPLATES.get(reminder_type, {})
        template = templates.get(tone, templates.get("friendly", "Reminder: {activity}"))
        
        try:
            return template.format(**template_vars)
        except KeyError:
            # Fallback if template variables don't match
            return f"Reminder: {template_vars.get('activity', template_vars.get('habit_name', 'scheduled activity'))}"
    
    def get_due_reminders(self, user_id: str) -> List[Reminder]:
        """Get reminders that are due for a user."""
        now = datetime.now(timezone.utc)
        due_reminders = []
        
        for reminder in self.user_reminders[user_id]:
            if (reminder.status == ReminderStatus.SCHEDULED and 
                reminder.scheduled_time <= now):
                due_reminders.append(reminder)
        
        return sorted(due_reminders, key=lambda r: r.priority.value, reverse=True)
    
    def get_upcoming_reminders(
        self, user_id: str, hours_ahead: int = 24
    ) -> List[Reminder]:
        """Get upcoming reminders for a user."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        
        upcoming = []
        for reminder in self.user_reminders[user_id]:
            if (reminder.status == ReminderStatus.SCHEDULED and
                now < reminder.scheduled_time <= cutoff):
                upcoming.append(reminder)
        
        return sorted(upcoming, key=lambda r: r.scheduled_time)
    
    def snooze_reminder(
        self, reminder_id: str, snooze_duration: Optional[timedelta] = None
    ) -> bool:
        """Snooze a reminder."""
        reminder = self._find_reminder(reminder_id)
        if not reminder:
            return False
        
        if reminder.snooze_count >= reminder.max_snoozes:
            logger.warning(f"Maximum snoozes reached for reminder {reminder_id}")
            return False
        
        snooze_duration = snooze_duration or reminder.snooze_duration
        reminder.scheduled_time += snooze_duration
        reminder.snooze_count += 1
        reminder.status = ReminderStatus.SNOOZED
        reminder.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"Snoozed reminder {reminder_id} for {snooze_duration}")
        return True
    
    def acknowledge_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as acknowledged."""
        reminder = self._find_reminder(reminder_id)
        if not reminder:
            return False
        
        reminder.status = ReminderStatus.ACKNOWLEDGED
        reminder.acknowledged_at = datetime.now(timezone.utc)
        reminder.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"Acknowledged reminder {reminder_id}")
        return True
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed."""
        reminder = self._find_reminder(reminder_id)
        if not reminder:
            return False
        
        reminder.status = ReminderStatus.COMPLETED
        reminder.completed_at = datetime.now(timezone.utc)
        reminder.updated_at = datetime.now(timezone.utc)
        
        # Calculate effectiveness
        if reminder.delivered_at and reminder.completed_at:
            reminder.user_response_time = reminder.completed_at - reminder.delivered_at
            
            # Higher effectiveness for faster response
            response_minutes = reminder.user_response_time.total_seconds() / 60
            if response_minutes <= 5:
                reminder.effectiveness_score = 1.0
            elif response_minutes <= 30:
                reminder.effectiveness_score = 0.8
            elif response_minutes <= 120:
                reminder.effectiveness_score = 0.6
            else:
                reminder.effectiveness_score = 0.4
        
        logger.info(f"Completed reminder {reminder_id}")
        return True
    
    def dismiss_reminder(self, reminder_id: str) -> bool:
        """Dismiss a reminder."""
        reminder = self._find_reminder(reminder_id)
        if not reminder:
            return False
        
        reminder.status = ReminderStatus.DISMISSED
        reminder.updated_at = datetime.now(timezone.utc)
        
        # Lower effectiveness for dismissed reminders
        reminder.effectiveness_score = max(0.0, reminder.effectiveness_score - 0.2)
        
        logger.info(f"Dismissed reminder {reminder_id}")
        return True
    
    def update_user_preferences(
        self, user_id: str, preferences: ReminderPreferences
    ):
        """Update user reminder preferences."""
        self.user_preferences[user_id] = preferences
        logger.info(f"Updated reminder preferences for user {user_id}")
    
    def get_user_preferences(self, user_id: str) -> ReminderPreferences:
        """Get user reminder preferences."""
        return self.user_preferences.get(user_id, ReminderPreferences(user_id=user_id))
    
    def adapt_reminders_to_context(
        self,
        user_id: str,
        emotional_state: Optional[EmotionalState] = None,
        current_activity: Optional[str] = None,
        busy_level: Optional[float] = None
    ):
        """Adapt reminders based on current context."""
        preferences = self.get_user_preferences(user_id)
        
        if not preferences.use_emotional_context:
            return
        
        # Adjust reminder tone based on emotional state
        if emotional_state and preferences.adapt_to_mood:
            if emotional_state.primary_emotion in ["sadness", "stress", "anxiety"]:
                # Use gentler tone for negative emotions
                for reminder in self.user_reminders[user_id]:
                    if reminder.status == ReminderStatus.SCHEDULED:
                        reminder.tone = "gentle"
                        reminder.delivery_method = ReminderDeliveryMethod.GENTLE_NUDGE
            elif emotional_state.primary_emotion in ["joy", "excitement"]:
                # Use more energetic tone for positive emotions
                for reminder in self.user_reminders[user_id]:
                    if reminder.status == ReminderStatus.SCHEDULED:
                        reminder.tone = "enthusiastic"
        
        # Respect busy periods
        if busy_level and busy_level > 0.7 and preferences.respect_busy_periods:
            # Delay non-urgent reminders during busy periods
            now = datetime.now(timezone.utc)
            for reminder in self.user_reminders[user_id]:
                if (reminder.status == ReminderStatus.SCHEDULED and
                    reminder.priority != ReminderPriority.URGENT and
                    reminder.scheduled_time <= now + timedelta(hours=1)):
                    
                    reminder.scheduled_time += timedelta(hours=2)
                    reminder.updated_at = datetime.now(timezone.utc)
    
    def get_reminder_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get reminder statistics for a user."""
        reminders = self.user_reminders[user_id]
        
        if not reminders:
            return {}
        
        total_reminders = len(reminders)
        completed_reminders = len([r for r in reminders if r.status == ReminderStatus.COMPLETED])
        dismissed_reminders = len([r for r in reminders if r.status == ReminderStatus.DISMISSED])
        
        avg_effectiveness = sum(r.effectiveness_score for r in reminders) / total_reminders
        
        completion_rate = completed_reminders / total_reminders if total_reminders > 0 else 0
        dismissal_rate = dismissed_reminders / total_reminders if total_reminders > 0 else 0
        
        return {
            "total_reminders": total_reminders,
            "completed_reminders": completed_reminders,
            "dismissed_reminders": dismissed_reminders,
            "completion_rate": completion_rate,
            "dismissal_rate": dismissal_rate,
            "average_effectiveness": avg_effectiveness,
            "reminder_types": {
                rt.value: len([r for r in reminders if r.reminder_type == rt])
                for rt in ReminderType
            }
        }
    
    def _find_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """Find a reminder by ID."""
        for user_reminders in self.user_reminders.values():
            for reminder in user_reminders:
                if reminder.reminder_id == reminder_id:
                    return reminder
        return None
    
    def _get_next_occurrence(self, target_time: time) -> datetime:
        """Get next occurrence of a specific time."""
        now = datetime.now(timezone.utc)
        today = now.date()
        
        target_datetime = datetime.combine(today, target_time)
        
        if target_datetime <= now:
            target_datetime += timedelta(days=1)
        
        return target_datetime
    
    def _subtract_time_from_time(self, target_time: time, delta: timedelta) -> time:
        """Subtract timedelta from time."""
        dummy_date = datetime.combine(datetime.today(), target_time)
        result_datetime = dummy_date - delta
        return result_datetime.time()
    
    def _is_in_quiet_hours(self, check_time: time, preferences: ReminderPreferences) -> bool:
        """Check if time is within user's quiet hours."""
        start = preferences.quiet_hours_start
        end = preferences.quiet_hours_end
        
        if start <= end:
            # Same day quiet hours (e.g., 22:00 to 07:00 next day)
            return start <= check_time <= end
        else:
            # Quiet hours span midnight
            return check_time >= start or check_time <= end
    
    async def process_due_reminders(self, user_id: str):
        """Process all due reminders for a user."""
        due_reminders = self.get_due_reminders(user_id)
        
        for reminder in due_reminders:
            await self._deliver_reminder(reminder)
    
    async def _deliver_reminder(self, reminder: Reminder):
        """Deliver a reminder to the user."""
        logger.info(f"Delivering reminder {reminder.reminder_id} to user {reminder.user_id}")
        
        reminder.status = ReminderStatus.ACTIVE
        reminder.delivered_at = datetime.now(timezone.utc)
        reminder.updated_at = datetime.now(timezone.utc)
        
        # Execute delivery callback if registered
        callback_name = f"deliver_{reminder.delivery_method.value}"
        if callback_name in self.reminder_callbacks:
            callback = self.reminder_callbacks[callback_name]
            await callback(reminder)
        
        # Move to active reminders
        self.active_reminders[reminder.user_id].append(reminder)
    
    def register_delivery_callback(self, method: ReminderDeliveryMethod, callback: Callable):
        """Register a callback for reminder delivery."""
        callback_name = f"deliver_{method.value}"
        self.reminder_callbacks[callback_name] = callback
        logger.info(f"Registered delivery callback for {method.value}")