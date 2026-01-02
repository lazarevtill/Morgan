"""
Routine-Based Interaction Scheduler for Morgan RAG.

Schedules proactive interactions based on detected user habits and routines
to provide timely assistance and maintain engagement.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from .detector import HabitPattern, HabitType, HabitAnalysis
from ..emotional.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RoutineType(Enum):
    """Types of routine-based interactions."""
    
    PROACTIVE_CHECK_IN = "proactive_check_in"  # Regular check-ins
    HABIT_REMINDER = "habit_reminder"  # Reminders for habits
    CONTEXT_PREPARATION = "context_preparation"  # Prepare context for expected interactions
    WELLNESS_CHECK = "wellness_check"  # Wellness-related check-ins
    LEARNING_SUPPORT = "learning_support"  # Learning-related support
    PRODUCTIVITY_BOOST = "productivity_boost"  # Productivity assistance
    SOCIAL_ENGAGEMENT = "social_engagement"  # Social interaction prompts


class RoutineStatus(Enum):
    """Status of routine events."""
    
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RoutinePriority(Enum):
    """Priority levels for routine events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class RoutineEvent:
    """Represents a scheduled routine-based interaction."""
    
    event_id: str
    user_id: str
    routine_type: RoutineType
    priority: RoutinePriority
    
    # Scheduling information
    scheduled_time: datetime
    estimated_duration: timedelta
    recurrence_pattern: Optional[str] = None  # "daily", "weekly", etc.
    
    # Content and context
    title: str = ""
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    related_habits: List[str] = field(default_factory=list)  # Habit IDs
    
    # Execution details
    status: RoutineStatus = RoutineStatus.SCHEDULED
    callback_function: Optional[str] = None  # Function to call when triggered
    callback_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    
    # Adaptation
    success_rate: float = 1.0  # Success rate of this routine
    user_engagement: float = 0.5  # User engagement with this routine
    adaptation_score: float = 0.0  # How well adapted this routine is


@dataclass
class RoutineSchedule:
    """Complete routine schedule for a user."""
    
    user_id: str
    events: List[RoutineEvent] = field(default_factory=list)
    active_events: List[RoutineEvent] = field(default_factory=list)
    completed_events: List[RoutineEvent] = field(default_factory=list)
    
    # Schedule metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    total_events_created: int = 0
    successful_events: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_events_created == 0:
            return 0.0
        return self.successful_events / self.total_events_created


class RoutineScheduler:
    """
    Schedules routine-based interactions based on user habits.
    
    Analyzes user habits to create proactive interaction schedules,
    manages routine events, and adapts based on user engagement.
    """
    
    # Default scheduling parameters
    DEFAULT_CHECK_IN_INTERVAL = timedelta(hours=24)  # Daily check-ins
    DEFAULT_REMINDER_ADVANCE = timedelta(minutes=15)  # 15 minutes before habit
    DEFAULT_CONTEXT_PREP_ADVANCE = timedelta(minutes=30)  # 30 minutes before expected interaction
    
    # Routine templates
    ROUTINE_TEMPLATES = {
        RoutineType.PROACTIVE_CHECK_IN: {
            "title": "Daily Check-in",
            "description": "How are you doing today? Anything I can help with?",
            "duration": timedelta(minutes=5),
            "priority": RoutinePriority.MEDIUM
        },
        RoutineType.HABIT_REMINDER: {
            "title": "Habit Reminder",
            "description": "Gentle reminder about your routine",
            "duration": timedelta(minutes=2),
            "priority": RoutinePriority.LOW
        },
        RoutineType.CONTEXT_PREPARATION: {
            "title": "Context Preparation",
            "description": "Preparing context for your upcoming interaction",
            "duration": timedelta(minutes=1),
            "priority": RoutinePriority.HIGH
        },
        RoutineType.WELLNESS_CHECK: {
            "title": "Wellness Check",
            "description": "How are you feeling? Taking care of yourself?",
            "duration": timedelta(minutes=3),
            "priority": RoutinePriority.MEDIUM
        },
        RoutineType.LEARNING_SUPPORT: {
            "title": "Learning Support",
            "description": "Ready to continue learning? I'm here to help!",
            "duration": timedelta(minutes=10),
            "priority": RoutinePriority.MEDIUM
        },
        RoutineType.PRODUCTIVITY_BOOST: {
            "title": "Productivity Check",
            "description": "How's your productivity today? Need any assistance?",
            "duration": timedelta(minutes=5),
            "priority": RoutinePriority.MEDIUM
        },
        RoutineType.SOCIAL_ENGAGEMENT: {
            "title": "Social Check-in",
            "description": "How are your social connections? Anything to share?",
            "duration": timedelta(minutes=5),
            "priority": RoutinePriority.LOW
        }
    }
    
    def __init__(self):
        """Initialize routine scheduler."""
        self.user_schedules: Dict[str, RoutineSchedule] = {}
        self.event_callbacks: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        logger.info("Routine scheduler initialized")
    
    def create_routine_schedule(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Optional[Dict[str, Any]] = None
    ) -> RoutineSchedule:
        """
        Create a routine schedule based on user habits.
        
        Args:
            user_id: User identifier
            habit_analysis: Detected user habits
            preferences: User preferences for routine scheduling
            
        Returns:
            RoutineSchedule: Created schedule
        """
        logger.info(f"Creating routine schedule for user {user_id}")
        
        preferences = preferences or {}
        schedule = RoutineSchedule(user_id=user_id)
        
        # Create routine events based on detected habits
        events = []
        
        # Schedule proactive check-ins based on communication habits
        check_in_events = self._schedule_check_ins(user_id, habit_analysis, preferences)
        events.extend(check_in_events)
        
        # Schedule habit reminders
        reminder_events = self._schedule_habit_reminders(user_id, habit_analysis, preferences)
        events.extend(reminder_events)
        
        # Schedule context preparation
        context_events = self._schedule_context_preparation(user_id, habit_analysis, preferences)
        events.extend(context_events)
        
        # Schedule wellness checks
        wellness_events = self._schedule_wellness_checks(user_id, habit_analysis, preferences)
        events.extend(wellness_events)
        
        # Schedule learning support
        learning_events = self._schedule_learning_support(user_id, habit_analysis, preferences)
        events.extend(learning_events)
        
        # Schedule productivity boosts
        productivity_events = self._schedule_productivity_boosts(user_id, habit_analysis, preferences)
        events.extend(productivity_events)
        
        # Schedule social engagement
        social_events = self._schedule_social_engagement(user_id, habit_analysis, preferences)
        events.extend(social_events)
        
        schedule.events = events
        schedule.total_events_created = len(events)
        
        # Store schedule
        self.user_schedules[user_id] = schedule
        
        logger.info(f"Created routine schedule with {len(events)} events for user {user_id}")
        return schedule
    
    def _schedule_check_ins(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule proactive check-in events."""
        events = []
        
        # Find communication habits to determine best check-in times
        comm_habits = habit_analysis.habit_clusters.get(HabitType.COMMUNICATION, [])
        routine_habits = habit_analysis.habit_clusters.get(HabitType.ROUTINE, [])
        
        if not comm_habits and not routine_habits:
            # Default daily check-in at 9 AM
            events.append(self._create_routine_event(
                user_id=user_id,
                routine_type=RoutineType.PROACTIVE_CHECK_IN,
                scheduled_time=self._get_next_occurrence(time(9, 0)),
                recurrence_pattern="daily"
            ))
            return events
        
        # Schedule check-ins based on typical interaction times
        for habit in comm_habits + routine_habits:
            if habit.typical_times:
                # Schedule check-in 30 minutes before typical interaction time
                for typical_time in habit.typical_times[:2]:  # Max 2 check-ins per day
                    check_in_time = self._subtract_time(typical_time, timedelta(minutes=30))
                    
                    event = self._create_routine_event(
                        user_id=user_id,
                        routine_type=RoutineType.PROACTIVE_CHECK_IN,
                        scheduled_time=self._get_next_occurrence(check_in_time),
                        recurrence_pattern="daily",
                        related_habits=[habit.habit_id]
                    )
                    
                    # Customize based on habit type
                    if habit.habit_type == HabitType.WORK:
                        event.title = "Work Day Check-in"
                        event.description = "Ready to tackle your work tasks today?"
                    elif habit.habit_type == HabitType.LEARNING:
                        event.title = "Learning Check-in"
                        event.description = "Ready for today's learning session?"
                    
                    events.append(event)
        
        return events
    
    def _schedule_habit_reminders(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule habit reminder events."""
        events = []
        
        # Only create reminders for habits that benefit from them
        reminder_worthy_types = [
            HabitType.WELLNESS,
            HabitType.LEARNING,
            HabitType.PRODUCTIVITY
        ]
        
        for habit_type in reminder_worthy_types:
            habits = habit_analysis.habit_clusters.get(habit_type, [])
            
            for habit in habits:
                if habit.typical_times and habit.consistency_score < 0.8:
                    # Only remind for habits that aren't very consistent
                    for typical_time in habit.typical_times[:1]:  # One reminder per habit
                        reminder_time = self._subtract_time(
                            typical_time, 
                            self.DEFAULT_REMINDER_ADVANCE
                        )
                        
                        event = self._create_routine_event(
                            user_id=user_id,
                            routine_type=RoutineType.HABIT_REMINDER,
                            scheduled_time=self._get_next_occurrence(reminder_time),
                            recurrence_pattern="daily",
                            related_habits=[habit.habit_id]
                        )
                        
                        event.title = f"{habit.name} Reminder"
                        event.description = f"Gentle reminder: {habit.description}"
                        event.priority = RoutinePriority.LOW
                        
                        events.append(event)
        
        return events
    
    def _schedule_context_preparation(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule context preparation events."""
        events = []
        
        # Prepare context before expected interactions
        comm_habits = habit_analysis.habit_clusters.get(HabitType.COMMUNICATION, [])
        
        for habit in comm_habits:
            if habit.typical_times and habit.consistency_score > 0.6:
                # Only prepare for consistent habits
                for typical_time in habit.typical_times[:1]:  # One prep per habit
                    prep_time = self._subtract_time(
                        typical_time,
                        self.DEFAULT_CONTEXT_PREP_ADVANCE
                    )
                    
                    event = self._create_routine_event(
                        user_id=user_id,
                        routine_type=RoutineType.CONTEXT_PREPARATION,
                        scheduled_time=self._get_next_occurrence(prep_time),
                        recurrence_pattern="daily",
                        related_habits=[habit.habit_id]
                    )
                    
                    event.title = "Context Preparation"
                    event.description = "Preparing for your upcoming interaction"
                    event.priority = RoutinePriority.HIGH
                    event.callback_function = "prepare_interaction_context"
                    event.callback_params = {"habit_id": habit.habit_id}
                    
                    events.append(event)
        
        return events
    
    def _schedule_wellness_checks(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule wellness check events."""
        events = []
        
        wellness_habits = habit_analysis.habit_clusters.get(HabitType.WELLNESS, [])
        
        if wellness_habits:
            # Schedule wellness checks based on wellness habits
            for habit in wellness_habits[:1]:  # One wellness check per day
                if habit.typical_times:
                    check_time = habit.typical_times[0]
                else:
                    check_time = time(18, 0)  # Default 6 PM
                
                event = self._create_routine_event(
                    user_id=user_id,
                    routine_type=RoutineType.WELLNESS_CHECK,
                    scheduled_time=self._get_next_occurrence(check_time),
                    recurrence_pattern="daily",
                    related_habits=[habit.habit_id]
                )
                
                events.append(event)
        else:
            # Default wellness check at 6 PM
            event = self._create_routine_event(
                user_id=user_id,
                routine_type=RoutineType.WELLNESS_CHECK,
                scheduled_time=self._get_next_occurrence(time(18, 0)),
                recurrence_pattern="daily"
            )
            events.append(event)
        
        return events
    
    def _schedule_learning_support(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule learning support events."""
        events = []
        
        learning_habits = habit_analysis.habit_clusters.get(HabitType.LEARNING, [])
        
        for habit in learning_habits:
            if habit.typical_times:
                for typical_time in habit.typical_times[:1]:  # One support per habit
                    event = self._create_routine_event(
                        user_id=user_id,
                        routine_type=RoutineType.LEARNING_SUPPORT,
                        scheduled_time=self._get_next_occurrence(typical_time),
                        recurrence_pattern="daily",
                        related_habits=[habit.habit_id]
                    )
                    
                    event.title = "Learning Support"
                    event.description = f"Ready to continue with {habit.name}?"
                    
                    events.append(event)
        
        return events
    
    def _schedule_productivity_boosts(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule productivity boost events."""
        events = []
        
        work_habits = habit_analysis.habit_clusters.get(HabitType.WORK, [])
        productivity_habits = habit_analysis.habit_clusters.get(HabitType.PRODUCTIVITY, [])
        
        all_productivity_habits = work_habits + productivity_habits
        
        for habit in all_productivity_habits[:2]:  # Max 2 productivity boosts per day
            if habit.typical_times:
                # Schedule boost 1 hour after typical start time
                boost_time = self._add_time(habit.typical_times[0], timedelta(hours=1))
                
                event = self._create_routine_event(
                    user_id=user_id,
                    routine_type=RoutineType.PRODUCTIVITY_BOOST,
                    scheduled_time=self._get_next_occurrence(boost_time),
                    recurrence_pattern="daily",
                    related_habits=[habit.habit_id]
                )
                
                event.title = "Productivity Check"
                event.description = "How's your productivity? Need any assistance?"
                
                events.append(event)
        
        return events
    
    def _schedule_social_engagement(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Dict[str, Any]
    ) -> List[RoutineEvent]:
        """Schedule social engagement events."""
        events = []
        
        social_habits = habit_analysis.habit_clusters.get(HabitType.SOCIAL, [])
        
        if social_habits:
            # Weekly social check-in based on social habits
            habit = social_habits[0]
            if habit.typical_days:
                # Schedule on the most common social day
                day_name = habit.typical_days[0]
                check_time = time(19, 0)  # 7 PM
                
                event = self._create_routine_event(
                    user_id=user_id,
                    routine_type=RoutineType.SOCIAL_ENGAGEMENT,
                    scheduled_time=self._get_next_occurrence_on_day(day_name, check_time),
                    recurrence_pattern="weekly",
                    related_habits=[habit.habit_id]
                )
                
                events.append(event)
        else:
            # Default weekly social check-in on Friday evening
            event = self._create_routine_event(
                user_id=user_id,
                routine_type=RoutineType.SOCIAL_ENGAGEMENT,
                scheduled_time=self._get_next_occurrence_on_day("Friday", time(19, 0)),
                recurrence_pattern="weekly"
            )
            events.append(event)
        
        return events
    
    def _create_routine_event(
        self,
        user_id: str,
        routine_type: RoutineType,
        scheduled_time: datetime,
        recurrence_pattern: Optional[str] = None,
        related_habits: Optional[List[str]] = None
    ) -> RoutineEvent:
        """Create a routine event with default values."""
        template = self.ROUTINE_TEMPLATES.get(routine_type, {})
        
        event_id = f"{routine_type.value}_{user_id}_{scheduled_time.timestamp()}"
        
        return RoutineEvent(
            event_id=event_id,
            user_id=user_id,
            routine_type=routine_type,
            priority=template.get("priority", RoutinePriority.MEDIUM),
            scheduled_time=scheduled_time,
            estimated_duration=template.get("duration", timedelta(minutes=5)),
            recurrence_pattern=recurrence_pattern,
            title=template.get("title", routine_type.value.replace("_", " ").title()),
            description=template.get("description", "Routine interaction"),
            related_habits=related_habits or []
        )
    
    def _get_next_occurrence(self, target_time: time) -> datetime:
        """Get next occurrence of a specific time."""
        now = datetime.utcnow()
        today = now.date()
        
        target_datetime = datetime.combine(today, target_time)
        
        if target_datetime <= now:
            # If time has passed today, schedule for tomorrow
            target_datetime += timedelta(days=1)
        
        return target_datetime
    
    def _get_next_occurrence_on_day(self, day_name: str, target_time: time) -> datetime:
        """Get next occurrence of a specific day and time."""
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        target_weekday = day_names.index(day_name)
        
        now = datetime.utcnow()
        today = now.date()
        current_weekday = today.weekday()
        
        # Calculate days until target weekday
        days_ahead = target_weekday - current_weekday
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        target_date = today + timedelta(days=days_ahead)
        return datetime.combine(target_date, target_time)
    
    def _subtract_time(self, target_time: time, delta: timedelta) -> time:
        """Subtract timedelta from time."""
        dummy_date = datetime.combine(datetime.today(), target_time)
        result_datetime = dummy_date - delta
        return result_datetime.time()
    
    def _add_time(self, target_time: time, delta: timedelta) -> time:
        """Add timedelta to time."""
        dummy_date = datetime.combine(datetime.today(), target_time)
        result_datetime = dummy_date + delta
        return result_datetime.time()
    
    def register_callback(self, callback_name: str, callback_function: Callable):
        """Register a callback function for routine events."""
        self.event_callbacks[callback_name] = callback_function
        logger.info(f"Registered callback: {callback_name}")
    
    async def start_scheduler(self, user_id: str):
        """Start the scheduler for a user."""
        if user_id in self.running_tasks:
            logger.warning(f"Scheduler already running for user {user_id}")
            return
        
        schedule = self.user_schedules.get(user_id)
        if not schedule:
            logger.warning(f"No schedule found for user {user_id}")
            return
        
        task = asyncio.create_task(self._run_scheduler(user_id))
        self.running_tasks[user_id] = task
        
        logger.info(f"Started scheduler for user {user_id}")
    
    async def stop_scheduler(self, user_id: str):
        """Stop the scheduler for a user."""
        task = self.running_tasks.get(user_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.running_tasks[user_id]
            logger.info(f"Stopped scheduler for user {user_id}")
    
    async def _run_scheduler(self, user_id: str):
        """Run the scheduler loop for a user."""
        logger.info(f"Running scheduler for user {user_id}")
        
        while True:
            try:
                await self._process_scheduled_events(user_id)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                logger.info(f"Scheduler cancelled for user {user_id}")
                break
            except Exception as e:
                logger.error(f"Error in scheduler for user {user_id}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _process_scheduled_events(self, user_id: str):
        """Process scheduled events for a user."""
        schedule = self.user_schedules.get(user_id)
        if not schedule:
            return
        
        now = datetime.utcnow()
        
        # Find events that should be triggered
        events_to_trigger = [
            event for event in schedule.events
            if event.status == RoutineStatus.SCHEDULED and event.scheduled_time <= now
        ]
        
        for event in events_to_trigger:
            await self._trigger_event(event)
    
    async def _trigger_event(self, event: RoutineEvent):
        """Trigger a routine event."""
        logger.info(f"Triggering event: {event.title} for user {event.user_id}")
        
        event.status = RoutineStatus.ACTIVE
        event.updated_at = datetime.utcnow()
        
        try:
            # Execute callback if specified
            if event.callback_function and event.callback_function in self.event_callbacks:
                callback = self.event_callbacks[event.callback_function]
                await callback(event)
            
            # Mark as completed
            event.status = RoutineStatus.COMPLETED
            event.last_executed = datetime.utcnow()
            event.execution_count += 1
            
            # Schedule next occurrence if recurring
            if event.recurrence_pattern:
                self._schedule_next_occurrence(event)
            
            # Update schedule statistics
            schedule = self.user_schedules.get(event.user_id)
            if schedule:
                schedule.successful_events += 1
            
            logger.info(f"Successfully completed event: {event.title}")
            
        except Exception as e:
            logger.error(f"Error executing event {event.title}: {e}")
            event.status = RoutineStatus.FAILED
            event.updated_at = datetime.utcnow()
    
    def _schedule_next_occurrence(self, event: RoutineEvent):
        """Schedule the next occurrence of a recurring event."""
        if event.recurrence_pattern == "daily":
            next_time = event.scheduled_time + timedelta(days=1)
        elif event.recurrence_pattern == "weekly":
            next_time = event.scheduled_time + timedelta(weeks=1)
        elif event.recurrence_pattern == "monthly":
            next_time = event.scheduled_time + timedelta(days=30)  # Approximate
        else:
            return  # No recurrence
        
        # Create new event for next occurrence
        new_event = RoutineEvent(
            event_id=f"{event.routine_type.value}_{event.user_id}_{next_time.timestamp()}",
            user_id=event.user_id,
            routine_type=event.routine_type,
            priority=event.priority,
            scheduled_time=next_time,
            estimated_duration=event.estimated_duration,
            recurrence_pattern=event.recurrence_pattern,
            title=event.title,
            description=event.description,
            context=event.context.copy(),
            related_habits=event.related_habits.copy(),
            callback_function=event.callback_function,
            callback_params=event.callback_params.copy()
        )
        
        # Add to schedule
        schedule = self.user_schedules.get(event.user_id)
        if schedule:
            schedule.events.append(new_event)
            schedule.total_events_created += 1
    
    def get_upcoming_events(
        self, user_id: str, hours_ahead: int = 24
    ) -> List[RoutineEvent]:
        """Get upcoming events for a user."""
        schedule = self.user_schedules.get(user_id)
        if not schedule:
            return []
        
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        
        upcoming = [
            event for event in schedule.events
            if event.status == RoutineStatus.SCHEDULED
            and now <= event.scheduled_time <= cutoff
        ]
        
        return sorted(upcoming, key=lambda e: e.scheduled_time)
    
    def update_event_engagement(
        self, event_id: str, user_engagement: float, success: bool = True
    ):
        """Update event engagement metrics."""
        for schedule in self.user_schedules.values():
            for event in schedule.events:
                if event.event_id == event_id:
                    event.user_engagement = user_engagement
                    if success:
                        event.success_rate = min(1.0, event.success_rate + 0.1)
                    else:
                        event.success_rate = max(0.0, event.success_rate - 0.1)
                    
                    event.updated_at = datetime.utcnow()
                    return
    
    def adapt_schedule(self, user_id: str, new_habit_analysis: HabitAnalysis):
        """Adapt schedule based on updated habit analysis."""
        logger.info(f"Adapting schedule for user {user_id}")
        
        # Remove low-performing events
        schedule = self.user_schedules.get(user_id)
        if not schedule:
            return
        
        # Filter out events with low success rate or engagement
        schedule.events = [
            event for event in schedule.events
            if event.success_rate > 0.3 and event.user_engagement > 0.2
        ]
        
        # Create new schedule based on updated habits
        new_schedule = self.create_routine_schedule(user_id, new_habit_analysis)
        
        # Merge with existing high-performing events
        high_performing_events = [
            event for event in schedule.events
            if event.success_rate > 0.7 and event.user_engagement > 0.6
        ]
        
        new_schedule.events.extend(high_performing_events)
        self.user_schedules[user_id] = new_schedule
        
        logger.info(f"Adapted schedule for user {user_id} with {len(new_schedule.events)} events")
    
    def get_schedule_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get schedule statistics for a user."""
        schedule = self.user_schedules.get(user_id)
        if not schedule:
            return {}
        
        stats = {
            "total_events": len(schedule.events),
            "active_events": len([e for e in schedule.events if e.status == RoutineStatus.ACTIVE]),
            "completed_events": len([e for e in schedule.events if e.status == RoutineStatus.COMPLETED]),
            "success_rate": schedule.success_rate,
            "average_engagement": sum(e.user_engagement for e in schedule.events) / len(schedule.events) if schedule.events else 0.0,
            "routine_types": {
                rt.value: len([e for e in schedule.events if e.routine_type == rt])
                for rt in RoutineType
            }
        }
        
        return stats