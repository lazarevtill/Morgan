"""
Habit Pattern Detection for Morgan RAG.

Analyzes user interaction patterns to identify habits, routines, and behavioral patterns
for personalized assistance and proactive support.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..emotional.models import InteractionData
from ..learning.patterns import InteractionPatterns, TimingPattern
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HabitType(Enum):
    """Types of habits that can be detected."""
    
    COMMUNICATION = "communication"  # Communication patterns
    WORK = "work"  # Work-related habits
    LEARNING = "learning"  # Learning and study habits
    WELLNESS = "wellness"  # Health and wellness habits
    SOCIAL = "social"  # Social interaction habits
    PRODUCTIVITY = "productivity"  # Productivity and task habits
    ENTERTAINMENT = "entertainment"  # Entertainment and leisure habits
    ROUTINE = "routine"  # Daily routine habits


class HabitFrequency(Enum):
    """Frequency of habit occurrence."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    IRREGULAR = "irregular"


class HabitConfidence(Enum):
    """Confidence level in habit detection."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class HabitPattern:
    """Represents a detected habit pattern."""
    
    habit_id: str
    user_id: str
    habit_type: HabitType
    name: str
    description: str
    frequency: HabitFrequency
    confidence: HabitConfidence
    
    # Timing information
    typical_times: List[time]  # Typical times when habit occurs
    typical_days: List[str]  # Days of week when habit occurs
    duration_minutes: Optional[int]  # Typical duration
    
    # Pattern details
    triggers: List[str]  # What triggers this habit
    context: List[str]  # Context in which habit occurs
    keywords: List[str]  # Keywords associated with habit
    
    # Statistics
    occurrence_count: int
    first_observed: datetime
    last_observed: datetime
    consistency_score: float  # 0.0 to 1.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HabitAnalysis:
    """Complete habit analysis for a user."""
    
    user_id: str
    analysis_period: timedelta
    detected_habits: List[HabitPattern]
    habit_clusters: Dict[HabitType, List[HabitPattern]]
    routine_strength: float  # Overall routine consistency (0.0-1.0)
    predictability_score: float  # How predictable user behavior is
    analysis_confidence: float  # Overall confidence in analysis
    last_analysis: datetime = field(default_factory=datetime.utcnow)


class HabitDetector:
    """
    Detects user habits and behavioral patterns from interaction data.
    
    Analyzes timing patterns, communication habits, topic preferences,
    and behavioral consistency to identify user routines and habits.
    """
    
    # Minimum occurrences to consider a pattern a habit
    MIN_OCCURRENCES_FOR_HABIT = 3
    MIN_DAYS_FOR_ANALYSIS = 7
    
    # Habit detection patterns
    HABIT_KEYWORDS = {
        HabitType.WORK: [
            r"\b(work|job|office|meeting|project|deadline|task)\b",
            r"\b(email|report|presentation|document|spreadsheet)\b",
            r"\b(schedule|calendar|appointment|conference)\b",
        ],
        HabitType.LEARNING: [
            r"\b(learn|study|course|tutorial|lesson|practice)\b",
            r"\b(book|read|research|understand|explain)\b",
            r"\b(skill|knowledge|training|education)\b",
        ],
        HabitType.WELLNESS: [
            r"\b(exercise|workout|gym|run|walk|health)\b",
            r"\b(sleep|rest|tired|energy|meditation)\b",
            r"\b(eat|food|meal|nutrition|diet|water)\b",
        ],
        HabitType.SOCIAL: [
            r"\b(friend|family|social|party|event|gathering)\b",
            r"\b(call|text|message|chat|visit|meet)\b",
            r"\b(birthday|celebration|holiday|weekend)\b",
        ],
        HabitType.PRODUCTIVITY: [
            r"\b(organize|plan|todo|list|priority|goal)\b",
            r"\b(focus|concentrate|productive|efficient)\b",
            r"\b(clean|tidy|organize|sort|arrange)\b",
        ],
        HabitType.ENTERTAINMENT: [
            r"\b(movie|show|game|music|book|video)\b",
            r"\b(watch|play|listen|read|enjoy|fun)\b",
            r"\b(hobby|interest|passion|creative)\b",
        ],
    }
    
    # Time-based habit indicators
    MORNING_HOURS = list(range(6, 12))  # 6 AM - 12 PM
    AFTERNOON_HOURS = list(range(12, 18))  # 12 PM - 6 PM
    EVENING_HOURS = list(range(18, 22))  # 6 PM - 10 PM
    NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))  # 10 PM - 6 AM
    
    def __init__(self):
        """Initialize habit detector."""
        self.user_habits: Dict[str, HabitAnalysis] = {}
        logger.info("Habit detector initialized")
    
    def detect_habits(
        self,
        user_id: str,
        interactions: List[InteractionData],
        interaction_patterns: Optional[InteractionPatterns] = None,
        analysis_period: timedelta = timedelta(days=30)
    ) -> HabitAnalysis:
        """
        Detect user habits from interaction data.
        
        Args:
            user_id: User identifier
            interactions: List of user interactions
            interaction_patterns: Pre-analyzed interaction patterns
            analysis_period: Time period for analysis
            
        Returns:
            HabitAnalysis: Detected habits and patterns
        """
        logger.info(
            f"Detecting habits for user {user_id} from {len(interactions)} interactions"
        )
        
        if len(interactions) < self.MIN_OCCURRENCES_FOR_HABIT:
            logger.warning(
                f"Insufficient interactions ({len(interactions)}) for habit detection"
            )
            return self._create_empty_analysis(user_id, analysis_period)
        
        # Filter interactions to analysis period
        cutoff_date = datetime.utcnow() - analysis_period
        recent_interactions = [
            interaction for interaction in interactions
            if hasattr(interaction, 'timestamp') and interaction.timestamp >= cutoff_date
        ]
        
        if len(recent_interactions) < self.MIN_OCCURRENCES_FOR_HABIT:
            logger.warning("Insufficient recent interactions for habit detection")
            return self._create_empty_analysis(user_id, analysis_period)
        
        # Detect different types of habits
        detected_habits = []
        
        # Communication habits
        comm_habits = self._detect_communication_habits(user_id, recent_interactions)
        detected_habits.extend(comm_habits)
        
        # Topic-based habits
        topic_habits = self._detect_topic_habits(user_id, recent_interactions)
        detected_habits.extend(topic_habits)
        
        # Timing-based habits
        timing_habits = self._detect_timing_habits(user_id, recent_interactions)
        detected_habits.extend(timing_habits)
        
        # Behavioral habits
        behavioral_habits = self._detect_behavioral_habits(user_id, recent_interactions)
        detected_habits.extend(behavioral_habits)
        
        # Cluster habits by type
        habit_clusters = self._cluster_habits_by_type(detected_habits)
        
        # Calculate overall metrics
        routine_strength = self._calculate_routine_strength(detected_habits)
        predictability_score = self._calculate_predictability_score(recent_interactions)
        analysis_confidence = self._calculate_analysis_confidence(detected_habits)
        
        analysis = HabitAnalysis(
            user_id=user_id,
            analysis_period=analysis_period,
            detected_habits=detected_habits,
            habit_clusters=habit_clusters,
            routine_strength=routine_strength,
            predictability_score=predictability_score,
            analysis_confidence=analysis_confidence
        )
        
        # Store analysis for future reference
        self.user_habits[user_id] = analysis
        
        logger.info(
            f"Habit detection complete for user {user_id}. "
            f"Found {len(detected_habits)} habits with confidence {analysis_confidence:.2f}"
        )
        
        return analysis
    
    def _detect_communication_habits(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[HabitPattern]:
        """Detect communication-related habits."""
        logger.debug(f"Detecting communication habits for user {user_id}")
        
        habits = []
        
        # Analyze session timing patterns
        session_times = self._extract_session_times(interactions)
        if session_times:
            timing_habit = self._analyze_session_timing_habit(
                user_id, session_times, interactions
            )
            if timing_habit:
                habits.append(timing_habit)
        
        # Analyze response length preferences
        response_habit = self._analyze_response_length_habit(user_id, interactions)
        if response_habit:
            habits.append(response_habit)
        
        # Analyze question asking patterns
        question_habit = self._analyze_question_asking_habit(user_id, interactions)
        if question_habit:
            habits.append(question_habit)
        
        return habits
    
    def _detect_topic_habits(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[HabitPattern]:
        """Detect topic-related habits."""
        logger.debug(f"Detecting topic habits for user {user_id}")
        
        habits = []
        
        # Extract text content from interactions
        text_content = self._extract_text_content(interactions)
        
        # Detect habits for each habit type
        for habit_type, keywords in self.HABIT_KEYWORDS.items():
            habit = self._detect_keyword_based_habit(
                user_id, habit_type, keywords, text_content, interactions
            )
            if habit:
                habits.append(habit)
        
        return habits
    
    def _detect_timing_habits(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[HabitPattern]:
        """Detect timing-based habits."""
        logger.debug(f"Detecting timing habits for user {user_id}")
        
        habits = []
        
        # Extract timestamps
        timestamps = [
            interaction.timestamp for interaction in interactions
            if hasattr(interaction, 'timestamp') and interaction.timestamp
        ]
        
        if not timestamps:
            return habits
        
        # Detect daily routine patterns
        daily_habit = self._detect_daily_routine_habit(user_id, timestamps)
        if daily_habit:
            habits.append(daily_habit)
        
        # Detect weekly patterns
        weekly_habit = self._detect_weekly_routine_habit(user_id, timestamps)
        if weekly_habit:
            habits.append(weekly_habit)
        
        # Detect time-of-day preferences
        time_habits = self._detect_time_preference_habits(user_id, timestamps)
        habits.extend(time_habits)
        
        return habits
    
    def _detect_behavioral_habits(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[HabitPattern]:
        """Detect behavioral habits."""
        logger.debug(f"Detecting behavioral habits for user {user_id}")
        
        habits = []
        
        # Detect feedback giving habit
        feedback_habit = self._detect_feedback_habit(user_id, interactions)
        if feedback_habit:
            habits.append(feedback_habit)
        
        # Detect help-seeking patterns
        help_habit = self._detect_help_seeking_habit(user_id, interactions)
        if help_habit:
            habits.append(help_habit)
        
        return habits
    
    def _extract_session_times(self, interactions: List[InteractionData]) -> List[datetime]:
        """Extract session start times from interactions."""
        session_times = []
        
        for interaction in interactions:
            if hasattr(interaction, 'timestamp') and interaction.timestamp:
                session_times.append(interaction.timestamp)
        
        return sorted(session_times)
    
    def _analyze_session_timing_habit(
        self, user_id: str, session_times: List[datetime], interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Analyze session timing patterns to detect habits."""
        if len(session_times) < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        # Group sessions by hour of day
        hour_counter = Counter(dt.hour for dt in session_times)
        most_common_hours = hour_counter.most_common(3)
        
        if not most_common_hours or most_common_hours[0][1] < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        # Determine typical times
        typical_times = [time(hour=hour) for hour, _ in most_common_hours]
        
        # Determine frequency
        total_days = (session_times[-1] - session_times[0]).days + 1
        sessions_per_day = len(session_times) / total_days
        
        if sessions_per_day >= 0.8:
            frequency = HabitFrequency.DAILY
        elif sessions_per_day >= 0.3:
            frequency = HabitFrequency.WEEKLY
        else:
            frequency = HabitFrequency.IRREGULAR
        
        # Calculate consistency
        consistency_score = self._calculate_timing_consistency(session_times)
        
        # Determine confidence
        confidence = self._determine_confidence(len(session_times), consistency_score)
        
        return HabitPattern(
            habit_id=f"comm_timing_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.COMMUNICATION,
            name="Regular Communication Sessions",
            description=f"Tends to interact at {', '.join(f'{t.hour}:00' for t in typical_times)}",
            frequency=frequency,
            confidence=confidence,
            typical_times=typical_times,
            typical_days=self._get_typical_days(session_times),
            duration_minutes=self._estimate_session_duration(interactions),
            triggers=["time-based"],
            context=["regular schedule"],
            keywords=["communication", "session", "timing"],
            occurrence_count=len(session_times),
            first_observed=session_times[0],
            last_observed=session_times[-1],
            consistency_score=consistency_score
        )
    
    def _analyze_response_length_habit(
        self, user_id: str, interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Analyze response length preferences."""
        # This would analyze user's preferred response lengths
        # For now, return None as this requires more complex analysis
        return None
    
    def _analyze_question_asking_habit(
        self, user_id: str, interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Analyze question asking patterns."""
        text_content = self._extract_text_content(interactions)
        
        question_count = 0
        total_messages = len(text_content)
        
        for text in text_content:
            if '?' in text:
                question_count += 1
        
        if question_count < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        question_ratio = question_count / total_messages if total_messages > 0 else 0
        
        if question_ratio < 0.3:  # Less than 30% questions
            return None
        
        return HabitPattern(
            habit_id=f"question_habit_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.COMMUNICATION,
            name="Frequent Question Asking",
            description=f"Asks questions in {question_ratio:.1%} of interactions",
            frequency=HabitFrequency.DAILY,
            confidence=HabitConfidence.MEDIUM,
            typical_times=[],
            typical_days=[],
            duration_minutes=None,
            triggers=["curiosity", "learning"],
            context=["information seeking"],
            keywords=["questions", "inquiry", "learning"],
            occurrence_count=question_count,
            first_observed=interactions[0].timestamp if interactions else datetime.utcnow(),
            last_observed=interactions[-1].timestamp if interactions else datetime.utcnow(),
            consistency_score=question_ratio
        )
    
    def _extract_text_content(self, interactions: List[InteractionData]) -> List[str]:
        """Extract text content from interactions."""
        text_content = []
        
        for interaction in interactions:
            # Extract message text
            if hasattr(interaction.conversation_context, 'message_text'):
                text = interaction.conversation_context.message_text
                if text and isinstance(text, str):
                    text_content.append(text.lower())
            
            # Extract user message
            if hasattr(interaction, 'user_message') and interaction.user_message:
                text_content.append(interaction.user_message.lower())
        
        return text_content
    
    def _detect_keyword_based_habit(
        self,
        user_id: str,
        habit_type: HabitType,
        keywords: List[str],
        text_content: List[str],
        interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Detect habit based on keyword patterns."""
        if not text_content:
            return None
        
        # Count keyword matches
        match_count = 0
        matching_texts = []
        
        for text in text_content:
            text_matches = 0
            for pattern in keywords:
                matches = re.findall(pattern, text)
                text_matches += len(matches)
            
            if text_matches > 0:
                match_count += 1
                matching_texts.append(text)
        
        if match_count < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        # Calculate frequency and consistency
        match_ratio = match_count / len(text_content)
        
        if match_ratio < 0.1:  # Less than 10% of interactions
            return None
        
        # Determine frequency based on match ratio
        if match_ratio >= 0.5:
            frequency = HabitFrequency.DAILY
        elif match_ratio >= 0.2:
            frequency = HabitFrequency.WEEKLY
        else:
            frequency = HabitFrequency.IRREGULAR
        
        # Extract timing information
        timestamps = [
            interaction.timestamp for interaction in interactions
            if hasattr(interaction, 'timestamp') and interaction.timestamp
        ]
        
        typical_times = self._extract_typical_times(timestamps) if timestamps else []
        typical_days = self._get_typical_days(timestamps) if timestamps else []
        
        confidence = self._determine_confidence(match_count, match_ratio)
        
        return HabitPattern(
            habit_id=f"{habit_type.value}_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=habit_type,
            name=f"{habit_type.value.title()} Interest",
            description=f"Shows consistent interest in {habit_type.value} topics ({match_ratio:.1%} of interactions)",
            frequency=frequency,
            confidence=confidence,
            typical_times=typical_times,
            typical_days=typical_days,
            duration_minutes=None,
            triggers=["topic interest"],
            context=[habit_type.value],
            keywords=[habit_type.value],
            occurrence_count=match_count,
            first_observed=timestamps[0] if timestamps else datetime.utcnow(),
            last_observed=timestamps[-1] if timestamps else datetime.utcnow(),
            consistency_score=match_ratio
        )
    
    def _detect_daily_routine_habit(
        self, user_id: str, timestamps: List[datetime]
    ) -> Optional[HabitPattern]:
        """Detect daily routine patterns."""
        if len(timestamps) < 7:  # Need at least a week of data
            return None
        
        # Group by day and analyze consistency
        daily_sessions = defaultdict(list)
        for ts in timestamps:
            day_key = ts.date()
            daily_sessions[day_key].append(ts)
        
        # Check if user has daily interaction pattern
        days_with_sessions = len(daily_sessions)
        total_days = (timestamps[-1].date() - timestamps[0].date()).days + 1
        daily_consistency = days_with_sessions / total_days
        
        if daily_consistency < 0.5:  # Less than 50% of days
            return None
        
        return HabitPattern(
            habit_id=f"daily_routine_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.ROUTINE,
            name="Daily Interaction Routine",
            description=f"Interacts daily with {daily_consistency:.1%} consistency",
            frequency=HabitFrequency.DAILY,
            confidence=self._determine_confidence(days_with_sessions, daily_consistency),
            typical_times=self._extract_typical_times(timestamps),
            typical_days=list(range(7)),  # All days
            duration_minutes=None,
            triggers=["daily routine"],
            context=["regular schedule"],
            keywords=["daily", "routine", "regular"],
            occurrence_count=days_with_sessions,
            first_observed=timestamps[0],
            last_observed=timestamps[-1],
            consistency_score=daily_consistency
        )
    
    def _detect_weekly_routine_habit(
        self, user_id: str, timestamps: List[datetime]
    ) -> Optional[HabitPattern]:
        """Detect weekly routine patterns."""
        if len(timestamps) < 14:  # Need at least two weeks
            return None
        
        # Analyze day-of-week patterns
        weekday_counter = Counter(ts.weekday() for ts in timestamps)
        most_common_days = weekday_counter.most_common(3)
        
        if not most_common_days or most_common_days[0][1] < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        # Check for weekly consistency
        total_weeks = (timestamps[-1] - timestamps[0]).days // 7 + 1
        weekly_consistency = most_common_days[0][1] / total_weeks
        
        if weekly_consistency < 0.3:
            return None
        
        typical_days = [
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]
            for day, _ in most_common_days
        ]
        
        return HabitPattern(
            habit_id=f"weekly_routine_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.ROUTINE,
            name="Weekly Interaction Pattern",
            description=f"Most active on {', '.join(typical_days[:2])}",
            frequency=HabitFrequency.WEEKLY,
            confidence=self._determine_confidence(len(most_common_days), weekly_consistency),
            typical_times=self._extract_typical_times(timestamps),
            typical_days=typical_days,
            duration_minutes=None,
            triggers=["weekly routine"],
            context=["weekly schedule"],
            keywords=["weekly", "routine", "schedule"],
            occurrence_count=sum(count for _, count in most_common_days),
            first_observed=timestamps[0],
            last_observed=timestamps[-1],
            consistency_score=weekly_consistency
        )
    
    def _detect_time_preference_habits(
        self, user_id: str, timestamps: List[datetime]
    ) -> List[HabitPattern]:
        """Detect time-of-day preference habits."""
        habits = []
        
        if len(timestamps) < self.MIN_OCCURRENCES_FOR_HABIT:
            return habits
        
        # Analyze time-of-day patterns
        time_periods = {
            'Morning': self.MORNING_HOURS,
            'Afternoon': self.AFTERNOON_HOURS,
            'Evening': self.EVENING_HOURS,
            'Night': self.NIGHT_HOURS
        }
        
        period_counts = {}
        for period_name, hours in time_periods.items():
            count = sum(1 for ts in timestamps if ts.hour in hours)
            period_counts[period_name] = count
        
        # Find dominant time period
        max_period = max(period_counts, key=period_counts.get)
        max_count = period_counts[max_period]
        
        if max_count < self.MIN_OCCURRENCES_FOR_HABIT:
            return habits
        
        # Calculate preference strength
        total_interactions = len(timestamps)
        preference_strength = max_count / total_interactions
        
        if preference_strength < 0.4:  # Less than 40% preference
            return habits
        
        habit = HabitPattern(
            habit_id=f"time_pref_{max_period.lower()}_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.ROUTINE,
            name=f"{max_period} Preference",
            description=f"Prefers to interact during {max_period.lower()} hours ({preference_strength:.1%})",
            frequency=HabitFrequency.DAILY,
            confidence=self._determine_confidence(max_count, preference_strength),
            typical_times=[time(hour=h) for h in time_periods[max_period][:3]],
            typical_days=[],
            duration_minutes=None,
            triggers=["time preference"],
            context=[max_period.lower()],
            keywords=[max_period.lower(), "time", "preference"],
            occurrence_count=max_count,
            first_observed=timestamps[0],
            last_observed=timestamps[-1],
            consistency_score=preference_strength
        )
        
        habits.append(habit)
        return habits
    
    def _detect_feedback_habit(
        self, user_id: str, interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Detect feedback giving habits."""
        feedback_count = 0
        
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'user_feedback'):
                if interaction.conversation_context.user_feedback is not None:
                    feedback_count += 1
        
        if feedback_count < self.MIN_OCCURRENCES_FOR_HABIT:
            return None
        
        feedback_ratio = feedback_count / len(interactions)
        
        if feedback_ratio < 0.2:  # Less than 20% feedback rate
            return None
        
        return HabitPattern(
            habit_id=f"feedback_habit_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            habit_type=HabitType.COMMUNICATION,
            name="Regular Feedback Provider",
            description=f"Provides feedback in {feedback_ratio:.1%} of interactions",
            frequency=HabitFrequency.DAILY,
            confidence=self._determine_confidence(feedback_count, feedback_ratio),
            typical_times=[],
            typical_days=[],
            duration_minutes=None,
            triggers=["helpfulness", "engagement"],
            context=["feedback"],
            keywords=["feedback", "rating", "helpful"],
            occurrence_count=feedback_count,
            first_observed=interactions[0].timestamp if interactions else datetime.utcnow(),
            last_observed=interactions[-1].timestamp if interactions else datetime.utcnow(),
            consistency_score=feedback_ratio
        )
    
    def _detect_help_seeking_habit(
        self, user_id: str, interactions: List[InteractionData]
    ) -> Optional[HabitPattern]:
        """Detect help-seeking patterns."""
        # This would analyze how user seeks help
        # For now, return None as this requires more complex analysis
        return None
    
    def _cluster_habits_by_type(
        self, habits: List[HabitPattern]
    ) -> Dict[HabitType, List[HabitPattern]]:
        """Cluster habits by type."""
        clusters = defaultdict(list)
        
        for habit in habits:
            clusters[habit.habit_type].append(habit)
        
        return dict(clusters)
    
    def _calculate_routine_strength(self, habits: List[HabitPattern]) -> float:
        """Calculate overall routine strength."""
        if not habits:
            return 0.0
        
        # Weight by confidence and consistency
        total_strength = 0.0
        total_weight = 0.0
        
        for habit in habits:
            confidence_weight = {
                HabitConfidence.LOW: 0.3,
                HabitConfidence.MEDIUM: 0.6,
                HabitConfidence.HIGH: 0.9
            }.get(habit.confidence, 0.3)
            
            strength = habit.consistency_score * confidence_weight
            total_strength += strength
            total_weight += confidence_weight
        
        return total_strength / total_weight if total_weight > 0 else 0.0
    
    def _calculate_predictability_score(self, interactions: List[InteractionData]) -> float:
        """Calculate how predictable user behavior is."""
        if len(interactions) < 7:
            return 0.0
        
        # Analyze timing predictability
        timestamps = [
            interaction.timestamp for interaction in interactions
            if hasattr(interaction, 'timestamp') and interaction.timestamp
        ]
        
        if not timestamps:
            return 0.0
        
        timing_consistency = self._calculate_timing_consistency(timestamps)
        
        # Could add more predictability metrics here
        return timing_consistency
    
    def _calculate_analysis_confidence(self, habits: List[HabitPattern]) -> float:
        """Calculate overall confidence in habit analysis."""
        if not habits:
            return 0.0
        
        confidence_values = []
        for habit in habits:
            conf_value = {
                HabitConfidence.LOW: 0.3,
                HabitConfidence.MEDIUM: 0.6,
                HabitConfidence.HIGH: 0.9
            }.get(habit.confidence, 0.3)
            confidence_values.append(conf_value)
        
        return sum(confidence_values) / len(confidence_values)
    
    def _calculate_timing_consistency(self, timestamps: List[datetime]) -> float:
        """Calculate consistency of timing patterns."""
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate variance in hour-of-day
        hours = [ts.hour for ts in timestamps]
        if not hours:
            return 0.0
        
        mean_hour = sum(hours) / len(hours)
        variance = sum((hour - mean_hour) ** 2 for hour in hours) / len(hours)
        
        # Convert variance to consistency score (lower variance = higher consistency)
        max_variance = 144  # Maximum possible variance for hours (0-23)
        consistency = max(0.0, 1.0 - (variance / max_variance))
        
        return consistency
    
    def _extract_typical_times(self, timestamps: List[datetime]) -> List[time]:
        """Extract typical times from timestamps."""
        if not timestamps:
            return []
        
        hour_counter = Counter(ts.hour for ts in timestamps)
        most_common_hours = hour_counter.most_common(3)
        
        return [time(hour=hour) for hour, _ in most_common_hours]
    
    def _get_typical_days(self, timestamps: List[datetime]) -> List[str]:
        """Get typical days of week from timestamps."""
        if not timestamps:
            return []
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counter = Counter(ts.weekday() for ts in timestamps)
        most_common_days = weekday_counter.most_common(3)
        
        return [day_names[day] for day, _ in most_common_days]
    
    def _estimate_session_duration(self, interactions: List[InteractionData]) -> Optional[int]:
        """Estimate typical session duration in minutes."""
        durations = []
        
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'session_duration'):
                duration = interaction.conversation_context.session_duration
                if duration:
                    durations.append(duration.total_seconds() / 60)  # Convert to minutes
        
        if not durations:
            return None
        
        return int(sum(durations) / len(durations))
    
    def _determine_confidence(self, occurrence_count: int, consistency_score: float) -> HabitConfidence:
        """Determine confidence level based on occurrences and consistency."""
        if occurrence_count >= 20 and consistency_score >= 0.7:
            return HabitConfidence.HIGH
        elif occurrence_count >= 10 and consistency_score >= 0.5:
            return HabitConfidence.MEDIUM
        else:
            return HabitConfidence.LOW
    
    def _create_empty_analysis(
        self, user_id: str, analysis_period: timedelta
    ) -> HabitAnalysis:
        """Create empty habit analysis for insufficient data."""
        return HabitAnalysis(
            user_id=user_id,
            analysis_period=analysis_period,
            detected_habits=[],
            habit_clusters={},
            routine_strength=0.0,
            predictability_score=0.0,
            analysis_confidence=0.0
        )
    
    def get_user_habits(self, user_id: str) -> Optional[HabitAnalysis]:
        """Get stored habit analysis for a user."""
        return self.user_habits.get(user_id)
    
    def get_habits_by_type(
        self, user_id: str, habit_type: HabitType
    ) -> List[HabitPattern]:
        """Get habits of a specific type for a user."""
        analysis = self.user_habits.get(user_id)
        if not analysis:
            return []
        
        return analysis.habit_clusters.get(habit_type, [])
    
    def predict_next_interaction_time(self, user_id: str) -> Optional[datetime]:
        """Predict when user is likely to interact next."""
        analysis = self.user_habits.get(user_id)
        if not analysis or not analysis.detected_habits:
            return None
        
        # Find timing-based habits
        timing_habits = [
            habit for habit in analysis.detected_habits
            if habit.habit_type == HabitType.ROUTINE and habit.typical_times
        ]
        
        if not timing_habits:
            return None
        
        # Use the most confident timing habit
        best_habit = max(timing_habits, key=lambda h: h.consistency_score)
        
        # Predict next occurrence based on typical times
        now = datetime.utcnow()
        today = now.date()
        
        for typical_time in best_habit.typical_times:
            next_time = datetime.combine(today, typical_time)
            if next_time > now:
                return next_time
        
        # If no time today, try tomorrow
        tomorrow = today + timedelta(days=1)
        return datetime.combine(tomorrow, best_habit.typical_times[0])