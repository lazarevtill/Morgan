"""
Wellness Habit Tracking for Morgan RAG.

Tracks user wellness habits, health patterns, and provides personalized
wellness support and recommendations based on detected patterns.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .detector import HabitPattern, HabitType, HabitAnalysis
from ..intelligence.core.models import InteractionData, EmotionalState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WellnessCategory(Enum):
    """Categories of wellness tracking."""
    
    PHYSICAL_HEALTH = "physical_health"  # Exercise, movement, physical activity
    MENTAL_HEALTH = "mental_health"  # Stress, mood, mental wellbeing
    SLEEP = "sleep"  # Sleep patterns and quality
    NUTRITION = "nutrition"  # Eating habits and nutrition
    HYDRATION = "hydration"  # Water intake and hydration
    STRESS_MANAGEMENT = "stress_management"  # Stress relief activities
    SOCIAL_WELLNESS = "social_wellness"  # Social connections and relationships
    WORK_LIFE_BALANCE = "work_life_balance"  # Balance between work and personal life


class WellnessMetricType(Enum):
    """Types of wellness metrics."""
    
    FREQUENCY = "frequency"  # How often something occurs
    DURATION = "duration"  # How long something lasts
    INTENSITY = "intensity"  # How intense something is (1-10 scale)
    QUALITY = "quality"  # Quality rating (1-10 scale)
    BINARY = "binary"  # Yes/No or True/False
    COUNT = "count"  # Numerical count
    RATING = "rating"  # User rating (1-10 scale)


class WellnessGoalStatus(Enum):
    """Status of wellness goals."""
    
    ACTIVE = "active"
    ACHIEVED = "achieved"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    OVERDUE = "overdue"


@dataclass
class WellnessMetric:
    """Represents a wellness metric measurement."""
    
    metric_id: str
    user_id: str
    category: WellnessCategory
    metric_type: WellnessMetricType
    
    # Metric details
    name: str
    description: str
    value: Any  # The actual measurement value
    unit: Optional[str] = None  # Unit of measurement (e.g., "minutes", "glasses")
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Timing
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
    
    # Source
    source: str = "user_input"  # user_input, habit_detection, automatic
    confidence: float = 1.0  # Confidence in the measurement (0.0-1.0)


@dataclass
class WellnessGoal:
    """Represents a wellness goal."""
    
    goal_id: str
    user_id: str
    category: WellnessCategory
    
    # Goal details
    title: str
    description: str
    target_value: Any
    target_unit: Optional[str] = None
    
    # Timeline
    start_date: datetime = field(default_factory=datetime.utcnow)
    target_date: Optional[datetime] = None
    frequency: str = "daily"  # daily, weekly, monthly
    
    # Progress tracking
    current_value: Any = 0
    progress_percentage: float = 0.0
    status: WellnessGoalStatus = WellnessGoalStatus.ACTIVE
    
    # Motivation and context
    motivation: str = ""
    related_habits: List[str] = field(default_factory=list)  # Habit IDs
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    achieved_at: Optional[datetime] = None


@dataclass
class WellnessInsight:
    """Represents a wellness insight or recommendation."""
    
    insight_id: str
    user_id: str
    category: WellnessCategory
    
    # Insight content
    title: str
    message: str
    insight_type: str  # trend, recommendation, alert, achievement
    
    # Supporting data
    supporting_metrics: List[str] = field(default_factory=list)  # Metric IDs
    confidence: float = 0.5
    
    # Actionability
    actionable: bool = True
    suggested_actions: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float = 0.5  # How relevant this insight is (0.0-1.0)


@dataclass
class WellnessProfile:
    """Complete wellness profile for a user."""
    
    user_id: str
    
    # Current metrics
    current_metrics: Dict[WellnessCategory, List[WellnessMetric]] = field(default_factory=dict)
    
    # Goals
    active_goals: List[WellnessGoal] = field(default_factory=list)
    achieved_goals: List[WellnessGoal] = field(default_factory=list)
    
    # Insights and trends
    recent_insights: List[WellnessInsight] = field(default_factory=list)
    wellness_trends: Dict[WellnessCategory, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall wellness score
    overall_wellness_score: float = 0.5  # 0.0-1.0
    category_scores: Dict[WellnessCategory, float] = field(default_factory=dict)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    tracking_since: datetime = field(default_factory=datetime.utcnow)


class WellnessHabitTracker:
    """
    Tracks wellness habits and provides personalized wellness insights.
    
    Analyzes user wellness patterns, tracks goals, and provides actionable
    recommendations for improving overall wellbeing.
    """
    
    # Default wellness metrics to track
    DEFAULT_WELLNESS_METRICS = {
        WellnessCategory.PHYSICAL_HEALTH: [
            {"name": "Exercise Duration", "type": WellnessMetricType.DURATION, "unit": "minutes"},
            {"name": "Steps Taken", "type": WellnessMetricType.COUNT, "unit": "steps"},
            {"name": "Physical Activity", "type": WellnessMetricType.BINARY, "unit": None}
        ],
        WellnessCategory.MENTAL_HEALTH: [
            {"name": "Stress Level", "type": WellnessMetricType.RATING, "unit": "1-10"},
            {"name": "Mood Rating", "type": WellnessMetricType.RATING, "unit": "1-10"},
            {"name": "Meditation", "type": WellnessMetricType.DURATION, "unit": "minutes"}
        ],
        WellnessCategory.SLEEP: [
            {"name": "Sleep Duration", "type": WellnessMetricType.DURATION, "unit": "hours"},
            {"name": "Sleep Quality", "type": WellnessMetricType.RATING, "unit": "1-10"},
            {"name": "Bedtime", "type": WellnessMetricType.FREQUENCY, "unit": "time"}
        ],
        WellnessCategory.NUTRITION: [
            {"name": "Meals Eaten", "type": WellnessMetricType.COUNT, "unit": "meals"},
            {"name": "Nutrition Quality", "type": WellnessMetricType.RATING, "unit": "1-10"},
            {"name": "Healthy Snacks", "type": WellnessMetricType.COUNT, "unit": "snacks"}
        ],
        WellnessCategory.HYDRATION: [
            {"name": "Water Intake", "type": WellnessMetricType.COUNT, "unit": "glasses"},
            {"name": "Hydration Level", "type": WellnessMetricType.RATING, "unit": "1-10"}
        ]
    }
    
    # Wellness goal templates
    WELLNESS_GOAL_TEMPLATES = {
        WellnessCategory.PHYSICAL_HEALTH: [
            {"title": "Daily Exercise", "target": 30, "unit": "minutes", "frequency": "daily"},
            {"title": "Weekly Workouts", "target": 3, "unit": "sessions", "frequency": "weekly"},
            {"title": "Daily Steps", "target": 10000, "unit": "steps", "frequency": "daily"}
        ],
        WellnessCategory.SLEEP: [
            {"title": "Sleep Duration", "target": 8, "unit": "hours", "frequency": "daily"},
            {"title": "Consistent Bedtime", "target": "22:00", "unit": "time", "frequency": "daily"}
        ],
        WellnessCategory.HYDRATION: [
            {"title": "Daily Water Intake", "target": 8, "unit": "glasses", "frequency": "daily"}
        ]
    }
    
    def __init__(self):
        """Initialize wellness habit tracker."""
        self.user_profiles: Dict[str, WellnessProfile] = {}
        self.wellness_metrics: Dict[str, List[WellnessMetric]] = defaultdict(list)
        self.wellness_goals: Dict[str, List[WellnessGoal]] = defaultdict(list)
        logger.info("Wellness habit tracker initialized")
    
    def create_wellness_profile(
        self,
        user_id: str,
        habit_analysis: Optional[HabitAnalysis] = None
    ) -> WellnessProfile:
        """
        Create a wellness profile for a user.
        
        Args:
            user_id: User identifier
            habit_analysis: Optional habit analysis to inform profile creation
            
        Returns:
            WellnessProfile: Created wellness profile
        """
        logger.info(f"Creating wellness profile for user {user_id}")
        
        profile = WellnessProfile(user_id=user_id)
        
        # Initialize category scores
        for category in WellnessCategory:
            profile.category_scores[category] = 0.5  # Neutral starting score
        
        # Extract wellness habits if analysis provided
        if habit_analysis:
            wellness_habits = habit_analysis.habit_clusters.get(HabitType.WELLNESS, [])
            self._initialize_from_habits(profile, wellness_habits)
        
        # Store profile
        self.user_profiles[user_id] = profile
        
        logger.info(f"Created wellness profile for user {user_id}")
        return profile
    
    def track_wellness_metric(
        self,
        user_id: str,
        category: WellnessCategory,
        metric_name: str,
        value: Any,
        unit: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WellnessMetric:
        """Track a wellness metric for a user."""
        metric_id = f"{user_id}_{category.value}_{metric_name}_{datetime.now(timezone.utc).timestamp()}"
        
        metric = WellnessMetric(
            metric_id=metric_id,
            user_id=user_id,
            category=category,
            metric_type=self._determine_metric_type(value),
            name=metric_name,
            description=f"{metric_name} measurement",
            value=value,
            unit=unit,
            context=context or {}
        )
        
        # Store metric
        self.wellness_metrics[user_id].append(metric)
        
        # Update profile
        profile = self.user_profiles.get(user_id)
        if profile:
            if category not in profile.current_metrics:
                profile.current_metrics[category] = []
            profile.current_metrics[category].append(metric)
            profile.last_updated = datetime.now(timezone.utc)
            
            # Update category score
            self._update_category_score(profile, category)
        
        logger.debug(f"Tracked wellness metric: {metric_name} = {value} for user {user_id}")
        return metric
    
    def create_wellness_goal(
        self,
        user_id: str,
        category: WellnessCategory,
        title: str,
        target_value: Any,
        target_unit: Optional[str] = None,
        target_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> WellnessGoal:
        """Create a wellness goal for a user."""
        goal_id = f"{user_id}_{category.value}_{title.replace(' ', '_').lower()}_{datetime.now(timezone.utc).timestamp()}"
        
        goal = WellnessGoal(
            goal_id=goal_id,
            user_id=user_id,
            category=category,
            title=title,
            description=f"Goal to achieve {target_value} {target_unit or ''} {frequency}",
            target_value=target_value,
            target_unit=target_unit,
            target_date=target_date,
            frequency=frequency
        )
        
        # Store goal
        self.wellness_goals[user_id].append(goal)
        
        # Update profile
        profile = self.user_profiles.get(user_id)
        if profile:
            profile.active_goals.append(goal)
            profile.last_updated = datetime.now(timezone.utc)
        
        logger.info(f"Created wellness goal: {title} for user {user_id}")
        return goal
    
    def update_goal_progress(self, goal_id: str, current_value: Any) -> bool:
        """Update progress for a wellness goal."""
        for goals in self.wellness_goals.values():
            for goal in goals:
                if goal.goal_id == goal_id:
                    goal.current_value = current_value
                    goal.updated_at = datetime.now(timezone.utc)
                    
                    # Calculate progress percentage
                    if isinstance(goal.target_value, (int, float)) and isinstance(current_value, (int, float)):
                        goal.progress_percentage = min(100.0, (current_value / goal.target_value) * 100)
                    
                    # Check if goal is achieved
                    if self._is_goal_achieved(goal):
                        goal.status = WellnessGoalStatus.ACHIEVED
                        goal.achieved_at = datetime.now(timezone.utc)
                        
                        # Move to achieved goals in profile
                        profile = self.user_profiles.get(goal.user_id)
                        if profile:
                            if goal in profile.active_goals:
                                profile.active_goals.remove(goal)
                            profile.achieved_goals.append(goal)
                    
                    logger.info(f"Updated goal progress: {goal.title} = {current_value}")
                    return True
        
        return False
    
    def generate_wellness_insights(self, user_id: str) -> List[WellnessInsight]:
        """Generate wellness insights for a user."""
        logger.info(f"Generating wellness insights for user {user_id}")
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return []
        
        insights = []
        
        # Generate trend insights
        trend_insights = self._generate_trend_insights(user_id, profile)
        insights.extend(trend_insights)
        
        # Generate goal progress insights
        goal_insights = self._generate_goal_insights(user_id, profile)
        insights.extend(goal_insights)
        
        # Generate recommendation insights
        recommendation_insights = self._generate_recommendation_insights(user_id, profile)
        insights.extend(recommendation_insights)
        
        # Generate alert insights
        alert_insights = self._generate_alert_insights(user_id, profile)
        insights.extend(alert_insights)
        
        # Store insights in profile
        profile.recent_insights = insights
        profile.last_updated = datetime.now(timezone.utc)
        
        logger.info(f"Generated {len(insights)} wellness insights for user {user_id}")
        return insights
    
    def get_wellness_summary(self, user_id: str) -> Dict[str, Any]:
        """Get wellness summary for a user."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}
        
        # Calculate recent metrics
        recent_metrics = self._get_recent_metrics(user_id, days=7)
        
        # Calculate goal progress
        goal_progress = {
            "active_goals": len(profile.active_goals),
            "achieved_goals": len(profile.achieved_goals),
            "goals_on_track": len([g for g in profile.active_goals if g.progress_percentage >= 70]),
            "goals_behind": len([g for g in profile.active_goals if g.progress_percentage < 50])
        }
        
        # Calculate wellness trends
        trends = self._calculate_wellness_trends(user_id)
        
        summary = {
            "overall_wellness_score": profile.overall_wellness_score,
            "category_scores": profile.category_scores,
            "recent_metrics_count": len(recent_metrics),
            "goal_progress": goal_progress,
            "trends": trends,
            "recent_insights_count": len(profile.recent_insights),
            "tracking_duration": (datetime.now(timezone.utc) - profile.tracking_since).days,
            "last_updated": profile.last_updated
        }
        
        return summary
    
    def detect_wellness_habits_from_interactions(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[WellnessMetric]:
        """Detect wellness habits from user interactions."""
        logger.info(f"Detecting wellness habits from {len(interactions)} interactions for user {user_id}")
        
        detected_metrics = []
        
        # Analyze interaction content for wellness indicators
        for interaction in interactions:
            metrics = self._extract_wellness_from_interaction(user_id, interaction)
            detected_metrics.extend(metrics)
        
        # Store detected metrics
        for metric in detected_metrics:
            self.wellness_metrics[user_id].append(metric)
        
        logger.info(f"Detected {len(detected_metrics)} wellness metrics from interactions")
        return detected_metrics
    
    def _initialize_from_habits(self, profile: WellnessProfile, wellness_habits: List[HabitPattern]):
        """Initialize wellness profile from detected habits."""
        for habit in wellness_habits:
            # Create initial metrics based on habit
            category = self._map_habit_to_wellness_category(habit)
            if category:
                # Create a metric representing this habit
                metric = WellnessMetric(
                    metric_id=f"habit_{habit.habit_id}",
                    user_id=profile.user_id,
                    category=category,
                    metric_type=WellnessMetricType.FREQUENCY,
                    name=habit.name,
                    description=habit.description,
                    value=habit.consistency_score,
                    source="habit_detection",
                    confidence=0.8
                )
                
                if category not in profile.current_metrics:
                    profile.current_metrics[category] = []
                profile.current_metrics[category].append(metric)
    
    def _map_habit_to_wellness_category(self, habit: HabitPattern) -> Optional[WellnessCategory]:
        """Map a habit to a wellness category."""
        keyword_mapping = {
            WellnessCategory.PHYSICAL_HEALTH: ["exercise", "workout", "gym", "run", "walk", "fitness"],
            WellnessCategory.MENTAL_HEALTH: ["stress", "meditation", "mindfulness", "anxiety", "mood"],
            WellnessCategory.SLEEP: ["sleep", "rest", "tired", "bedtime", "wake"],
            WellnessCategory.NUTRITION: ["eat", "food", "meal", "nutrition", "diet"],
            WellnessCategory.HYDRATION: ["water", "drink", "hydration"],
            WellnessCategory.WORK_LIFE_BALANCE: ["work", "balance", "break", "vacation"]
        }
        
        for category, keywords in keyword_mapping.items():
            if any(keyword in habit.keywords for keyword in keywords):
                return category
        
        return None
    
    def _determine_metric_type(self, value: Any) -> WellnessMetricType:
        """Determine metric type based on value."""
        if isinstance(value, bool):
            return WellnessMetricType.BINARY
        elif isinstance(value, int):
            if 1 <= value <= 10:
                return WellnessMetricType.RATING
            else:
                return WellnessMetricType.COUNT
        elif isinstance(value, float):
            return WellnessMetricType.DURATION
        else:
            return WellnessMetricType.FREQUENCY
    
    def _update_category_score(self, profile: WellnessProfile, category: WellnessCategory):
        """Update wellness score for a category."""
        metrics = profile.current_metrics.get(category, [])
        if not metrics:
            return
        
        # Calculate score based on recent metrics
        recent_metrics = [m for m in metrics if (datetime.now(timezone.utc) - m.recorded_at).days <= 7]
        
        if not recent_metrics:
            return
        
        # Simple scoring based on metric values
        total_score = 0.0
        for metric in recent_metrics:
            if metric.metric_type == WellnessMetricType.RATING:
                total_score += metric.value / 10.0  # Normalize to 0-1
            elif metric.metric_type == WellnessMetricType.BINARY:
                total_score += 1.0 if metric.value else 0.0
            else:
                total_score += 0.5  # Neutral score for other types
        
        profile.category_scores[category] = total_score / len(recent_metrics)
        
        # Update overall wellness score
        profile.overall_wellness_score = sum(profile.category_scores.values()) / len(profile.category_scores)
    
    def _is_goal_achieved(self, goal: WellnessGoal) -> bool:
        """Check if a goal is achieved."""
        if isinstance(goal.target_value, (int, float)) and isinstance(goal.current_value, (int, float)):
            return goal.current_value >= goal.target_value
        elif isinstance(goal.target_value, str) and isinstance(goal.current_value, str):
            return goal.current_value == goal.target_value
        else:
            return False
    
    def _generate_trend_insights(self, user_id: str, profile: WellnessProfile) -> List[WellnessInsight]:
        """Generate trend-based insights."""
        insights = []
        
        for category, metrics in profile.current_metrics.items():
            if len(metrics) < 3:  # Need at least 3 data points for trends
                continue
            
            # Analyze trend in recent metrics
            recent_metrics = sorted(metrics, key=lambda m: m.recorded_at)[-7:]  # Last 7 metrics
            
            if len(recent_metrics) >= 3:
                trend = self._calculate_trend(recent_metrics)
                
                if trend > 0.1:  # Improving trend
                    insight = WellnessInsight(
                        insight_id=f"trend_positive_{user_id}_{category.value}_{datetime.now(timezone.utc).timestamp()}",
                        user_id=user_id,
                        category=category,
                        title=f"Improving {category.value.replace('_', ' ').title()}",
                        message=f"Your {category.value.replace('_', ' ')} has been improving over the past week! Keep up the great work.",
                        insight_type="trend",
                        confidence=0.8,
                        actionable=True,
                        suggested_actions=[f"Continue your current {category.value.replace('_', ' ')} routine"]
                    )
                    insights.append(insight)
                
                elif trend < -0.1:  # Declining trend
                    insight = WellnessInsight(
                        insight_id=f"trend_negative_{user_id}_{category.value}_{datetime.now(timezone.utc).timestamp()}",
                        user_id=user_id,
                        category=category,
                        title=f"Declining {category.value.replace('_', ' ').title()}",
                        message=f"Your {category.value.replace('_', ' ')} has been declining. Consider focusing more attention on this area.",
                        insight_type="trend",
                        confidence=0.7,
                        actionable=True,
                        suggested_actions=[f"Set a specific goal for {category.value.replace('_', ' ')}"]
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_goal_insights(self, user_id: str, profile: WellnessProfile) -> List[WellnessInsight]:
        """Generate goal-related insights."""
        insights = []
        
        for goal in profile.active_goals:
            if goal.progress_percentage >= 90:
                insight = WellnessInsight(
                    insight_id=f"goal_near_completion_{goal.goal_id}",
                    user_id=user_id,
                    category=goal.category,
                    title="Goal Almost Achieved!",
                    message=f"You're almost there! Only {100 - goal.progress_percentage:.1f}% left to achieve your goal: {goal.title}",
                    insight_type="achievement",
                    confidence=0.9,
                    actionable=True,
                    suggested_actions=["Keep pushing to complete this goal!"]
                )
                insights.append(insight)
            
            elif goal.progress_percentage < 30 and (datetime.now(timezone.utc) - goal.created_at).days > 7:
                insight = WellnessInsight(
                    insight_id=f"goal_behind_{goal.goal_id}",
                    user_id=user_id,
                    category=goal.category,
                    title="Goal Needs Attention",
                    message=f"Your goal '{goal.title}' is behind schedule. Consider adjusting your approach or breaking it into smaller steps.",
                    insight_type="alert",
                    confidence=0.8,
                    actionable=True,
                    suggested_actions=["Break the goal into smaller, manageable steps", "Set daily reminders"]
                )
                insights.append(insight)
        
        return insights
    
    def _generate_recommendation_insights(self, user_id: str, profile: WellnessProfile) -> List[WellnessInsight]:
        """Generate recommendation insights."""
        insights = []
        
        # Recommend goals for categories with low scores
        for category, score in profile.category_scores.items():
            if score < 0.4 and not any(g.category == category for g in profile.active_goals):
                templates = self.WELLNESS_GOAL_TEMPLATES.get(category, [])
                if templates:
                    template = templates[0]  # Use first template
                    insight = WellnessInsight(
                        insight_id=f"recommend_goal_{user_id}_{category.value}_{datetime.now(timezone.utc).timestamp()}",
                        user_id=user_id,
                        category=category,
                        title=f"Improve Your {category.value.replace('_', ' ').title()}",
                        message=f"Consider setting a goal for {category.value.replace('_', ' ')}. How about: {template['title']}?",
                        insight_type="recommendation",
                        confidence=0.6,
                        actionable=True,
                        suggested_actions=[f"Set a goal: {template['title']}"]
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_alert_insights(self, user_id: str, profile: WellnessProfile) -> List[WellnessInsight]:
        """Generate alert insights."""
        insights = []
        
        # Alert for very low wellness scores
        if profile.overall_wellness_score < 0.3:
            insight = WellnessInsight(
                insight_id=f"wellness_alert_{user_id}_{datetime.now(timezone.utc).timestamp()}",
                user_id=user_id,
                category=WellnessCategory.MENTAL_HEALTH,  # Default to mental health
                title="Wellness Needs Attention",
                message="Your overall wellness score is quite low. Consider focusing on self-care and reaching out for support if needed.",
                insight_type="alert",
                confidence=0.9,
                actionable=True,
                suggested_actions=["Schedule time for self-care", "Consider talking to a healthcare professional"]
            )
            insights.append(insight)
        
        return insights
    
    def _get_recent_metrics(self, user_id: str, days: int = 7) -> List[WellnessMetric]:
        """Get recent wellness metrics for a user."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            metric for metric in self.wellness_metrics[user_id]
            if metric.recorded_at >= cutoff_date
        ]
    
    def _calculate_wellness_trends(self, user_id: str) -> Dict[str, str]:
        """Calculate wellness trends for each category."""
        trends = {}
        
        for category in WellnessCategory:
            category_metrics = [
                m for m in self.wellness_metrics[user_id]
                if m.category == category and (datetime.now(timezone.utc) - m.recorded_at).days <= 14
            ]
            
            if len(category_metrics) >= 3:
                trend_value = self._calculate_trend(category_metrics)
                if trend_value > 0.1:
                    trends[category.value] = "improving"
                elif trend_value < -0.1:
                    trends[category.value] = "declining"
                else:
                    trends[category.value] = "stable"
            else:
                trends[category.value] = "insufficient_data"
        
        return trends
    
    def _calculate_trend(self, metrics: List[WellnessMetric]) -> float:
        """Calculate trend from a list of metrics."""
        if len(metrics) < 2:
            return 0.0
        
        # Sort by date
        sorted_metrics = sorted(metrics, key=lambda m: m.recorded_at)
        
        # Calculate simple trend (difference between first and last)
        try:
            first_value = float(sorted_metrics[0].value) if isinstance(sorted_metrics[0].value, (int, float)) else 0.5
            last_value = float(sorted_metrics[-1].value) if isinstance(sorted_metrics[-1].value, (int, float)) else 0.5
            
            # Normalize to -1 to 1 range
            return (last_value - first_value) / max(abs(first_value), abs(last_value), 1.0)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_wellness_from_interaction(
        self, user_id: str, interaction: InteractionData
    ) -> List[WellnessMetric]:
        """Extract wellness metrics from an interaction."""
        metrics = []
        
        # Extract text content
        text_content = ""
        if hasattr(interaction, 'user_message') and interaction.user_message:
            text_content = interaction.user_message.lower()
        
        # Look for wellness indicators in text
        wellness_indicators = {
            WellnessCategory.PHYSICAL_HEALTH: ["exercised", "workout", "gym", "ran", "walked"],
            WellnessCategory.MENTAL_HEALTH: ["stressed", "anxious", "happy", "sad", "mood"],
            WellnessCategory.SLEEP: ["slept", "tired", "exhausted", "rested"],
            WellnessCategory.NUTRITION: ["ate", "hungry", "meal", "food"],
            WellnessCategory.HYDRATION: ["water", "thirsty", "drink"]
        }
        
        for category, indicators in wellness_indicators.items():
            for indicator in indicators:
                if indicator in text_content:
                    metric = WellnessMetric(
                        metric_id=f"detected_{user_id}_{category.value}_{datetime.now(timezone.utc).timestamp()}",
                        user_id=user_id,
                        category=category,
                        metric_type=WellnessMetricType.BINARY,
                        name=f"Mentioned {indicator}",
                        description=f"User mentioned {indicator} in conversation",
                        value=True,
                        source="automatic",
                        confidence=0.6
                    )
                    metrics.append(metric)
                    break  # Only one metric per category per interaction
        
        return metrics
    
    def get_wellness_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get personalized wellness recommendations."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return []
        
        recommendations = []
        
        # Recommend based on low category scores
        for category, score in profile.category_scores.items():
            if score < 0.5:
                category_recommendations = self._get_category_recommendations(category)
                recommendations.extend(category_recommendations)
        
        # Recommend based on goal progress
        for goal in profile.active_goals:
            if goal.progress_percentage < 50:
                recommendations.append({
                    "type": "goal_support",
                    "category": goal.category.value,
                    "title": f"Boost Your {goal.title} Progress",
                    "description": f"You're at {goal.progress_percentage:.1f}% progress. Here are some tips to help you succeed.",
                    "actions": ["Set daily reminders", "Break into smaller steps", "Track daily progress"]
                })
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _get_category_recommendations(self, category: WellnessCategory) -> List[Dict[str, Any]]:
        """Get recommendations for a specific wellness category."""
        recommendations = {
            WellnessCategory.PHYSICAL_HEALTH: [
                {
                    "type": "activity",
                    "category": category.value,
                    "title": "Start with 10-Minute Walks",
                    "description": "Begin with short, daily walks to build a foundation for physical activity.",
                    "actions": ["Schedule daily 10-minute walks", "Track your steps", "Gradually increase duration"]
                }
            ],
            WellnessCategory.MENTAL_HEALTH: [
                {
                    "type": "mindfulness",
                    "category": category.value,
                    "title": "Try 5-Minute Meditation",
                    "description": "Start with short meditation sessions to improve mental wellbeing.",
                    "actions": ["Use a meditation app", "Set a daily reminder", "Find a quiet space"]
                }
            ],
            WellnessCategory.SLEEP: [
                {
                    "type": "sleep_hygiene",
                    "category": category.value,
                    "title": "Establish a Bedtime Routine",
                    "description": "Create a consistent bedtime routine to improve sleep quality.",
                    "actions": ["Set a consistent bedtime", "Avoid screens before bed", "Create a relaxing environment"]
                }
            ],
            WellnessCategory.HYDRATION: [
                {
                    "type": "hydration",
                    "category": category.value,
                    "title": "Increase Water Intake",
                    "description": "Aim to drink more water throughout the day for better hydration.",
                    "actions": ["Keep a water bottle nearby", "Set hourly reminders", "Track daily intake"]
                }
            ]
        }
        
        return recommendations.get(category, [])