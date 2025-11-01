"""
Companion experience metrics and user satisfaction tracking for Morgan RAG.

Provides specialized metrics for emotional intelligence, relationship building,
and companion experience quality as specified in the requirements.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import structlog

from .metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)


class EmotionType(Enum):
    """Supported emotion types for tracking."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class MilestoneType(Enum):
    """Types of relationship milestones."""
    FIRST_CONVERSATION = "first_conversation"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    GOAL_ACHIEVED = "goal_achieved"
    TRUST_MILESTONE = "trust_milestone"
    EMOTIONAL_SUPPORT = "emotional_support"
    LEARNING_MILESTONE = "learning_milestone"


@dataclass
class EmotionalInteraction:
    """Represents an emotional interaction with the user."""
    user_id: str
    detected_emotion: EmotionType
    emotion_intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    response_empathy_score: float  # 0.0 to 1.0
    user_satisfaction: Optional[float] = None  # 0.0 to 1.0, if provided
    interaction_duration: Optional[float] = None  # seconds
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipMetrics:
    """Metrics for a user's relationship with Morgan."""
    user_id: str
    total_interactions: int
    relationship_duration_days: int
    average_satisfaction: float
    emotional_support_count: int
    milestones_reached: List[MilestoneType]
    preferred_interaction_style: str
    trust_level: float  # 0.0 to 1.0
    engagement_score: float  # 0.0 to 1.0
    last_interaction: datetime
    emotional_pattern: Dict[EmotionType, int] = field(default_factory=dict)


@dataclass
class CompanionQualityMetrics:
    """Overall companion experience quality metrics."""
    total_users: int
    active_users_24h: int
    active_users_7d: int
    average_satisfaction_score: float
    empathy_accuracy_rate: float
    emotional_support_success_rate: float
    relationship_retention_rate: float
    milestone_completion_rate: float
    response_personalization_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class CompanionMetrics:
    """
    Companion experience metrics and user satisfaction tracking system.
    
    Tracks emotional intelligence effectiveness, relationship building progress,
    and overall companion experience quality.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
        # User relationship tracking
        self._user_relationships: Dict[str, RelationshipMetrics] = {}
        self._emotional_interactions: deque = deque(maxlen=10000)
        
        # Satisfaction tracking
        self._satisfaction_history: Dict[str, List[Tuple[float, datetime]]] = defaultdict(list)
        
        # Empathy and emotional intelligence tracking
        self._empathy_scores: deque = deque(maxlen=1000)
        self._emotion_detection_accuracy: Dict[str, List[bool]] = defaultdict(list)
        
        # Milestone tracking
        self._milestone_history: List[Tuple[str, MilestoneType, datetime]] = []
        
        logger.info("CompanionMetrics initialized")
    
    def record_emotional_interaction(self, user_id: str, detected_emotion: EmotionType,
                                   emotion_intensity: float, confidence: float,
                                   response_empathy_score: float,
                                   user_satisfaction: Optional[float] = None,
                                   interaction_duration: Optional[float] = None,
                                   context: Optional[Dict[str, Any]] = None):
        """Record an emotional interaction with a user."""
        interaction = EmotionalInteraction(
            user_id=user_id,
            detected_emotion=detected_emotion,
            emotion_intensity=emotion_intensity,
            confidence=confidence,
            response_empathy_score=response_empathy_score,
            user_satisfaction=user_satisfaction,
            interaction_duration=interaction_duration,
            context=context or {}
        )
        
        # Store interaction
        self._emotional_interactions.append(interaction)
        
        # Update user relationship metrics
        self._update_user_relationship(user_id, interaction)
        
        # Record in metrics collector
        self.metrics_collector.record_emotional_analysis_time(
            analysis_type="emotion_detection",
            duration=context.get("analysis_duration", 0) if context else 0
        )
        
        self.metrics_collector.record_empathy_score(
            response_type="emotional_response",
            empathy_score=response_empathy_score
        )
        
        if user_satisfaction is not None:
            self.metrics_collector.record_user_satisfaction(
                interaction_type="emotional_interaction",
                satisfaction_score=user_satisfaction
            )
            
            # Store satisfaction history
            self._satisfaction_history[user_id].append((user_satisfaction, datetime.now()))
        
        # Track empathy scores
        self._empathy_scores.append(response_empathy_score)
        
        logger.debug("Recorded emotional interaction",
                    user_id=user_id,
                    emotion=detected_emotion.value,
                    intensity=emotion_intensity,
                    empathy_score=response_empathy_score,
                    satisfaction=user_satisfaction)
    
    def record_relationship_milestone(self, user_id: str, milestone_type: MilestoneType,
                                    significance_score: float = 1.0,
                                    context: Optional[Dict[str, Any]] = None):
        """Record a relationship milestone achievement."""
        timestamp = datetime.now()
        
        # Store milestone
        self._milestone_history.append((user_id, milestone_type, timestamp))
        
        # Update user relationship
        if user_id in self._user_relationships:
            relationship = self._user_relationships[user_id]
            if milestone_type not in relationship.milestones_reached:
                relationship.milestones_reached.append(milestone_type)
                
                # Adjust trust level based on milestone
                trust_boost = {
                    MilestoneType.FIRST_CONVERSATION: 0.1,
                    MilestoneType.BREAKTHROUGH_MOMENT: 0.2,
                    MilestoneType.GOAL_ACHIEVED: 0.15,
                    MilestoneType.TRUST_MILESTONE: 0.25,
                    MilestoneType.EMOTIONAL_SUPPORT: 0.2,
                    MilestoneType.LEARNING_MILESTONE: 0.1
                }.get(milestone_type, 0.1)
                
                relationship.trust_level = min(1.0, relationship.trust_level + trust_boost)
        
        # Record in metrics collector
        self.metrics_collector.record_relationship_milestone(milestone_type.value)
        
        logger.info("Recorded relationship milestone",
                   user_id=user_id,
                   milestone_type=milestone_type.value,
                   significance_score=significance_score)
    
    def record_emotion_detection_feedback(self, user_id: str, detected_emotion: EmotionType,
                                        actual_emotion: EmotionType, confidence: float):
        """Record feedback on emotion detection accuracy."""
        is_correct = detected_emotion == actual_emotion
        
        # Store accuracy data
        emotion_key = f"{detected_emotion.value}_to_{actual_emotion.value}"
        self._emotion_detection_accuracy[emotion_key].append(is_correct)
        
        # Update user relationship trust based on accuracy
        if user_id in self._user_relationships:
            relationship = self._user_relationships[user_id]
            if is_correct:
                # Boost trust for accurate emotion detection
                relationship.trust_level = min(1.0, relationship.trust_level + 0.05)
            else:
                # Slight decrease for inaccurate detection
                relationship.trust_level = max(0.0, relationship.trust_level - 0.02)
        
        logger.debug("Recorded emotion detection feedback",
                    user_id=user_id,
                    detected=detected_emotion.value,
                    actual=actual_emotion.value,
                    correct=is_correct,
                    confidence=confidence)
    
    def record_personalization_effectiveness(self, user_id: str, personalization_score: float,
                                           interaction_type: str):
        """Record effectiveness of personalization."""
        self.metrics_collector.record_user_satisfaction(
            interaction_type=f"personalized_{interaction_type}",
            satisfaction_score=personalization_score
        )
        
        # Update user relationship engagement
        if user_id in self._user_relationships:
            relationship = self._user_relationships[user_id]
            # Update engagement score as moving average
            current_engagement = relationship.engagement_score
            relationship.engagement_score = (current_engagement * 0.8) + (personalization_score * 0.2)
        
        logger.debug("Recorded personalization effectiveness",
                    user_id=user_id,
                    score=personalization_score,
                    interaction_type=interaction_type)
    
    def _update_user_relationship(self, user_id: str, interaction: EmotionalInteraction):
        """Update user relationship metrics based on interaction."""
        if user_id not in self._user_relationships:
            # Create new relationship
            self._user_relationships[user_id] = RelationshipMetrics(
                user_id=user_id,
                total_interactions=0,
                relationship_duration_days=0,
                average_satisfaction=0.0,
                emotional_support_count=0,
                milestones_reached=[],
                preferred_interaction_style="adaptive",
                trust_level=0.5,  # Start with neutral trust
                engagement_score=0.5,  # Start with neutral engagement
                last_interaction=datetime.now(),
                emotional_pattern={}
            )
        
        relationship = self._user_relationships[user_id]
        
        # Update basic metrics
        relationship.total_interactions += 1
        relationship.last_interaction = interaction.timestamp
        
        # Update relationship duration
        if relationship.total_interactions == 1:
            relationship.relationship_duration_days = 0
        else:
            duration = (datetime.now() - relationship.last_interaction).days
            relationship.relationship_duration_days = max(relationship.relationship_duration_days, duration)
        
        # Update emotional pattern
        emotion = interaction.detected_emotion
        if emotion not in relationship.emotional_pattern:
            relationship.emotional_pattern[emotion] = 0
        relationship.emotional_pattern[emotion] += 1
        
        # Update satisfaction average
        if interaction.user_satisfaction is not None:
            current_avg = relationship.average_satisfaction
            total_interactions = relationship.total_interactions
            relationship.average_satisfaction = (
                (current_avg * (total_interactions - 1) + interaction.user_satisfaction) / total_interactions
            )
        
        # Count emotional support interactions
        if interaction.response_empathy_score > 0.7:  # High empathy threshold
            relationship.emotional_support_count += 1
        
        # Update engagement based on interaction quality
        interaction_quality = (
            interaction.response_empathy_score * 0.4 +
            (interaction.user_satisfaction or 0.5) * 0.4 +
            interaction.confidence * 0.2
        )
        relationship.engagement_score = (
            relationship.engagement_score * 0.9 + interaction_quality * 0.1
        )
    
    def get_user_relationship_metrics(self, user_id: str) -> Optional[RelationshipMetrics]:
        """Get relationship metrics for a specific user."""
        return self._user_relationships.get(user_id)
    
    def get_companion_quality_metrics(self, time_window: timedelta = timedelta(days=7)) -> CompanionQualityMetrics:
        """Get overall companion experience quality metrics."""
        cutoff_time = datetime.now() - time_window
        
        # Filter recent interactions
        recent_interactions = [
            interaction for interaction in self._emotional_interactions
            if interaction.timestamp >= cutoff_time
        ]
        
        # Calculate metrics
        total_users = len(self._user_relationships)
        
        # Active users
        active_users_24h = len(set(
            interaction.user_id for interaction in recent_interactions
            if interaction.timestamp >= datetime.now() - timedelta(days=1)
        ))
        
        active_users_7d = len(set(
            interaction.user_id for interaction in recent_interactions
        ))
        
        # Average satisfaction
        satisfaction_scores = [
            interaction.user_satisfaction for interaction in recent_interactions
            if interaction.user_satisfaction is not None
        ]
        average_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0
        
        # Empathy accuracy (based on recent empathy scores)
        recent_empathy_scores = list(self._empathy_scores)[-100:]  # Last 100 scores
        empathy_accuracy = sum(recent_empathy_scores) / len(recent_empathy_scores) if recent_empathy_scores else 0.0
        
        # Emotional support success rate (high empathy + high satisfaction)
        emotional_support_interactions = [
            interaction for interaction in recent_interactions
            if interaction.response_empathy_score > 0.7 and interaction.user_satisfaction is not None
        ]
        emotional_support_success = sum(
            1 for interaction in emotional_support_interactions
            if interaction.user_satisfaction > 0.7
        )
        emotional_support_success_rate = (
            emotional_support_success / len(emotional_support_interactions)
            if emotional_support_interactions else 0.0
        )
        
        # Relationship retention (users with interactions in last 7 days vs total)
        relationship_retention_rate = active_users_7d / total_users if total_users > 0 else 0.0
        
        # Milestone completion rate (users with milestones vs total)
        users_with_milestones = len([
            relationship for relationship in self._user_relationships.values()
            if relationship.milestones_reached
        ])
        milestone_completion_rate = users_with_milestones / total_users if total_users > 0 else 0.0
        
        # Response personalization score (average engagement score)
        engagement_scores = [
            relationship.engagement_score for relationship in self._user_relationships.values()
        ]
        response_personalization_score = (
            sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
        )
        
        return CompanionQualityMetrics(
            total_users=total_users,
            active_users_24h=active_users_24h,
            active_users_7d=active_users_7d,
            average_satisfaction_score=average_satisfaction,
            empathy_accuracy_rate=empathy_accuracy,
            emotional_support_success_rate=emotional_support_success_rate,
            relationship_retention_rate=relationship_retention_rate,
            milestone_completion_rate=milestone_completion_rate,
            response_personalization_score=response_personalization_score
        )
    
    def get_emotion_detection_accuracy(self) -> Dict[str, float]:
        """Get emotion detection accuracy statistics."""
        accuracy_stats = {}
        
        for emotion_key, results in self._emotion_detection_accuracy.items():
            if results:
                accuracy = sum(results) / len(results)
                accuracy_stats[emotion_key] = accuracy
        
        return accuracy_stats
    
    def get_user_emotional_patterns(self, user_id: str) -> Dict[EmotionType, float]:
        """Get emotional patterns for a specific user."""
        if user_id not in self._user_relationships:
            return {}
        
        relationship = self._user_relationships[user_id]
        total_interactions = sum(relationship.emotional_pattern.values())
        
        if total_interactions == 0:
            return {}
        
        return {
            emotion: count / total_interactions
            for emotion, count in relationship.emotional_pattern.items()
        }
    
    def get_satisfaction_trends(self, user_id: str, days: int = 30) -> List[Tuple[float, datetime]]:
        """Get satisfaction score trends for a user."""
        if user_id not in self._satisfaction_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        return [
            (score, timestamp) for score, timestamp in self._satisfaction_history[user_id]
            if timestamp >= cutoff_time
        ]
    
    def get_milestone_statistics(self, time_window: timedelta = timedelta(days=30)) -> Dict[MilestoneType, int]:
        """Get milestone achievement statistics."""
        cutoff_time = datetime.now() - time_window
        
        milestone_counts = defaultdict(int)
        for user_id, milestone_type, timestamp in self._milestone_history:
            if timestamp >= cutoff_time:
                milestone_counts[milestone_type] += 1
        
        return dict(milestone_counts)
    
    def generate_companion_insights(self) -> Dict[str, Any]:
        """Generate insights about companion performance and user relationships."""
        quality_metrics = self.get_companion_quality_metrics()
        emotion_accuracy = self.get_emotion_detection_accuracy()
        milestone_stats = self.get_milestone_statistics()
        
        # Calculate insights
        insights = {
            "overall_health": {
                "status": "healthy" if quality_metrics.average_satisfaction_score > 0.7 else "needs_attention",
                "satisfaction_score": quality_metrics.average_satisfaction_score,
                "empathy_accuracy": quality_metrics.empathy_accuracy_rate,
                "user_retention": quality_metrics.relationship_retention_rate
            },
            "user_engagement": {
                "total_users": quality_metrics.total_users,
                "active_users_24h": quality_metrics.active_users_24h,
                "active_users_7d": quality_metrics.active_users_7d,
                "engagement_trend": "growing" if quality_metrics.active_users_7d > quality_metrics.active_users_24h * 3 else "stable"
            },
            "emotional_intelligence": {
                "empathy_effectiveness": quality_metrics.empathy_accuracy_rate,
                "emotional_support_success": quality_metrics.emotional_support_success_rate,
                "emotion_detection_accuracy": emotion_accuracy,
                "improvement_areas": self._identify_emotion_improvement_areas(emotion_accuracy)
            },
            "relationship_building": {
                "milestone_completion_rate": quality_metrics.milestone_completion_rate,
                "recent_milestones": milestone_stats,
                "personalization_effectiveness": quality_metrics.response_personalization_score,
                "trust_building_success": self._calculate_trust_building_success()
            },
            "recommendations": self._generate_companion_recommendations(quality_metrics, emotion_accuracy)
        }
        
        return insights
    
    def _identify_emotion_improvement_areas(self, emotion_accuracy: Dict[str, float]) -> List[str]:
        """Identify areas where emotion detection needs improvement."""
        improvement_areas = []
        
        for emotion_key, accuracy in emotion_accuracy.items():
            if accuracy < 0.7:  # Below 70% accuracy threshold
                improvement_areas.append(f"Improve detection accuracy for {emotion_key}")
        
        return improvement_areas
    
    def _calculate_trust_building_success(self) -> float:
        """Calculate overall trust building success rate."""
        if not self._user_relationships:
            return 0.0
        
        trust_levels = [relationship.trust_level for relationship in self._user_relationships.values()]
        return sum(trust_levels) / len(trust_levels)
    
    def _generate_companion_recommendations(self, quality_metrics: CompanionQualityMetrics,
                                         emotion_accuracy: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving companion experience."""
        recommendations = []
        
        # Satisfaction recommendations
        if quality_metrics.average_satisfaction_score < 0.7:
            recommendations.append(
                "User satisfaction is below target (70%). Focus on improving response quality and personalization."
            )
        
        # Empathy recommendations
        if quality_metrics.empathy_accuracy_rate < 0.8:
            recommendations.append(
                "Empathy accuracy needs improvement. Consider refining emotional response generation algorithms."
            )
        
        # Retention recommendations
        if quality_metrics.relationship_retention_rate < 0.6:
            recommendations.append(
                "User retention is low. Implement more engaging conversation patterns and milestone celebrations."
            )
        
        # Emotional support recommendations
        if quality_metrics.emotional_support_success_rate < 0.7:
            recommendations.append(
                "Emotional support effectiveness is below target. Enhance empathetic response training."
            )
        
        # Personalization recommendations
        if quality_metrics.response_personalization_score < 0.7:
            recommendations.append(
                "Response personalization needs improvement. Enhance user preference learning and adaptation."
            )
        
        return recommendations