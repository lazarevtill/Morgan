"""
Relationship Timeline for Morgan Assistant.

Manages and visualizes the chronological history of user relationships,
tracking key events, milestones, and relationship progression over time.

Requirements: 9.4, 9.5, 10.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..emotional.models import (
    CompanionProfile,
    EmotionalState,
    InteractionData,
    RelationshipMilestone,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TimelineEventType(Enum):
    """Types of timeline events."""

    MILESTONE = "milestone"
    INTERACTION = "interaction"
    EMOTIONAL_SHIFT = "emotional_shift"
    TRUST_CHANGE = "trust_change"
    ENGAGEMENT_CHANGE = "engagement_change"
    PREFERENCE_UPDATE = "preference_update"
    RELATIONSHIP_STAGE = "relationship_stage"


@dataclass
class TimelineEvent:
    """Individual event in relationship timeline."""

    event_id: str
    event_type: TimelineEventType
    timestamp: datetime
    title: str
    description: str
    significance: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "description": self.description,
            "significance": self.significance,
            "metadata": self.metadata,
        }


@dataclass
class TimelinePeriod:
    """Period analysis in relationship timeline."""

    start_date: datetime
    end_date: datetime
    period_name: str
    event_count: int
    milestone_count: int
    average_significance: float
    dominant_themes: List[str]
    relationship_growth: float
    key_events: List[TimelineEvent]


@dataclass
class RelationshipSummary:
    """Summary of relationship development."""

    total_duration: timedelta
    total_interactions: int
    total_milestones: int
    relationship_stages: List[str]
    growth_trajectory: str  # "accelerating", "steady", "slowing"
    key_achievements: List[str]
    emotional_journey: Dict[str, Any]
    trust_progression: List[Tuple[datetime, float]]
    engagement_progression: List[Tuple[datetime, float]]


class RelationshipTimeline:
    """
    Manages chronological relationship history and progression tracking.

    Provides comprehensive timeline management, event tracking, and
    relationship development visualization.
    """

    def __init__(self):
        """Initialize relationship timeline manager."""
        self.timeline_cache = {}  # Cache timelines per user
        self.event_processors = self._initialize_event_processors()
        logger.info("Relationship timeline manager initialized")

    def build_timeline(
        self,
        user_profile: CompanionProfile,
        interaction_history: Optional[List[InteractionData]] = None,
        emotional_history: Optional[List[EmotionalState]] = None,
    ) -> List[TimelineEvent]:
        """
        Build comprehensive relationship timeline.

        Args:
            user_profile: User's companion profile
            interaction_history: Historical interaction data
            emotional_history: Historical emotional states

        Returns:
            Chronologically ordered list of timeline events
        """
        events = []

        # Add milestone events
        for milestone in user_profile.relationship_milestones:
            event = self._create_milestone_event(milestone, user_profile)
            events.append(event)

        # Add interaction events if history provided
        if interaction_history:
            interaction_events = self._process_interaction_history(
                interaction_history, user_profile
            )
            events.extend(interaction_events)

        # Add emotional shift events if history provided
        if emotional_history:
            emotional_events = self._process_emotional_history(
                emotional_history, user_profile
            )
            events.extend(emotional_events)

        # Add relationship stage progression events
        stage_events = self._infer_stage_progression(user_profile)
        events.extend(stage_events)

        # Sort chronologically
        events.sort(key=lambda e: e.timestamp)

        # Cache timeline
        self.timeline_cache[user_profile.user_id] = events

        logger.info(
            f"Built timeline with {len(events)} events for user {user_profile.user_id}"
        )

        return events

    def analyze_timeline_periods(
        self, timeline: List[TimelineEvent], period_length_days: int = 7
    ) -> List[TimelinePeriod]:
        """
        Analyze timeline in periods for trend identification.

        Args:
            timeline: Timeline events to analyze
            period_length_days: Length of each analysis period

        Returns:
            List of timeline periods with analysis
        """
        if not timeline:
            return []

        periods = []
        start_date = timeline[0].timestamp
        end_date = timeline[-1].timestamp

        current_date = start_date
        period_num = 1

        while current_date < end_date:
            period_end = current_date + timedelta(days=period_length_days)

            # Get events in this period
            period_events = [
                event
                for event in timeline
                if current_date <= event.timestamp < period_end
            ]

            if period_events:
                period = self._analyze_period(
                    period_events, current_date, period_end, period_num
                )
                periods.append(period)

            current_date = period_end
            period_num += 1

        logger.debug(f"Analyzed timeline into {len(periods)} periods")
        return periods

    def get_relationship_summary(
        self,
        user_profile: CompanionProfile,
        timeline: Optional[List[TimelineEvent]] = None,
    ) -> RelationshipSummary:
        """
        Generate comprehensive relationship summary.

        Args:
            user_profile: User's companion profile
            timeline: Timeline events (will build if not provided)

        Returns:
            Comprehensive relationship summary
        """
        if timeline is None:
            timeline = self.build_timeline(user_profile)

        # Calculate basic metrics
        total_duration = user_profile.relationship_duration
        total_interactions = user_profile.interaction_count
        total_milestones = len(user_profile.relationship_milestones)

        # Analyze relationship stages
        relationship_stages = self._extract_relationship_stages(timeline)

        # Determine growth trajectory
        growth_trajectory = self._analyze_growth_trajectory(timeline, user_profile)

        # Extract key achievements
        key_achievements = self._extract_key_achievements(timeline)

        # Analyze emotional journey
        emotional_journey = self._analyze_emotional_journey(timeline)

        # Build progression data
        trust_progression = self._build_trust_progression(timeline, user_profile)
        engagement_progression = self._build_engagement_progression(
            timeline, user_profile
        )

        summary = RelationshipSummary(
            total_duration=total_duration,
            total_interactions=total_interactions,
            total_milestones=total_milestones,
            relationship_stages=relationship_stages,
            growth_trajectory=growth_trajectory,
            key_achievements=key_achievements,
            emotional_journey=emotional_journey,
            trust_progression=trust_progression,
            engagement_progression=engagement_progression,
        )

        logger.info(f"Generated relationship summary for user {user_profile.user_id}")
        return summary

    def identify_significant_moments(
        self, timeline: List[TimelineEvent], significance_threshold: float = 0.7
    ) -> List[TimelineEvent]:
        """
        Identify most significant moments in relationship.

        Args:
            timeline: Timeline events to analyze
            significance_threshold: Minimum significance score

        Returns:
            List of significant timeline events
        """
        significant_events = [
            event for event in timeline if event.significance >= significance_threshold
        ]

        # Sort by significance
        significant_events.sort(key=lambda e: e.significance, reverse=True)

        logger.debug(
            f"Identified {len(significant_events)} significant moments "
            f"(threshold: {significance_threshold})"
        )

        return significant_events

    def predict_future_milestones(
        self,
        user_profile: CompanionProfile,
        timeline: Optional[List[TimelineEvent]] = None,
        prediction_horizon_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Predict likely future milestones based on timeline patterns.

        Args:
            user_profile: User's companion profile
            timeline: Timeline events for analysis
            prediction_horizon_days: How far ahead to predict

        Returns:
            List of predicted milestone opportunities
        """
        if timeline is None:
            timeline = self.build_timeline(user_profile)

        predictions = []

        # Analyze milestone patterns
        milestone_events = [
            e for e in timeline if e.event_type == TimelineEventType.MILESTONE
        ]

        if len(milestone_events) >= 2:
            # Calculate average time between milestones
            time_diffs = []
            for i in range(1, len(milestone_events)):
                diff = milestone_events[i].timestamp - milestone_events[i - 1].timestamp
                time_diffs.append(diff.days)

            avg_milestone_interval = sum(time_diffs) / len(time_diffs)

            # Predict next milestone timing
            last_milestone = milestone_events[-1]
            days_since_last = (datetime.utcnow() - last_milestone.timestamp).days

            if days_since_last >= avg_milestone_interval * 0.8:
                # Milestone is due soon
                predictions.append(
                    {
                        "type": "milestone_due",
                        "predicted_date": datetime.utcnow()
                        + timedelta(days=avg_milestone_interval - days_since_last),
                        "confidence": 0.7,
                        "description": "Next milestone expected based on historical pattern",
                        "suggested_focus": self._suggest_milestone_focus(user_profile),
                    }
                )

        # Predict based on relationship stage progression
        current_stage = self._infer_current_stage(user_profile)
        stage_predictions = self._predict_stage_progression(current_stage, user_profile)
        predictions.extend(stage_predictions)

        # Predict based on trust/engagement trends
        trend_predictions = self._predict_from_trends(timeline, user_profile)
        predictions.extend(trend_predictions)

        # Sort by confidence and filter by horizon
        predictions = [
            p
            for p in predictions
            if (p["predicted_date"] - datetime.utcnow()).days <= prediction_horizon_days
        ]
        predictions.sort(key=lambda p: p["confidence"], reverse=True)

        logger.info(f"Generated {len(predictions)} future milestone predictions")
        return predictions[:5]  # Top 5 predictions

    def export_timeline_data(
        self, timeline: List[TimelineEvent], format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Export timeline data in specified format.

        Args:
            timeline: Timeline events to export
            format_type: Export format ("json", "summary", "detailed")

        Returns:
            Formatted timeline data
        """
        if format_type == "json":
            return {
                "timeline": [event.to_dict() for event in timeline],
                "event_count": len(timeline),
                "date_range": {
                    "start": timeline[0].timestamp.isoformat() if timeline else None,
                    "end": timeline[-1].timestamp.isoformat() if timeline else None,
                },
                "exported_at": datetime.utcnow().isoformat(),
            }

        elif format_type == "summary":
            return self._create_timeline_summary(timeline)

        elif format_type == "detailed":
            return self._create_detailed_export(timeline)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _create_milestone_event(
        self, milestone: RelationshipMilestone, user_profile: CompanionProfile
    ) -> TimelineEvent:
        """Create timeline event from milestone."""
        return TimelineEvent(
            event_id=milestone.milestone_id,
            event_type=TimelineEventType.MILESTONE,
            timestamp=milestone.timestamp,
            title=f"Milestone: {milestone.milestone_type.value.replace('_', ' ').title()}",
            description=milestone.description,
            significance=milestone.emotional_significance,
            metadata={
                "milestone_type": milestone.milestone_type.value,
                "user_feedback": milestone.user_feedback,
                "celebration_acknowledged": milestone.celebration_acknowledged,
            },
            related_data=milestone,
        )

    def _process_interaction_history(
        self, interaction_history: List[InteractionData], user_profile: CompanionProfile
    ) -> List[TimelineEvent]:
        """Process interaction history into timeline events."""
        events = []

        # Group interactions by significance
        significant_interactions = [
            interaction
            for interaction in interaction_history
            if (interaction.user_satisfaction and interaction.user_satisfaction > 0.8)
            or len(interaction.conversation_context.message_text) > 200
            or interaction.emotional_state.intensity > 0.7
        ]

        for interaction in significant_interactions:
            event = TimelineEvent(
                event_id=f"interaction_{interaction.conversation_context.conversation_id}",
                event_type=TimelineEventType.INTERACTION,
                timestamp=interaction.conversation_context.timestamp,
                title="Significant Interaction",
                description=f"Meaningful conversation: {interaction.conversation_context.message_text[:100]}...",
                significance=min(0.6, (interaction.user_satisfaction or 0.5) * 0.8),
                metadata={
                    "message_length": len(
                        interaction.conversation_context.message_text
                    ),
                    "user_satisfaction": interaction.user_satisfaction,
                    "topics": interaction.topics_discussed,
                    "emotional_state": interaction.emotional_state.primary_emotion.value,
                },
                related_data=interaction,
            )
            events.append(event)

        return events

    def _process_emotional_history(
        self, emotional_history: List[EmotionalState], user_profile: CompanionProfile
    ) -> List[TimelineEvent]:
        """Process emotional history into timeline events."""
        events = []

        # Detect significant emotional shifts
        for i in range(1, len(emotional_history)):
            prev_emotion = emotional_history[i - 1]
            curr_emotion = emotional_history[i]

            # Detect major emotional shifts
            if (
                prev_emotion.primary_emotion != curr_emotion.primary_emotion
                and curr_emotion.intensity > 0.6
            ):

                event = TimelineEvent(
                    event_id=f"emotion_shift_{curr_emotion.timestamp.timestamp()}",
                    event_type=TimelineEventType.EMOTIONAL_SHIFT,
                    timestamp=curr_emotion.timestamp,
                    title=f"Emotional Shift: {curr_emotion.primary_emotion.value.title()}",
                    description=f"Emotional state changed from {prev_emotion.primary_emotion.value} to {curr_emotion.primary_emotion.value}",
                    significance=curr_emotion.intensity * 0.6,
                    metadata={
                        "previous_emotion": prev_emotion.primary_emotion.value,
                        "current_emotion": curr_emotion.primary_emotion.value,
                        "intensity": curr_emotion.intensity,
                        "confidence": curr_emotion.confidence,
                    },
                    related_data=curr_emotion,
                )
                events.append(event)

        return events

    def _infer_stage_progression(
        self, user_profile: CompanionProfile
    ) -> List[TimelineEvent]:
        """Infer relationship stage progression events."""
        events = []

        # Simple stage inference based on interaction count and trust
        stages = []

        if user_profile.interaction_count >= 1:
            stages.append(("initial", user_profile.profile_created, 0.5))

        if user_profile.interaction_count >= 5 and user_profile.trust_level >= 0.3:
            stage_date = user_profile.profile_created + timedelta(days=3)
            stages.append(("building", stage_date, 0.6))

        if user_profile.interaction_count >= 15 and user_profile.trust_level >= 0.5:
            stage_date = user_profile.profile_created + timedelta(days=10)
            stages.append(("established", stage_date, 0.7))

        if user_profile.interaction_count >= 30 and user_profile.trust_level >= 0.7:
            stage_date = user_profile.profile_created + timedelta(days=21)
            stages.append(("deep", stage_date, 0.8))

        for stage_name, stage_date, significance in stages:
            event = TimelineEvent(
                event_id=f"stage_{stage_name}_{user_profile.user_id}",
                event_type=TimelineEventType.RELATIONSHIP_STAGE,
                timestamp=stage_date,
                title=f"Relationship Stage: {stage_name.title()}",
                description=f"Relationship progressed to {stage_name} stage",
                significance=significance,
                metadata={"stage": stage_name},
                related_data=None,
            )
            events.append(event)

        return events

    def _analyze_period(
        self,
        period_events: List[TimelineEvent],
        start_date: datetime,
        end_date: datetime,
        period_num: int,
    ) -> TimelinePeriod:
        """Analyze a specific time period."""
        milestone_events = [
            e for e in period_events if e.event_type == TimelineEventType.MILESTONE
        ]

        # Calculate average significance
        avg_significance = (
            sum(e.significance for e in period_events) / len(period_events)
            if period_events
            else 0.0
        )

        # Identify dominant themes
        themes = []
        if milestone_events:
            themes.append("milestone_achievement")
        if any(
            e.event_type == TimelineEventType.EMOTIONAL_SHIFT for e in period_events
        ):
            themes.append("emotional_development")
        if any(e.event_type == TimelineEventType.INTERACTION for e in period_events):
            themes.append("active_engagement")

        # Calculate relationship growth (simplified)
        growth = len(milestone_events) * 0.3 + len(period_events) * 0.1

        # Select key events (top 3 by significance)
        key_events = sorted(period_events, key=lambda e: e.significance, reverse=True)[
            :3
        ]

        return TimelinePeriod(
            start_date=start_date,
            end_date=end_date,
            period_name=f"Period {period_num}",
            event_count=len(period_events),
            milestone_count=len(milestone_events),
            average_significance=avg_significance,
            dominant_themes=themes,
            relationship_growth=growth,
            key_events=key_events,
        )

    def _extract_relationship_stages(self, timeline: List[TimelineEvent]) -> List[str]:
        """Extract relationship stages from timeline."""
        stage_events = [
            e for e in timeline if e.event_type == TimelineEventType.RELATIONSHIP_STAGE
        ]
        return [e.metadata.get("stage", "unknown") for e in stage_events]

    def _analyze_growth_trajectory(
        self, timeline: List[TimelineEvent], user_profile: CompanionProfile
    ) -> str:
        """Analyze overall growth trajectory."""
        if len(timeline) < 3:
            return "emerging"

        # Analyze milestone frequency over time
        milestone_events = [
            e for e in timeline if e.event_type == TimelineEventType.MILESTONE
        ]

        if len(milestone_events) < 2:
            return "steady"

        # Compare first half vs second half milestone rates
        mid_point = len(milestone_events) // 2
        first_half = milestone_events[:mid_point]
        second_half = milestone_events[mid_point:]

        if not first_half or not second_half:
            return "steady"

        first_timespan = (first_half[-1].timestamp - first_half[0].timestamp).days or 1
        second_timespan = (
            second_half[-1].timestamp - second_half[0].timestamp
        ).days or 1

        first_rate = len(first_half) / first_timespan
        second_rate = len(second_half) / second_timespan

        if second_rate > first_rate * 1.3:
            return "accelerating"
        elif second_rate < first_rate * 0.7:
            return "slowing"
        else:
            return "steady"

    def _extract_key_achievements(self, timeline: List[TimelineEvent]) -> List[str]:
        """Extract key achievements from timeline."""
        achievements = []

        milestone_events = [
            e for e in timeline if e.event_type == TimelineEventType.MILESTONE
        ]
        significant_events = [e for e in timeline if e.significance >= 0.8]

        # Add milestone achievements
        for event in milestone_events:
            achievements.append(event.title)

        # Add other significant achievements
        for event in significant_events:
            if event.event_type != TimelineEventType.MILESTONE:
                achievements.append(event.title)

        return achievements[:10]  # Top 10 achievements

    def _analyze_emotional_journey(
        self, timeline: List[TimelineEvent]
    ) -> Dict[str, Any]:
        """Analyze emotional journey from timeline."""
        emotional_events = [
            e for e in timeline if e.event_type == TimelineEventType.EMOTIONAL_SHIFT
        ]

        if not emotional_events:
            return {
                "emotional_shifts": 0,
                "dominant_emotions": [],
                "emotional_stability": "stable",
            }

        emotions = []
        for event in emotional_events:
            if "current_emotion" in event.metadata:
                emotions.append(event.metadata["current_emotion"])

        # Count emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        dominant_emotions = sorted(
            emotion_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Assess emotional stability
        if len(emotional_events) > len(timeline) * 0.3:
            stability = "volatile"
        elif len(emotional_events) > len(timeline) * 0.1:
            stability = "dynamic"
        else:
            stability = "stable"

        return {
            "emotional_shifts": len(emotional_events),
            "dominant_emotions": [emotion for emotion, count in dominant_emotions],
            "emotional_stability": stability,
            "emotion_distribution": dict(dominant_emotions),
        }

    def _build_trust_progression(
        self, timeline: List[TimelineEvent], user_profile: CompanionProfile
    ) -> List[Tuple[datetime, float]]:
        """Build trust progression over time."""
        # Simplified trust progression based on milestones and current level
        progression = [(user_profile.profile_created, 0.0)]

        trust_milestones = [
            e
            for e in timeline
            if e.event_type == TimelineEventType.MILESTONE
            and "trust" in e.title.lower()
        ]

        current_trust = user_profile.trust_level
        trust_increment = current_trust / max(len(trust_milestones), 1)

        for i, milestone in enumerate(trust_milestones):
            trust_level = trust_increment * (i + 1)
            progression.append((milestone.timestamp, trust_level))

        # Add current trust level
        progression.append((datetime.utcnow(), current_trust))

        return progression

    def _build_engagement_progression(
        self, timeline: List[TimelineEvent], user_profile: CompanionProfile
    ) -> List[Tuple[datetime, float]]:
        """Build engagement progression over time."""
        # Simplified engagement progression
        progression = [(user_profile.profile_created, 0.0)]

        interaction_events = [
            e for e in timeline if e.event_type == TimelineEventType.INTERACTION
        ]

        current_engagement = user_profile.engagement_score
        engagement_increment = current_engagement / max(len(interaction_events), 1)

        for i, interaction in enumerate(interaction_events):
            engagement_level = engagement_increment * (i + 1)
            progression.append((interaction.timestamp, engagement_level))

        # Add current engagement level
        progression.append((datetime.utcnow(), current_engagement))

        return progression

    def _initialize_event_processors(self) -> Dict[str, Any]:
        """Initialize event processing configurations."""
        return {
            "milestone_significance_boost": 0.2,
            "interaction_significance_threshold": 0.5,
            "emotional_shift_intensity_threshold": 0.6,
            "trust_change_threshold": 0.1,
            "engagement_change_threshold": 0.1,
        }

    def _suggest_milestone_focus(self, user_profile: CompanionProfile) -> str:
        """Suggest focus area for next milestone."""
        existing_types = [
            m.milestone_type.value for m in user_profile.relationship_milestones
        ]

        if "trust_building" not in existing_types:
            return "Focus on trust-building conversations"
        elif "learning_milestone" not in existing_types:
            return "Create learning opportunities"
        elif "emotional_support" not in existing_types:
            return "Provide emotional support when needed"
        else:
            return "Continue meaningful conversations"

    def _infer_current_stage(self, user_profile: CompanionProfile) -> str:
        """Infer current relationship stage."""
        if user_profile.trust_level >= 0.8 and user_profile.interaction_count >= 30:
            return "deep"
        elif user_profile.trust_level >= 0.5 and user_profile.interaction_count >= 15:
            return "established"
        elif user_profile.interaction_count >= 5:
            return "building"
        else:
            return "initial"

    def _predict_stage_progression(
        self, current_stage: str, user_profile: CompanionProfile
    ) -> List[Dict[str, Any]]:
        """Predict stage progression opportunities."""
        predictions = []

        stage_requirements = {
            "initial": {"interactions": 5, "trust": 0.3, "next": "building"},
            "building": {"interactions": 15, "trust": 0.5, "next": "established"},
            "established": {"interactions": 30, "trust": 0.8, "next": "deep"},
        }

        if current_stage in stage_requirements:
            req = stage_requirements[current_stage]

            # Check if close to next stage
            interaction_progress = user_profile.interaction_count / req["interactions"]
            trust_progress = user_profile.trust_level / req["trust"]

            if interaction_progress >= 0.7 and trust_progress >= 0.7:
                days_to_stage = max(
                    1, int((1 - min(interaction_progress, trust_progress)) * 14)
                )

                predictions.append(
                    {
                        "type": "stage_progression",
                        "predicted_date": datetime.utcnow()
                        + timedelta(days=days_to_stage),
                        "confidence": min(interaction_progress, trust_progress),
                        "description": f"Progression to {req['next']} stage",
                        "suggested_focus": "Continue building trust and engagement",
                    }
                )

        return predictions

    def _predict_from_trends(
        self, timeline: List[TimelineEvent], user_profile: CompanionProfile
    ) -> List[Dict[str, Any]]:
        """Predict milestones from timeline trends."""
        predictions = []

        # Analyze recent activity trends
        recent_events = [
            e for e in timeline if (datetime.utcnow() - e.timestamp).days <= 14
        ]

        if len(recent_events) >= 3:
            avg_significance = sum(e.significance for e in recent_events) / len(
                recent_events
            )

            if avg_significance > 0.6:
                predictions.append(
                    {
                        "type": "high_engagement_milestone",
                        "predicted_date": datetime.utcnow() + timedelta(days=7),
                        "confidence": avg_significance,
                        "description": "High engagement trend suggests upcoming milestone",
                        "suggested_focus": "Maintain current engagement level",
                    }
                )

        return predictions

    def _create_timeline_summary(self, timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Create summary format of timeline."""
        return {
            "total_events": len(timeline),
            "event_types": {
                event_type.value: len(
                    [e for e in timeline if e.event_type == event_type]
                )
                for event_type in TimelineEventType
            },
            "date_range": {
                "start": timeline[0].timestamp.isoformat() if timeline else None,
                "end": timeline[-1].timestamp.isoformat() if timeline else None,
            },
            "average_significance": (
                sum(e.significance for e in timeline) / len(timeline) if timeline else 0
            ),
            "most_significant_events": [
                e.to_dict()
                for e in sorted(timeline, key=lambda x: x.significance, reverse=True)[
                    :5
                ]
            ],
        }

    def _create_detailed_export(self, timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Create detailed export format."""
        return {
            "timeline": [event.to_dict() for event in timeline],
            "summary": self._create_timeline_summary(timeline),
            "periods": [
                {
                    "start": period.start_date.isoformat(),
                    "end": period.end_date.isoformat(),
                    "name": period.period_name,
                    "event_count": period.event_count,
                    "themes": period.dominant_themes,
                }
                for period in self.analyze_timeline_periods(timeline)
            ],
            "exported_at": datetime.utcnow().isoformat(),
        }
