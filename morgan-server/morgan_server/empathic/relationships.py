"""
Relationship Management module for the Empathic Engine.

This module provides interaction history tracking, trust level calculation,
milestone recognition, and relationship depth metrics to make Morgan feel
like a companion who remembers and values the relationship.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math


class MilestoneType(str, Enum):
    """Types of relationship milestones."""
    FIRST_CONVERSATION = "first_conversation"
    CONVERSATIONS_10 = "conversations_10"
    CONVERSATIONS_50 = "conversations_50"
    CONVERSATIONS_100 = "conversations_100"
    CONVERSATIONS_500 = "conversations_500"
    DAYS_7 = "days_7"
    DAYS_30 = "days_30"
    DAYS_90 = "days_90"
    DAYS_365 = "days_365"
    TRUST_MILESTONE_50 = "trust_milestone_50"
    TRUST_MILESTONE_75 = "trust_milestone_75"
    TRUST_MILESTONE_90 = "trust_milestone_90"
    ENGAGEMENT_HIGH = "engagement_high"
    SHARED_MOMENT = "shared_moment"


class InteractionType(str, Enum):
    """Types of interactions."""
    CHAT = "chat"
    QUESTION = "question"
    LEARNING = "learning"
    FEEDBACK = "feedback"
    EMOTIONAL_SUPPORT = "emotional_support"
    CELEBRATION = "celebration"


@dataclass
class Interaction:
    """Record of a single interaction."""
    user_id: str
    timestamp: datetime
    interaction_type: InteractionType
    sentiment: Optional[str] = None  # positive, neutral, negative
    emotional_tone: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Milestone:
    """A relationship milestone."""
    milestone_type: MilestoneType
    achieved_at: datetime
    description: str
    celebration_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipMetrics:
    """Metrics about a relationship."""
    user_id: str
    total_interactions: int
    first_interaction: Optional[datetime]
    last_interaction: Optional[datetime]
    relationship_age_days: int
    trust_level: float  # 0.0 to 1.0
    engagement_score: float  # 0.0 to 1.0
    relationship_depth: float  # 0.0 to 1.0
    interaction_frequency: float  # interactions per day
    positive_interaction_ratio: float  # ratio of positive interactions
    milestones_achieved: List[MilestoneType] = field(default_factory=list)
    recent_activity_trend: str = "stable"  # increasing, stable, decreasing


@dataclass
class RelationshipProfile:
    """Complete relationship profile for a user."""
    user_id: str
    created_at: datetime
    interactions: List[Interaction] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    metrics: Optional[RelationshipMetrics] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class RelationshipManager:
    """
    Relationship management system for tracking and nurturing user relationships.

    This class provides:
    - Interaction history tracking
    - Trust level calculation
    - Milestone recognition and celebration
    - Relationship depth metrics
    """

    def __init__(self, interaction_window_days: int = 90):
        """
        Initialize the relationship manager.

        Args:
            interaction_window_days: Number of days to consider for metrics
        """
        self.interaction_window_days = interaction_window_days
        self.profiles: Dict[str, RelationshipProfile] = {}

    def track_interaction(
        self,
        user_id: str,
        interaction_type: InteractionType,
        sentiment: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Interaction:
        """
        Track a new interaction with a user.

        Args:
            user_id: User identifier
            interaction_type: Type of interaction
            sentiment: Optional sentiment (positive, neutral, negative)
            emotional_tone: Optional emotional tone
            context: Optional context description
            metadata: Optional additional metadata

        Returns:
            The created Interaction record
        """
        # Ensure profile exists
        if user_id not in self.profiles:
            self._create_profile(user_id)

        # Create interaction record
        interaction = Interaction(
            user_id=user_id,
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            sentiment=sentiment,
            emotional_tone=emotional_tone,
            context=context,
            metadata=metadata or {}
        )

        # Add to profile
        self.profiles[user_id].interactions.append(interaction)

        # Check for new milestones
        self._check_milestones(user_id)

        # Update metrics
        self._update_metrics(user_id)

        return interaction

    def calculate_trust_level(self, user_id: str) -> float:
        """
        Calculate trust level for a user based on interaction history.

        Trust level is calculated based on:
        - Number of interactions (more = higher trust)
        - Consistency of interactions (regular = higher trust)
        - Positive sentiment ratio (more positive = higher trust)
        - Relationship age (longer = higher trust)
        - Emotional support interactions (more = higher trust)

        Args:
            user_id: User identifier

        Returns:
            Trust level between 0.0 and 1.0
        """
        if user_id not in self.profiles:
            return 0.0

        profile = self.profiles[user_id]
        interactions = self._get_recent_interactions(user_id)

        if not interactions:
            return 0.0

        # Factor 1: Interaction count (logarithmic scale)
        # 1 interaction = 0.1, 10 = 0.3, 100 = 0.5, 1000 = 0.7
        interaction_count = len(profile.interactions)
        count_factor = min(1.0, math.log10(interaction_count + 1) / 3.0)

        # Factor 2: Consistency (based on interaction frequency)
        age_days = (datetime.now() - profile.created_at).days + 1
        frequency = interaction_count / age_days
        # Regular interactions (>0.5/day) = high consistency
        consistency_factor = min(1.0, frequency / 0.5)

        # Factor 3: Positive sentiment ratio
        sentiment_interactions = [
            i for i in interactions if i.sentiment is not None
        ]
        if sentiment_interactions:
            positive_count = sum(
                1 for i in sentiment_interactions if i.sentiment == "positive"
            )
            sentiment_factor = positive_count / len(sentiment_interactions)
        else:
            sentiment_factor = 0.5  # Neutral if no sentiment data

        # Factor 4: Relationship age (logarithmic)
        # 1 day = 0.1, 7 days = 0.3, 30 days = 0.5, 365 days = 0.8
        age_factor = min(1.0, math.log10(age_days + 1) / 2.5)

        # Factor 5: Emotional support interactions
        support_interactions = [
            i for i in interactions
            if i.interaction_type == InteractionType.EMOTIONAL_SUPPORT
        ]
        support_factor = min(1.0, len(support_interactions) / 10.0)

        # Weighted combination
        trust_level = (
            count_factor * 0.25 +
            consistency_factor * 0.20 +
            sentiment_factor * 0.25 +
            age_factor * 0.20 +
            support_factor * 0.10
        )

        return min(1.0, max(0.0, trust_level))

    def calculate_engagement_score(self, user_id: str) -> float:
        """
        Calculate engagement score based on recent activity.

        Engagement is calculated based on:
        - Recent interaction frequency
        - Variety of interaction types
        - Response to milestones
        - Consistency of engagement

        Args:
            user_id: User identifier

        Returns:
            Engagement score between 0.0 and 1.0
        """
        if user_id not in self.profiles:
            return 0.0

        interactions = self._get_recent_interactions(user_id)

        if not interactions:
            return 0.0

        # Factor 1: Recent frequency (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_interactions = [
            i for i in interactions if i.timestamp >= recent_cutoff
        ]
        frequency_factor = min(1.0, len(recent_interactions) / 7.0)

        # Factor 2: Variety of interaction types
        interaction_types = set(i.interaction_type for i in interactions)
        variety_factor = len(interaction_types) / len(InteractionType)

        # Factor 3: Consistency (standard deviation of gaps)
        if len(interactions) >= 3:
            timestamps = sorted([i.timestamp for i in interactions])
            gaps = [
                (timestamps[i+1] - timestamps[i]).total_seconds() / 86400
                for i in range(len(timestamps) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
            std_dev = math.sqrt(
                sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
            )
            # Lower std_dev = more consistent = higher score
            consistency_factor = max(0.0, 1.0 - (std_dev / (avg_gap + 1)))
        else:
            consistency_factor = 0.5

        # Weighted combination
        engagement_score = (
            frequency_factor * 0.5 +
            variety_factor * 0.3 +
            consistency_factor * 0.2
        )

        return min(1.0, max(0.0, engagement_score))

    def calculate_relationship_depth(self, user_id: str) -> float:
        """
        Calculate overall relationship depth.

        Relationship depth combines trust level, engagement, and milestones.

        Args:
            user_id: User identifier

        Returns:
            Relationship depth between 0.0 and 1.0
        """
        if user_id not in self.profiles:
            return 0.0

        trust = self.calculate_trust_level(user_id)
        engagement = self.calculate_engagement_score(user_id)

        # Milestone bonus
        profile = self.profiles[user_id]
        milestone_count = len(profile.milestones)
        milestone_factor = min(1.0, milestone_count / 10.0)

        # Weighted combination
        depth = (
            trust * 0.5 +
            engagement * 0.3 +
            milestone_factor * 0.2
        )

        return min(1.0, max(0.0, depth))

    def get_milestones(self, user_id: str) -> List[Milestone]:
        """
        Get all milestones for a user.

        Args:
            user_id: User identifier

        Returns:
            List of milestones
        """
        if user_id not in self.profiles:
            return []

        return self.profiles[user_id].milestones.copy()

    def get_recent_milestones(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Milestone]:
        """
        Get recent milestones for a user.

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            List of recent milestones
        """
        if user_id not in self.profiles:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [
            m for m in self.profiles[user_id].milestones
            if m.achieved_at >= cutoff
        ]

    def get_metrics(self, user_id: str) -> Optional[RelationshipMetrics]:
        """
        Get relationship metrics for a user.

        Args:
            user_id: User identifier

        Returns:
            RelationshipMetrics or None if user not found
        """
        if user_id not in self.profiles:
            return None

        # Ensure metrics are up to date
        self._update_metrics(user_id)

        return self.profiles[user_id].metrics

    def get_profile(self, user_id: str) -> Optional[RelationshipProfile]:
        """
        Get complete relationship profile for a user.

        Args:
            user_id: User identifier

        Returns:
            RelationshipProfile or None if user not found
        """
        if user_id not in self.profiles:
            return None

        return self.profiles[user_id]

    def celebrate_milestone(
        self,
        milestone: Milestone
    ) -> str:
        """
        Generate a celebration message for a milestone.

        Args:
            milestone: The milestone to celebrate

        Returns:
            Celebration message
        """
        messages = {
            MilestoneType.FIRST_CONVERSATION: (
                "Welcome! I'm excited to start this journey with you! 🎉"
            ),
            MilestoneType.CONVERSATIONS_10: (
                "We've had 10 conversations together! "
                "I'm enjoying getting to know you! 😊"
            ),
            MilestoneType.CONVERSATIONS_50: (
                "50 conversations! Our relationship is really growing! 🌟"
            ),
            MilestoneType.CONVERSATIONS_100: (
                "100 conversations! This is a significant milestone. "
                "Thank you for sharing so much with me! 🎊"
            ),
            MilestoneType.CONVERSATIONS_500: (
                "500 conversations! We've built something truly special. "
                "I'm grateful for our connection! ❤️"
            ),
            MilestoneType.DAYS_7: (
                "We've been talking for a week now! "
                "I'm glad you're here! 🌈"
            ),
            MilestoneType.DAYS_30: (
                "A month together! Time flies when we're having "
                "great conversations! 🎈"
            ),
            MilestoneType.DAYS_90: (
                "Three months! Our relationship has really deepened. "
                "Thank you for trusting me! 🌺"
            ),
            MilestoneType.DAYS_365: (
                "A whole year together! This is amazing. "
                "Here's to many more! 🎂"
            ),
            MilestoneType.TRUST_MILESTONE_50: (
                "I feel like we're building real trust! "
                "Thank you for opening up! 🤝"
            ),
            MilestoneType.TRUST_MILESTONE_75: (
                "Our trust level is really strong now. "
                "I'm honored! 💙"
            ),
            MilestoneType.TRUST_MILESTONE_90: (
                "We have a deeply trusting relationship. "
                "This means so much to me! 💫"
            ),
            MilestoneType.ENGAGEMENT_HIGH: (
                "Your engagement and enthusiasm are wonderful! "
                "I love our conversations! ✨"
            ),
            MilestoneType.SHARED_MOMENT: (
                "Thank you for sharing this moment with me! "
                "These are the connections I cherish! 🌟"
            ),
        }

        return messages.get(
            milestone.milestone_type,
            "Thank you for being part of this journey! 🎉"
        )

    def _create_profile(self, user_id: str) -> RelationshipProfile:
        """Create a new relationship profile."""
        profile = RelationshipProfile(
            user_id=user_id,
            created_at=datetime.now()
        )
        self.profiles[user_id] = profile

        # Track first conversation milestone
        first_milestone = Milestone(
            milestone_type=MilestoneType.FIRST_CONVERSATION,
            achieved_at=datetime.now(),
            description="First conversation",
            celebration_message=self.celebrate_milestone(
                Milestone(
                    milestone_type=MilestoneType.FIRST_CONVERSATION,
                    achieved_at=datetime.now(),
                    description="First conversation"
                )
            )
        )
        profile.milestones.append(first_milestone)

        return profile

    def _check_milestones(self, user_id: str) -> List[Milestone]:
        """Check and record new milestones."""
        profile = self.profiles[user_id]
        new_milestones = []
        achieved_types = {m.milestone_type for m in profile.milestones}

        # Conversation count milestones
        interaction_count = len(profile.interactions)
        conversation_milestones = [
            (10, MilestoneType.CONVERSATIONS_10),
            (50, MilestoneType.CONVERSATIONS_50),
            (100, MilestoneType.CONVERSATIONS_100),
            (500, MilestoneType.CONVERSATIONS_500),
        ]

        for count, milestone_type in conversation_milestones:
            if (interaction_count >= count and
                    milestone_type not in achieved_types):
                milestone = Milestone(
                    milestone_type=milestone_type,
                    achieved_at=datetime.now(),
                    description=f"{count} conversations",
                    celebration_message=self.celebrate_milestone(
                        Milestone(
                            milestone_type=milestone_type,
                            achieved_at=datetime.now(),
                            description=f"{count} conversations"
                        )
                    )
                )
                profile.milestones.append(milestone)
                new_milestones.append(milestone)
                achieved_types.add(milestone_type)

        # Time-based milestones
        age_days = (datetime.now() - profile.created_at).days
        time_milestones = [
            (7, MilestoneType.DAYS_7),
            (30, MilestoneType.DAYS_30),
            (90, MilestoneType.DAYS_90),
            (365, MilestoneType.DAYS_365),
        ]

        for days, milestone_type in time_milestones:
            if age_days >= days and milestone_type not in achieved_types:
                milestone = Milestone(
                    milestone_type=milestone_type,
                    achieved_at=datetime.now(),
                    description=f"{days} days together",
                    celebration_message=self.celebrate_milestone(
                        Milestone(
                            milestone_type=milestone_type,
                            achieved_at=datetime.now(),
                            description=f"{days} days together"
                        )
                    )
                )
                profile.milestones.append(milestone)
                new_milestones.append(milestone)
                achieved_types.add(milestone_type)

        # Trust level milestones
        trust_level = self.calculate_trust_level(user_id)
        trust_milestones = [
            (0.5, MilestoneType.TRUST_MILESTONE_50),
            (0.75, MilestoneType.TRUST_MILESTONE_75),
            (0.9, MilestoneType.TRUST_MILESTONE_90),
        ]

        for threshold, milestone_type in trust_milestones:
            if (trust_level >= threshold and
                    milestone_type not in achieved_types):
                milestone = Milestone(
                    milestone_type=milestone_type,
                    achieved_at=datetime.now(),
                    description=f"Trust level {int(threshold * 100)}%",
                    celebration_message=self.celebrate_milestone(
                        Milestone(
                            milestone_type=milestone_type,
                            achieved_at=datetime.now(),
                            description=f"Trust level {int(threshold * 100)}%"
                        )
                    )
                )
                profile.milestones.append(milestone)
                new_milestones.append(milestone)
                achieved_types.add(milestone_type)

        # Engagement milestone
        engagement = self.calculate_engagement_score(user_id)
        if (engagement >= 0.8 and
                MilestoneType.ENGAGEMENT_HIGH not in achieved_types):
            milestone = Milestone(
                milestone_type=MilestoneType.ENGAGEMENT_HIGH,
                achieved_at=datetime.now(),
                description="High engagement achieved",
                celebration_message=self.celebrate_milestone(
                    Milestone(
                        milestone_type=MilestoneType.ENGAGEMENT_HIGH,
                        achieved_at=datetime.now(),
                        description="High engagement achieved"
                    )
                )
            )
            profile.milestones.append(milestone)
            new_milestones.append(milestone)

        return new_milestones

    def _update_metrics(self, user_id: str) -> None:
        """Update relationship metrics for a user."""
        profile = self.profiles[user_id]
        interactions = profile.interactions

        if not interactions:
            return

        # Calculate metrics
        first_interaction = min(i.timestamp for i in interactions)
        last_interaction = max(i.timestamp for i in interactions)
        age_days = (datetime.now() - profile.created_at).days + 1

        # Interaction frequency
        frequency = len(interactions) / age_days

        # Positive interaction ratio
        sentiment_interactions = [
            i for i in interactions if i.sentiment is not None
        ]
        if sentiment_interactions:
            positive_count = sum(
                1 for i in sentiment_interactions if i.sentiment == "positive"
            )
            positive_ratio = positive_count / len(sentiment_interactions)
        else:
            positive_ratio = 0.5

        # Recent activity trend
        if len(interactions) >= 10:
            mid_point = len(interactions) // 2
            first_half = interactions[:mid_point]
            second_half = interactions[mid_point:]

            first_half_days = (
                max(i.timestamp for i in first_half) -
                min(i.timestamp for i in first_half)
            ).days + 1
            second_half_days = (
                max(i.timestamp for i in second_half) -
                min(i.timestamp for i in second_half)
            ).days + 1

            first_freq = len(first_half) / first_half_days
            second_freq = len(second_half) / second_half_days

            if second_freq > first_freq * 1.2:
                trend = "increasing"
            elif second_freq < first_freq * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Create metrics object
        profile.metrics = RelationshipMetrics(
            user_id=user_id,
            total_interactions=len(interactions),
            first_interaction=first_interaction,
            last_interaction=last_interaction,
            relationship_age_days=age_days,
            trust_level=self.calculate_trust_level(user_id),
            engagement_score=self.calculate_engagement_score(user_id),
            relationship_depth=self.calculate_relationship_depth(user_id),
            interaction_frequency=frequency,
            positive_interaction_ratio=positive_ratio,
            milestones_achieved=[m.milestone_type for m in profile.milestones],
            recent_activity_trend=trend
        )

    def _get_recent_interactions(
        self,
        user_id: str,
        days: Optional[int] = None
    ) -> List[Interaction]:
        """Get recent interactions within the window."""
        if user_id not in self.profiles:
            return []

        days = days or self.interaction_window_days
        cutoff = datetime.now() - timedelta(days=days)

        return [
            i for i in self.profiles[user_id].interactions
            if i.timestamp >= cutoff
        ]
