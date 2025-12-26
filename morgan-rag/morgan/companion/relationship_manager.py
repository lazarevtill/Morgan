"""
Companion Relationship Manager.

Builds and maintains meaningful relationships with users over time through
profile building, conversation adaptation, milestone tracking, and personalized
interactions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from morgan.intelligence.core.models import (
    CommunicationStyle,
    CompanionProfile,
    ConversationContext,
    ConversationStyle,
    ConversationTopic,
    EmotionalState,
    InteractionData,
    MilestoneType,
    PersonalizedGreeting,
    RelationshipMilestone,
    ResponseLength,
    UserPreferences,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Interaction:
    """Represents a single user interaction."""

    interaction_id: str
    user_id: str
    timestamp: datetime
    message_content: str
    emotional_state: Optional[EmotionalState] = None
    topics_discussed: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None
    duration: Optional[timedelta] = None


class CompanionRelationshipManager:
    """
    Manages companion relationships with users.

    Provides functionality for building user profiles, adapting conversation
    styles, tracking relationship milestones, and generating personalized
    interactions.
    """

    def __init__(self):
        """Initialize the companion relationship manager."""
        self.profiles: Dict[str, CompanionProfile] = {}
        self.interaction_history: Dict[str, List[Interaction]] = {}

    def build_user_profile(
        self, user_id: str, interactions: List[Interaction]
    ) -> CompanionProfile:
        """
        Build or update a user profile based on interactions.

        Args:
            user_id: Unique identifier for the user
            interactions: List of user interactions to analyze

        Returns:
            CompanionProfile: Updated user profile
        """
        logger.info(f"Building user profile for user {user_id}")

        # Get existing profile or create new one
        existing_profile = self.profiles.get(user_id)

        if existing_profile:
            profile = existing_profile
            # Update interaction count and duration
            profile.interaction_count += len(interactions)
            if interactions:
                profile.last_interaction = max(
                    interaction.timestamp for interaction in interactions
                )
                # Update relationship duration
                profile.relationship_duration = (
                    profile.last_interaction - profile.profile_created
                )
        else:
            # Create new profile
            if not interactions:
                logger.warning(f"No interactions provided for new user {user_id}")
                interactions = []

            first_interaction = (
                min(interactions, key=lambda x: x.timestamp).timestamp
                if interactions
                else datetime.utcnow()
            )
            last_interaction = (
                max(interactions, key=lambda x: x.timestamp).timestamp
                if interactions
                else datetime.utcnow()
            )

            profile = CompanionProfile(
                user_id=user_id,
                relationship_duration=last_interaction - first_interaction,
                interaction_count=len(interactions),
                preferred_name=self._extract_preferred_name(interactions),
                communication_preferences=self._analyze_communication_preferences(
                    interactions
                ),
                last_interaction=last_interaction,
                profile_created=first_interaction,
            )

        # Update profile with new interaction data
        profile = self._update_profile_from_interactions(profile, interactions)

        # Re-extract preferred name from all interactions (including new ones)
        all_interactions = self.interaction_history.get(user_id, []) + interactions
        new_preferred_name = self._extract_preferred_name(all_interactions)
        if new_preferred_name != user_id and new_preferred_name != "Friend":
            profile.preferred_name = new_preferred_name

        # Store updated profile
        self.profiles[user_id] = profile

        # Update interaction history
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        self.interaction_history[user_id].extend(interactions)

        logger.info(
            f"Profile updated for user {user_id}: "
            f"{profile.interaction_count} interactions, "
            f"{profile.relationship_duration.days} days relationship"
        )

        return profile

    def adapt_conversation_style(
        self, user_profile: CompanionProfile, current_mood: EmotionalState
    ) -> ConversationStyle:
        """
        Adapt conversation style based on user profile and current mood.

        Args:
            user_profile: User's companion profile
            current_mood: User's current emotional state

        Returns:
            ConversationStyle: Adapted conversation style
        """
        logger.debug(f"Adapting conversation style for user {user_profile.user_id}")

        # Base style from user preferences
        prefs = user_profile.communication_preferences

        # Adjust formality based on communication style and relationship age
        formality_base = {
            CommunicationStyle.FORMAL: 0.8,
            CommunicationStyle.PROFESSIONAL: 0.7,
            CommunicationStyle.TECHNICAL: 0.6,
            CommunicationStyle.FRIENDLY: 0.3,
            CommunicationStyle.CASUAL: 0.1,
        }.get(prefs.communication_style, 0.5)

        # Reduce formality as relationship develops
        relationship_factor = min(user_profile.get_relationship_age_days() / 30, 1.0)
        formality_level = formality_base * (1 - relationship_factor * 0.3)

        # Adjust technical depth based on preferences and topics
        technical_depth = 0.5  # Default
        if "technical" in [topic.lower() for topic in prefs.topics_of_interest]:
            technical_depth = 0.8
        elif prefs.communication_style == CommunicationStyle.TECHNICAL:
            technical_depth = 0.9

        # Adjust empathy based on emotional state and trust level
        empathy_base = 0.6
        if current_mood.primary_emotion.value in ["sadness", "fear", "anger"]:
            empathy_base = 0.9
        elif current_mood.primary_emotion.value == "joy":
            empathy_base = 0.4

        empathy_emphasis = min(empathy_base + user_profile.trust_level * 0.3, 1.0)

        # Determine personality traits based on profile
        personality_traits = []
        if user_profile.trust_level > 0.7:
            personality_traits.append("warm")
        if user_profile.engagement_score > 0.8:
            personality_traits.append("enthusiastic")
        if len(user_profile.relationship_milestones) > 3:
            personality_traits.append("supportive")

        style = ConversationStyle(
            formality_level=formality_level,
            technical_depth=technical_depth,
            empathy_emphasis=empathy_emphasis,
            response_length_target=prefs.preferred_response_length,
            personality_traits=personality_traits,
            adaptation_confidence=min(user_profile.interaction_count / 10, 1.0),
        )

        logger.debug(
            f"Conversation style adapted: formality={style.formality_level:.2f}, "
            f"empathy={style.empathy_emphasis:.2f}, "
            f"technical={style.technical_depth:.2f}"
        )

        return style

    def track_relationship_milestones(
        self, user_id: str, milestone_type: str
    ) -> RelationshipMilestone:
        """
        Track and create relationship milestones.

        Args:
            user_id: User identifier
            milestone_type: Type of milestone to track

        Returns:
            RelationshipMilestone: Created milestone
        """
        logger.info(f"Tracking milestone '{milestone_type}' for user {user_id}")

        profile = self.profiles.get(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return None

        # Convert string to MilestoneType enum
        try:
            milestone_enum = MilestoneType(milestone_type.lower())
        except ValueError:
            logger.error(f"Invalid milestone type: {milestone_type}")
            return None

        # Generate milestone description and significance
        description, significance = self._generate_milestone_details(
            milestone_enum, profile
        )

        milestone = RelationshipMilestone(
            milestone_id=str(uuid.uuid4()),
            milestone_type=milestone_enum,
            description=description,
            timestamp=datetime.utcnow(),
            emotional_significance=significance,
        )

        # Add milestone to profile
        profile.add_milestone(milestone)

        # Update trust and engagement based on milestone
        self._update_profile_from_milestone(profile, milestone)

        logger.info(
            f"Milestone created: {milestone.description} "
            f"(significance: {milestone.emotional_significance:.2f})"
        )

        return milestone

    def generate_personalized_greeting(
        self, user_profile: CompanionProfile, time_since_last_interaction: timedelta
    ) -> PersonalizedGreeting:
        """
        Generate a personalized greeting based on user profile and time gap.

        Args:
            user_profile: User's companion profile
            time_since_last_interaction: Time since last interaction

        Returns:
            PersonalizedGreeting: Personalized greeting
        """
        logger.debug(
            f"Generating personalized greeting for user {user_profile.user_id}"
        )

        context_elements = []
        personalization_level = 0.0

        # Base greeting based on time gap
        hours_since = time_since_last_interaction.total_seconds() / 3600

        if hours_since < 1:
            base_greeting = "Welcome back"
            time_awareness = False
        elif hours_since < 24:
            base_greeting = "Good to see you again"
            time_awareness = True
            context_elements.append("same_day_return")
        elif hours_since < 168:  # 1 week
            base_greeting = "It's great to have you back"
            time_awareness = True
            context_elements.append("weekly_return")
        else:
            base_greeting = "Welcome back! It's been a while"
            time_awareness = True
            context_elements.append("long_absence")

        # Personalize with name if available
        if (
            user_profile.preferred_name
            and user_profile.preferred_name != user_profile.user_id
        ):
            base_greeting += f", {user_profile.preferred_name}"
            personalization_level += 0.3
            context_elements.append("personal_name")
        elif user_profile.preferred_name and user_profile.preferred_name != "Friend":
            base_greeting += f", {user_profile.preferred_name}"
            personalization_level += 0.3
            context_elements.append("personal_name")

        # Add relationship context for established relationships
        relationship_reference = False
        if user_profile.get_relationship_age_days() > 7:
            if user_profile.trust_level > 0.5:
                base_greeting += "! I've been looking forward to our conversation"
                relationship_reference = True
                personalization_level += 0.4
                context_elements.append("established_relationship")

        # Add recent milestone reference
        recent_milestones = [
            m
            for m in user_profile.relationship_milestones
            if (datetime.utcnow() - m.timestamp).days < 7
        ]
        if recent_milestones and not recent_milestones[-1].celebration_acknowledged:
            milestone = recent_milestones[-1]
            base_greeting += f". I hope you're still feeling good about {milestone.description.lower()}"
            personalization_level += 0.3
            context_elements.append("recent_milestone")
            # Mark as acknowledged
            milestone.celebration_acknowledged = True

        greeting = PersonalizedGreeting(
            greeting_text=base_greeting,
            personalization_level=min(personalization_level, 1.0),
            context_elements=context_elements,
            time_awareness=time_awareness,
            relationship_reference=relationship_reference,
        )

        logger.debug(
            f"Generated greeting with personalization level: "
            f"{greeting.personalization_level:.2f}"
        )

        return greeting

    def suggest_conversation_topics(
        self, user_interests: List[str], recent_context: ConversationContext
    ) -> List[ConversationTopic]:
        """
        Suggest conversation topics based on user interests and recent context.

        Args:
            user_interests: List of user's interests
            recent_context: Recent conversation context

        Returns:
            List[ConversationTopic]: Suggested conversation topics
        """
        logger.debug("Generating conversation topic suggestions")

        topics = []

        # Interest-based topics
        for interest in user_interests[:3]:  # Top 3 interests
            topic = ConversationTopic(
                topic=f"Let's explore more about {interest}",
                relevance_score=0.8,
                category="interest_based",
                reasoning=f"Based on your interest in {interest}",
                user_interest_match=1.0,
            )
            topics.append(topic)

        # Context-based follow-up topics
        if recent_context.previous_messages:
            last_message = recent_context.previous_messages[-1]
            if len(last_message) > 50:  # Substantial previous message
                topic = ConversationTopic(
                    topic="Would you like to dive deeper into what we just discussed?",
                    relevance_score=0.9,
                    category="context_followup",
                    reasoning="Following up on recent conversation",
                    user_interest_match=0.7,
                )
                topics.append(topic)

        # Learning and growth topics
        learning_topic = ConversationTopic(
            topic="Is there something new you'd like to learn about today?",
            relevance_score=0.6,
            category="learning_growth",
            reasoning="Encouraging continuous learning",
            user_interest_match=0.5,
        )
        topics.append(learning_topic)

        # Personal check-in topic
        personal_topic = ConversationTopic(
            topic="How are you feeling about your recent projects or goals?",
            relevance_score=0.7,
            category="personal_checkin",
            reasoning="Maintaining personal connection",
            user_interest_match=0.6,
        )
        topics.append(personal_topic)

        # Sort by relevance score
        topics.sort(key=lambda t: t.relevance_score, reverse=True)

        logger.debug(f"Generated {len(topics)} conversation topic suggestions")

        return topics[:5]  # Return top 5 suggestions

    def _extract_preferred_name(self, interactions: List[Interaction]) -> str:
        """Extract preferred name from interactions."""
        # Simple implementation - look for "call me" patterns
        for interaction in interactions:
            content = interaction.message_content.lower()
            if "call me" in content:
                # Extract name after "call me"
                parts = content.split("call me")
                if len(parts) > 1:
                    name_part = parts[1].strip()
                    # Remove punctuation and get first word
                    name = name_part.replace(".", "").replace(",", "").split()[0]
                    if name and name.isalpha():
                        return name.title()

        # Default to user_id if no preferred name found
        return interactions[0].user_id if interactions else "Friend"

    def _analyze_communication_preferences(
        self, interactions: List[Interaction]
    ) -> UserPreferences:
        """Analyze communication preferences from interactions."""
        topics = []
        total_length = 0
        formal_indicators = 0

        for interaction in interactions:
            # Extract topics
            topics.extend(interaction.topics_discussed)

            # Analyze message length and formality
            content = interaction.message_content
            total_length += len(content)

            # Simple formality detection
            if any(
                word in content.lower()
                for word in ["please", "thank you", "could you", "would you"]
            ):
                formal_indicators += 1

        # Determine communication style
        avg_length = total_length / len(interactions) if interactions else 0
        formality_ratio = formal_indicators / len(interactions) if interactions else 0

        if formality_ratio > 0.5:
            comm_style = CommunicationStyle.FORMAL
        elif avg_length > 200:
            comm_style = CommunicationStyle.TECHNICAL
        else:
            comm_style = CommunicationStyle.FRIENDLY

        # Determine response length preference
        if avg_length > 300:
            response_length = ResponseLength.COMPREHENSIVE
        elif avg_length > 100:
            response_length = ResponseLength.DETAILED
        else:
            response_length = ResponseLength.BRIEF

        return UserPreferences(
            topics_of_interest=list(set(topics)),
            communication_style=comm_style,
            preferred_response_length=response_length,
        )

    def _update_profile_from_interactions(
        self, profile: CompanionProfile, interactions: List[Interaction]
    ) -> CompanionProfile:
        """Update profile based on new interactions."""
        # Update emotional patterns
        emotions = []
        satisfaction_scores = []

        for interaction in interactions:
            if interaction.emotional_state:
                emotions.append(interaction.emotional_state.primary_emotion.value)
            if interaction.user_satisfaction:
                satisfaction_scores.append(interaction.user_satisfaction)

        if emotions:
            profile.emotional_patterns["recent_emotions"] = emotions[-10:]  # Last 10

        # Update engagement score based on satisfaction
        if satisfaction_scores:
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
            profile.engagement_score = (
                profile.engagement_score * 0.7 + avg_satisfaction * 0.3
            )

        # Update trust level based on interaction frequency and satisfaction
        if profile.interaction_count > 0:
            trust_boost = min(len(interactions) * 0.05, 0.2)
            if (
                satisfaction_scores
                and sum(satisfaction_scores) / len(satisfaction_scores) > 0.7
            ):
                trust_boost *= 1.5
            profile.trust_level = min(profile.trust_level + trust_boost, 1.0)

        return profile

    def _generate_milestone_details(
        self, milestone_type: MilestoneType, profile: CompanionProfile
    ) -> tuple[str, float]:
        """Generate milestone description and emotional significance."""
        descriptions = {
            MilestoneType.FIRST_CONVERSATION: (
                "Your first conversation with Morgan",
                0.8,
            ),
            MilestoneType.BREAKTHROUGH_MOMENT: (
                "A breakthrough moment in our conversation",
                0.9,
            ),
            MilestoneType.GOAL_ACHIEVED: ("Achieving an important goal together", 0.9),
            MilestoneType.LEARNING_MILESTONE: (
                "A significant learning achievement",
                0.7,
            ),
            MilestoneType.EMOTIONAL_SUPPORT: (
                "Providing meaningful emotional support",
                0.8,
            ),
            MilestoneType.TRUST_BUILDING: (
                "Building deeper trust in our relationship",
                0.8,
            ),
        }

        base_desc, base_significance = descriptions.get(
            milestone_type, ("A meaningful moment", 0.5)
        )

        # Adjust significance based on relationship maturity
        relationship_factor = min(profile.get_relationship_age_days() / 30, 1.0)
        adjusted_significance = base_significance * (0.5 + relationship_factor * 0.5)

        return base_desc, adjusted_significance

    def _update_profile_from_milestone(
        self, profile: CompanionProfile, milestone: RelationshipMilestone
    ):
        """Update profile metrics based on milestone."""
        # Boost trust and engagement based on milestone significance
        trust_boost = milestone.emotional_significance * 0.1
        engagement_boost = milestone.emotional_significance * 0.05

        profile.trust_level = min(profile.trust_level + trust_boost, 1.0)
        profile.engagement_score = min(profile.engagement_score + engagement_boost, 1.0)
