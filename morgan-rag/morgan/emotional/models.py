"""
Data models for emotional intelligence and companion features.

Defines the core data structures for tracking emotions, user preferences,
relationship milestones, and empathetic responses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum


class EmotionType(Enum):
    """Primary emotion types based on Ekman's basic emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class CommunicationStyle(Enum):
    """User communication style preferences."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"


class ResponseLength(Enum):
    """User preferred response length."""
    BRIEF = "brief"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class MilestoneType(Enum):
    """Types of relationship milestones."""
    FIRST_CONVERSATION = "first_conversation"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    GOAL_ACHIEVED = "goal_achieved"
    LEARNING_MILESTONE = "learning_milestone"
    EMOTIONAL_SUPPORT = "emotional_support"
    TRUST_BUILDING = "trust_building"


@dataclass
class EmotionalState:
    """User's emotional state analysis."""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    secondary_emotions: List[EmotionType] = field(default_factory=list)
    emotional_indicators: List[str] = field(default_factory=list)  # text patterns that indicated emotion
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate emotional state values."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class MoodPattern:
    """User's mood patterns over time."""
    user_id: str
    timeframe: str  # e.g., "7d", "30d"
    dominant_emotions: List[EmotionType]
    average_intensity: float
    mood_stability: float  # 0.0 (very unstable) to 1.0 (very stable)
    emotional_trends: Dict[str, Any]  # trends like "improving", "declining"
    pattern_confidence: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreferences:
    """User's learned preferences and interests."""
    topics_of_interest: List[str] = field(default_factory=list)
    communication_style: CommunicationStyle = CommunicationStyle.FRIENDLY
    preferred_response_length: ResponseLength = ResponseLength.DETAILED
    learning_goals: List[str] = field(default_factory=list)
    personal_context: Dict[str, Any] = field(default_factory=dict)
    interaction_preferences: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RelationshipMilestone:
    """Significant moments in the user-Morgan relationship."""
    milestone_id: str
    milestone_type: MilestoneType
    description: str
    timestamp: datetime
    emotional_significance: float  # 0.0 to 1.0
    related_memories: List[str] = field(default_factory=list)  # memory IDs
    user_feedback: Optional[str] = None
    celebration_acknowledged: bool = False


@dataclass
class EmpatheticResponse:
    """Emotionally aware response generation."""
    response_text: str
    emotional_tone: str
    empathy_level: float  # 0.0 to 1.0
    relationship_context: str
    confidence_score: float
    personalization_elements: List[str] = field(default_factory=list)
    generation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationContext:
    """Context for a conversation turn."""
    user_id: str
    conversation_id: str
    message_text: str
    timestamp: datetime
    previous_messages: List[str] = field(default_factory=list)
    user_feedback: Optional[int] = None  # 1-5 rating
    session_duration: Optional[timedelta] = None


@dataclass
class InteractionData:
    """Data from a user interaction for profile updates."""
    conversation_context: ConversationContext
    emotional_state: EmotionalState
    user_satisfaction: Optional[float] = None
    topics_discussed: List[str] = field(default_factory=list)
    learning_indicators: List[str] = field(default_factory=list)
    relationship_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanionProfile:
    """Complete companion relationship profile."""
    user_id: str
    relationship_duration: timedelta
    interaction_count: int
    preferred_name: str  # what user likes to be called
    communication_preferences: UserPreferences
    emotional_patterns: Dict[str, Any] = field(default_factory=dict)
    shared_memories: List[str] = field(default_factory=list)  # memory IDs
    relationship_milestones: List[RelationshipMilestone] = field(default_factory=list)
    last_interaction: datetime = field(default_factory=datetime.utcnow)
    trust_level: float = 0.0  # 0.0 to 1.0
    engagement_score: float = 0.0  # 0.0 to 1.0
    profile_created: datetime = field(default_factory=datetime.utcnow)
    
    def get_relationship_age_days(self) -> int:
        """Get relationship age in days."""
        return self.relationship_duration.days
    
    def add_milestone(self, milestone: RelationshipMilestone):
        """Add a new relationship milestone."""
        self.relationship_milestones.append(milestone)
        # Sort by timestamp to maintain chronological order
        self.relationship_milestones.sort(key=lambda m: m.timestamp)


@dataclass
class PersonalizedGreeting:
    """Personalized greeting based on user profile."""
    greeting_text: str
    personalization_level: float  # 0.0 to 1.0
    context_elements: List[str] = field(default_factory=list)
    time_awareness: bool = False
    relationship_reference: bool = False


@dataclass
class ConversationTopic:
    """Suggested conversation topic."""
    topic: str
    relevance_score: float  # 0.0 to 1.0
    category: str
    reasoning: str
    user_interest_match: float  # 0.0 to 1.0


@dataclass
class ConversationStyle:
    """Adapted conversation style for user."""
    formality_level: float  # 0.0 (very casual) to 1.0 (very formal)
    technical_depth: float  # 0.0 (simple) to 1.0 (highly technical)
    empathy_emphasis: float  # 0.0 (minimal) to 1.0 (high empathy)
    response_length_target: ResponseLength
    personality_traits: List[str] = field(default_factory=list)
    adaptation_confidence: float = 0.0