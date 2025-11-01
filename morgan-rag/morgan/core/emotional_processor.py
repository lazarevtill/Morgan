"""
Emotional processing integration for Morgan Assistant.

Handles emotional intelligence integration, user profile management,
and emotional context processing following KISS principles.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..emotional.models import (
    EmotionalState, CompanionProfile, InteractionData, 
    ConversationContext, RelationshipMilestone
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalProcessor:
    """
    Processes emotional intelligence and user relationships.
    
    KISS: Single responsibility - handle emotional processing and user profiles.
    """
    
    def __init__(self, emotional_engine, relationship_manager, memory_processor):
        """Initialize emotional processor."""
        self.emotional_engine = emotional_engine
        self.relationship_manager = relationship_manager
        self.memory_processor = memory_processor
        logger.info("Emotional processor initialized")
    
    def get_or_create_user_profile(self, user_id: str) -> CompanionProfile:
        """Get existing user profile or create new one."""
        if user_id in self.relationship_manager.profiles:
            return self.relationship_manager.profiles[user_id]
        
        # Create new profile with minimal interaction data
        from ..companion.relationship_manager import Interaction
        
        initial_interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_content="Initial interaction"
        )
        
        return self.relationship_manager.build_user_profile(
            user_id, [initial_interaction]
        )
    
    def should_generate_greeting(self, user_profile: CompanionProfile) -> bool:
        """Check if we should generate a personalized greeting."""
        # Generate greeting if it's been more than 1 hour since last interaction
        from datetime import timedelta
        time_since_last = datetime.utcnow() - user_profile.last_interaction
        return time_since_last > timedelta(hours=1)
    
    def generate_personalized_greeting(self, user_profile: CompanionProfile):
        """Generate personalized greeting for user."""
        from datetime import timedelta
        time_since_last = datetime.utcnow() - user_profile.last_interaction
        return self.relationship_manager.generate_personalized_greeting(
            user_profile, time_since_last
        )
    
    def check_for_milestones(
        self,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState
    ) -> Optional[RelationshipMilestone]:
        """Check if this interaction represents a milestone."""
        # Simple milestone detection logic
        if user_profile.interaction_count == 1:
            return self.relationship_manager.track_relationship_milestones(
                user_profile.user_id, "first_conversation"
            )
        
        # Check for breakthrough moments (high positive emotion + engagement)
        if (emotional_state.primary_emotion.value == "joy" and 
            emotional_state.intensity > 0.7 and
            len(conversation_context.message_text) > 100):
            return self.relationship_manager.track_relationship_milestones(
                user_profile.user_id, "breakthrough_moment"
            )
        
        # Check for trust building (personal sharing)
        if any(word in conversation_context.message_text.lower() 
               for word in ['personal', 'share', 'trust', 'private']):
            return self.relationship_manager.track_relationship_milestones(
                user_profile.user_id, "trust_building"
            )
        
        return None
    
    def process_conversation_memory(
        self,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
        response_answer: str,
        response_sources: List[str]
    ):
        """Process and store conversation memories with emotional context."""
        try:
            # Create conversation turn for memory processing
            from ..core.memory import ConversationTurn
            
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                conversation_id=conversation_context.conversation_id,
                question=conversation_context.message_text,
                answer=response_answer,
                timestamp=conversation_context.timestamp.isoformat(),
                sources=response_sources,
                feedback_rating=conversation_context.user_feedback
            )
            
            # Extract memories with emotional context
            memory_result = self.memory_processor.extract_memories(
                turn, conversation_context.user_feedback, emotional_state
            )
            
            # Store memories
            for memory in memory_result.memories:
                self.memory_processor.store_memory(memory)
            
            logger.debug(f"Processed {len(memory_result.memories)} memories with emotional context")
            
        except Exception as e:
            logger.error(f"Failed to process conversation memory: {e}")
    
    def update_user_profile(
        self,
        user_profile: CompanionProfile,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
        response_confidence: float,
        topics_discussed: List[str]
    ):
        """Update user profile with interaction data."""
        try:
            interaction_data = InteractionData(
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                user_satisfaction=response_confidence,  # Use confidence as satisfaction proxy
                topics_discussed=topics_discussed
            )
            
            self.emotional_engine.update_user_profile(
                user_profile.user_id, interaction_data
            )
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    def generate_milestone_celebration(self, milestone: RelationshipMilestone) -> str:
        """Generate celebration message for milestone."""
        celebrations = {
            "first_conversation": "Welcome! I'm excited to start this journey of learning and discovery with you.",
            "breakthrough_moment": "What a wonderful breakthrough! I'm so glad I could help you reach this understanding.",
            "goal_achieved": "Congratulations on achieving your goal! Your dedication and hard work have paid off.",
            "learning_milestone": "It's amazing to see how much you've learned! Your curiosity and persistence inspire me.",
            "emotional_support": "I'm honored that you trust me to support you through this. You're stronger than you know.",
            "trust_building": "Thank you for sharing something so personal with me. Our growing trust means a lot."
        }
        
        return celebrations.get(
            milestone.milestone_type.value,
            "I'm grateful for this meaningful moment in our relationship."
        )
    
    def get_relationship_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about the relationship with a user."""
        try:
            profile = self.relationship_manager.profiles.get(user_id)
            if not profile:
                return {"message": "We're just getting to know each other!"}
            
            # Get mood patterns
            mood_pattern = self.emotional_engine.track_mood_patterns(user_id)
            
            insights = {
                "relationship_age_days": profile.get_relationship_age_days(),
                "interaction_count": profile.interaction_count,
                "trust_level": profile.trust_level,
                "engagement_score": profile.engagement_score,
                "preferred_name": profile.preferred_name,
                "communication_style": profile.communication_preferences.communication_style.value,
                "dominant_emotions": [e.value for e in mood_pattern.dominant_emotions],
                "mood_stability": mood_pattern.mood_stability,
                "recent_milestones": [
                    {
                        "type": m.milestone_type.value,
                        "description": m.description,
                        "date": m.timestamp.strftime("%Y-%m-%d")
                    }
                    for m in profile.relationship_milestones[-3:]  # Last 3 milestones
                ],
                "topics_of_interest": profile.communication_preferences.topics_of_interest[:5]
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get relationship insights: {e}")
            return {"error": "Unable to retrieve relationship insights"}
    
    def suggest_conversation_topics(self, user_id: str) -> List[str]:
        """Suggest conversation topics based on user interests and history."""
        try:
            profile = self.relationship_manager.profiles.get(user_id)
            if not profile:
                return [
                    "What would you like to learn about today?",
                    "Tell me about your current projects",
                    "What's been on your mind lately?"
                ]
            
            # Create recent context
            recent_context = ConversationContext(
                user_id=user_id,
                conversation_id="",
                message_text="",
                timestamp=datetime.utcnow()
            )
            
            topics = self.relationship_manager.suggest_conversation_topics(
                profile.communication_preferences.topics_of_interest,
                recent_context
            )
            
            return [topic.topic for topic in topics]
            
        except Exception as e:
            logger.error(f"Failed to suggest topics: {e}")
            return ["How can I help you today?"]
    
    def celebrate_milestone(
        self,
        user_id: str,
        milestone_type: str,
        custom_message: Optional[str] = None
    ) -> str:
        """Celebrate a relationship milestone with the user."""
        try:
            milestone = self.relationship_manager.track_relationship_milestones(
                user_id, milestone_type
            )
            
            if milestone:
                if custom_message:
                    celebration = custom_message
                else:
                    celebration = self.generate_milestone_celebration(milestone)
                
                logger.info(f"Celebrated milestone for {user_id}: {milestone.description}")
                return celebration
            else:
                return "I appreciate our growing relationship!"
                
        except Exception as e:
            logger.error(f"Failed to celebrate milestone: {e}")
            return "I'm grateful for our conversation!"