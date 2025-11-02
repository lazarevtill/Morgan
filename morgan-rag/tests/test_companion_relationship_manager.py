"""
Tests for the Companion Relationship Manager.

Tests the core functionality of building user profiles, adapting conversation
styles, tracking milestones, and generating personalized interactions.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from morgan.companion.relationship_manager import (
    CompanionRelationshipManager, Interaction
)
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext,
    CommunicationStyle, ResponseLength, MilestoneType
)


class TestCompanionRelationshipManager:
    """Test cases for CompanionRelationshipManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CompanionRelationshipManager()
        self.user_id = "test_user_123"
        
        # Create sample interactions
        self.interactions = [
            Interaction(
                interaction_id="int_1",
                user_id=self.user_id,
                timestamp=datetime.utcnow() - timedelta(days=5),
                message_content="Hello, I'm interested in learning about Python programming",
                topics_discussed=["python", "programming"],
                user_satisfaction=0.8
            ),
            Interaction(
                interaction_id="int_2", 
                user_id=self.user_id,
                timestamp=datetime.utcnow() - timedelta(days=3),
                message_content="Could you please help me understand machine learning concepts?",
                topics_discussed=["machine learning", "concepts"],
                user_satisfaction=0.9
            ),
            Interaction(
                interaction_id="int_3",
                user_id=self.user_id,
                timestamp=datetime.utcnow() - timedelta(days=1),
                message_content="Thank you for the detailed explanation. Call me Alex.",
                topics_discussed=["feedback"],
                user_satisfaction=0.95
            )
        ]
    
    def test_build_user_profile_new_user(self):
        """Test building a profile for a new user."""
        profile = self.manager.build_user_profile(self.user_id, self.interactions)
        
        assert profile.user_id == self.user_id
        assert profile.interaction_count == 3
        assert profile.preferred_name == "Alex"
        assert profile.relationship_duration.days == 4  # 5 days - 1 day
        assert "python" in profile.communication_preferences.topics_of_interest
        assert "machine learning" in profile.communication_preferences.topics_of_interest
        assert profile.communication_preferences.communication_style == CommunicationStyle.FORMAL
        assert profile.trust_level > 0.0  # Should have some trust from interactions
        assert profile.engagement_score > 0.0  # Should have engagement from satisfaction
    
    def test_build_user_profile_existing_user(self):
        """Test updating an existing user profile."""
        # Create initial profile
        initial_profile = self.manager.build_user_profile(self.user_id, self.interactions[:2])
        initial_count = initial_profile.interaction_count
        
        # Add more interactions
        new_interactions = [self.interactions[2]]
        updated_profile = self.manager.build_user_profile(self.user_id, new_interactions)
        
        assert updated_profile.interaction_count == initial_count + 1
        assert updated_profile.preferred_name == "Alex"  # Should extract from new interaction
        assert updated_profile.user_id == self.user_id
    
    def test_adapt_conversation_style(self):
        """Test conversation style adaptation."""
        profile = self.manager.build_user_profile(self.user_id, self.interactions)
        
        # Test with happy mood
        happy_mood = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )
        
        style = self.manager.adapt_conversation_style(profile, happy_mood)
        
        assert 0.0 <= style.formality_level <= 1.0
        assert 0.0 <= style.technical_depth <= 1.0
        assert 0.0 <= style.empathy_emphasis <= 1.0
        assert style.response_length_target in [ResponseLength.BRIEF, ResponseLength.DETAILED, ResponseLength.COMPREHENSIVE]
        assert style.adaptation_confidence > 0.0
        
        # Test with sad mood - should increase empathy
        sad_mood = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.8
        )
        
        sad_style = self.manager.adapt_conversation_style(profile, sad_mood)
        assert sad_style.empathy_emphasis > style.empathy_emphasis
    
    def test_track_relationship_milestones(self):
        """Test milestone tracking."""
        # Build profile first
        profile = self.manager.build_user_profile(self.user_id, self.interactions)
        
        # Track a milestone
        milestone = self.manager.track_relationship_milestones(
            self.user_id, "first_conversation"
        )
        
        assert milestone is not None
        assert milestone.milestone_type == MilestoneType.FIRST_CONVERSATION
        assert milestone.emotional_significance > 0.0
        assert len(milestone.description) > 0
        
        # Check that milestone was added to profile
        updated_profile = self.manager.profiles[self.user_id]
        assert len(updated_profile.relationship_milestones) == 1
        assert updated_profile.relationship_milestones[0].milestone_id == milestone.milestone_id
    
    def test_generate_personalized_greeting(self):
        """Test personalized greeting generation."""
        profile = self.manager.build_user_profile(self.user_id, self.interactions)
        
        # Test recent interaction (1 hour ago)
        recent_time = timedelta(hours=1)
        greeting = self.manager.generate_personalized_greeting(profile, recent_time)
        
        assert "Alex" in greeting.greeting_text  # Should use preferred name
        assert greeting.personalization_level > 0.0
        assert len(greeting.context_elements) > 0
        
        # Test long absence (1 week ago)
        long_time = timedelta(days=7)
        long_greeting = self.manager.generate_personalized_greeting(profile, long_time)
        
        assert "while" in long_greeting.greeting_text.lower()
        assert long_greeting.time_awareness is True
        assert "long_absence" in long_greeting.context_elements
    
    def test_suggest_conversation_topics(self):
        """Test conversation topic suggestions."""
        user_interests = ["python", "machine learning", "data science"]
        
        context = ConversationContext(
            user_id=self.user_id,
            conversation_id="conv_123",
            message_text="I want to learn more about neural networks",
            timestamp=datetime.utcnow(),
            previous_messages=["Tell me about machine learning algorithms"]
        )
        
        topics = self.manager.suggest_conversation_topics(user_interests, context)
        
        assert len(topics) > 0
        assert len(topics) <= 5  # Should return max 5 topics
        
        # Check that topics are properly structured
        for topic in topics:
            assert len(topic.topic) > 0
            assert 0.0 <= topic.relevance_score <= 1.0
            assert 0.0 <= topic.user_interest_match <= 1.0
            assert len(topic.category) > 0
            assert len(topic.reasoning) > 0
        
        # Topics should be sorted by relevance
        for i in range(len(topics) - 1):
            assert topics[i].relevance_score >= topics[i + 1].relevance_score
    
    def test_empty_interactions_handling(self):
        """Test handling of empty interactions list."""
        profile = self.manager.build_user_profile(self.user_id, [])
        
        assert profile.user_id == self.user_id
        assert profile.interaction_count == 0
        assert profile.preferred_name == "Friend"  # Default name
        assert profile.trust_level == 0.0
        assert profile.engagement_score == 0.0
    
    def test_invalid_milestone_type(self):
        """Test handling of invalid milestone type."""
        # Build profile first
        self.manager.build_user_profile(self.user_id, self.interactions)
        
        # Try to track invalid milestone
        milestone = self.manager.track_relationship_milestones(
            self.user_id, "invalid_milestone_type"
        )
        
        assert milestone is None
    
    def test_profile_persistence(self):
        """Test that profiles are stored and retrievable."""
        profile1 = self.manager.build_user_profile(self.user_id, self.interactions)
        
        # Profile should be stored
        assert self.user_id in self.manager.profiles
        assert self.manager.profiles[self.user_id] == profile1
        
        # Interaction history should be stored
        assert self.user_id in self.manager.interaction_history
        assert len(self.manager.interaction_history[self.user_id]) == 3