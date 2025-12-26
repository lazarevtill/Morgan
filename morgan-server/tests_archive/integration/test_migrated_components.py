"""
Integration tests for migrated components from old morgan-rag system.

Tests the integration of:
- Reranking with search
- Emotional intelligence integration
- Relationship management integration
- Communication preferences

Requirements: All
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from morgan_server.knowledge.reranking import RerankingService, RerankResult
from morgan_server.empathic.emotional import EmotionalIntelligence, EmotionalTone
from morgan_server.empathic.relationships import RelationshipManager
from morgan_server.personalization.preferences import PreferenceManager


class TestRerankingIntegration:
    """Test reranking integration with search."""

    @pytest.mark.asyncio
    async def test_reranking_with_search_results(self):
        """Test that reranking properly reorders search results."""
        # Create reranking service (local only for testing)
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        # Sample search results
        query = "What is Python programming?"
        documents = [
            "Java is a programming language used for enterprise applications.",
            "Python is a high-level programming language known for simplicity.",
            "JavaScript is used for web development and runs in browsers.",
            "Python is widely used in data science and machine learning.",
            "C++ is a low-level language used for system programming.",
        ]

        # Rerank documents
        async with reranking_service:
            results = await reranking_service.rerank(
                query=query,
                documents=documents,
                top_k=3
            )

        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)

        # Verify Python-related documents are ranked higher
        top_texts = [r.text for r in results]
        assert any("Python" in text for text in top_texts[:2])

        # Verify scores are in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_reranking_preserves_metadata(self):
        """Test that reranking preserves document metadata."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        query = "machine learning"
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Data science involves statistics."
        ]
        metadata = [
            {"source": "doc1", "page": 1},
            {"source": "doc2", "page": 5},
            {"source": "doc3", "page": 10}
        ]

        async with reranking_service:
            results = await reranking_service.rerank(
                query=query,
                documents=documents,
                metadata=metadata
            )

        # Verify metadata is preserved
        assert all(r.metadata for r in results)
        assert all("source" in r.metadata for r in results)

    @pytest.mark.asyncio
    async def test_reranking_statistics(self):
        """Test that reranking tracks statistics correctly."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        async with reranking_service:
            await reranking_service.rerank(query, documents)
            await reranking_service.rerank(query, documents)

        stats = reranking_service.get_stats()

        assert stats["total_requests"] == 2
        assert stats["total_pairs"] == 6  # 2 requests * 3 documents
        assert stats["local_requests"] == 2
        assert stats["errors"] == 0


class TestEmotionalIntelligenceIntegration:
    """Test emotional intelligence integration."""

    def test_emotional_detection_with_pattern_tracking(self):
        """Test that emotional detection tracks patterns over time."""
        ei = EmotionalIntelligence(pattern_window_days=30)
        user_id = "test_user_123"

        # Simulate a series of interactions
        messages = [
            "I'm so happy today!",
            "This is amazing!",
            "I'm feeling a bit down",
            "I'm worried about this",
            "Things are looking up!",
        ]

        for message in messages:
            detection = ei.detect_tone(message, user_id=user_id)
            # Just verify detection works, don't enforce specific tones
            assert detection.primary_tone is not None
            assert detection.confidence > 0

        # Verify patterns were tracked
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 5

        # Verify trend analysis
        trend = ei.analyze_emotional_trend(user_id)
        assert trend["dominant_tone"] is not None
        assert "tone_distribution" in trend
        assert trend["trend"] in ["improving", "declining", "stable"]

    def test_emotional_adjustment_provides_appropriate_support(self):
        """Test that emotional adjustment provides appropriate support."""
        ei = EmotionalIntelligence()
        user_id = "test_user_456"

        # Test sad emotion
        sad_detection = ei.detect_tone("I'm feeling really sad today", user_id=user_id)
        adjustment = ei.adjust_response_tone(sad_detection, user_id=user_id)

        assert adjustment.target_tone == EmotionalTone.CONCERNED
        assert adjustment.support is not None
        assert "support" in adjustment.suggestions[0].lower() or "gentle" in adjustment.suggestions[0].lower()

        # Test joyful emotion
        joy_detection = ei.detect_tone("I'm so excited and happy!", user_id=user_id)
        adjustment = ei.adjust_response_tone(joy_detection, user_id=user_id)

        assert adjustment.target_tone in [EmotionalTone.JOYFUL, EmotionalTone.EXCITED]
        assert adjustment.intensity > 0.7

    def test_emotional_celebration_on_positive_shift(self):
        """Test that system celebrates positive emotional shifts."""
        ei = EmotionalIntelligence()
        user_id = "test_user_789"

        # Simulate negative to positive shift
        ei.detect_tone("I'm feeling sad", user_id=user_id)
        ei.detect_tone("I'm anxious about this", user_id=user_id)
        joy_detection = ei.detect_tone("I'm so happy now!", user_id=user_id)

        adjustment = ei.adjust_response_tone(joy_detection, user_id=user_id)

        # Should have celebration message
        assert adjustment.celebration is not None
        assert "glad" in adjustment.celebration.lower() or "happy" in adjustment.celebration.lower()


class TestRelationshipManagementIntegration:
    """Test relationship management integration."""

    def test_relationship_tracking_with_emotional_context(self):
        """Test that relationship tracking integrates with emotional context."""
        rm = RelationshipManager()
        ei = EmotionalIntelligence()
        user_id = "test_user_rel_123"

        # Create user profile
        profile = rm.get_or_create_profile(user_id)

        # Simulate interactions with emotional context
        messages = [
            "Hi, I need help with something",
            "Thank you so much for your help!",
            "I really appreciate your support",
            "You've been so helpful to me"
        ]

        for message in messages:
            detection = ei.detect_tone(message, user_id=user_id)
            rm.record_interaction(
                user_id=user_id,
                interaction_type="chat",
                quality_score=0.8,
                emotional_tone=detection.primary_tone.value
            )

        # Verify relationship metrics updated
        updated_profile = rm.get_profile(user_id)
        assert updated_profile.interaction_count == 4
        assert updated_profile.trust_level > profile.trust_level

    def test_milestone_detection_with_emotional_patterns(self):
        """Test that milestone detection considers emotional patterns."""
        rm = RelationshipManager()
        ei = EmotionalIntelligence()
        user_id = "test_user_milestone_456"

        # Create profile
        rm.get_or_create_profile(user_id)

        # Simulate breakthrough moment with strong positive emotion
        breakthrough_message = "Oh wow, I finally understand! This is amazing! Thank you so much!"
        detection = ei.detect_tone(breakthrough_message, user_id=user_id)

        # Record high-quality interaction
        rm.record_interaction(
            user_id=user_id,
            interaction_type="chat",
            quality_score=0.95,
            emotional_tone=detection.primary_tone.value
        )

        # Check for milestones
        milestones = rm.get_milestones(user_id)

        # Should have at least first interaction milestone
        assert len(milestones) > 0

    def test_trust_building_over_time(self):
        """Test that trust builds appropriately over time."""
        rm = RelationshipManager()
        user_id = "test_user_trust_789"

        profile = rm.get_or_create_profile(user_id)
        initial_trust = profile.trust_level

        # Simulate multiple positive interactions
        for i in range(10):
            rm.record_interaction(
                user_id=user_id,
                interaction_type="chat",
                quality_score=0.8,
                emotional_tone="grateful"
            )

        updated_profile = rm.get_profile(user_id)

        # Trust should increase
        assert updated_profile.trust_level > initial_trust
        assert updated_profile.interaction_count == 10


class TestCommunicationPreferencesIntegration:
    """Test communication preferences integration."""

    def test_preference_learning_from_interactions(self):
        """Test that preferences are learned from user interactions."""
        pm = PreferenceManager()
        user_id = "test_user_pref_123"

        # Simulate interactions with different message lengths
        short_messages = ["ok", "thanks", "yes", "got it"]
        long_messages = [
            "I really appreciate your detailed explanation. It helped me understand the concept much better.",
            "Could you please provide more information about this topic? I'm very interested in learning more.",
            "Thank you for taking the time to explain this thoroughly. Your examples were particularly helpful."
        ]

        # Record short message interactions
        for msg in short_messages:
            pm.update_from_interaction(
                user_id=user_id,
                message=msg,
                response_length="brief",
                satisfaction_score=0.8
            )

        # Get preferences
        prefs = pm.get_preferences(user_id)

        # Should prefer brief responses
        assert prefs is not None
        assert prefs.preferred_response_length == "brief"

    def test_preference_adaptation_over_time(self):
        """Test that preferences adapt as user behavior changes."""
        pm = PreferenceManager()
        user_id = "test_user_adapt_456"

        # Initially prefer brief responses
        for i in range(5):
            pm.update_from_interaction(
                user_id=user_id,
                message="quick question",
                response_length="brief",
                satisfaction_score=0.8
            )

        initial_prefs = pm.get_preferences(user_id)

        # Then shift to detailed responses
        for i in range(10):
            pm.update_from_interaction(
                user_id=user_id,
                message="I'd like to understand this in detail, with examples and explanations.",
                response_length="detailed",
                satisfaction_score=0.9
            )

        updated_prefs = pm.get_preferences(user_id)

        # Preferences should adapt
        assert updated_prefs.preferred_response_length != initial_prefs.preferred_response_length

    def test_topic_interest_tracking(self):
        """Test that topic interests are tracked correctly."""
        pm = PreferenceManager()
        user_id = "test_user_topics_789"

        # Simulate interactions about specific topics
        tech_messages = [
            "Tell me about Python programming",
            "How does machine learning work?",
            "Explain neural networks"
        ]

        for msg in tech_messages:
            pm.update_from_interaction(
                user_id=user_id,
                message=msg,
                response_length="moderate",
                satisfaction_score=0.9
            )

        prefs = pm.get_preferences(user_id)

        # Should have technology-related interests
        assert prefs is not None
        assert len(prefs.topics_of_interest) > 0


class TestIntegratedWorkflow:
    """Test complete integrated workflow across all migrated components."""

    @pytest.mark.asyncio
    async def test_complete_interaction_workflow(self):
        """Test a complete interaction workflow using all migrated components."""
        # Initialize all components
        ei = EmotionalIntelligence()
        rm = RelationshipManager()
        pm = PreferenceManager()
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        user_id = "test_user_workflow_123"

        # Step 1: User sends message
        user_message = "I'm really excited to learn about Python! Can you help me?"

        # Step 2: Detect emotional tone
        emotion = ei.detect_tone(user_message, user_id=user_id)
        assert emotion.primary_tone == EmotionalTone.EXCITED

        # Step 3: Get response tone adjustment
        adjustment = ei.adjust_response_tone(emotion, user_id=user_id)
        assert adjustment.target_tone in [EmotionalTone.EXCITED, EmotionalTone.JOYFUL]

        # Step 4: Search for relevant content (simulated)
        query = "Python programming tutorial"
        documents = [
            "Python is a high-level programming language.",
            "Java is used for enterprise applications.",
            "Python is great for beginners and data science.",
            "JavaScript runs in web browsers.",
        ]

        # Step 5: Rerank results
        async with reranking_service:
            reranked = await reranking_service.rerank(
                query=query,
                documents=documents,
                top_k=2
            )

        assert len(reranked) == 2
        assert "Python" in reranked[0].text

        # Step 6: Record interaction
        rm.record_interaction(
            user_id=user_id,
            interaction_type="chat",
            quality_score=0.9,
            emotional_tone=emotion.primary_tone.value
        )

        # Step 7: Update preferences
        pm.update_from_interaction(
            user_id=user_id,
            message=user_message,
            response_length="detailed",
            satisfaction_score=0.9
        )

        # Step 8: Verify all components updated correctly
        profile = rm.get_profile(user_id)
        assert profile.interaction_count == 1

        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1

        prefs = pm.get_preferences(user_id)
        assert prefs is not None

    def test_emotional_context_influences_relationship(self):
        """Test that emotional context influences relationship metrics."""
        ei = EmotionalIntelligence()
        rm = RelationshipManager()
        user_id = "test_user_context_456"

        # Create profile
        rm.get_or_create_profile(user_id)

        # Simulate consistently positive interactions
        positive_messages = [
            "Thank you so much!",
            "This is really helpful!",
            "I appreciate your support!",
            "You're amazing!"
        ]

        for msg in positive_messages:
            emotion = ei.detect_tone(msg, user_id=user_id)
            rm.record_interaction(
                user_id=user_id,
                interaction_type="chat",
                quality_score=0.9,
                emotional_tone=emotion.primary_tone.value
            )

        profile = rm.get_profile(user_id)

        # Trust and engagement should be high
        assert profile.trust_level > 0.5
        assert profile.engagement_score > 0.5

    def test_preferences_influence_content_selection(self):
        """Test that learned preferences can influence content selection."""
        pm = PreferenceManager()
        user_id = "test_user_content_789"

        # Learn that user prefers technical content
        technical_messages = [
            "Explain the technical details of how this works",
            "I want to understand the underlying algorithms",
            "Show me the implementation details"
        ]

        for msg in technical_messages:
            pm.update_from_interaction(
                user_id=user_id,
                message=msg,
                response_length="detailed",
                satisfaction_score=0.95
            )

        prefs = pm.get_preferences(user_id)

        # Should prefer detailed, technical responses
        assert prefs.preferred_response_length == "detailed"
        assert prefs.communication_style in ["professional", "technical"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
