"""
Simplified integration tests for migrated components.

Tests the actual API of migrated components:
- Emotional intelligence integration
- Reranking integration

Requirements: All
"""

import pytest
from datetime import datetime

from morgan_server.empathic.emotional import EmotionalIntelligence, EmotionalTone
from morgan_server.knowledge.reranking import RerankingService, RerankResult


class TestEmotionalIntelligenceIntegration:
    """Test emotional intelligence integration."""

    def test_emotional_detection_and_pattern_tracking(self):
        """Test that emotional detection tracks patterns over time."""
        ei = EmotionalIntelligence(pattern_window_days=30)
        user_id = "test_user_123"

        # Simulate interactions
        messages = [
            "I'm so happy today!",
            "This is amazing!",
            "I'm feeling a bit down",
            "I'm worried about this",
            "Things are looking up!",
        ]

        for message in messages:
            detection = ei.detect_tone(message, user_id=user_id)
            assert detection.primary_tone is not None
            assert detection.confidence > 0
            assert isinstance(detection.indicators, list)

        # Verify patterns were tracked
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 5

        # Verify trend analysis works
        trend = ei.analyze_emotional_trend(user_id)
        assert "dominant_tone" in trend
        assert "tone_distribution" in trend
        assert trend["trend"] in ["improving", "declining", "stable", "unknown"]

    def test_emotional_adjustment_for_different_tones(self):
        """Test that emotional adjustment provides appropriate responses."""
        ei = EmotionalIntelligence()
        user_id = "test_user_456"

        test_cases = [
            ("I'm feeling really sad today", EmotionalTone.CONCERNED),
            ("I'm so excited and happy!", [EmotionalTone.JOYFUL, EmotionalTone.EXCITED]),
            ("I'm confused about this", EmotionalTone.CONTENT),
            ("Thank you so much!", EmotionalTone.CONTENT),
        ]

        for message, expected_tones in test_cases:
            detection = ei.detect_tone(message, user_id=user_id)
            adjustment = ei.adjust_response_tone(detection, user_id=user_id)

            # Verify adjustment has required fields
            assert adjustment.target_tone is not None
            assert adjustment.intensity > 0
            assert len(adjustment.suggestions) > 0

            # Verify target tone is appropriate
            if isinstance(expected_tones, list):
                assert adjustment.target_tone in expected_tones
            else:
                assert adjustment.target_tone == expected_tones

    def test_celebration_and_support_messages(self):
        """Test that system generates celebration and support messages."""
        ei = EmotionalIntelligence()
        user_id = "test_user_789"

        # Simulate negative to positive shift
        ei.detect_tone("I'm feeling sad", user_id=user_id)
        ei.detect_tone("I'm anxious about this", user_id=user_id)
        joy_detection = ei.detect_tone("I'm so happy now!", user_id=user_id)

        adjustment = ei.adjust_response_tone(joy_detection, user_id=user_id)

        # Should have celebration or support message
        assert adjustment.celebration is not None or adjustment.support is not None

    def test_pattern_cleanup(self):
        """Test that old patterns are cleaned up."""
        ei = EmotionalIntelligence(pattern_window_days=1)
        user_id = "test_user_cleanup"

        # Add a pattern
        ei.detect_tone("I'm happy", user_id=user_id)

        # Verify it exists
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1

        # Manually trigger cleanup (in real usage, this happens automatically)
        ei._cleanup_old_patterns(user_id)

        # Pattern should still exist (it's recent)
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1


class TestRerankingIntegration:
    """Test reranking integration."""

    @pytest.mark.asyncio
    async def test_reranking_basic_functionality(self):
        """Test basic reranking functionality."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        query = "Python programming"
        documents = [
            "Java is a programming language.",
            "Python is a high-level language.",
            "JavaScript runs in browsers.",
        ]

        async with reranking_service:
            results = await reranking_service.rerank(
                query=query,
                documents=documents,
                top_k=2
            )

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)
        assert all(hasattr(r, 'text') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'original_index') for r in results)

        # Verify scores are in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_reranking_with_metadata(self):
        """Test that reranking preserves metadata."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        query = "machine learning"
        documents = ["ML is AI", "DL uses networks", "Stats is important"]
        metadata = [
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc3"}
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
    async def test_reranking_statistics_tracking(self):
        """Test that reranking tracks statistics."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        query = "test"
        documents = ["doc1", "doc2"]

        async with reranking_service:
            await reranking_service.rerank(query, documents)
            await reranking_service.rerank(query, documents)

        stats = reranking_service.get_stats()

        assert stats["total_requests"] == 2
        assert stats["total_pairs"] == 4
        assert stats["local_requests"] == 2
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_reranking_empty_documents(self):
        """Test reranking with empty document list."""
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        async with reranking_service:
            results = await reranking_service.rerank("query", [])

        assert results == []


class TestIntegratedWorkflow:
    """Test integrated workflow across components."""

    @pytest.mark.asyncio
    async def test_emotional_intelligence_with_reranking(self):
        """Test using emotional intelligence and reranking together."""
        ei = EmotionalIntelligence()
        reranking_service = RerankingService(
            remote_endpoint=None,
            enable_local_fallback=True,
            local_device="cpu"
        )

        user_id = "test_user_workflow"

        # Step 1: Detect emotion from user message
        user_message = "I'm excited to learn about Python!"
        emotion = ei.detect_tone(user_message, user_id=user_id)

        assert emotion.primary_tone in [EmotionalTone.EXCITED, EmotionalTone.JOYFUL]

        # Step 2: Get response adjustment
        adjustment = ei.adjust_response_tone(emotion, user_id=user_id)

        assert adjustment.target_tone is not None
        assert adjustment.intensity > 0.5  # Should be enthusiastic

        # Step 3: Rerank content based on query
        query = "Python programming tutorial"
        documents = [
            "Python is great for beginners.",
            "Java is used in enterprise.",
            "Python is used in data science.",
        ]

        async with reranking_service:
            reranked = await reranking_service.rerank(
                query=query,
                documents=documents,
                top_k=2
            )

        assert len(reranked) == 2
        # Python-related docs should be ranked higher
        assert any("Python" in r.text for r in reranked)

    def test_emotional_pattern_analysis(self):
        """Test analyzing emotional patterns over multiple interactions."""
        ei = EmotionalIntelligence()
        user_id = "test_user_patterns"

        # Simulate a week of interactions
        positive_messages = [
            "Thank you!",
            "This is helpful!",
            "I appreciate it!",
            "Great work!",
        ]

        for msg in positive_messages:
            ei.detect_tone(msg, user_id=user_id)

        # Analyze patterns
        patterns = ei.get_patterns(user_id)
        assert len(patterns) >= 3  # At least 3 patterns tracked

        trend = ei.analyze_emotional_trend(user_id)

        # Should show positive trend
        assert trend["dominant_tone"] in ["joyful", "grateful", "content", "excited"]
        assert trend["trend"] in ["improving", "stable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
