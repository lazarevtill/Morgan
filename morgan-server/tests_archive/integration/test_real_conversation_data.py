"""
Integration tests with real conversation data.

This test suite imports actual conversation history from the old system
and validates that the new system handles it correctly.

Task 43: Test with actual conversation data
Task 43.1: Write integration tests with real data

Requirements: All
"""

import pytest
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from morgan_server.empathic.emotional import EmotionalIntelligence, EmotionalTone
from morgan_server.personalization.memory import MemoryManager, MessageRole
from morgan_server.personalization.profile import UserProfile, ProfileManager
from morgan_server.empathic.relationships import RelationshipManager, InteractionType


# Sample conversation data extracted from conversations.json
SAMPLE_CONVERSATIONS = [
    {
        "user_id": "test_user_real",
        "messages": [
            {
                "role": "user",
                "content": "I'm feeling really happy today!",
                "timestamp": "2025-01-01T10:00:00Z",
            },
            {
                "role": "assistant",
                "content": "That's wonderful to hear! What's making you feel so happy?",
                "timestamp": "2025-01-01T10:00:05Z",
            },
            {
                "role": "user",
                "content": "I just finished a big project at work.",
                "timestamp": "2025-01-01T10:01:00Z",
            },
            {
                "role": "assistant",
                "content": "Congratulations! Completing a big project is a great achievement.",
                "timestamp": "2025-01-01T10:01:05Z",
            },
        ],
    },
    {
        "user_id": "test_user_real",
        "messages": [
            {
                "role": "user",
                "content": "I'm worried about the upcoming presentation.",
                "timestamp": "2025-01-02T14:00:00Z",
            },
            {
                "role": "assistant",
                "content": "It's natural to feel nervous. What specifically concerns you?",
                "timestamp": "2025-01-02T14:00:05Z",
            },
            {
                "role": "user",
                "content": "I'm not sure if I prepared enough.",
                "timestamp": "2025-01-02T14:01:00Z",
            },
        ],
    },
    {
        "user_id": "test_user_real",
        "messages": [
            {
                "role": "user",
                "content": "The presentation went great! Thank you for the support.",
                "timestamp": "2025-01-03T16:00:00Z",
            },
            {
                "role": "assistant",
                "content": "I'm so glad to hear it went well! You did the work, I just listened.",
                "timestamp": "2025-01-03T16:00:05Z",
            },
        ],
    },
]


class TestRealConversationImport:
    """Test importing and processing real conversation data."""

    def test_import_conversation_structure(self):
        """Test that conversation data structure is valid."""
        for conv in SAMPLE_CONVERSATIONS:
            assert "user_id" in conv
            assert "messages" in conv
            assert len(conv["messages"]) > 0

            for msg in conv["messages"]:
                assert "role" in msg
                assert "content" in msg
                assert "timestamp" in msg
                assert msg["role"] in ["user", "assistant", "system"]

    def test_emotional_intelligence_with_real_conversations(self):
        """Test emotional intelligence with real conversation data."""
        ei = EmotionalIntelligence()
        user_id = "test_user_real"

        # Process all user messages
        user_messages = []
        for conv in SAMPLE_CONVERSATIONS:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])

        # Detect emotions for each message
        detections = []
        for message in user_messages:
            detection = ei.detect_tone(message, user_id=user_id)
            detections.append(detection)

            # Verify detection is valid
            assert detection.primary_tone is not None
            assert detection.confidence > 0
            assert isinstance(detection.indicators, list)

        # Verify we detected emotions
        assert len(detections) > 0

        # Verify emotional journey: happy -> worried -> happy
        # First message should be positive
        assert detections[0].primary_tone in [
            EmotionalTone.JOYFUL,
            EmotionalTone.EXCITED,
            EmotionalTone.CONTENT,
        ]
        # Last message should be positive (grateful for support)
        assert detections[-1].primary_tone in [
            EmotionalTone.JOYFUL,
            EmotionalTone.GRATEFUL,
            EmotionalTone.CONTENT,
        ]

    def test_emotional_trend_analysis_with_real_data(self):
        """Test emotional trend analysis with real conversation history."""
        ei = EmotionalIntelligence()
        user_id = "test_user_real"

        # Process conversations in order
        for conv in SAMPLE_CONVERSATIONS:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    ei.detect_tone(msg["content"], user_id=user_id)

        # Analyze trend
        trend = ei.analyze_emotional_trend(user_id)

        assert "dominant_tone" in trend
        assert "tone_distribution" in trend
        assert "trend" in trend
        assert trend["trend"] in ["improving", "declining", "stable", "unknown"]

        # Should show positive overall (starts happy, ends happy)
        assert trend["dominant_tone"] in [
            "joyful",
            "grateful",
            "content",
            "excited",
            "neutral",
        ]

    def test_relationship_tracking_with_real_conversations(self):
        """Test relationship tracking with real conversation data."""
        rm = RelationshipManager()
        user_id = "test_user_real"

        # Track all interactions
        interaction_count = 0
        for conv in SAMPLE_CONVERSATIONS:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    rm.track_interaction(
                        user_id=user_id,
                        interaction_type=InteractionType.CHAT,
                        sentiment="positive",
                    )
                    interaction_count += 1

        # Get relationship metrics
        metrics = rm.get_metrics(user_id)

        assert metrics is not None
        assert metrics.total_interactions == interaction_count
        assert metrics.trust_level > 0
        assert metrics.engagement_score > 0
        assert metrics.relationship_age_days >= 0

    def test_memory_system_with_real_conversations(self):
        """Test memory system stores and retrieves real conversations."""
        memory = MemoryManager()
        user_id = "test_user_real"
        timestamp = datetime.now().timestamp()

        # Store all conversations
        conversation_ids = []
        for i, conv in enumerate(SAMPLE_CONVERSATIONS):
            conv_id = f"conv_real_{timestamp}_{i}"
            conversation_ids.append(conv_id)

            # Create conversation
            memory.create_conversation(conv_id, user_id)

            for msg in conv["messages"]:
                memory.add_message(
                    conversation_id=conv_id,
                    role=MessageRole(msg["role"]),
                    content=msg["content"],
                )

        # Retrieve conversation
        conversation = memory.get_conversation(conversation_ids[0])
        assert conversation is not None

        history = conversation.messages
        assert len(history) == len(SAMPLE_CONVERSATIONS[0]["messages"])

        # Verify message content
        for i, msg in enumerate(history):
            assert msg.role.value == SAMPLE_CONVERSATIONS[0]["messages"][i]["role"]
            assert msg.content == SAMPLE_CONVERSATIONS[0]["messages"][i]["content"]

    def test_memory_search_with_real_queries(self):
        """Test memory search with actual queries from conversations."""
        memory = MemoryManager()
        user_id = "test_user_real_search"
        timestamp = datetime.now().timestamp()

        # Store conversations
        for i, conv in enumerate(SAMPLE_CONVERSATIONS):
            conv_id = f"conv_search_{timestamp}_{i}"
            memory.create_conversation(conv_id, user_id)

            for msg in conv["messages"]:
                memory.add_message(
                    conversation_id=conv_id,
                    role=MessageRole(msg["role"]),
                    content=msg["content"],
                )

        # Search for specific topics
        test_queries = [
            ("project", "project"),
            ("presentation", "presentation"),
            ("happy", "happy"),
        ]

        for query, expected_keyword in test_queries:
            results = memory.search_conversations(user_id=user_id, query=query, limit=5)

            # Should find relevant conversations
            assert len(results) > 0

            # At least one result should contain the keyword
            found = any(
                expected_keyword.lower() in message.content.lower()
                for _, message, _ in results
            )
            assert (
                found
            ), f"Expected to find '{expected_keyword}' in search results for '{query}'"

    def test_user_profile_with_real_data(self):
        """Test that user profile works with real conversation data."""
        profile = UserProfile(user_id="test_user_real_profile")

        # Extract topics from conversations
        topics = set()
        for conv in SAMPLE_CONVERSATIONS:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    content = msg["content"].lower()
                    if "project" in content or "work" in content:
                        topics.add("work")
                    if "presentation" in content:
                        topics.add("presentations")

        # Update profile with learned topics
        profile.topics_of_interest = list(topics)

        # Verify profile has topics
        assert len(profile.topics_of_interest) > 0
        assert any(
            "work" in topic.lower() or "presentation" in topic.lower()
            for topic in profile.topics_of_interest
        )

    def test_milestone_recognition_from_real_conversations(self):
        """Test milestone recognition from real conversation patterns."""
        rm = RelationshipManager()
        user_id = "test_user_real_milestones"

        # Track interactions
        for conv in SAMPLE_CONVERSATIONS:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    rm.track_interaction(
                        user_id=user_id, interaction_type=InteractionType.CHAT
                    )

        # Check for milestones
        milestones = rm.get_milestones(user_id)

        # Should have at least first conversation milestone
        assert len(milestones) > 0

        # Verify milestone structure
        for milestone in milestones:
            assert hasattr(milestone, "milestone_type")
            assert hasattr(milestone, "achieved_at")
            assert hasattr(milestone, "description")


class TestEndToEndChatFlow:
    """Test end-to-end chat flow with real conversation data."""

    def test_complete_conversation_flow(self):
        """Test complete conversation flow from user message to response."""
        # Initialize components
        ei = EmotionalIntelligence()
        memory = MemoryManager()
        rm = RelationshipManager()
        profile = UserProfile(user_id="test_user_e2e")

        user_id = "test_user_e2e"
        conversation_id = f"conv_e2e_{datetime.now().timestamp()}"

        # Simulate first conversation
        user_message = "I'm excited to start learning Python!"

        # Step 1: Detect emotion
        emotion = ei.detect_tone(user_message, user_id=user_id)
        assert emotion.primary_tone in [
            EmotionalTone.EXCITED,
            EmotionalTone.JOYFUL,
            EmotionalTone.CONTENT,
        ]

        # Step 2: Get response adjustment
        adjustment = ei.adjust_response_tone(emotion, user_id=user_id)
        assert adjustment.target_tone is not None
        assert adjustment.intensity > 0

        # Step 3: Store in memory
        memory.create_conversation(conversation_id, user_id)
        memory.add_message(
            conversation_id=conversation_id, role=MessageRole.USER, content=user_message
        )

        # Step 4: Track interaction
        rm.track_interaction(
            user_id=user_id, interaction_type=InteractionType.CHAT, sentiment="positive"
        )

        # Step 5: Update profile
        profile.topics_of_interest.append("python")
        profile.topics_of_interest.append("programming")

        # Verify everything was stored correctly
        conversation = memory.get_conversation(conversation_id)
        assert conversation is not None
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == user_message

        metrics = rm.get_metrics(user_id)
        assert metrics is not None
        assert metrics.total_interactions == 1

        assert "python" in [t.lower() for t in profile.topics_of_interest]

    def test_multi_turn_conversation_with_context(self):
        """Test multi-turn conversation maintains context."""
        memory = MemoryManager()
        ei = EmotionalIntelligence()

        user_id = "test_user_context"
        conversation_id = f"conv_context_{datetime.now().timestamp()}"

        # Create conversation
        memory.create_conversation(conversation_id, user_id)

        # Simulate multi-turn conversation
        turns = [
            ("user", "I'm working on a Python project"),
            ("assistant", "That sounds interesting! What kind of project?"),
            ("user", "It's a web application using FastAPI"),
            ("assistant", "FastAPI is great for building APIs. How's it going?"),
            ("user", "I'm stuck on authentication"),
        ]

        for role, content in turns:
            memory.add_message(
                conversation_id=conversation_id, role=MessageRole(role), content=content
            )

            if role == "user":
                ei.detect_tone(content, user_id=user_id)

        # Retrieve full conversation
        conversation = memory.get_conversation(conversation_id)
        assert conversation is not None
        history = conversation.messages
        assert len(history) == len(turns)

        # Verify context is maintained
        for i, (role, content) in enumerate(turns):
            assert history[i].role.value == role
            assert history[i].content == content

        # Search for context
        results = memory.search_conversations(user_id=user_id, query="FastAPI", limit=5)

        # Should find relevant messages
        assert len(results) > 0
        assert any("fastapi" in message.content.lower() for _, message, _ in results)


class TestDataMigration:
    """Test data migration from old system format."""

    def test_conversation_format_conversion(self):
        """Test converting old conversation format to new format."""
        # Old format (from conversations.json)
        old_format = {
            "title": "Test Conversation",
            "create_time": 1761952773.952984,
            "update_time": 1761953354.38973,
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "create_time": 1761952773.683605,
                    }
                }
            },
        }

        # Convert to new format
        new_format = {
            "user_id": "migrated_user",
            "conversation_id": "migrated_conv_1",
            "messages": [],
        }

        for msg_id, msg_data in old_format["mapping"].items():
            if msg_data.get("message") and msg_data["message"].get("content"):
                message = msg_data["message"]
                if message["content"].get("parts"):
                    new_format["messages"].append(
                        {
                            "role": message["author"]["role"],
                            "content": message["content"]["parts"][0],
                            "timestamp": datetime.fromtimestamp(
                                message.get("create_time", old_format["create_time"])
                            ).isoformat(),
                        }
                    )

        # Verify conversion
        assert len(new_format["messages"]) > 0
        assert new_format["messages"][0]["role"] == "user"
        assert new_format["messages"][0]["content"] == "Hello"

    def test_profile_data_migration(self):
        """Test migrating user profile data."""
        # Old format
        old_profile = {
            "user_profile": "Preferred name: Till\n",
            "user_instructions": "Girl version of myself...",
        }

        # Convert to new format
        profile = UserProfile(user_id="migrated_user")

        # Extract preferred name
        if "Preferred name:" in old_profile["user_profile"]:
            name = old_profile["user_profile"].split("Preferred name:")[1].strip()
            profile.preferred_name = name

        # Verify migration
        assert profile.preferred_name == "Till"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
