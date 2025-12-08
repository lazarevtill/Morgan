"""
Unit tests for the Memory System module.

Tests cover:
- Conversation storage
- Memory retrieval
- Summarization
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from morgan_server.personalization.memory import (
    MemoryManager,
    Conversation,
    Message,
    MessageRole,
)


class TestConversationCreation:
    """Test conversation creation."""
    
    def test_create_conversation_basic(self):
        """Test creating a basic conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.create_conversation(
                "conv123", "user123"
            )
            
            assert conversation.conversation_id == "conv123"
            assert conversation.user_id == "user123"
            assert len(conversation.messages) == 0
            assert conversation.summary is None
    
    def test_create_conversation_with_metadata(self):
        """Test creating a conversation with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            metadata = {"topic": "AI", "priority": "high"}
            conversation = manager.create_conversation(
                "conv123", "user123", metadata=metadata
            )
            
            assert conversation.metadata["topic"] == "AI"
            assert conversation.metadata["priority"] == "high"
    
    def test_create_duplicate_conversation_raises_error(self):
        """Test that creating duplicate conversation raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            with pytest.raises(
                ValueError, match="Conversation already exists"
            ):
                manager.create_conversation("conv123", "user123")
    
    def test_conversation_has_timestamps(self):
        """Test that conversation has creation and update timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            before = datetime.now()
            conversation = manager.create_conversation("conv123", "user123")
            after = datetime.now()
            
            assert before <= conversation.created_at <= after
            assert before <= conversation.last_updated <= after
    
    def test_get_or_create_conversation_creates_new(self):
        """Test get_or_create creates new conversation if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.get_or_create_conversation(
                "conv123", "user123"
            )
            
            assert conversation.conversation_id == "conv123"
            assert "conv123" in manager.conversations
    
    def test_get_or_create_conversation_returns_existing(self):
        """Test get_or_create returns existing conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            # Create conversation
            original = manager.create_conversation("conv123", "user123")
            original.add_message(MessageRole.USER, "Hello")
            
            # Get or create should return existing
            conversation = manager.get_or_create_conversation(
                "conv123", "user123"
            )
            
            assert conversation.conversation_id == "conv123"
            assert len(conversation.messages) == 1
            assert conversation is original


class TestConversationRetrieval:
    """Test conversation retrieval."""
    
    def test_get_conversation_exists(self):
        """Test getting an existing conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            conversation = manager.get_conversation("conv123")
            
            assert conversation is not None
            assert conversation.conversation_id == "conv123"
    
    def test_get_conversation_not_exists(self):
        """Test getting a non-existent conversation returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.get_conversation("nonexistent")
            
            assert conversation is None
    
    def test_get_user_conversations(self):
        """Test getting all conversations for a user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.create_conversation("conv2", "user123")
            manager.create_conversation("conv3", "user456")
            
            conversations = manager.get_user_conversations("user123")
            
            assert len(conversations) == 2
            conv_ids = [c.conversation_id for c in conversations]
            assert "conv1" in conv_ids
            assert "conv2" in conv_ids
            assert "conv3" not in conv_ids
    
    def test_get_user_conversations_empty(self):
        """Test getting conversations for user with none."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversations = manager.get_user_conversations("user123")
            
            assert conversations == []


class TestMessageManagement:
    """Test message management."""
    
    def test_add_message_to_conversation(self):
        """Test adding a message to a conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            message = manager.add_message(
                "conv123", MessageRole.USER, "Hello, Morgan!"
            )
            
            assert message.role == MessageRole.USER
            assert message.content == "Hello, Morgan!"
            
            conversation = manager.get_conversation("conv123")
            assert len(conversation.messages) == 1
            assert conversation.messages[0].content == "Hello, Morgan!"
    
    def test_add_message_with_metadata(self):
        """Test adding a message with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            metadata = {"emotion": "happy", "confidence": 0.9}
            message = manager.add_message(
                "conv123",
                MessageRole.ASSISTANT,
                "I'm here to help!",
                metadata=metadata
            )
            
            assert message.metadata["emotion"] == "happy"
            assert message.metadata["confidence"] == 0.9
    
    def test_add_message_to_nonexistent_conversation(self):
        """Test adding message to non-existent conversation raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            with pytest.raises(ValueError, match="Conversation not found"):
                manager.add_message(
                    "nonexistent", MessageRole.USER, "Hello"
                )
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages to a conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            manager.add_message("conv123", MessageRole.USER, "Hello")
            manager.add_message("conv123", MessageRole.ASSISTANT, "Hi!")
            manager.add_message("conv123", MessageRole.USER, "How are you?")
            
            conversation = manager.get_conversation("conv123")
            assert len(conversation.messages) == 3
            assert conversation.messages[0].role == MessageRole.USER
            assert conversation.messages[1].role == MessageRole.ASSISTANT
            assert conversation.messages[2].role == MessageRole.USER
    
    def test_message_has_timestamp(self):
        """Test that messages have timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            before = datetime.now()
            message = manager.add_message(
                "conv123", MessageRole.USER, "Hello"
            )
            after = datetime.now()
            
            assert before <= message.timestamp <= after
    
    def test_get_messages_by_role(self):
        """Test getting messages filtered by role."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.create_conversation("conv123", "user123")
            conversation.add_message(MessageRole.USER, "Hello")
            conversation.add_message(MessageRole.ASSISTANT, "Hi!")
            conversation.add_message(MessageRole.USER, "How are you?")
            conversation.add_message(MessageRole.ASSISTANT, "I'm good!")
            
            user_messages = conversation.get_messages(role=MessageRole.USER)
            assistant_messages = conversation.get_messages(
                role=MessageRole.ASSISTANT
            )
            
            assert len(user_messages) == 2
            assert len(assistant_messages) == 2
            assert all(m.role == MessageRole.USER for m in user_messages)
            assert all(
                m.role == MessageRole.ASSISTANT for m in assistant_messages
            )
    
    def test_get_messages_with_limit(self):
        """Test getting messages with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.create_conversation("conv123", "user123")
            for i in range(10):
                conversation.add_message(MessageRole.USER, f"Message {i}")
            
            messages = conversation.get_messages(limit=5)
            
            assert len(messages) == 5
            # Should get the most recent 5 messages
            assert messages[0].content == "Message 5"
            assert messages[4].content == "Message 9"


class TestContextRetrieval:
    """Test context retrieval from memory."""
    
    def test_get_conversation_context(self):
        """Test getting conversation context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            for i in range(15):
                manager.add_message(
                    "conv123", MessageRole.USER, f"Message {i}"
                )
            
            # Default context window is 10 messages
            context = manager.get_conversation_context("conv123")
            
            assert len(context) == 10
            # Should get the most recent 10 messages
            assert context[0].content == "Message 5"
            assert context[9].content == "Message 14"
    
    def test_get_conversation_context_custom_limit(self):
        """Test getting conversation context with custom limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            for i in range(15):
                manager.add_message(
                    "conv123", MessageRole.USER, f"Message {i}"
                )
            
            context = manager.get_conversation_context("conv123", max_messages=5)
            
            assert len(context) == 5
            assert context[0].content == "Message 10"
            assert context[4].content == "Message 14"
    
    def test_get_conversation_context_nonexistent(self):
        """Test getting context for non-existent conversation raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            with pytest.raises(ValueError, match="Conversation not found"):
                manager.get_conversation_context("nonexistent")
    
    def test_get_context_window_fewer_messages(self):
        """Test getting context when fewer messages than limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager.create_conversation("conv123", "user123")
            conversation.add_message(MessageRole.USER, "Message 1")
            conversation.add_message(MessageRole.ASSISTANT, "Response 1")
            
            context = conversation.get_context_window(max_messages=10)
            
            assert len(context) == 2


class TestMemorySearch:
    """Test memory search functionality."""
    
    def test_search_conversations_basic(self):
        """Test basic conversation search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "Tell me about AI")
            manager.add_message(
                "conv1", MessageRole.ASSISTANT, "AI is fascinating"
            )
            
            manager.create_conversation("conv2", "user123")
            manager.add_message("conv2", MessageRole.USER, "What's the weather?")
            
            results = manager.search_conversations("user123", "AI")
            
            assert len(results) > 0
            # Should find messages containing "AI"
            assert any("AI" in msg.content for _, msg, _ in results)
    
    def test_search_conversations_case_insensitive(self):
        """Test that search is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "Tell me about AI")
            
            results = manager.search_conversations("user123", "ai")
            
            assert len(results) > 0
    
    def test_search_conversations_with_limit(self):
        """Test search with result limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            for i in range(10):
                manager.add_message(
                    "conv1", MessageRole.USER, f"Message about AI {i}"
                )
            
            results = manager.search_conversations("user123", "AI", limit=5)
            
            assert len(results) == 5
    
    def test_search_conversations_no_results(self):
        """Test search with no matching results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "Hello")
            
            results = manager.search_conversations("user123", "nonexistent")
            
            assert len(results) == 0
    
    def test_search_conversations_relevance_scoring(self):
        """Test that search results are sorted by relevance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "AI")
            manager.add_message("conv1", MessageRole.USER, "Tell me about AI")
            manager.add_message(
                "conv1", MessageRole.USER, "AI AI AI AI"
            )
            
            results = manager.search_conversations("user123", "AI")
            
            # Results should be sorted by relevance (descending)
            assert len(results) >= 2
            # Higher relevance should come first
            assert results[0][2] >= results[1][2]


class TestMemorySummarization:
    """Test memory summarization."""
    
    def test_summarize_conversation_basic(self):
        """Test basic conversation summarization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            manager.add_message("conv123", MessageRole.USER, "Hello")
            manager.add_message("conv123", MessageRole.ASSISTANT, "Hi!")
            
            summary = manager.summarize_conversation("conv123")
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "2 messages" in summary
    
    def test_summarize_conversation_cached(self):
        """Test that summary is cached after first generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            manager.add_message("conv123", MessageRole.USER, "Hello")
            
            summary1 = manager.summarize_conversation("conv123")
            summary2 = manager.summarize_conversation("conv123")
            
            assert summary1 == summary2
            
            # Check that summary is stored in conversation
            conversation = manager.get_conversation("conv123")
            assert conversation.summary == summary1
    
    def test_summarize_conversation_nonexistent(self):
        """Test summarizing non-existent conversation raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            with pytest.raises(ValueError, match="Conversation not found"):
                manager.summarize_conversation("nonexistent")
    
    def test_summarize_conversation_with_topics(self):
        """Test that summary includes topics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            manager.add_message(
                "conv123",
                MessageRole.USER,
                "Tell me about artificial intelligence and machine learning"
            )
            manager.add_message(
                "conv123",
                MessageRole.ASSISTANT,
                "Artificial intelligence and machine learning are related"
            )
            
            summary = manager.summarize_conversation("conv123")
            
            assert "Topics discussed:" in summary


class TestMemoryStats:
    """Test memory statistics."""
    
    def test_get_memory_stats_empty(self):
        """Test getting stats when no conversations exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            stats = manager.get_memory_stats()
            
            assert stats["total_conversations"] == 0
            assert stats["total_messages"] == 0
            assert stats["oldest_conversation"] is None
            assert stats["newest_conversation"] is None
    
    def test_get_memory_stats_all(self):
        """Test getting stats for all conversations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "Hello")
            manager.add_message("conv1", MessageRole.ASSISTANT, "Hi!")
            
            manager.create_conversation("conv2", "user456")
            manager.add_message("conv2", MessageRole.USER, "Hey")
            
            stats = manager.get_memory_stats()
            
            assert stats["total_conversations"] == 2
            assert stats["total_messages"] == 3
            assert stats["oldest_conversation"] is not None
            assert stats["newest_conversation"] is not None
    
    def test_get_memory_stats_for_user(self):
        """Test getting stats for specific user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.add_message("conv1", MessageRole.USER, "Hello")
            
            manager.create_conversation("conv2", "user456")
            manager.add_message("conv2", MessageRole.USER, "Hey")
            manager.add_message("conv2", MessageRole.ASSISTANT, "Hi!")
            
            stats = manager.get_memory_stats(user_id="user123")
            
            assert stats["total_conversations"] == 1
            assert stats["total_messages"] == 1


class TestMemoryPersistence:
    """Test memory persistence."""
    
    def test_conversation_saved_to_disk(self):
        """Test that conversation is saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            # Check file exists
            conv_files = list(Path(tmpdir).glob("*.json"))
            assert len(conv_files) == 1
    
    def test_conversation_loaded_from_disk(self):
        """Test that conversation is loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save conversation
            manager1 = MemoryManager(storage_dir=tmpdir)
            manager1.create_conversation("conv123", "user123")
            manager1.add_message("conv123", MessageRole.USER, "Hello")
            
            # Create new manager (should load from disk)
            manager2 = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager2.get_conversation("conv123")
            assert conversation is not None
            assert conversation.conversation_id == "conv123"
            assert len(conversation.messages) == 1
            assert conversation.messages[0].content == "Hello"
    
    def test_message_addition_persisted(self):
        """Test that message additions are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create conversation and add message
            manager1 = MemoryManager(storage_dir=tmpdir)
            manager1.create_conversation("conv123", "user123")
            manager1.add_message("conv123", MessageRole.USER, "Hello")
            manager1.add_message("conv123", MessageRole.ASSISTANT, "Hi!")
            
            # Load from disk
            manager2 = MemoryManager(storage_dir=tmpdir)
            
            conversation = manager2.get_conversation("conv123")
            assert len(conversation.messages) == 2
            assert conversation.messages[0].content == "Hello"
            assert conversation.messages[1].content == "Hi!"
    
    def test_multiple_conversations_persisted(self):
        """Test that multiple conversations are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple conversations
            manager1 = MemoryManager(storage_dir=tmpdir)
            manager1.create_conversation("conv1", "user123")
            manager1.create_conversation("conv2", "user123")
            manager1.create_conversation("conv3", "user456")
            
            # Load from disk
            manager2 = MemoryManager(storage_dir=tmpdir)
            
            assert len(manager2.conversations) == 3
            assert manager2.get_conversation("conv1") is not None
            assert manager2.get_conversation("conv2") is not None
            assert manager2.get_conversation("conv3") is not None
    
    def test_conversation_deletion_removes_file(self):
        """Test that deleting conversation removes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            # Verify file exists
            conv_files = list(Path(tmpdir).glob("*.json"))
            assert len(conv_files) == 1
            
            # Delete conversation
            manager.delete_conversation("conv123")
            
            # Verify file removed
            conv_files = list(Path(tmpdir).glob("*.json"))
            assert len(conv_files) == 0
    
    def test_delete_conversation_success(self):
        """Test successful conversation deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv123", "user123")
            
            result = manager.delete_conversation("conv123")
            
            assert result is True
            assert manager.get_conversation("conv123") is None
    
    def test_delete_nonexistent_conversation(self):
        """Test deleting non-existent conversation returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            result = manager.delete_conversation("nonexistent")
            
            assert result is False
    
    def test_conversation_serialization_deserialization(self):
        """Test conversation to_dict and from_dict."""
        conversation = Conversation(
            conversation_id="conv123",
            user_id="user123"
        )
        conversation.add_message(MessageRole.USER, "Hello")
        conversation.add_message(MessageRole.ASSISTANT, "Hi!")
        conversation.summary = "Test summary"
        
        # Serialize
        data = conversation.to_dict()
        
        # Deserialize
        restored = Conversation.from_dict(data)
        
        assert restored.conversation_id == conversation.conversation_id
        assert restored.user_id == conversation.user_id
        assert len(restored.messages) == len(conversation.messages)
        assert restored.messages[0].content == "Hello"
        assert restored.messages[1].content == "Hi!"
        assert restored.summary == "Test summary"


class TestMemoryCleanup:
    """Test memory cleanup functionality."""
    
    def test_cleanup_old_conversations(self):
        """Test cleaning up old conversations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            # Create 5 conversations
            for i in range(5):
                conv = manager.create_conversation(f"conv{i}", "user123")
                # Manually set last_updated to different times
                conv.last_updated = datetime.now() - timedelta(days=i)
            
            # Keep only 3 most recent
            deleted = manager.cleanup_old_conversations("user123", keep_recent=3)
            
            assert deleted == 2
            
            # Check that only 3 conversations remain
            conversations = manager.get_user_conversations("user123")
            assert len(conversations) == 3
    
    def test_cleanup_no_conversations_to_delete(self):
        """Test cleanup when no conversations need to be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            manager.create_conversation("conv1", "user123")
            manager.create_conversation("conv2", "user123")
            
            # Keep 5 most recent (but only 2 exist)
            deleted = manager.cleanup_old_conversations("user123", keep_recent=5)
            
            assert deleted == 0
            
            conversations = manager.get_user_conversations("user123")
            assert len(conversations) == 2
    
    def test_cleanup_keeps_most_recent(self):
        """Test that cleanup keeps the most recent conversations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_dir=tmpdir)
            
            # Create conversations with different timestamps
            old_conv = manager.create_conversation("old", "user123")
            old_conv.last_updated = datetime.now() - timedelta(days=10)
            
            recent_conv = manager.create_conversation("recent", "user123")
            recent_conv.last_updated = datetime.now()
            
            # Keep only 1 most recent
            manager.cleanup_old_conversations("user123", keep_recent=1)
            
            # Recent should remain, old should be deleted
            assert manager.get_conversation("recent") is not None
            assert manager.get_conversation("old") is None


class TestMessageModel:
    """Test Message model."""
    
    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        metadata = {"emotion": "happy", "confidence": 0.9}
        message = Message(
            role=MessageRole.ASSISTANT,
            content="I'm here to help!",
            metadata=metadata
        )
        
        assert message.metadata["emotion"] == "happy"
        assert message.metadata["confidence"] == 0.9
    
    def test_message_serialization(self):
        """Test message serialization."""
        message = Message(
            role=MessageRole.USER,
            content="Test message",
            metadata={"key": "value"}
        )
        
        data = message.to_dict()
        
        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "timestamp" in data
        assert data["metadata"]["key"] == "value"
    
    def test_message_deserialization(self):
        """Test message deserialization."""
        data = {
            "role": "assistant",
            "content": "Test response",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"key": "value"}
        }
        
        message = Message.from_dict(data)
        
        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Test response"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata["key"] == "value"


class TestConversationModel:
    """Test Conversation model."""
    
    def test_conversation_message_count(self):
        """Test getting message count."""
        conversation = Conversation(
            conversation_id="conv123",
            user_id="user123"
        )
        
        assert conversation.message_count() == 0
        
        conversation.add_message(MessageRole.USER, "Hello")
        assert conversation.message_count() == 1
        
        conversation.add_message(MessageRole.ASSISTANT, "Hi!")
        assert conversation.message_count() == 2
    
    def test_conversation_add_message_updates_timestamp(self):
        """Test that adding message updates last_updated."""
        conversation = Conversation(
            conversation_id="conv123",
            user_id="user123"
        )
        
        original_timestamp = conversation.last_updated
        
        # Wait a tiny bit to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        conversation.add_message(MessageRole.USER, "Hello")
        
        assert conversation.last_updated > original_timestamp
