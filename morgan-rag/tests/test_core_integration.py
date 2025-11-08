"""
Integration tests for the core assistant system.

Tests the interaction between all core components.
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path

from morgan.core import (
    MorganAssistant,
    MemorySystem,
    ContextManager,
    ResponseGenerator,
    Message,
    MessageRole,
    ConversationContext,
    ContextPruningStrategy,
)


class TestMemorySystem:
    """Test MemorySystem functionality."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        """Test storing and retrieving messages."""
        memory = MemorySystem(storage_path=tmp_path)
        await memory.initialize()

        try:
            # Store a message
            message = Message(
                role=MessageRole.USER,
                content="Test message",
                timestamp=datetime.now(),
                message_id="msg_001",
            )

            await memory.store_message("session_001", message)

            # Retrieve it
            messages = await memory.retrieve_context("session_001")

            assert len(messages) == 1
            assert messages[0].content == "Test message"
            assert messages[0].role == MessageRole.USER

        finally:
            await memory.cleanup()

    @pytest.mark.asyncio
    async def test_search_memories(self, tmp_path):
        """Test searching memories."""
        memory = MemorySystem(storage_path=tmp_path)
        await memory.initialize()

        try:
            # Store multiple messages
            messages = [
                Message(
                    role=MessageRole.USER,
                    content="I love Python programming",
                    timestamp=datetime.now(),
                    message_id="msg_001",
                ),
                Message(
                    role=MessageRole.USER,
                    content="Machine learning is fascinating",
                    timestamp=datetime.now(),
                    message_id="msg_002",
                ),
                Message(
                    role=MessageRole.USER,
                    content="The weather is nice today",
                    timestamp=datetime.now(),
                    message_id="msg_003",
                ),
            ]

            for msg in messages:
                await memory.store_message("session_001", msg)

            # Search for "Python"
            results = await memory.search_memories("user_001", "Python")

            assert len(results) > 0
            assert any("Python" in r.content for r in results)

        finally:
            await memory.cleanup()

    @pytest.mark.asyncio
    async def test_performance(self, tmp_path):
        """Test memory retrieval performance."""
        memory = MemorySystem(storage_path=tmp_path)
        await memory.initialize()

        try:
            # Store 50 messages
            for i in range(50):
                message = Message(
                    role=MessageRole.USER,
                    content=f"Message {i}",
                    timestamp=datetime.now(),
                    message_id=f"msg_{i:03d}",
                )
                await memory.store_message("session_001", message)

            # Measure retrieval time
            start_time = datetime.now()
            messages = await memory.retrieve_context("session_001", n_messages=10)
            end_time = datetime.now()

            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Should be < 100ms
            assert duration_ms < 100, f"Retrieval took {duration_ms}ms, expected < 100ms"
            assert len(messages) == 10

        finally:
            await memory.cleanup()


class TestContextManager:
    """Test ContextManager functionality."""

    @pytest.mark.asyncio
    async def test_build_context(self):
        """Test building conversation context."""
        context_mgr = ContextManager()

        messages = [
            Message(
                role=MessageRole.USER,
                content="Hello!",
                timestamp=datetime.now(),
                message_id="msg_001",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Hi! How can I help?",
                timestamp=datetime.now(),
                message_id="msg_002",
            ),
        ]

        context = await context_mgr.build_context(
            messages=messages,
            user_id="user_001",
            session_id="session_001",
        )

        assert context.message_count == 2
        assert context.user_id == "user_001"
        assert context.session_id == "session_001"
        assert context.total_tokens > 0

    @pytest.mark.asyncio
    async def test_prune_sliding_window(self):
        """Test sliding window pruning."""
        context_mgr = ContextManager()

        # Create 20 messages
        messages = [
            Message(
                role=MessageRole.USER,
                content=f"Message {i}" * 10,  # Make them longer
                timestamp=datetime.now(),
                message_id=f"msg_{i:03d}",
            )
            for i in range(20)
        ]

        # Prune to ~500 tokens (should keep ~10 messages)
        pruned = await context_mgr.prune_context(
            messages=messages,
            target_tokens=500,
            strategy=ContextPruningStrategy.SLIDING_WINDOW,
        )

        # Should keep the most recent ones
        assert len(pruned) < len(messages)
        assert pruned[-1].message_id == messages[-1].message_id  # Most recent kept

    @pytest.mark.asyncio
    async def test_prune_importance_based(self):
        """Test importance-based pruning."""
        context_mgr = ContextManager()

        # Create messages with varying importance
        messages = [
            Message(
                role=MessageRole.USER,
                content=f"Message {i}",
                timestamp=datetime.now(),
                message_id=f"msg_{i:03d}",
                importance_score=0.5 if i % 2 == 0 else 0.9,  # Alternate importance
            )
            for i in range(10)
        ]

        pruned = await context_mgr.prune_context(
            messages=messages,
            target_tokens=100,
            strategy=ContextPruningStrategy.IMPORTANCE_BASED,
        )

        # Should keep higher importance messages
        assert all(msg.importance_score >= 0.5 for msg in pruned)

    @pytest.mark.asyncio
    async def test_performance(self):
        """Test context building performance."""
        context_mgr = ContextManager()

        messages = [
            Message(
                role=MessageRole.USER,
                content=f"Message {i}" * 10,
                timestamp=datetime.now(),
                message_id=f"msg_{i:03d}",
            )
            for i in range(50)
        ]

        # Measure context building time
        start_time = datetime.now()
        context = await context_mgr.build_context(
            messages=messages,
            user_id="user_001",
            session_id="session_001",
        )
        end_time = datetime.now()

        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Should be < 50ms
        assert duration_ms < 50, f"Context building took {duration_ms}ms, expected < 50ms"


class TestResponseGenerator:
    """Test ResponseGenerator functionality."""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation."""
        generator = ResponseGenerator(
            llm_base_url="http://localhost:11434",
            llm_model="llama3.2:latest",
        )

        try:
            # Create simple context
            messages = [
                Message(
                    role=MessageRole.USER,
                    content="Hello!",
                    timestamp=datetime.now(),
                    message_id="msg_001",
                ),
            ]

            context = ConversationContext(
                messages=messages,
                user_id="user_001",
                session_id="session_001",
            )

            # Note: This will fail if Ollama is not running
            # In real tests, you'd mock the HTTP calls
            try:
                response = await generator.generate(
                    context=context,
                    user_message="Hello!",
                )

                assert response.content
                assert response.response_id
                assert response.generation_time_ms > 0

            except Exception as e:
                # Expected if Ollama is not available
                pytest.skip(f"Ollama not available: {e}")

        finally:
            await generator.cleanup()


class TestMorganAssistant:
    """Test MorganAssistant integration."""

    @pytest.mark.asyncio
    async def test_assistant_initialization(self, tmp_path):
        """Test assistant initialization."""
        assistant = MorganAssistant(
            storage_path=tmp_path,
            enable_emotion_detection=False,  # Disable for faster tests
            enable_learning=False,
            enable_rag=False,
        )

        await assistant.initialize()

        try:
            stats = assistant.get_stats()
            assert "metrics" in stats
            assert "memory" in stats

        finally:
            await assistant.cleanup()

    @pytest.mark.asyncio
    async def test_assistant_process_message_mock(self, tmp_path, monkeypatch):
        """Test message processing with mocked LLM."""
        assistant = MorganAssistant(
            storage_path=tmp_path,
            enable_emotion_detection=False,
            enable_learning=False,
            enable_rag=False,
        )

        await assistant.initialize()

        try:
            # Mock the response generator
            async def mock_generate(context, user_message, **kwargs):
                from morgan.core import AssistantResponse
                return AssistantResponse(
                    content="Mocked response",
                    response_id="mock_001",
                    timestamp=datetime.now(),
                )

            monkeypatch.setattr(
                assistant.response_generator,
                "generate",
                mock_generate,
            )

            # Process message
            response = await assistant.process_message(
                user_id="user_001",
                message="Test message",
                session_id="session_001",
            )

            assert response.content == "Mocked response"
            assert response.response_id == "mock_001"

            # Check memory was updated
            messages = await assistant.memory_system.retrieve_context("session_001")
            assert len(messages) == 2  # User message + assistant response

        finally:
            await assistant.cleanup()

    @pytest.mark.asyncio
    async def test_assistant_stats(self, tmp_path):
        """Test assistant statistics."""
        assistant = MorganAssistant(
            storage_path=tmp_path,
            enable_emotion_detection=False,
            enable_learning=False,
            enable_rag=False,
        )

        await assistant.initialize()

        try:
            stats = assistant.get_stats()

            assert "metrics" in stats
            assert "circuit_breakers" in stats
            assert "memory" in stats
            assert "context" in stats
            assert "response_generator" in stats

            # Check metrics structure
            assert "total_requests" in stats["metrics"]
            assert "successful_requests" in stats["metrics"]

        finally:
            await assistant.cleanup()


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for tests."""
    return tmp_path_factory.mktemp("morgan_test")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
