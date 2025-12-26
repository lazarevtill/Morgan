"""
Integration tests for the Morgan Assistant.

This module tests the full integration of the assistant with all engines.
"""

import pytest
from unittest.mock import AsyncMock

from morgan_server.assistant import MorganAssistant
from morgan_server.empathic.emotional import EmotionalIntelligence
from morgan_server.empathic.personality import PersonalitySystem
from morgan_server.personalization.memory import MemoryManager
from morgan_server.personalization.preferences import PreferenceManager
from morgan_server.personalization.profile import ProfileManager


class TestAssistantIntegration:
    """Integration tests for the full assistant workflow."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        from morgan_server.llm import LLMResponse
        
        client = AsyncMock()
        client.generate = AsyncMock(return_value=LLMResponse(
            content="This is a test response from the assistant.",
            model="test-model"
        ))
        return client

    @pytest.fixture
    def mock_rag_system(self):
        """Create a mock RAG system."""
        rag = AsyncMock()
        rag.retrieve = AsyncMock(return_value={
            "context": "Test context",
            "sources": [],
            "confidence": 0.8
        })
        return rag

    @pytest.fixture
    def assistant(self, mock_llm_client, mock_rag_system, tmp_path):
        """Create an assistant instance with mocked dependencies."""
        emotional_intelligence = EmotionalIntelligence()
        personality_system = PersonalitySystem()
        memory_system = MemoryManager(str(tmp_path / "memory"))
        preference_manager = PreferenceManager(str(tmp_path / "preferences"))
        profile_manager = ProfileManager(str(tmp_path / "profiles"))

        assistant = MorganAssistant(
            llm_client=mock_llm_client,
            rag_system=mock_rag_system,
            emotional_intelligence=emotional_intelligence,
            personality_system=personality_system,
            memory_system=memory_system,
            preference_manager=preference_manager,
            profile_manager=profile_manager
        )

        return assistant

    @pytest.mark.asyncio
    async def test_full_chat_flow(self, assistant, mock_llm_client):
        """Test the full chat flow from message to response."""
        user_id = "test_user"
        message = "Hello, how are you?"

        response = await assistant.chat(user_id=user_id, message=message)

        # Verify response structure
        assert hasattr(response, "answer")
        assert hasattr(response, "conversation_id")
        assert response.answer == "This is a test response from the assistant."
        assert response.conversation_id.startswith("conv_")

        # Verify LLM was called
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_integration(self, assistant):
        """Test that all engines are integrated correctly."""
        user_id = "test_user"
        message = "I'm feeling great today!"

        response = await assistant.chat(user_id=user_id, message=message)

        # Verify emotional intelligence detected tone
        assert hasattr(response, "emotional_tone")

        # Verify response was generated
        assert hasattr(response, "answer")
        assert len(response.answer) > 0

    @pytest.mark.asyncio
    async def test_context_management(self, assistant):
        """Test that conversation context is managed correctly."""
        user_id = "test_user"

        # Send first message
        response1 = await assistant.chat(user_id=user_id, message="Hello")
        conversation_id = response1.conversation_id

        # Send second message in same conversation
        response2 = await assistant.chat(
            user_id=user_id,
            message="How are you?",
            conversation_id=conversation_id
        )

        # Verify same conversation
        assert response2.conversation_id == conversation_id
