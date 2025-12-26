import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from morgan.core.domain.entities import KnowledgeChunk, ConversationTurn
from morgan.core.infrastructure.repositories import KnowledgeRepository, MemoryRepository

class TestRepositories:
    @pytest.fixture
    def mock_vector_db(self):
        return MagicMock()

    @pytest.fixture
    def knowledge_repo(self, mock_vector_db):
        return KnowledgeRepository(mock_vector_db, "test_knowledge")

    @pytest.fixture
    def memory_repo(self, mock_vector_db):
        return MemoryRepository(mock_vector_db, "test_turns", "test_convs")

    def test_store_knowledge_chunks(self, knowledge_repo, mock_vector_db):
        chunks = [
            KnowledgeChunk(
                content="Test content",
                source="test.txt",
                chunk_id="chunk1",
                metadata={"key": "value"},
                embedding=[0.1, 0.2, 0.3],
                ingested_at=datetime(2023, 10, 27, 10, 0, 0),
                embedding_type="legacy"
            )
        ]
        
        knowledge_repo.store_chunks(chunks)
        
        mock_vector_db.upsert_points.assert_called_once()
        args, kwargs = mock_vector_db.upsert_points.call_args
        assert args[0] == "test_knowledge"
        assert len(args[1]) == 1
        assert args[1][0]["id"] == "chunk1"
        assert args[1][0]["payload"]["content"] == "Test content"

    def test_search_knowledge(self, knowledge_repo, mock_vector_db):
        mock_res = MagicMock()
        mock_res.id = "chunk1"
        mock_res.payload = {
            "content": "Test content",
            "source": "test.txt",
            "metadata": {"key": "value"},
            "ingested_at": "2023-10-27T10:00:00",
            "embedding_type": "legacy"
        }
        mock_vector_db.search.return_value = [mock_res]
        
        results = knowledge_repo.search([0.1, 0.2, 0.3], limit=5, score_threshold=0.7)
        
        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].chunk_id == "chunk1"

    def test_store_memory_turn(self, memory_repo, mock_vector_db):
        turn = ConversationTurn(
            turn_id="turn1",
            conversation_id="conv1",
            timestamp=datetime(2023, 10, 27, 10, 0, 0),
            question="What is DDD?",
            answer="Domain-Driven Design",
            sources=["book.pdf"]
        )
        
        memory_repo.store_turn(turn, [0.1, 0.2, 0.3])
        
        mock_vector_db.upsert_points.assert_called_once()
        args, kwargs = mock_vector_db.upsert_points.call_args
        assert args[0] == "test_turns"
        assert args[1][0]["id"] == "turn1"
        assert args[1][0]["payload"]["question"] == "What is DDD?"

    def test_get_memory_history(self, memory_repo, mock_vector_db):
        mock_res = MagicMock()
        mock_res.id = "turn1"
        mock_res.payload = {
            "conversation_id": "conv1",
            "timestamp": "2023-10-27T10:00:00",
            "question": "What is DDD?",
            "answer": "Domain-Driven Design",
            "sources": ["book.pdf"]
        }
        mock_vector_db.search_with_filter.return_value = [mock_res]
        
        results = memory_repo.get_recent_turns("conv1", limit=10)
        
        assert len(results) == 1
        assert results[0].question == "What is DDD?"
        assert results[0].conversation_id == "conv1"
