"""Unit tests for Knowledge API routes."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.knowledge import (
    router,
    set_rag_system,
    get_rag_system,
)
from morgan_server.knowledge.rag import RAGSystem, Source as RAGSource


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for testing."""
    system = MagicMock(spec=RAGSystem)

    # Mock index_document
    system.index_document = AsyncMock(return_value=5)

    # Mock search_similar
    test_sources = [
        RAGSource(
            document_id="doc_123",
            chunk_id="chunk_1",
            content="Test content about Python programming",
            score=0.85,
            metadata={"filename": "test.md", "doc_type": "markdown"},
        ),
        RAGSource(
            document_id="doc_456",
            chunk_id="chunk_2",
            content="More test content about machine learning",
            score=0.75,
            metadata={"filename": "ml.pdf", "doc_type": "pdf"},
        ),
    ]
    system.search_similar = AsyncMock(return_value=test_sources)

    # Mock get_stats
    system.get_stats = AsyncMock(
        return_value={
            "collection_name": "knowledge_base",
            "total_chunks": 100,
            "indexed_chunks": 100,
            "status": "green",
        }
    )

    return system


@pytest.fixture
def app(mock_rag_system):
    """Create a FastAPI app with the knowledge router for testing."""
    app = FastAPI()
    app.include_router(router)

    # Set the mock RAG system
    set_rag_system(mock_rag_system)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestLearnEndpoint:
    """Tests for POST /api/knowledge/learn endpoint."""

    def test_learn_from_file_success(self, client, mock_rag_system):
        """Test successful document learning from file."""
        request_data = {
            "source": "/path/to/document.pdf",
            "doc_type": "pdf",
            "metadata": {"author": "Test Author"},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "status" in data
        assert "documents_processed" in data
        assert "chunks_created" in data
        assert "processing_time_seconds" in data

        # Verify values
        assert data["status"] == "success"
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 5
        assert data["processing_time_seconds"] >= 0

        # Verify RAG system was called correctly
        mock_rag_system.index_document.assert_called_once_with(
            source="/path/to/document.pdf",
            doc_type="pdf",
            metadata={"author": "Test Author"},
        )

    def test_learn_from_url_success(self, client, mock_rag_system):
        """Test successful document learning from URL."""
        request_data = {
            "url": "https://example.com/article",
            "doc_type": "auto",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 5

        # Verify RAG system was called with URL
        mock_rag_system.index_document.assert_called_once_with(
            source="https://example.com/article",
            doc_type="auto",
            metadata={},
        )

    def test_learn_no_source_provided(self, client):
        """Test learning without any source (should fail)."""
        request_data = {
            "doc_type": "auto",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        # Should return 422 error (validation error from Pydantic)
        assert response.status_code == 422

    def test_learn_from_content_not_implemented(self, client):
        """Test learning from direct content (not yet implemented)."""
        request_data = {
            "content": "Direct content to learn",
            "doc_type": "text",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        # Should return 501 error (not implemented)
        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"].lower()

    def test_learn_invalid_doc_type(self, client):
        """Test learning with invalid document type."""
        request_data = {
            "source": "/path/to/document.txt",
            "doc_type": "invalid_type",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_learn_no_changes(self, client, mock_rag_system):
        """Test learning when document hasn't changed (no chunks created)."""
        # Mock no chunks created (document unchanged)
        mock_rag_system.index_document = AsyncMock(return_value=0)

        request_data = {
            "source": "/path/to/unchanged.pdf",
            "doc_type": "pdf",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "no_changes"
        assert data["documents_processed"] == 0
        assert data["chunks_created"] == 0

    def test_learn_with_metadata(self, client, mock_rag_system):
        """Test learning with custom metadata."""
        request_data = {
            "source": "/path/to/document.md",
            "doc_type": "markdown",
            "metadata": {
                "author": "John Doe",
                "category": "tutorial",
                "tags": ["python", "programming"],
            },
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        assert response.status_code == 200

        # Verify metadata was passed correctly
        call_args = mock_rag_system.index_document.call_args
        assert call_args.kwargs["metadata"]["author"] == "John Doe"
        assert call_args.kwargs["metadata"]["category"] == "tutorial"

    def test_learn_rag_system_error(self, client, mock_rag_system):
        """Test learning when RAG system raises an error."""
        # Make RAG system raise an error
        mock_rag_system.index_document = AsyncMock(side_effect=Exception("Test error"))

        request_data = {
            "source": "/path/to/document.pdf",
            "doc_type": "pdf",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to learn document" in response.json()["detail"]

    def test_learn_rag_system_not_initialized(self, client):
        """Test learning when RAG system is not initialized."""
        # Clear the global RAG system
        set_rag_system(None)

        request_data = {
            "source": "/path/to/document.pdf",
            "doc_type": "pdf",
            "metadata": {},
        }

        response = client.post("/api/knowledge/learn", json=request_data)

        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestSearchEndpoint:
    """Tests for GET /api/knowledge/search endpoint."""

    def test_search_success(self, client, mock_rag_system):
        """Test successful knowledge search."""
        response = client.get("/api/knowledge/search?query=python programming&limit=5")

        assert response.status_code == 200
        data = response.json()

        # Verify response is a list
        assert isinstance(data, list)
        assert len(data) == 2

        # Verify result structure
        result = data[0]
        assert "content" in result
        assert "document_id" in result
        assert "chunk_id" in result
        assert "score" in result
        assert "metadata" in result

        # Verify values
        assert result["document_id"] == "doc_123"
        assert result["chunk_id"] == "chunk_1"
        assert result["content"] == "Test content about Python programming"
        assert result["score"] == 0.85

        # Verify RAG system was called correctly
        mock_rag_system.search_similar.assert_called_once_with(
            query="python programming",
            limit=5,
            filter_conditions=None,
        )

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.get("/api/knowledge/search?query=")

        # Should fail validation
        assert response.status_code == 422

    def test_search_whitespace_query(self, client):
        """Test search with whitespace-only query."""
        response = client.get("/api/knowledge/search?query=   ")

        # Should return 400 error
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_search_missing_query(self, client):
        """Test search without query parameter."""
        response = client.get("/api/knowledge/search")

        # Should fail validation
        assert response.status_code == 422

    def test_search_custom_limit(self, client, mock_rag_system):
        """Test search with custom limit."""
        response = client.get("/api/knowledge/search?query=machine learning&limit=10")

        assert response.status_code == 200

        # Verify limit was passed correctly
        mock_rag_system.search_similar.assert_called_once_with(
            query="machine learning",
            limit=10,
            filter_conditions=None,
        )

    def test_search_default_limit(self, client, mock_rag_system):
        """Test search with default limit."""
        response = client.get("/api/knowledge/search?query=test")

        assert response.status_code == 200

        # Verify default limit (5) was used
        mock_rag_system.search_similar.assert_called_once_with(
            query="test",
            limit=5,
            filter_conditions=None,
        )

    def test_search_limit_too_high(self, client):
        """Test search with limit exceeding maximum."""
        response = client.get("/api/knowledge/search?query=test&limit=100")

        # Should fail validation (max is 50)
        assert response.status_code == 422

    def test_search_limit_too_low(self, client):
        """Test search with limit below minimum."""
        response = client.get("/api/knowledge/search?query=test&limit=0")

        # Should fail validation (min is 1)
        assert response.status_code == 422

    def test_search_with_score_threshold(self, client, mock_rag_system):
        """Test search with custom score threshold."""
        response = client.get("/api/knowledge/search?query=test&score_threshold=0.8")

        assert response.status_code == 200
        data = response.json()

        # Only results with score >= 0.8 should be returned
        assert len(data) == 1  # Only first result has score 0.85
        assert data[0]["score"] >= 0.8

    def test_search_score_threshold_filters_all(self, client, mock_rag_system):
        """Test search where threshold filters out all results."""
        response = client.get("/api/knowledge/search?query=test&score_threshold=0.9")

        assert response.status_code == 200
        data = response.json()

        # All results should be filtered out (max score is 0.85)
        assert len(data) == 0

    def test_search_no_results(self, client, mock_rag_system):
        """Test search with no matching results."""
        # Mock empty results
        mock_rag_system.search_similar = AsyncMock(return_value=[])

        response = client.get("/api/knowledge/search?query=nonexistent")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    def test_search_rag_system_error(self, client, mock_rag_system):
        """Test search when RAG system raises an error."""
        # Make RAG system raise an error
        mock_rag_system.search_similar = AsyncMock(side_effect=Exception("Test error"))

        response = client.get("/api/knowledge/search?query=test")

        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to search knowledge" in response.json()["detail"]

    def test_search_rag_system_not_initialized(self, client):
        """Test search when RAG system is not initialized."""
        # Clear the global RAG system
        set_rag_system(None)

        response = client.get("/api/knowledge/search?query=test")

        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestStatsEndpoint:
    """Tests for GET /api/knowledge/stats endpoint."""

    def test_get_stats_success(self, client, mock_rag_system):
        """Test successful stats retrieval."""
        response = client.get("/api/knowledge/stats")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "total_size_bytes" in data
        assert "collections" in data

        # Verify values
        assert data["total_chunks"] == 100
        assert isinstance(data["collections"], list)
        assert "knowledge_base" in data["collections"]

        # Verify RAG system was called
        mock_rag_system.get_stats.assert_called_once()

    def test_get_stats_empty_knowledge_base(self, client, mock_rag_system):
        """Test stats retrieval when knowledge base is empty."""
        # Mock empty stats
        mock_rag_system.get_stats = AsyncMock(
            return_value={
                "collection_name": "knowledge_base",
                "total_chunks": 0,
                "indexed_chunks": 0,
                "status": "green",
            }
        )

        response = client.get("/api/knowledge/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_chunks"] == 0

    def test_get_stats_rag_system_error(self, client, mock_rag_system):
        """Test stats retrieval when RAG system raises an error."""
        # Make RAG system raise an error
        mock_rag_system.get_stats = AsyncMock(side_effect=Exception("Test error"))

        response = client.get("/api/knowledge/stats")

        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to retrieve knowledge stats" in response.json()["detail"]

    def test_get_stats_rag_system_not_initialized(self, client):
        """Test stats retrieval when RAG system is not initialized."""
        # Clear the global RAG system
        set_rag_system(None)

        response = client.get("/api/knowledge/stats")

        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_set_and_get_rag_system(self, mock_rag_system):
        """Test setting and getting RAG system."""
        set_rag_system(mock_rag_system)

        retrieved = get_rag_system()

        assert retrieved == mock_rag_system

    def test_get_rag_system_not_initialized(self):
        """Test get_rag_system when not initialized."""
        # Clear the global RAG system
        set_rag_system(None)

        with pytest.raises(Exception) as exc_info:
            get_rag_system()

        # Should raise HTTPException with 503 status
        assert (
            "503" in str(exc_info.value)
            or "not initialized" in str(exc_info.value).lower()
        )
