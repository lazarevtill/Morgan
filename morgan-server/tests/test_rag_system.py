"""Unit tests for RAG system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morgan_server.knowledge.rag import (
    RAGSystem,
    EmbeddingService,
    RAGResult,
    Source,
    RAGError,
    EmbeddingError,
    RetrievalError,
)
from morgan_server.knowledge.vectordb import SearchResult


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def local_embedding_service(self):
        """Create a local embedding service for testing."""
        return EmbeddingService(
            provider="local",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

    @pytest.fixture
    def ollama_embedding_service(self):
        """Create an Ollama embedding service for testing."""
        return EmbeddingService(
            provider="ollama",
            model="nomic-embed-text",
            endpoint="http://localhost:11434",
        )

    @pytest.fixture
    def openai_embedding_service(self):
        """Create an OpenAI-compatible embedding service for testing."""
        return EmbeddingService(
            provider="openai-compatible",
            model="text-embedding-ada-002",
            endpoint="http://localhost:8000",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_embed_text_local(self, local_embedding_service):
        """Test embedding generation with local model."""
        with patch.object(
            local_embedding_service, "_embed_local", return_value=[0.1, 0.2, 0.3]
        ):
            embedding = await local_embedding_service.embed_text("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 3
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_local(self, local_embedding_service):
        """Test batch embedding generation with local model."""
        with patch.object(
            local_embedding_service,
            "_embed_batch_local",
            return_value=[[0.1, 0.2], [0.3, 0.4]],
        ):
            embeddings = await local_embedding_service.embed_batch(
                ["text1", "text2"]
            )
            assert isinstance(embeddings, list)
            assert len(embeddings) == 2
            assert all(isinstance(emb, list) for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_text_ollama(self, ollama_embedding_service):
        """Test embedding generation with Ollama."""
        with patch.object(
            ollama_embedding_service,
            "_embed_ollama",
            return_value=[0.1, 0.2, 0.3],
        ):
            embedding = await ollama_embedding_service.embed_text("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 3

    @pytest.mark.asyncio
    async def test_embed_text_openai(self, openai_embedding_service):
        """Test embedding generation with OpenAI-compatible API."""
        with patch.object(
            openai_embedding_service,
            "_embed_openai_compatible",
            return_value=[0.1, 0.2, 0.3],
        ):
            embedding = await openai_embedding_service.embed_text("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 3

    @pytest.mark.asyncio
    async def test_embed_text_unknown_provider(self):
        """Test embedding with unknown provider raises error."""
        service = EmbeddingService(provider="unknown")
        with pytest.raises(EmbeddingError, match="Unknown provider"):
            await service.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self, local_embedding_service):
        """Test error handling in embedding generation."""
        with patch.object(
            local_embedding_service,
            "_embed_local",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                await local_embedding_service.embed_text("test")

    def test_get_embedding_dimension(self, local_embedding_service):
        """Test getting embedding dimension."""
        dim = local_embedding_service.get_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2 dimension

    def test_get_embedding_dimension_unknown_model(self):
        """Test getting dimension for unknown model returns default."""
        service = EmbeddingService(model="unknown-model")
        dim = service.get_embedding_dimension()
        assert dim == 384  # Default dimension


class TestRAGSystem:
    """Tests for RAGSystem."""

    @pytest.fixture
    def mock_vectordb_client(self):
        """Create a mock vector database client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = AsyncMock()
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        service.embed_batch = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        return service

    @pytest.fixture
    def rag_system(self, mock_vectordb_client, mock_embedding_service):
        """Create a RAG system for testing."""
        return RAGSystem(
            vectordb_client=mock_vectordb_client,
            embedding_service=mock_embedding_service,
            collection_name="test_collection",
            top_k=10,
            rerank_top_k=5,
            score_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_retrieve_success(self, rag_system, mock_vectordb_client):
        """Test successful document retrieval."""
        # Mock search results
        mock_result = SearchResult(
            id="chunk1",
            score=0.95,
            payload={
                "document_id": "doc1",
                "content": "This is test content",
                "chunk_index": 0,
            },
        )
        mock_vectordb_client.search.return_value = [mock_result]

        result = await rag_system.retrieve("test query")

        assert isinstance(result, RAGResult)
        assert result.query == "test query"
        assert len(result.sources) == 1
        assert result.sources[0].document_id == "doc1"
        assert result.sources[0].content == "This is test content"
        assert result.confidence > 0.0
        assert result.context != ""

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self, rag_system, mock_vectordb_client):
        """Test retrieval with no results."""
        mock_vectordb_client.search.return_value = []

        result = await rag_system.retrieve("test query")

        assert isinstance(result, RAGResult)
        assert len(result.sources) == 0
        assert result.confidence == 0.0
        assert result.context == ""

    @pytest.mark.asyncio
    async def test_retrieve_with_reranking(self, rag_system, mock_vectordb_client):
        """Test retrieval with reranking enabled."""
        # Mock multiple search results
        mock_results = [
            SearchResult(
                id=f"chunk{i}",
                score=0.9 - i * 0.1,
                payload={
                    "document_id": f"doc{i}",
                    "content": f"Content {i} with test query terms",
                    "chunk_index": i,
                },
            )
            for i in range(3)
        ]
        mock_vectordb_client.search.return_value = mock_results

        result = await rag_system.retrieve("test query", rerank=True)

        assert len(result.sources) <= rag_system.rerank_top_k
        # Verify rerank scores were calculated
        assert all(s.rerank_score is not None for s in result.sources)

    @pytest.mark.asyncio
    async def test_retrieve_without_reranking(
        self, rag_system, mock_vectordb_client
    ):
        """Test retrieval without reranking."""
        mock_result = SearchResult(
            id="chunk1",
            score=0.95,
            payload={
                "document_id": "doc1",
                "content": "Test content",
                "chunk_index": 0,
            },
        )
        mock_vectordb_client.search.return_value = [mock_result]

        result = await rag_system.retrieve("test query", rerank=False)

        assert len(result.sources) == 1
        # Rerank scores should not be set
        assert result.sources[0].rerank_score is None

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_top_k(
        self, rag_system, mock_vectordb_client
    ):
        """Test retrieval with custom top_k parameter."""
        mock_results = [
            SearchResult(
                id=f"chunk{i}",
                score=0.9,
                payload={"document_id": f"doc{i}", "content": f"Content {i}"},
            )
            for i in range(5)
        ]
        mock_vectordb_client.search.return_value = mock_results

        result = await rag_system.retrieve("test query", top_k=3, rerank=False)

        # Verify search was called with custom top_k
        call_kwargs = mock_vectordb_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self, rag_system, mock_vectordb_client):
        """Test retrieval with filter conditions."""
        mock_vectordb_client.search.return_value = []
        filter_conditions = {"category": "tech"}

        await rag_system.retrieve("test query", filter_conditions=filter_conditions)

        # Verify filter was passed to search
        call_kwargs = mock_vectordb_client.search.call_args.kwargs
        assert call_kwargs["filter_conditions"] == filter_conditions

    @pytest.mark.asyncio
    async def test_retrieve_error_handling(self, rag_system, mock_vectordb_client):
        """Test error handling in retrieval."""
        mock_vectordb_client.search.side_effect = Exception("Search failed")

        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            await rag_system.retrieve("test query")

    @pytest.mark.asyncio
    async def test_index_document_success(
        self, rag_system, mock_vectordb_client, mock_embedding_service
    ):
        """Test successful document indexing."""
        with patch(
            "morgan_server.knowledge.rag.DocumentProcessor"
        ) as mock_processor_class:
            # Mock document processor
            mock_processor = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.content = "Test content"
            mock_chunk.chunk_id = "chunk1"
            mock_chunk.chunk_index = 0
            mock_chunk.metadata = {
                "document_id": "doc1",
                "source": "test.txt",
            }
            mock_processor.process.return_value = [mock_chunk]
            mock_processor_class.return_value = mock_processor

            # Mock vector insertion
            mock_vectordb_client.insert_vectors.return_value = ["chunk1"]

            num_chunks = await rag_system.index_document("test.txt")

            assert num_chunks == 1
            mock_processor.process.assert_called_once()
            mock_embedding_service.embed_batch.assert_called_once()
            mock_vectordb_client.insert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_document_unchanged(
        self, rag_system, mock_vectordb_client
    ):
        """Test indexing unchanged document returns 0."""
        with patch(
            "morgan_server.knowledge.rag.DocumentProcessor"
        ) as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor.process.return_value = []  # No chunks (unchanged)
            mock_processor_class.return_value = mock_processor

            num_chunks = await rag_system.index_document("test.txt")

            assert num_chunks == 0
            # Should not call embedding or insertion
            mock_vectordb_client.insert_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_document_with_metadata(
        self, rag_system, mock_vectordb_client, mock_embedding_service
    ):
        """Test indexing document with additional metadata."""
        with patch(
            "morgan_server.knowledge.rag.DocumentProcessor"
        ) as mock_processor_class:
            mock_processor = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.content = "Test content"
            mock_chunk.chunk_id = "chunk1"
            mock_chunk.metadata = {"document_id": "doc1"}
            mock_processor.process.return_value = [mock_chunk]
            mock_processor_class.return_value = mock_processor

            metadata = {"category": "tech", "author": "test"}
            await rag_system.index_document("test.txt", metadata=metadata)

            # Verify metadata was passed to processor (3rd positional argument)
            call_args = mock_processor.process.call_args.args
            assert len(call_args) == 3
            assert call_args[2] == metadata

    @pytest.mark.asyncio
    async def test_index_document_error_handling(self, rag_system):
        """Test error handling in document indexing."""
        with patch(
            "morgan_server.knowledge.rag.DocumentProcessor"
        ) as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor.process.side_effect = Exception("Processing failed")
            mock_processor_class.return_value = mock_processor

            with pytest.raises(RAGError, match="Failed to index document"):
                await rag_system.index_document("test.txt")

    @pytest.mark.asyncio
    async def test_search_similar_success(self, rag_system, mock_vectordb_client):
        """Test simple semantic search."""
        mock_result = SearchResult(
            id="chunk1",
            score=0.95,
            payload={
                "document_id": "doc1",
                "content": "Test content",
            },
        )
        mock_vectordb_client.search.return_value = [mock_result]

        sources = await rag_system.search_similar("test query", limit=5)

        assert len(sources) == 1
        assert isinstance(sources[0], Source)
        assert sources[0].document_id == "doc1"
        assert sources[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_search_similar_with_filter(
        self, rag_system, mock_vectordb_client
    ):
        """Test semantic search with filter."""
        mock_vectordb_client.search.return_value = []
        filter_conditions = {"category": "tech"}

        await rag_system.search_similar(
            "test query", filter_conditions=filter_conditions
        )

        call_kwargs = mock_vectordb_client.search.call_args.kwargs
        assert call_kwargs["filter_conditions"] == filter_conditions

    @pytest.mark.asyncio
    async def test_search_similar_error_handling(
        self, rag_system, mock_vectordb_client
    ):
        """Test error handling in semantic search."""
        mock_vectordb_client.search.side_effect = Exception("Search failed")

        with pytest.raises(RetrievalError, match="Failed to search"):
            await rag_system.search_similar("test query")

    @pytest.mark.asyncio
    async def test_get_stats_success(self, rag_system, mock_vectordb_client):
        """Test getting knowledge base statistics."""
        from morgan_server.knowledge.vectordb import CollectionStats

        mock_stats = CollectionStats(
            name="test_collection",
            vectors_count=100,
            indexed_vectors_count=100,
            points_count=100,
            segments_count=1,
            status="green",
        )
        mock_vectordb_client.get_collection_stats.return_value = mock_stats

        stats = await rag_system.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 100
        assert stats["indexed_chunks"] == 100
        assert stats["status"] == "green"

    @pytest.mark.asyncio
    async def test_get_stats_collection_not_found(
        self, rag_system, mock_vectordb_client
    ):
        """Test getting stats when collection doesn't exist."""
        mock_vectordb_client.get_collection_stats.return_value = None

        stats = await rag_system.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 0
        assert stats["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_stats_error_handling(
        self, rag_system, mock_vectordb_client
    ):
        """Test error handling in get_stats."""
        mock_vectordb_client.get_collection_stats.side_effect = Exception(
            "Stats failed"
        )

        stats = await rag_system.get_stats()

        assert "error" in stats
        assert stats["collection_name"] == "test_collection"

    def test_build_context(self, rag_system):
        """Test context building from sources."""
        sources = [
            Source(
                document_id="doc1",
                chunk_id="chunk1",
                content="First content",
                score=0.95,
            ),
            Source(
                document_id="doc2",
                chunk_id="chunk2",
                content="Second content",
                score=0.85,
            ),
        ]

        context = rag_system._build_context(sources)

        assert "First content" in context
        assert "Second content" in context
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "0.950" in context  # Score formatting

    def test_build_context_empty(self, rag_system):
        """Test context building with no sources."""
        context = rag_system._build_context([])
        assert context == ""

    def test_calculate_confidence(self, rag_system):
        """Test confidence calculation."""
        sources = [
            Source("doc1", "chunk1", "content1", 0.95),
            Source("doc2", "chunk2", "content2", 0.85),
            Source("doc3", "chunk3", "content3", 0.75),
        ]

        confidence = rag_system._calculate_confidence(sources)

        assert 0.0 <= confidence <= 1.0
        # Higher scores should result in higher confidence
        assert confidence > 0.7

    def test_calculate_confidence_empty(self, rag_system):
        """Test confidence calculation with no sources."""
        confidence = rag_system._calculate_confidence([])
        assert confidence == 0.0

    def test_calculate_confidence_single_source(self, rag_system):
        """Test confidence calculation with single source."""
        sources = [Source("doc1", "chunk1", "content1", 0.95)]
        confidence = rag_system._calculate_confidence(sources)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_sources(self, rag_system):
        """Test source reranking."""
        sources = [
            Source("doc1", "chunk1", "content with query terms", 0.8),
            Source("doc2", "chunk2", "unrelated content", 0.9),
            Source("doc3", "chunk3", "query terms here", 0.7),
        ]

        reranked = await rag_system._rerank_sources("query terms", sources)

        # Verify rerank scores were calculated
        assert all(s.rerank_score is not None for s in reranked)
        # Verify sources are sorted by rerank score
        scores = [s.rerank_score for s in reranked]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieve_confidence_scoring(
        self, rag_system, mock_vectordb_client
    ):
        """Test confidence scoring in retrieval."""
        # High score results should give high confidence
        high_score_results = [
            SearchResult(
                id=f"chunk{i}",
                score=0.95 - i * 0.01,
                payload={"document_id": f"doc{i}", "content": f"Content {i}"},
            )
            for i in range(3)
        ]
        mock_vectordb_client.search.return_value = high_score_results

        result = await rag_system.retrieve("test query", rerank=False)
        assert result.confidence > 0.8

        # Low score results should give lower confidence
        low_score_results = [
            SearchResult(
                id=f"chunk{i}",
                score=0.55 - i * 0.01,
                payload={"document_id": f"doc{i}", "content": f"Content {i}"},
            )
            for i in range(3)
        ]
        mock_vectordb_client.search.return_value = low_score_results

        result = await rag_system.retrieve("test query", rerank=False)
        assert result.confidence < 0.7
