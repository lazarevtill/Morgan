"""
Unit tests for the search system.

Tests vector search, hybrid search, and reranking functionality.
"""

import pytest
from unittest.mock import AsyncMock

from morgan_server.knowledge.search import (
    SearchSystem,
    SearchQuery,
    RankedResult,
    KeywordSearcher,
    ResultReranker,
    SearchError,
)
from morgan_server.knowledge.vectordb import (
    VectorDBClient,
    SearchResult,
)


@pytest.fixture
def mock_vectordb_client():
    """Create a mock vector database client."""
    client = AsyncMock(spec=VectorDBClient)
    return client


@pytest.fixture
def search_system(mock_vectordb_client):
    """Create a search system instance."""
    return SearchSystem(
        vectordb_client=mock_vectordb_client,
        collection_name="test_collection",
        use_cross_encoder=False,
    )


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    return [
        SearchResult(
            id="doc1",
            score=0.9,
            payload={
                "content": "Python is a programming language",
                "document_id": "doc1",
            },
        ),
        SearchResult(
            id="doc2",
            score=0.8,
            payload={
                "content": "Machine learning with Python",
                "document_id": "doc2",
            },
        ),
        SearchResult(
            id="doc3",
            score=0.7,
            payload={
                "content": "Data science and analytics",
                "document_id": "doc3",
            },
        ),
    ]


class TestKeywordSearcher:
    """Tests for KeywordSearcher."""

    def test_tokenize(self):
        """Test text tokenization."""
        searcher = KeywordSearcher()
        text = "The quick brown fox jumps over the lazy dog"
        tokens = searcher.tokenize(text)

        # Should remove stop words and short words
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens  # Stop word
        assert "a" not in tokens  # Stop word

    def test_score_match_exact(self):
        """Test scoring with exact match."""
        searcher = KeywordSearcher()
        query = "machine learning"
        document = "This is about machine learning algorithms"

        score = searcher.score_match(query, document)
        assert score > 0.5  # Should have high score for exact match

    def test_score_match_partial(self):
        """Test scoring with partial match."""
        searcher = KeywordSearcher()
        query = "machine learning"
        document = "This is about machine algorithms"

        score = searcher.score_match(query, document)
        assert 0 < score < 0.5  # Should have moderate score

    def test_score_match_no_match(self):
        """Test scoring with no match."""
        searcher = KeywordSearcher()
        query = "machine learning"
        document = "This is about cooking recipes"

        score = searcher.score_match(query, document)
        assert score == 0.0  # Should have zero score

    def test_search(self, sample_search_results):
        """Test keyword search."""
        searcher = KeywordSearcher()
        query = "Python programming"

        results = searcher.search(query, sample_search_results, limit=2)

        assert len(results) <= 2
        assert all(isinstance(r, RankedResult) for r in results)
        # Results should be sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score


class TestResultReranker:
    """Tests for ResultReranker."""

    def test_rerank_heuristic(self):
        """Test heuristic reranking."""
        reranker = ResultReranker(use_cross_encoder=False)

        results = [
            RankedResult(
                id="doc1",
                content="Python is great for machine learning",
                score=0.7,
                metadata={},
            ),
            RankedResult(
                id="doc2",
                content="Machine learning with Python is powerful",
                score=0.8,
                metadata={},
            ),
            RankedResult(
                id="doc3",
                content="Data science tools",
                score=0.9,
                metadata={},
            ),
        ]

        query = "machine learning Python"
        reranked = reranker.rerank(query, results)

        assert len(reranked) == 3
        # Check that rerank scores were assigned
        assert all(r.rerank_score is not None for r in reranked)
        # Check that ranks were assigned
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2
        assert reranked[2].rank == 3

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        reranker = ResultReranker(use_cross_encoder=False)

        results = [
            RankedResult(
                id=f"doc{i}",
                content=f"Document {i}",
                score=0.5,
                metadata={},
            )
            for i in range(10)
        ]

        query = "test query"
        reranked = reranker.rerank(query, results, top_k=3)

        assert len(reranked) == 3

    def test_rerank_empty_results(self):
        """Test reranking with empty results."""
        reranker = ResultReranker(use_cross_encoder=False)

        results = []
        query = "test query"
        reranked = reranker.rerank(query, results)

        assert len(reranked) == 0


class TestSearchSystem:
    """Tests for SearchSystem."""

    @pytest.mark.asyncio
    async def test_vector_search(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test vector similarity search."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query
        query = SearchQuery(
            query="test query",
            limit=5,
            search_type="vector",
            rerank=False,
        )
        query_vector = [0.1] * 384

        # Execute search
        results = await search_system.search(query, query_vector)

        # Verify
        assert len(results) <= query.limit
        assert all(isinstance(r, RankedResult) for r in results)
        mock_vectordb_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_without_vector(
        self,
        search_system,
    ):
        """Test vector search fails without query vector."""
        query = SearchQuery(
            query="test query",
            search_type="vector",
        )

        with pytest.raises(SearchError, match="Query vector required"):
            await search_system.search(query, query_vector=None)

    @pytest.mark.asyncio
    async def test_keyword_search(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test keyword-based search."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query
        query = SearchQuery(
            query="Python programming",
            limit=5,
            search_type="keyword",
            rerank=False,
        )

        # Execute search
        results = await search_system.search(query)

        # Verify
        assert len(results) <= query.limit
        assert all(isinstance(r, RankedResult) for r in results)
        mock_vectordb_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test hybrid search (vector + keyword)."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query
        query = SearchQuery(
            query="Python programming",
            limit=5,
            search_type="hybrid",
            keyword_weight=0.3,
            rerank=False,
        )
        query_vector = [0.1] * 384

        # Execute search
        results = await search_system.search(query, query_vector)

        # Verify
        assert len(results) <= query.limit
        assert all(isinstance(r, RankedResult) for r in results)
        # Should have both vector and keyword scores
        for result in results:
            assert result.vector_score is not None
            assert result.keyword_score is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_without_vector(
        self,
        search_system,
    ):
        """Test hybrid search fails without query vector."""
        query = SearchQuery(
            query="test query",
            search_type="hybrid",
        )

        with pytest.raises(SearchError, match="Query vector required"):
            await search_system.search(query, query_vector=None)

    @pytest.mark.asyncio
    async def test_search_with_reranking(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test search with reranking enabled."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query with reranking
        query = SearchQuery(
            query="Python programming",
            limit=2,
            search_type="vector",
            rerank=True,
        )
        query_vector = [0.1] * 384

        # Execute search
        results = await search_system.search(query, query_vector)

        # Verify
        assert len(results) <= query.limit
        # Check that rerank scores were assigned
        assert all(r.rerank_score is not None for r in results)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test search with score threshold filtering."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query with high threshold
        query = SearchQuery(
            query="test query",
            limit=10,
            score_threshold=0.85,
            search_type="vector",
            rerank=False,
        )
        query_vector = [0.1] * 384

        # Execute search
        results = await search_system.search(query, query_vector)

        # Verify - should filter out results below threshold
        assert all(r.score >= 0.85 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter_conditions(
        self,
        search_system,
        mock_vectordb_client,
        sample_search_results,
    ):
        """Test search with metadata filters."""
        # Setup mock
        mock_vectordb_client.search.return_value = sample_search_results

        # Create query with filters
        query = SearchQuery(
            query="test query",
            limit=5,
            search_type="vector",
            filter_conditions={"doc_type": "pdf"},
            rerank=False,
        )
        query_vector = [0.1] * 384

        # Execute search
        await search_system.search(query, query_vector)

        # Verify that filter was passed to vectordb
        call_args = mock_vectordb_client.search.call_args
        assert call_args[1]["filter_conditions"] == {"doc_type": "pdf"}

    @pytest.mark.asyncio
    async def test_search_unknown_type(
        self,
        search_system,
    ):
        """Test search with unknown search type."""
        query = SearchQuery(
            query="test query",
            search_type="unknown",
        )

        with pytest.raises(SearchError, match="Unknown search type"):
            await search_system.search(query)

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self,
        search_system,
        mock_vectordb_client,
    ):
        """Test search error handling."""
        # Setup mock to raise error
        mock_vectordb_client.search.side_effect = Exception("Database error")

        query = SearchQuery(
            query="test query",
            search_type="vector",
        )
        query_vector = [0.1] * 384

        with pytest.raises(SearchError, match="Search failed"):
            await search_system.search(query, query_vector)

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        search_system,
        mock_vectordb_client,
    ):
        """Test getting search system statistics."""
        # Setup mock
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

        # Get stats
        stats = await search_system.get_stats()

        # Verify
        assert stats["collection_name"] == "test_collection"
        assert stats["total_documents"] == 100
        assert stats["indexed_documents"] == 100
        assert stats["status"] == "green"

    @pytest.mark.asyncio
    async def test_get_stats_collection_not_found(
        self,
        search_system,
        mock_vectordb_client,
    ):
        """Test getting stats when collection doesn't exist."""
        # Setup mock
        mock_vectordb_client.get_collection_stats.return_value = None

        # Get stats
        stats = await search_system.get_stats()

        # Verify
        assert stats["collection_name"] == "test_collection"
        assert stats["total_documents"] == 0
        assert stats["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_stats_error(
        self,
        search_system,
        mock_vectordb_client,
    ):
        """Test getting stats with error."""
        # Setup mock to raise error
        mock_vectordb_client.get_collection_stats.side_effect = Exception(
            "Database error"
        )

        # Get stats
        stats = await search_system.get_stats()

        # Verify error is captured
        assert "error" in stats
        assert "Database error" in stats["error"]


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_default_values(self):
        """Test SearchQuery default values."""
        query = SearchQuery(query="test")

        assert query.query == "test"
        assert query.limit == 10
        assert query.score_threshold == 0.5
        assert query.filter_conditions is None
        assert query.search_type == "vector"
        assert query.rerank is True
        assert query.keyword_weight == 0.3

    def test_custom_values(self):
        """Test SearchQuery with custom values."""
        query = SearchQuery(
            query="test",
            limit=20,
            score_threshold=0.7,
            filter_conditions={"type": "pdf"},
            search_type="hybrid",
            rerank=False,
            keyword_weight=0.5,
        )

        assert query.limit == 20
        assert query.score_threshold == 0.7
        assert query.filter_conditions == {"type": "pdf"}
        assert query.search_type == "hybrid"
        assert query.rerank is False
        assert query.keyword_weight == 0.5


class TestRankedResult:
    """Tests for RankedResult dataclass."""

    def test_creation(self):
        """Test RankedResult creation."""
        result = RankedResult(
            id="doc1",
            content="Test content",
            score=0.9,
            metadata={"key": "value"},
        )

        assert result.id == "doc1"
        assert result.content == "Test content"
        assert result.score == 0.9
        assert result.metadata == {"key": "value"}
        assert result.vector_score is None
        assert result.keyword_score is None
        assert result.rerank_score is None
        assert result.rank == 0

    def test_with_all_scores(self):
        """Test RankedResult with all score types."""
        result = RankedResult(
            id="doc1",
            content="Test content",
            score=0.9,
            vector_score=0.85,
            keyword_score=0.75,
            rerank_score=0.95,
            rank=1,
        )

        assert result.vector_score == 0.85
        assert result.keyword_score == 0.75
        assert result.rerank_score == 0.95
        assert result.rank == 1
