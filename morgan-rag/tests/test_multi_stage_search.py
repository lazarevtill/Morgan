"""
Tests for Multi-Stage Search Engine.

Tests core functionality of the multi-stage search engine including:
- Search strategy execution
- Hierarchical filtering
- Result fusion
- Deduplication
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from morgan.search.multi_stage_search import (
    MultiStageSearchEngine,
    SearchStrategy,
    SearchResult,
    SearchResults,
)


class TestMultiStageSearchEngine:
    """Test multi-stage search engine functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch(
            "morgan.search.multi_stage_search.get_embedding_service"
        ) as mock_embedding, patch(
            "morgan.search.multi_stage_search.get_hierarchical_embedding_service"
        ) as mock_hierarchical, patch(
            "morgan.search.multi_stage_search.VectorDBClient"
        ) as mock_vector_db:

            # Setup embedding service mock
            mock_embedding_service = Mock()
            mock_embedding_service.encode.return_value = [0.1] * 384
            mock_embedding_service.encode_batch.return_value = [
                [0.1] * 384,
                [0.2] * 384,
                [0.3] * 384,
            ]
            mock_embedding.return_value = mock_embedding_service

            # Setup hierarchical service mock
            mock_hierarchical_service = Mock()
            mock_hierarchical.return_value = mock_hierarchical_service

            # Setup vector DB mock
            mock_db = Mock()
            mock_db.search.return_value = []
            mock_db.search_with_filter.return_value = []
            mock_vector_db.return_value = mock_db

            yield {
                "embedding_service": mock_embedding_service,
                "hierarchical_service": mock_hierarchical_service,
                "vector_db": mock_db,
            }

    @pytest.fixture
    def search_engine(self, mock_dependencies):
        """Create search engine with mocked dependencies."""
        return MultiStageSearchEngine()

    @pytest.fixture
    def sample_db_results(self):
        """Sample database results for testing."""
        return [
            Mock(
                score=0.9,
                payload={
                    "content": "Docker is a containerization platform",
                    "source": "docker-guide.md",
                    "metadata": {"category": "documentation"},
                    "coarse_embedding": [0.1] * 384,
                    "medium_embedding": [0.2] * 384,
                    "fine_embedding": [0.3] * 384,
                },
            ),
            Mock(
                score=0.8,
                payload={
                    "content": "How to install Docker on Ubuntu",
                    "source": "installation.md",
                    "metadata": {"category": "documentation"},
                    "coarse_embedding": [0.2] * 384,
                    "medium_embedding": [0.3] * 384,
                    "fine_embedding": [0.4] * 384,
                },
            ),
        ]

    def test_search_engine_initialization(self, search_engine):
        """Test search engine initializes correctly."""
        assert search_engine is not None
        assert search_engine.knowledge_collection == "morgan_knowledge"
        assert search_engine.memory_collection == "morgan_turns"
        assert search_engine.coarse_filter_ratio == 0.1
        assert search_engine.rrf_k == 60

    def test_basic_search_execution(
        self, search_engine, mock_dependencies, sample_db_results
    ):
        """Test basic search execution with single strategy."""
        # Setup mock to return sample results
        mock_dependencies["vector_db"].search.return_value = sample_db_results

        # Execute search
        results = search_engine.search(
            query="Docker installation",
            max_results=5,
            strategies=[SearchStrategy.SEMANTIC],
        )

        # Verify results
        assert isinstance(results, SearchResults)
        assert len(results.strategies_used) == 1
        assert SearchStrategy.SEMANTIC.value in results.strategies_used
        assert results.search_time > 0

    def test_multiple_strategy_search(
        self, search_engine, mock_dependencies, sample_db_results
    ):
        """Test search with multiple strategies and fusion."""
        # Setup mocks
        mock_dependencies["vector_db"].search.return_value = sample_db_results
        mock_dependencies["vector_db"].search_with_filter.return_value = (
            sample_db_results
        )

        # Execute search with multiple strategies
        results = search_engine.search(
            query="Docker installation",
            max_results=5,
            strategies=[
                SearchStrategy.SEMANTIC,
                SearchStrategy.KEYWORD,
                SearchStrategy.CATEGORY,
            ],
        )

        # Verify fusion was applied
        assert results.fusion_applied == True
        assert len(results.strategies_used) == 3
        assert SearchStrategy.SEMANTIC.value in results.strategies_used
        assert SearchStrategy.KEYWORD.value in results.strategies_used
        assert SearchStrategy.CATEGORY.value in results.strategies_used

    def test_hierarchical_search_filtering(
        self, search_engine, mock_dependencies, sample_db_results
    ):
        """Test hierarchical search with coarse-to-fine filtering."""
        # Setup mock to return results for hierarchical search
        mock_dependencies["vector_db"].search.return_value = sample_db_results

        # Mock similarity calculation to return valid scores
        with patch.object(search_engine, "_calculate_similarity", return_value=0.8):
            results = search_engine.search(
                query="Docker installation",
                max_results=5,
                strategies=[SearchStrategy.SEMANTIC],
                use_hierarchical=True,
            )

        # Verify hierarchical search was used
        assert isinstance(results, SearchResults)
        # Should have some candidate reduction
        assert results.total_candidates >= results.filtered_candidates

    def test_search_result_structure(
        self, search_engine, mock_dependencies, sample_db_results
    ):
        """Test search result structure and metadata."""
        mock_dependencies["vector_db"].search.return_value = sample_db_results

        results = search_engine.search(
            query="Docker installation",
            max_results=5,
            strategies=[SearchStrategy.SEMANTIC],
        )

        # Check result structure
        if len(results) > 0:
            result = results[0]
            assert isinstance(result, SearchResult)
            assert hasattr(result, "content")
            assert hasattr(result, "source")
            assert hasattr(result, "score")
            assert hasattr(result, "result_type")
            assert hasattr(result, "metadata")
            assert hasattr(result, "strategy")

    def test_keyword_extraction(self, search_engine):
        """Test keyword extraction from queries."""
        keywords = search_engine._extract_keywords(
            "How to install Docker on Ubuntu server"
        )

        # Should extract meaningful keywords
        assert "install" in keywords
        assert "docker" in keywords
        assert "ubuntu" in keywords
        assert "server" in keywords

        # Should not include stop words
        assert "how" not in keywords
        assert "to" not in keywords
        assert "on" not in keywords

    def test_category_detection(self, search_engine):
        """Test query category detection."""
        # Test code-related query
        code_category = search_engine._detect_query_category(
            "function definition in Python"
        )
        assert code_category == "code"

        # Test documentation query
        doc_category = search_engine._detect_query_category(
            "how to install Docker tutorial"
        )
        assert doc_category == "documentation"

        # Test API query
        api_category = search_engine._detect_query_category(
            "REST API endpoint authentication"
        )
        assert api_category == "api"

        # Test troubleshooting query
        trouble_category = search_engine._detect_query_category(
            "Docker error connection refused"
        )
        assert trouble_category == "troubleshooting"

    def test_similarity_calculation(self, search_engine):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Identical vectors should have similarity 1.0
        similarity_identical = search_engine._calculate_similarity(vec1, vec2)
        assert abs(similarity_identical - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0.0
        similarity_orthogonal = search_engine._calculate_similarity(vec1, vec3)
        assert abs(similarity_orthogonal - 0.0) < 0.001

    def test_result_deduplication(self, search_engine, mock_dependencies):
        """Test result deduplication functionality."""
        # Create duplicate results
        results = [
            SearchResult(
                content="Docker is a containerization platform",
                source="docker-guide.md",
                score=0.9,
                result_type="knowledge",
                metadata={},
            ),
            SearchResult(
                content="Docker is a containerization platform",  # Duplicate content
                source="docker-intro.md",
                score=0.8,
                result_type="knowledge",
                metadata={},
            ),
            SearchResult(
                content="Kubernetes orchestrates containers",  # Different content
                source="k8s-guide.md",
                score=0.7,
                result_type="knowledge",
                metadata={},
            ),
        ]

        # Mock embedding service for deduplication
        mock_dependencies["embedding_service"].encode.side_effect = [
            [0.1] * 384,  # First result
            [0.1] * 384,  # Duplicate (same embedding)
            [0.9] * 384,  # Different result
        ]

        # Mock similarity calculation
        with patch.object(search_engine, "_calculate_similarity") as mock_sim:
            mock_sim.side_effect = [
                0.99,
                0.1,
            ]  # First comparison is duplicate, second is not

            deduplicated = search_engine._deduplicate_results(results)

        # Should remove one duplicate
        assert len(deduplicated) == 2
        assert deduplicated[0].content == "Docker is a containerization platform"
        assert deduplicated[1].content == "Kubernetes orchestrates containers"

    def test_rrf_fusion(self, search_engine, mock_dependencies):
        """Test Reciprocal Rank Fusion algorithm."""
        # Create results from different strategies
        strategy1_results = [
            SearchResult("Result A", "source1.md", 0.9, "knowledge", {}),
            SearchResult("Result B", "source2.md", 0.8, "knowledge", {}),
        ]

        strategy2_results = [
            SearchResult(
                "Result B", "source2.md", 0.85, "knowledge", {}
            ),  # Same as strategy1[1]
            SearchResult("Result C", "source3.md", 0.7, "knowledge", {}),
        ]

        # Mock embedding service to return diverse embeddings (low similarity)
        mock_dependencies["embedding_service"].encode.side_effect = [
            [1.0, 0.0, 0.0],  # Result A - orthogonal to others
            [0.0, 1.0, 0.0],  # Result B - orthogonal to others
            [0.0, 0.0, 1.0],  # Result C - orthogonal to others
        ]

        # Mock result signature generation
        with patch.object(search_engine, "_get_result_signature") as mock_sig:
            mock_sig.side_effect = ["sig_A", "sig_B", "sig_B", "sig_C"]

            fused_results = search_engine._fusion_results(
                [strategy1_results, strategy2_results], 10
            )

        # Should have 3 unique results
        assert len(fused_results) == 3

        # Results should have RRF scores
        for result in fused_results:
            assert result.rrf_score is not None
            assert result.rrf_score > 0

        # Check that RRF metadata is present
        for result in fused_results:
            assert "rrf_raw_score" in result.metadata
            assert "strategy_count" in result.metadata
            assert "strategy_boost" in result.metadata
            assert "strategies_found_in" in result.metadata

        # Result B should have higher RRF score since it appears in both strategies
        result_b = next(r for r in fused_results if r.content == "Result B")
        assert result_b.metadata["strategy_count"] == 2  # Found in 2 strategies
        assert result_b.metadata["strategy_boost"] > 1.0  # Should have boost

    def test_performance_stats_tracking(
        self, search_engine, mock_dependencies, sample_db_results
    ):
        """Test performance statistics tracking."""
        mock_dependencies["vector_db"].search.return_value = sample_db_results

        # Execute a few searches
        for i in range(3):
            search_engine.search(
                query=f"test query {i}", strategies=[SearchStrategy.SEMANTIC]
            )

        # Check stats were updated
        stats = search_engine.get_performance_stats()
        assert stats["total_searches"] == 3
        assert stats["average_search_time"] > 0
        assert "average_candidates" in stats
        assert "average_reduction" in stats

    def test_search_results_methods(self):
        """Test SearchResults helper methods."""
        results = [
            SearchResult(
                "Content A", "source1.md", 0.9, "knowledge", {}, strategy="semantic"
            ),
            SearchResult(
                "Content B", "source2.md", 0.8, "knowledge", {}, strategy="keyword"
            ),
            SearchResult(
                "Content C", "source3.md", 0.7, "knowledge", {}, strategy="semantic"
            ),
        ]

        search_results = SearchResults(
            results=results,
            total_candidates=100,
            filtered_candidates=3,
            strategies_used=["semantic", "keyword"],
        )

        # Test basic methods
        assert len(search_results) == 3
        assert search_results[0].content == "Content A"

        # Test strategy filtering
        semantic_results = search_results.get_by_strategy("semantic")
        assert len(semantic_results) == 2

        # Test reduction ratio
        reduction = search_results.get_reduction_ratio()
        assert reduction == 0.97  # (100 - 3) / 100

        # Test performance summary
        summary = search_results.get_performance_summary()
        assert summary["total_results"] == 3
        assert summary["reduction_ratio"] == 0.97
        assert "semantic" in summary["strategies_used"]

    def test_error_handling(self, search_engine, mock_dependencies):
        """Test error handling in search operations."""
        # Mock vector DB to raise exception
        mock_dependencies["vector_db"].search.side_effect = Exception("Database error")

        # Search should not crash and return empty results
        results = search_engine.search(
            query="test query", strategies=[SearchStrategy.SEMANTIC]
        )

        assert isinstance(results, SearchResults)
        assert len(results) == 0
        assert results.search_time > 0  # Should still track time

    def test_memory_search_formatting(self, search_engine, mock_dependencies):
        """Test memory search result formatting."""
        # Mock memory search results
        memory_results = [
            Mock(
                score=0.9,
                payload={
                    "question": "How do I install Docker?",
                    "answer": "You can install Docker by downloading it from docker.com",
                    "conversation_id": "conv123",
                    "turn_id": "turn456",
                    "timestamp": "2024-01-01T10:00:00Z",
                },
            )
        ]

        mock_dependencies["vector_db"].search.return_value = memory_results

        results = search_engine.search(
            query="Docker installation", strategies=[SearchStrategy.MEMORY]
        )

        # Check memory result formatting
        if len(results) > 0:
            result = results[0]
            assert "Q:" in result.content
            assert "A:" in result.content
            assert result.result_type == "memory"
            assert "Conversation" in result.source


if __name__ == "__main__":
    pytest.main([__file__])
