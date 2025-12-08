"""Unit tests for vector database client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morgan_server.knowledge.vectordb import (
    VectorDBClient,
    VectorDBError,
    VectorDBCollectionError,
    VectorDBSearchError,
    SearchResult,
    CollectionStats,
)


class TestVectorDBClient:
    """Tests for VectorDBClient."""

    @pytest.fixture
    def vectordb_client(self):
        """Create a vector database client for testing."""
        return VectorDBClient(
            url="http://localhost:6333",
            api_key=None,
            timeout=30,
            max_retries=2,
            retry_delay=0.1,
        )

    @pytest.mark.asyncio
    async def test_health_check_success(self, vectordb_client):
        """Test successful health check."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.health_check()
            assert result is True
            mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, vectordb_client):
        """Test failed health check."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("Connection failed")

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_create_collection_success(self, vectordb_client):
        """Test successful collection creation."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.create_collection.return_value = True

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.create_collection(
                collection_name="test_collection",
                vector_size=384,
                distance="Cosine",
            )
            assert result is True
            mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, vectordb_client):
        """Test creating a collection that already exists."""
        mock_client = AsyncMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.create_collection(
                collection_name="test_collection",
                vector_size=384,
            )
            assert result is True
            # Should not call create_collection if it already exists
            mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_with_different_distances(
        self, vectordb_client
    ):
        """Test creating collections with different distance metrics."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            # Test Cosine
            await vectordb_client.create_collection(
                "test_cosine", 384, "Cosine"
            )
            # Test Euclid
            await vectordb_client.create_collection(
                "test_euclid", 384, "Euclid"
            )
            # Test Dot
            await vectordb_client.create_collection("test_dot", 384, "Dot")

            assert mock_client.create_collection.call_count == 3

    @pytest.mark.asyncio
    async def test_create_collection_error(self, vectordb_client):
        """Test collection creation error handling."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"",
            headers={},
        )

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            with pytest.raises(VectorDBCollectionError):
                await vectordb_client.create_collection(
                    "test_collection", 384
                )

    @pytest.mark.asyncio
    async def test_delete_collection_success(self, vectordb_client):
        """Test successful collection deletion."""
        mock_client = AsyncMock()
        mock_client.delete_collection.return_value = True

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.delete_collection("test_collection")
            assert result is True
            mock_client.delete_collection.assert_called_once_with(
                collection_name="test_collection"
            )

    @pytest.mark.asyncio
    async def test_delete_collection_error(self, vectordb_client):
        """Test collection deletion error handling."""
        mock_client = AsyncMock()
        mock_client.delete_collection.side_effect = Exception("Delete failed")

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            with pytest.raises(VectorDBCollectionError):
                await vectordb_client.delete_collection("test_collection")

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, vectordb_client):
        """Test checking if collection exists (true case)."""
        mock_client = AsyncMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.collection_exists("test_collection")
            assert result is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, vectordb_client):
        """Test checking if collection exists (false case)."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.collection_exists("test_collection")
            assert result is False

    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self, vectordb_client):
        """Test getting collection statistics."""
        mock_client = AsyncMock()
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.indexed_vectors_count = 100
        mock_info.points_count = 100
        mock_info.segments_count = 1
        mock_status = MagicMock()
        mock_status.value = "green"
        mock_info.status = mock_status
        mock_client.get_collection.return_value = mock_info

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            stats = await vectordb_client.get_collection_stats("test_collection")
            assert stats is not None
            assert isinstance(stats, CollectionStats)
            assert stats.name == "test_collection"
            assert stats.vectors_count == 100
            assert stats.points_count == 100
            assert stats.status == "green"

    @pytest.mark.asyncio
    async def test_get_collection_stats_error(self, vectordb_client):
        """Test getting collection statistics error handling."""
        mock_client = AsyncMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            stats = await vectordb_client.get_collection_stats("test_collection")
            assert stats is None

    @pytest.mark.asyncio
    async def test_insert_vectors_success(self, vectordb_client):
        """Test successful vector insertion."""
        mock_client = AsyncMock()
        mock_client.upsert.return_value = True

        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        payloads = [{"text": "doc1"}, {"text": "doc2"}]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            ids = await vectordb_client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
            )
            assert len(ids) == 2
            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_vectors_with_ids(self, vectordb_client):
        """Test vector insertion with provided IDs."""
        mock_client = AsyncMock()
        mock_client.upsert.return_value = True

        vectors = [[0.1, 0.2, 0.3]]
        payloads = [{"text": "doc1"}]
        ids = ["custom-id-1"]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result_ids = await vectordb_client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
                ids=ids,
            )
            assert result_ids == ids

    @pytest.mark.asyncio
    async def test_insert_vectors_length_mismatch(self, vectordb_client):
        """Test vector insertion with mismatched lengths."""
        vectors = [[0.1, 0.2, 0.3]]
        payloads = [{"text": "doc1"}, {"text": "doc2"}]

        with pytest.raises(VectorDBError, match="must have same length"):
            await vectordb_client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
            )

    @pytest.mark.asyncio
    async def test_insert_vectors_ids_length_mismatch(self, vectordb_client):
        """Test vector insertion with mismatched IDs length."""
        vectors = [[0.1, 0.2, 0.3]]
        payloads = [{"text": "doc1"}]
        ids = ["id1", "id2"]

        with pytest.raises(VectorDBError, match="must have same length"):
            await vectordb_client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
                ids=ids,
            )

    @pytest.mark.asyncio
    async def test_insert_vectors_error(self, vectordb_client):
        """Test vector insertion error handling."""
        mock_client = AsyncMock()
        mock_client.upsert.side_effect = Exception("Insert failed")

        vectors = [[0.1, 0.2, 0.3]]
        payloads = [{"text": "doc1"}]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            with pytest.raises(VectorDBError):
                await vectordb_client.insert_vectors(
                    collection_name="test_collection",
                    vectors=vectors,
                    payloads=payloads,
                )

    @pytest.mark.asyncio
    async def test_search_success(self, vectordb_client):
        """Test successful vector search."""
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.id = "doc1"
        mock_result.score = 0.95
        mock_result.payload = {"text": "document 1"}
        mock_result.vector = None
        mock_client.search.return_value = [mock_result]

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            results = await vectordb_client.search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=10,
            )
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].id == "doc1"
            assert results[0].score == 0.95
            assert results[0].payload["text"] == "document 1"

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, vectordb_client):
        """Test vector search with score threshold."""
        mock_client = AsyncMock()
        mock_client.search.return_value = []

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            results = await vectordb_client.search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=10,
                score_threshold=0.8,
            )
            assert len(results) == 0
            # Verify score_threshold was passed
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs["score_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_search_with_filter(self, vectordb_client):
        """Test vector search with filter conditions."""
        mock_client = AsyncMock()
        mock_client.search.return_value = []

        query_vector = [0.1, 0.2, 0.3]
        filter_conditions = {"category": "tech"}

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            await vectordb_client.search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=10,
                filter_conditions=filter_conditions,
            )
            # Verify filter was passed
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_with_vectors(self, vectordb_client):
        """Test vector search with vectors included in results."""
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.id = "doc1"
        mock_result.score = 0.95
        mock_result.payload = {"text": "document 1"}
        mock_result.vector = [0.1, 0.2, 0.3]
        mock_client.search.return_value = [mock_result]

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            results = await vectordb_client.search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=10,
                with_vectors=True,
            )
            assert len(results) == 1
            assert results[0].vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_search_error(self, vectordb_client):
        """Test vector search error handling."""
        mock_client = AsyncMock()
        mock_client.search.side_effect = Exception("Search failed")

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            with pytest.raises(VectorDBSearchError):
                await vectordb_client.search(
                    collection_name="test_collection",
                    query_vector=query_vector,
                )

    @pytest.mark.asyncio
    async def test_delete_vectors_success(self, vectordb_client):
        """Test successful vector deletion."""
        mock_client = AsyncMock()
        mock_client.delete.return_value = True

        ids = ["doc1", "doc2"]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            result = await vectordb_client.delete_vectors(
                collection_name="test_collection",
                ids=ids,
            )
            assert result is True
            mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_vectors_error(self, vectordb_client):
        """Test vector deletion error handling."""
        mock_client = AsyncMock()
        mock_client.delete.side_effect = Exception("Delete failed")

        ids = ["doc1"]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            with pytest.raises(VectorDBError):
                await vectordb_client.delete_vectors(
                    collection_name="test_collection",
                    ids=ids,
                )

    @pytest.mark.asyncio
    async def test_close(self, vectordb_client):
        """Test closing the client."""
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        vectordb_client._async_client = mock_async_client
        vectordb_client._sync_client = mock_sync_client

        await vectordb_client.close()

        mock_async_client.close.assert_called_once()
        mock_sync_client.close.assert_called_once()
        assert vectordb_client._async_client is None
        assert vectordb_client._sync_client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, vectordb_client):
        """Test using client as async context manager."""
        mock_async_client = AsyncMock()

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_async_client
        ):
            async with vectordb_client as client:
                assert client is vectordb_client

            # Verify close was called
            # Note: We can't easily verify this without mocking close itself
            # but the context manager protocol ensures it's called

    @pytest.mark.asyncio
    async def test_retry_logic_on_insert(self, vectordb_client):
        """Test retry logic on vector insertion."""
        mock_client = AsyncMock()
        # Fail once, succeed on second attempt (within max_retries=2)
        mock_client.upsert.side_effect = [
            Exception("Temporary failure"),
            True,
        ]

        vectors = [[0.1, 0.2, 0.3]]
        payloads = [{"text": "doc1"}]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            # Should succeed after retry
            ids = await vectordb_client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
            )
            assert len(ids) == 1
            # Verify it was called 2 times (initial + 1 retry)
            assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_logic_on_search(self, vectordb_client):
        """Test retry logic on vector search."""
        mock_client = AsyncMock()
        # Fail once, succeed on second attempt
        mock_result = MagicMock()
        mock_result.id = "doc1"
        mock_result.score = 0.95
        mock_result.payload = {}
        mock_result.vector = None
        mock_client.search.side_effect = [
            Exception("Temporary failure"),
            [mock_result],
        ]

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(
            vectordb_client, "_get_async_client", return_value=mock_client
        ):
            # Should succeed after retry
            results = await vectordb_client.search(
                collection_name="test_collection",
                query_vector=query_vector,
            )
            assert len(results) == 1
            # Verify it was called 2 times (initial + 1 retry)
            assert mock_client.search.call_count == 2
