"""Vector database client for Qdrant."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import UnexpectedResponse


logger = structlog.get_logger(__name__)


class VectorDBError(Exception):
    """Base exception for vector database errors."""


class VectorDBConnectionError(VectorDBError):
    """Exception raised when connection to vector database fails."""


class VectorDBCollectionError(VectorDBError):
    """Exception raised when collection operations fail."""


class VectorDBSearchError(VectorDBError):
    """Exception raised when search operations fail."""


@dataclass
class SearchResult:
    """Result from a vector search."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionStats:
    """Statistics about a collection."""
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    status: str


class VectorDBClient:
    """Client for interacting with Qdrant vector database.
    
    Provides methods for collection management, vector insertion, and search
    with connection pooling and error handling.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the vector database client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize clients (lazy initialization)
        self._sync_client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        
        logger.info(
            "vector_db_client_initialized",
            url=url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _get_sync_client(self) -> QdrantClient:
        """Get or create synchronous Qdrant client."""
        if self._sync_client is None:
            self._sync_client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create asynchronous Qdrant client."""
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client

    async def health_check(self) -> bool:
        """Check if the vector database is accessible.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            client = self._get_async_client()
            await client.get_collections()
            logger.info("vector_db_health_check_passed")
            return True
        except Exception as e:
            logger.error("vector_db_health_check_failed", error=str(e))
            return False

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        on_disk: bool = False,
    ) -> bool:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Euclid, Dot)
            on_disk: Whether to store vectors on disk
            
        Returns:
            True if collection was created successfully
            
        Raises:
            VectorDBCollectionError: If collection creation fails
        """
        try:
            client = self._get_async_client()
            
            # Map distance string to Distance enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            distance_metric = distance_map.get(distance, Distance.COSINE)
            
            # Check if collection already exists
            collections = await client.get_collections()
            if any(c.name == collection_name for c in collections.collections):
                logger.info(
                    "collection_already_exists",
                    collection_name=collection_name,
                )
                return True
            
            # Create collection
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric,
                    on_disk=on_disk,
                ),
            )
            
            logger.info(
                "collection_created",
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
            )
            return True
            
        except UnexpectedResponse as e:
            logger.error(
                "collection_creation_failed",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBCollectionError(
                f"Failed to create collection {collection_name}: {e}"
            )
        except Exception as e:
            logger.error(
                "collection_creation_error",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBCollectionError(
                f"Unexpected error creating collection {collection_name}: {e}"
            )

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if collection was deleted successfully
            
        Raises:
            VectorDBCollectionError: If collection deletion fails
        """
        try:
            client = self._get_async_client()
            await client.delete_collection(collection_name=collection_name)
            
            logger.info("collection_deleted", collection_name=collection_name)
            return True
            
        except Exception as e:
            logger.error(
                "collection_deletion_failed",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBCollectionError(
                f"Failed to delete collection {collection_name}: {e}"
            )

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            client = self._get_async_client()
            collections = await client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(
                "collection_exists_check_failed",
                collection_name=collection_name,
                error=str(e),
            )
            return False

    async def get_collection_stats(
        self, collection_name: str
    ) -> Optional[CollectionStats]:
        """Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionStats object or None if collection doesn't exist
        """
        try:
            client = self._get_async_client()
            info = await client.get_collection(collection_name=collection_name)
            
            return CollectionStats(
                name=collection_name,
                vectors_count=info.vectors_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0,
                points_count=info.points_count or 0,
                segments_count=info.segments_count or 0,
                status=info.status.value if info.status else "unknown",
            )
        except Exception as e:
            logger.error(
                "get_collection_stats_failed",
                collection_name=collection_name,
                error=str(e),
            )
            return None

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors into a collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to insert
            payloads: List of payload dictionaries (metadata)
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of IDs for the inserted vectors
            
        Raises:
            VectorDBError: If insertion fails
        """
        if len(vectors) != len(payloads):
            raise VectorDBError(
                f"Vectors and payloads must have same length: "
                f"{len(vectors)} != {len(payloads)}"
            )
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise VectorDBError(
                f"IDs and vectors must have same length: "
                f"{len(ids)} != {len(vectors)}"
            )
        
        try:
            client = self._get_async_client()
            
            # Create points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Insert with retry logic
            for attempt in range(self.max_retries):
                try:
                    await client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                    
                    logger.info(
                        "vectors_inserted",
                        collection_name=collection_name,
                        count=len(vectors),
                    )
                    return ids
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "vector_insertion_retry",
                            attempt=attempt + 1,
                            error=str(e),
                        )
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise
            
        except Exception as e:
            logger.error(
                "vector_insertion_failed",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBError(f"Failed to insert vectors: {e}")

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            with_vectors: Whether to include vectors in results
            
        Returns:
            List of SearchResult objects
            
        Raises:
            VectorDBSearchError: If search fails
        """
        try:
            client = self._get_async_client()
            
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search with retry logic
            for attempt in range(self.max_retries):
                try:
                    results = await client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=limit,
                        score_threshold=score_threshold,
                        query_filter=query_filter,
                        with_vectors=with_vectors,
                    )
                    
                    # Convert to SearchResult objects
                    search_results = [
                        SearchResult(
                            id=str(result.id),
                            score=result.score,
                            payload=result.payload or {},
                            vector=result.vector if with_vectors else None,
                        )
                        for result in results
                    ]
                    
                    logger.info(
                        "vector_search_completed",
                        collection_name=collection_name,
                        results_count=len(search_results),
                    )
                    return search_results
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "vector_search_retry",
                            attempt=attempt + 1,
                            error=str(e),
                        )
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise
            
        except Exception as e:
            logger.error(
                "vector_search_failed",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBSearchError(f"Failed to search vectors: {e}")

    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str],
    ) -> bool:
        """Delete vectors by IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of vector IDs to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            VectorDBError: If deletion fails
        """
        try:
            client = self._get_async_client()
            
            await client.delete(
                collection_name=collection_name,
                points_selector=ids,
            )
            
            logger.info(
                "vectors_deleted",
                collection_name=collection_name,
                count=len(ids),
            )
            return True
            
        except Exception as e:
            logger.error(
                "vector_deletion_failed",
                collection_name=collection_name,
                error=str(e),
            )
            raise VectorDBError(f"Failed to delete vectors: {e}")

    async def close(self):
        """Close the client connections."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        logger.info("vector_db_client_closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
