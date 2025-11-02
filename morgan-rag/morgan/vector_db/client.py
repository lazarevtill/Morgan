"""
Enhanced Qdrant vector database client for Morgan RAG with companion features.

Provides comprehensive vector storage with support for:
- Knowledge documents with hierarchical embeddings
- Conversation memories with emotional context
- Companion profiles with relationship data
- Emotional states and mood patterns
- Relationship milestones and achievements
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.utils.error_handling import (
    StorageError, NetworkError, ValidationError, ErrorCategory, ErrorSeverity
)
from morgan.utils.error_decorators import (
    handle_storage_errors, monitor_performance, RetryConfig
)
from morgan.optimization.connection_pool import get_connection_pool_manager
from morgan.optimization.batch_processor import get_batch_processor

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Vector search result."""
    id: str
    score: float
    payload: Dict[str, Any]


@dataclass
class BatchOperationResult:
    """Result of batch operation."""
    success_count: int
    failure_count: int
    total_count: int
    errors: List[str]
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100.0


@dataclass
class CollectionInfo:
    """Extended collection information."""
    name: str
    points_count: int
    vectors_count: int
    disk_usage: int
    status: str
    schema_type: Optional[str] = None
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class VectorDBClient:
    """
    Enhanced Qdrant vector database client with companion features.
    
    Provides comprehensive vector operations for:
    - Knowledge documents with hierarchical embeddings
    - Conversation memories with emotional context
    - Companion profiles and relationship data
    - Emotional states and mood tracking
    - Batch operations for performance optimization
    """
    
    # Collection name constants
    KNOWLEDGE_COLLECTION = "morgan_knowledge"
    MEMORIES_COLLECTION = "morgan_memories"
    COMPANIONS_COLLECTION = "morgan_companions"
    EMOTIONS_COLLECTION = "morgan_emotions"
    MILESTONES_COLLECTION = "morgan_milestones"
    
    def __init__(self):
        """Initialize enhanced Qdrant client with companion support and connection pooling."""
        self.settings = get_settings()
        
        # Initialize connection pool manager
        self.pool_manager = get_connection_pool_manager()
        
        # Initialize Qdrant client (primary connection)
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30
        )
        
        # Initialize batch processor for optimized operations
        self.batch_processor = get_batch_processor()
        
        # Companion database schema will be loaded on demand
        self._schema = None
        
        # Batch operation settings with optimization
        self.batch_size = 100
        self.max_retries = 3
        self.retry_delay = 1.0
        self.use_connection_pooling = True
        
        logger.info(f"Connected to Qdrant at {self.settings.qdrant_url}")
        logger.info(
            "Enhanced vector database client initialized with companion features, "
            "connection pooling, and batch optimization"
        )
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def create_collection(
        self, 
        name: str, 
        vector_size: int, 
        distance: str = "cosine"
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            vector_size: Dimension of vectors
            distance: Distance metric (cosine, euclidean, dot)
            
        Returns:
            True if created successfully
        """
        try:
            # Map distance names
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclidean": models.Distance.EUCLID,
                "dot": models.Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, models.Distance.COSINE)
                )
            )
            
            logger.info(f"Created collection '{name}' with {vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    @handle_storage_errors("upsert_points", "vector_db_client", 
                          RetryConfig(max_attempts=3, base_delay=2.0))
    @monitor_performance("upsert_points", "vector_db_client")
    def upsert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]],
        use_batch_optimization: bool = True
    ) -> bool:
        """
        Insert or update points in collection with batch optimization.
        
        Args:
            collection_name: Target collection
            points: List of points with id, vector, payload
            use_batch_optimization: Use optimized batch processing (default: True)
            
        Returns:
            True if successful
        """
        try:
            # Use optimized batch processing for large point sets
            if use_batch_optimization and len(points) > 50:
                try:
                    # Create batch operation function
                    def upsert_operation(batch_points: List[Dict[str, Any]]) -> List[Any]:
                        qdrant_points = []
                        for point in batch_points:
                            qdrant_points.append(
                                models.PointStruct(
                                    id=point["id"],
                                    vector=point["vector"],
                                    payload=point.get("payload", {})
                                )
                            )
                        
                        # Use connection pooling if available
                        if self.use_connection_pooling:
                            try:
                                with self.pool_manager.get_connection("qdrant") as pooled_client:
                                    pooled_client.upsert(
                                        collection_name=collection_name,
                                        points=qdrant_points
                                    )
                            except Exception:
                                # Fall back to primary client
                                self.client.upsert(
                                    collection_name=collection_name,
                                    points=qdrant_points
                                )
                        else:
                            self.client.upsert(
                                collection_name=collection_name,
                                points=qdrant_points
                            )
                        
                        return qdrant_points
                    
                    # Process with batch optimizer
                    result = self.batch_processor.process_vector_operations_batch(
                        operations=points,
                        operation_function=upsert_operation,
                        operation_type="upsert"
                    )
                    
                    if result.success_rate >= 95.0:  # 95% success threshold for upserts
                        logger.info(
                            f"Optimized batch upsert completed: {result.processed_items}/{result.total_items} "
                            f"points ({result.success_rate:.1f}%) to '{collection_name}'"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Batch upsert had low success rate ({result.success_rate:.1f}%), "
                            f"falling back to standard processing"
                        )
                        
                except Exception as e:
                    logger.warning(f"Optimized batch upsert failed, falling back to standard: {e}")
            
            # Standard upsert processing
            # Convert to Qdrant format
            qdrant_points = []
            for point in points:
                qdrant_points.append(
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point.get("payload", {})
                    )
                )
            
            # Use connection pooling if available
            if self.use_connection_pooling and len(points) > 10:
                try:
                    with self.pool_manager.get_connection("qdrant") as pooled_client:
                        pooled_client.upsert(
                            collection_name=collection_name,
                            points=qdrant_points
                        )
                except Exception:
                    # Fall back to primary client
                    self.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points
                    )
            else:
                # Direct upsert for small batches
                self.client.upsert(
                    collection_name=collection_name,
                    points=qdrant_points
                )
            
            logger.debug(f"Upserted {len(points)} points to '{collection_name}'")
            return True
            
        except Exception as e:
            raise StorageError(
                f"Failed to upsert points to collection '{collection_name}': {e}",
                operation="upsert_points",
                component="vector_db_client",
                metadata={
                    "collection_name": collection_name,
                    "points_count": len(points),
                    "use_batch_optimization": use_batch_optimization,
                    "error_type": type(e).__name__
                }
            ) from e
    
    @handle_storage_errors("search", "vector_db_client", 
                          RetryConfig(max_attempts=2, base_delay=1.0))
    @monitor_performance("search", "vector_db_client")
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert to our format
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                ))
            
            logger.debug(
                f"Found {len(search_results)} results in '{collection_name}'"
            )
            return search_results
            
        except Exception as e:
            raise StorageError(
                f"Search failed in collection '{collection_name}': {e}",
                operation="search",
                component="vector_db_client",
                metadata={
                    "collection_name": collection_name,
                    "query_vector_dim": len(query_vector),
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "error_type": type(e).__name__
                }
            ) from e
    
    def search_with_filter(
        self,
        collection_name: str,
        filter_conditions: Dict[str, Any],
        limit: int = 100
    ) -> List[SearchResult]:
        """
        Search with payload filters (no vector query).
        
        Args:
            collection_name: Collection to search
            filter_conditions: Filter conditions
            limit: Maximum results
            
        Returns:
            List of matching results
        """
        try:
            # Build filter
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )
            
            # Scroll through results
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=limit
            )
            
            # Convert to our format
            search_results = []
            for result in results[0]:  # results is (points, next_page_offset)
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=1.0,  # No similarity score for filter-only search
                    payload=result.payload or {}
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Filter search failed in '{collection_name}': {e}")
            return []
    
    def delete_points(
        self, 
        collection_name: str, 
        point_ids: List[str]
    ) -> bool:
        """
        Delete points from collection.
        
        Args:
            collection_name: Target collection
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            logger.debug(f"Deleted {len(point_ids)} points from '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete points from '{collection_name}': {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection information.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection info dictionary
        """
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "disk_usage": getattr(info, 'disk_usage', 0),
                "status": info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return {"error": str(e)}
    
    def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Scroll through points in collection.
        
        Args:
            collection_name: Collection to scroll
            limit: Maximum points to return
            offset: Pagination offset
            
        Returns:
            List of points
        """
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset
            )
            
            # Convert to our format
            points = []
            for result in results[0]:  # results is (points, next_page_offset)
                points.append(SearchResult(
                    id=str(result.id),
                    score=1.0,  # No similarity score for scroll
                    payload=result.payload or {}
                ))
            
            return points
            
        except Exception as e:
            logger.error(f"Failed to scroll points in '{collection_name}': {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy.
        
        Returns:
            True if healthy
        """
        try:
            collections = self.client.get_collections()
            logger.debug(
                f"Qdrant health check passed "
                f"({len(collections.collections)} collections)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    # ========================================
    # Companion Feature Methods
    # ========================================
    
    def initialize_companion_collections(self) -> bool:
        """
        Initialize all companion-related collections with proper schemas.
        
        Returns:
            True if all collections created successfully
        """
        try:
            # Define schemas inline to avoid circular imports
            companion_schemas = [
                {
                    "name": self.COMPANIONS_COLLECTION,
                    "vector_size": 1536,
                    "distance": "cosine"
                },
                {
                    "name": self.MEMORIES_COLLECTION,
                    "vector_size": 1536,
                    "distance": "cosine"
                },
                {
                    "name": self.EMOTIONS_COLLECTION,
                    "vector_size": 768,
                    "distance": "cosine"
                },
                {
                    "name": self.MILESTONES_COLLECTION,
                    "vector_size": 512,
                    "distance": "cosine"
                }
            ]
            
            success_count = 0
            
            for schema_def in companion_schemas:
                if not self.collection_exists(schema_def["name"]):
                    success = self.create_collection(
                        name=schema_def["name"],
                        vector_size=schema_def["vector_size"],
                        distance=schema_def["distance"]
                    )
                    if success:
                        success_count += 1
                        logger.info(
                            f"Created companion collection: {schema_def['name']}"
                        )
                    else:
                        logger.error(
                            f"Failed to create collection: {schema_def['name']}"
                        )
                else:
                    success_count += 1
                    logger.debug(
                        f"Collection already exists: {schema_def['name']}"
                    )
            
            all_created = success_count == len(companion_schemas)
            if all_created:
                logger.info("All companion collections initialized successfully")
            else:
                logger.warning(
                    f"Only {success_count}/{len(companion_schemas)} "
                    f"collections initialized"
                )
            
            return all_created
            
        except Exception as e:
            logger.error(f"Failed to initialize companion collections: {e}")
            return False
    
    def create_hierarchical_collection(
        self,
        name: str,
        coarse_size: int = 384,
        medium_size: int = 768,
        fine_size: int = 1536,
        distance: str = "cosine"
    ) -> bool:
        """
        Create collection with hierarchical vector support.
        
        Args:
            name: Collection name
            coarse_size: Coarse embedding dimension
            medium_size: Medium embedding dimension  
            fine_size: Fine embedding dimension
            distance: Distance metric
            
        Returns:
            True if created successfully
        """
        try:
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclidean": models.Distance.EUCLID,
                "dot": models.Distance.DOT
            }
            
            # Create collection with multiple named vectors
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "coarse": models.VectorParams(
                        size=coarse_size,
                        distance=distance_map.get(distance, models.Distance.COSINE)
                    ),
                    "medium": models.VectorParams(
                        size=medium_size,
                        distance=distance_map.get(distance, models.Distance.COSINE)
                    ),
                    "fine": models.VectorParams(
                        size=fine_size,
                        distance=distance_map.get(distance, models.Distance.COSINE)
                    )
                }
            )
            
            logger.info(f"Created hierarchical collection '{name}' with {coarse_size}/{medium_size}/{fine_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create hierarchical collection '{name}': {e}")
            return False
    
    def upsert_companion_profile(
        self,
        user_id: str,
        profile_embedding: List[float],
        profile_data: Dict[str, Any]
    ) -> bool:
        """
        Upsert companion profile with validation.
        
        Args:
            user_id: User identifier
            profile_embedding: Profile embedding vector
            profile_data: Profile payload data
            
        Returns:
            True if successful
        """
        try:
            # Basic validation for required fields
            required_fields = ["user_id", "preferred_name", "communication_style"]
            for field in required_fields:
                if field not in profile_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Add metadata
            profile_data.update({
                "updated_at": datetime.now().isoformat(),
                "profile_version": "1.0"
            })
            
            point = {
                "id": user_id,
                "vector": profile_embedding,
                "payload": profile_data
            }
            
            return self.upsert_points(self.COMPANIONS_COLLECTION, [point])
            
        except Exception as e:
            logger.error(f"Failed to upsert companion profile for {user_id}: {e}")
            return False
    
    def upsert_memory_batch(
        self,
        memories: List[Dict[str, Any]]
    ) -> BatchOperationResult:
        """
        Batch upsert conversation memories with emotional context.
        
        Args:
            memories: List of memory dictionaries with id, vector, payload
            
        Returns:
            Batch operation result
        """
        start_time = time.time()
        success_count = 0
        errors = []
        
        try:
            # Process in batches
            for i in range(0, len(memories), self.batch_size):
                batch = memories[i:i + self.batch_size]
                
                # Convert to Qdrant format
                qdrant_points = []
                for memory in batch:
                    try:
                        # Add timestamp if not present
                        if "created_at" not in memory["payload"]:
                            memory["payload"]["created_at"] = datetime.now().isoformat()
                        
                        qdrant_points.append(
                            models.PointStruct(
                                id=memory["id"],
                                vector=memory["vector"],
                                payload=memory["payload"]
                            )
                        )
                    except Exception as e:
                        errors.append(f"Failed to prepare memory {memory.get('id', 'unknown')}: {e}")
                        continue
                
                # Batch upsert with retry
                batch_success = self._upsert_with_retry(
                    self.MEMORIES_COLLECTION, qdrant_points
                )
                
                if batch_success:
                    success_count += len(qdrant_points)
                else:
                    errors.append(f"Failed to upsert batch {i//self.batch_size + 1}")
            
            processing_time = time.time() - start_time
            failure_count = len(memories) - success_count
            
            result = BatchOperationResult(
                success_count=success_count,
                failure_count=failure_count,
                total_count=len(memories),
                errors=errors,
                processing_time=processing_time
            )
            
            logger.info(
                f"Memory batch upsert completed: {success_count}/{len(memories)} successful "
                f"({result.success_rate:.1f}%) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Memory batch upsert failed: {e}")
            
            return BatchOperationResult(
                success_count=success_count,
                failure_count=len(memories) - success_count,
                total_count=len(memories),
                errors=errors + [str(e)],
                processing_time=processing_time
            )
    
    def upsert_emotional_states_batch(
        self,
        emotional_states: List[Dict[str, Any]]
    ) -> BatchOperationResult:
        """
        Batch upsert emotional states for mood tracking.
        
        Args:
            emotional_states: List of emotional state dictionaries
            
        Returns:
            Batch operation result
        """
        start_time = time.time()
        success_count = 0
        errors = []
        
        try:
            # Basic validation for emotional states
            validated_states = []
            for state in emotional_states:
                payload = state.get("payload", {})
                required_fields = ["user_id", "primary_emotion", "intensity", "confidence"]
                
                # Check required fields
                missing_fields = [f for f in required_fields if f not in payload]
                if missing_fields:
                    errors.append(
                        f"Missing fields for {state.get('id', 'unknown')}: "
                        f"{missing_fields}"
                    )
                    continue
                    
                validated_states.append(state)
            
            # Process in batches
            for i in range(0, len(validated_states), self.batch_size):
                batch = validated_states[i:i + self.batch_size]
                
                # Convert to Qdrant format
                qdrant_points = []
                for state in batch:
                    try:
                        # Add metadata
                        if "created_at" not in state["payload"]:
                            state["payload"]["created_at"] = datetime.now().isoformat()
                        
                        qdrant_points.append(
                            models.PointStruct(
                                id=state["id"],
                                vector=state["vector"],
                                payload=state["payload"]
                            )
                        )
                    except Exception as e:
                        errors.append(f"Failed to prepare emotional state {state.get('id', 'unknown')}: {e}")
                        continue
                
                # Batch upsert with retry
                batch_success = self._upsert_with_retry(
                    self.EMOTIONS_COLLECTION, qdrant_points
                )
                
                if batch_success:
                    success_count += len(qdrant_points)
                else:
                    errors.append(f"Failed to upsert emotional batch {i//self.batch_size + 1}")
            
            processing_time = time.time() - start_time
            failure_count = len(emotional_states) - success_count
            
            result = BatchOperationResult(
                success_count=success_count,
                failure_count=failure_count,
                total_count=len(emotional_states),
                errors=errors,
                processing_time=processing_time
            )
            
            logger.info(
                f"Emotional states batch upsert completed: {success_count}/{len(emotional_states)} successful "
                f"({result.success_rate:.1f}%) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Emotional states batch upsert failed: {e}")
            
            return BatchOperationResult(
                success_count=success_count,
                failure_count=len(emotional_states) - success_count,
                total_count=len(emotional_states),
                errors=errors + [str(e)],
                processing_time=processing_time
            )
    
    def search_memories_by_user(
        self,
        user_id: str,
        query_vector: Optional[List[float]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        emotional_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search memories for specific user with optional filters.
        
        Args:
            user_id: User identifier
            query_vector: Optional query vector for semantic search
            limit: Maximum results to return
            min_importance: Minimum importance score filter
            emotional_filter: Optional emotion filter (e.g., "joy", "sadness")
            
        Returns:
            List of matching memories
        """
        try:
            # Build filter conditions
            filter_conditions = {"user_id": user_id}
            
            if min_importance > 0.0:
                filter_conditions["importance_score"] = {"gte": min_importance}
            
            if emotional_filter:
                filter_conditions["user_mood"] = emotional_filter
            
            # Build Qdrant filter
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, dict) and "gte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(gte=value["gte"])
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            filter_obj = models.Filter(must=must_conditions)
            
            # Perform search
            if query_vector:
                # Semantic search with filters
                results = self.client.search(
                    collection_name=self.MEMORIES_COLLECTION,
                    query_vector=query_vector,
                    query_filter=filter_obj,
                    limit=limit
                )
            else:
                # Filter-only search
                results = self.client.scroll(
                    collection_name=self.MEMORIES_COLLECTION,
                    scroll_filter=filter_obj,
                    limit=limit
                )
                results = results[0]  # Extract points from (points, next_page_offset)
            
            # Convert to SearchResult format
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=getattr(result, 'score', 1.0),
                    payload=result.payload or {}
                ))
            
            logger.debug(f"Found {len(search_results)} memories for user {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search memories for user {user_id}: {e}")
            return []
    
    def search_hierarchical(
        self,
        collection_name: str,
        coarse_vector: Optional[List[float]] = None,
        medium_vector: Optional[List[float]] = None,
        fine_vector: Optional[List[float]] = None,
        limit: int = 10,
        coarse_limit: int = 1000,
        medium_limit: int = 100,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform hierarchical search with coarse-to-fine filtering and emotional context integration.
        
        Enhanced to support:
        - Emotional context in search result ranking
        - Companion-aware result filtering and personalization
        - Multi-stage search with hierarchical embeddings
        
        Args:
            collection_name: Collection to search
            coarse_vector: Coarse-level query vector
            medium_vector: Medium-level query vector
            fine_vector: Fine-level query vector
            limit: Final result limit
            coarse_limit: Coarse filtering limit
            medium_limit: Medium filtering limit
            emotional_context: Optional emotional context for ranking
            user_id: Optional user ID for personalization
            
        Returns:
            Hierarchically filtered search results with emotional and companion enhancements
        """
        try:
            candidate_ids = None
            
            # Stage 1: Coarse filtering (if coarse vector provided)
            if coarse_vector:
                coarse_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("coarse", coarse_vector),
                    limit=coarse_limit
                )
                candidate_ids = [str(result.id) for result in coarse_results]
                logger.debug(f"Coarse filtering: {len(candidate_ids)} candidates")
            
            # Stage 2: Medium filtering (if medium vector provided)
            if medium_vector and candidate_ids:
                medium_filter = models.Filter(
                    must=[
                        models.HasIdCondition(has_id=candidate_ids)
                    ]
                )
                
                medium_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("medium", medium_vector),
                    query_filter=medium_filter,
                    limit=medium_limit
                )
                candidate_ids = [str(result.id) for result in medium_results]
                logger.debug(f"Medium filtering: {len(candidate_ids)} candidates")
            
            # Stage 3: Fine search (final results)
            if fine_vector:
                final_filter = None
                if candidate_ids:
                    final_filter = models.Filter(
                        must=[
                            models.HasIdCondition(has_id=candidate_ids)
                        ]
                    )
                
                final_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("fine", fine_vector),
                    query_filter=final_filter,
                    limit=limit * 2  # Get more results for emotional/companion filtering
                )
            else:
                # If no fine vector, return medium results or coarse results
                if candidate_ids:
                    final_filter = models.Filter(
                        must=[
                            models.HasIdCondition(has_id=candidate_ids[:limit * 2])
                        ]
                    )
                    final_results = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=final_filter,
                        limit=limit * 2
                    )
                    final_results = final_results[0]
                else:
                    final_results = []
            
            # Convert to SearchResult format
            search_results = []
            for result in final_results:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=getattr(result, 'score', 1.0),
                    payload=result.payload or {}
                ))
            
            # Apply emotional context and companion-aware ranking
            enhanced_results = self._apply_emotional_and_companion_ranking(
                search_results, emotional_context, user_id
            )
            
            logger.debug(f"Hierarchical search completed: {len(enhanced_results)} enhanced results")
            return enhanced_results[:limit]
            
        except Exception as e:
            logger.error(f"Hierarchical search failed in '{collection_name}': {e}")
            return []
    
    def get_companion_profile(self, user_id: str) -> Optional[SearchResult]:
        """
        Get companion profile for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Companion profile or None if not found
        """
        try:
            results = self.search_with_filter(
                collection_name=self.COMPANIONS_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=1
            )
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Failed to get companion profile for {user_id}: {e}")
            return None
    
    def get_user_emotional_history(
        self,
        user_id: str,
        days: int = 30,
        limit: int = 100
    ) -> List[SearchResult]:
        """
        Get emotional history for user within time period.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            limit: Maximum results to return
            
        Returns:
            List of emotional state records
        """
        try:
            # Calculate date threshold
            from datetime import datetime, timedelta
            threshold_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Build filter for user and date range
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    ),
                    models.FieldCondition(
                        key="timestamp",
                        range=models.DatetimeRange(gte=threshold_date)
                    )
                ]
            )
            
            # Scroll through results
            results = self.client.scroll(
                collection_name=self.EMOTIONS_COLLECTION,
                scroll_filter=filter_obj,
                limit=limit
            )
            
            # Convert to SearchResult format
            search_results = []
            for result in results[0]:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=1.0,
                    payload=result.payload or {}
                ))
            
            logger.debug(f"Found {len(search_results)} emotional records for user {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to get emotional history for {user_id}: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, CollectionInfo]:
        """
        Get statistics for all companion collections.
        
        Returns:
            Dictionary of collection statistics
        """
        stats = {}
        
        companion_collections = [
            self.KNOWLEDGE_COLLECTION,
            self.MEMORIES_COLLECTION,
            self.COMPANIONS_COLLECTION,
            self.EMOTIONS_COLLECTION,
            self.MILESTONES_COLLECTION
        ]
        
        for collection_name in companion_collections:
            try:
                if self.collection_exists(collection_name):
                    info_dict = self.get_collection_info(collection_name)
                    
                    stats[collection_name] = CollectionInfo(
                        name=collection_name,
                        points_count=info_dict.get("points_count", 0),
                        vectors_count=info_dict.get("vectors_count", 0),
                        disk_usage=info_dict.get("disk_usage", 0),
                        status=info_dict.get("status", "unknown"),
                        schema_type="companion",
                        created_at=None,  # Would need to be tracked separately
                        last_updated=datetime.now()
                    )
                else:
                    stats[collection_name] = CollectionInfo(
                        name=collection_name,
                        points_count=0,
                        vectors_count=0,
                        disk_usage=0,
                        status="not_created",
                        schema_type="companion"
                    )
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = CollectionInfo(
                    name=collection_name,
                    points_count=0,
                    vectors_count=0,
                    disk_usage=0,
                    status="error",
                    schema_type="companion"
                )
        
        return stats
    
    def _apply_emotional_and_companion_ranking(
        self,
        search_results: List[SearchResult],
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Apply emotional context and companion-aware ranking to search results.
        
        Integrates emotional intelligence and companion relationship data to
        personalize and enhance search result relevance.
        
        Args:
            search_results: Raw search results to enhance
            emotional_context: Optional emotional context for ranking
            user_id: Optional user ID for personalization
            
        Returns:
            Enhanced and re-ranked search results
        """
        try:
            if not search_results:
                return search_results
            
            enhanced_results = []
            
            for result in search_results:
                enhanced_score = result.score
                enhancement_factors = []
                
                # Apply emotional context enhancement
                if emotional_context:
                    emotional_boost = self._calculate_emotional_relevance_boost(
                        result, emotional_context
                    )
                    enhanced_score += emotional_boost
                    if emotional_boost > 0:
                        enhancement_factors.append(f"emotional_boost:{emotional_boost:.3f}")
                
                # Apply companion-aware personalization
                if user_id:
                    companion_boost = self._calculate_companion_personalization_boost(
                        result, user_id
                    )
                    enhanced_score += companion_boost
                    if companion_boost > 0:
                        enhancement_factors.append(f"companion_boost:{companion_boost:.3f}")
                
                # Apply content quality and relevance factors
                quality_boost = self._calculate_content_quality_boost(result)
                enhanced_score += quality_boost
                if quality_boost > 0:
                    enhancement_factors.append(f"quality_boost:{quality_boost:.3f}")
                
                # Ensure score doesn't exceed 1.0
                enhanced_score = min(1.0, enhanced_score)
                
                # Create enhanced result with metadata
                enhanced_result = SearchResult(
                    id=result.id,
                    score=enhanced_score,
                    payload={
                        **result.payload,
                        "original_score": result.score,
                        "enhancement_factors": enhancement_factors,
                        "emotional_enhanced": emotional_context is not None,
                        "companion_enhanced": user_id is not None
                    }
                )
                
                enhanced_results.append(enhanced_result)
            
            # Sort by enhanced score
            enhanced_results.sort(key=lambda r: r.score, reverse=True)
            
            logger.debug(
                f"Applied emotional and companion ranking: "
                f"avg boost = {sum(r.score - r.payload.get('original_score', r.score) for r in enhanced_results) / len(enhanced_results):.3f}"
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.warning(f"Failed to apply emotional and companion ranking: {e}")
            return search_results
    
    def _calculate_emotional_relevance_boost(
        self,
        result: SearchResult,
        emotional_context: Dict[str, Any]
    ) -> float:
        """
        Calculate emotional relevance boost for a search result.
        
        Args:
            result: Search result to evaluate
            emotional_context: Current emotional context
            
        Returns:
            Boost value (0.0 to 0.3)
        """
        try:
            boost = 0.0
            content = result.payload.get("content", "").lower()
            
            # Get emotional indicators from context
            primary_emotion = emotional_context.get("primary_emotion", "")
            intensity = emotional_context.get("intensity", 0.0)
            emotional_indicators = emotional_context.get("emotional_indicators", [])
            
            # Boost based on emotional resonance
            if primary_emotion and primary_emotion in content:
                boost += 0.1 * intensity
            
            # Boost based on emotional indicator matching
            indicator_matches = sum(1 for indicator in emotional_indicators 
                                  if indicator.lower() in content)
            if indicator_matches > 0:
                boost += min(0.15, indicator_matches * 0.05)
            
            # Boost supportive content for negative emotions
            negative_emotions = ["sadness", "anger", "fear", "disgust"]
            supportive_keywords = ["help", "support", "solution", "guide", "tutorial"]
            
            if primary_emotion in negative_emotions:
                supportive_matches = sum(1 for keyword in supportive_keywords 
                                       if keyword in content)
                if supportive_matches > 0:
                    boost += min(0.1, supportive_matches * 0.03)
            
            return min(0.3, boost)  # Cap at 30% boost
            
        except Exception as e:
            logger.warning(f"Failed to calculate emotional relevance boost: {e}")
            return 0.0
    
    def _calculate_companion_personalization_boost(
        self,
        result: SearchResult,
        user_id: str
    ) -> float:
        """
        Calculate companion-based personalization boost for a search result.
        
        Args:
            result: Search result to evaluate
            user_id: User identifier for personalization
            
        Returns:
            Boost value (0.0 to 0.25)
        """
        try:
            boost = 0.0
            content = result.payload.get("content", "").lower()
            source = result.payload.get("source", "").lower()
            category = result.payload.get("category", "").lower()
            
            # Get user profile for personalization (simplified approach)
            # In a full implementation, this would fetch actual user preferences
            user_interests = self._get_user_interests(user_id)
            
            # Boost based on user interest matching
            interest_matches = sum(1 for interest in user_interests 
                                 if interest.lower() in content or interest.lower() in category)
            if interest_matches > 0:
                boost += min(0.15, interest_matches * 0.05)
            
            # Boost based on preferred content types
            preferred_sources = ["documentation", "tutorial", "guide", "reference"]
            if any(pref_source in source for pref_source in preferred_sources):
                boost += 0.05
            
            # Boost based on content depth (longer content often more comprehensive)
            content_length = len(result.payload.get("content", ""))
            if content_length > 1000:
                boost += 0.03
            elif content_length > 500:
                boost += 0.02
            
            return min(0.25, boost)  # Cap at 25% boost
            
        except Exception as e:
            logger.warning(f"Failed to calculate companion personalization boost: {e}")
            return 0.0
    
    def _calculate_content_quality_boost(self, result: SearchResult) -> float:
        """
        Calculate content quality boost based on various quality indicators.
        
        Args:
            result: Search result to evaluate
            
        Returns:
            Boost value (0.0 to 0.1)
        """
        try:
            boost = 0.0
            content = result.payload.get("content", "")
            metadata = result.payload.get("metadata", {})
            
            # Boost based on content structure (headers, lists, code blocks)
            if any(indicator in content for indicator in ["#", "##", "```", "- ", "1. "]):
                boost += 0.03
            
            # Boost based on recency (if timestamp available)
            indexed_at = result.payload.get("indexed_at")
            if indexed_at:
                try:
                    from datetime import datetime
                    doc_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))
                    current_time = datetime.utcnow()
                    days_ago = (current_time - doc_time).days
                    
                    # Boost recent content (within last 90 days)
                    if days_ago <= 30:
                        boost += 0.02
                    elif days_ago <= 90:
                        boost += 0.01
                except (ValueError, TypeError):
                    pass
            
            # Boost based on content completeness
            if len(content) > 200 and len(content) < 5000:  # Sweet spot for completeness
                boost += 0.02
            
            return min(0.1, boost)  # Cap at 10% boost
            
        except Exception as e:
            logger.warning(f"Failed to calculate content quality boost: {e}")
            return 0.0
    
    def _get_user_interests(self, user_id: str) -> List[str]:
        """
        Get user interests for personalization (simplified implementation).
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user interests
        """
        try:
            # This is a simplified implementation
            # In practice, this would fetch from the companion profile
            default_interests = [
                "python", "javascript", "docker", "api", "database",
                "programming", "development", "software", "technology"
            ]
            
            # Try to get actual user profile
            companion_profile = self.get_companion_profile(user_id)
            if companion_profile and companion_profile.payload:
                interests = companion_profile.payload.get("interests", [])
                if interests:
                    return interests
            
            return default_interests
            
        except Exception as e:
            logger.warning(f"Failed to get user interests for {user_id}: {e}")
            return ["programming", "technology", "development"]
    
    def search_with_emotional_context(
        self,
        collection_name: str,
        query_vector: List[float],
        emotional_context: Dict[str, Any],
        user_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search with integrated emotional context and companion awareness.
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector for semantic search
            emotional_context: Emotional context for result enhancement
            user_id: Optional user ID for personalization
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            
        Returns:
            Emotionally and companion-aware search results
        """
        try:
            # Perform standard vector search with higher limit for filtering
            raw_results = self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit * 2,  # Get more results for emotional filtering
                score_threshold=score_threshold * 0.9 if score_threshold else None
            )
            
            # Apply emotional and companion enhancements
            enhanced_results = self._apply_emotional_and_companion_ranking(
                raw_results, emotional_context, user_id
            )
            
            # Filter by enhanced score threshold if provided
            if score_threshold:
                enhanced_results = [r for r in enhanced_results if r.score >= score_threshold]
            
            return enhanced_results[:limit]
            
        except Exception as e:
            logger.error(f"Emotional context search failed in '{collection_name}': {e}")
            return []
    
    def upsert_hierarchical_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        use_batch_optimization: bool = True
    ) -> bool:
        """
        Upsert points with named vectors for hierarchical collections.
        
        Supports dual-write functionality for hierarchical embeddings with
        coarse, medium, and fine named vectors.
        
        Args:
            collection_name: Target hierarchical collection
            points: List of points with named vector structure:
                   {
                       "id": str,
                       "vector": {
                           "coarse": List[float],
                           "medium": List[float], 
                           "fine": List[float]
                       },
                       "payload": Dict[str, Any]
                   }
            use_batch_optimization: Use optimized batch processing
            
        Returns:
            True if successful
        """
        try:
            # Validate hierarchical point structure
            for point in points:
                if "vector" not in point or not isinstance(point["vector"], dict):
                    raise ValidationError(
                        f"Point {point.get('id', 'unknown')} missing named vector structure",
                        operation="upsert_hierarchical_points",
                        component="vector_db_client"
                    )
                
                required_vectors = ["coarse", "medium", "fine"]
                missing_vectors = [v for v in required_vectors if v not in point["vector"]]
                if missing_vectors:
                    raise ValidationError(
                        f"Point {point.get('id', 'unknown')} missing vectors: {missing_vectors}",
                        operation="upsert_hierarchical_points", 
                        component="vector_db_client"
                    )
            
            # Use optimized batch processing for large point sets
            if use_batch_optimization and len(points) > 50:
                try:
                    def hierarchical_upsert_operation(batch_points: List[Dict[str, Any]]) -> List[Any]:
                        qdrant_points = []
                        for point in batch_points:
                            qdrant_points.append(
                                models.PointStruct(
                                    id=point["id"],
                                    vector=point["vector"],  # Named vectors dict
                                    payload=point.get("payload", {})
                                )
                            )
                        
                        # Use connection pooling if available
                        if self.use_connection_pooling:
                            try:
                                with self.pool_manager.get_connection("qdrant") as pooled_client:
                                    pooled_client.upsert(
                                        collection_name=collection_name,
                                        points=qdrant_points
                                    )
                            except Exception:
                                # Fall back to primary client
                                self.client.upsert(
                                    collection_name=collection_name,
                                    points=qdrant_points
                                )
                        else:
                            self.client.upsert(
                                collection_name=collection_name,
                                points=qdrant_points
                            )
                        
                        return qdrant_points
                    
                    # Process with batch optimizer
                    result = self.batch_processor.process_vector_operations_batch(
                        operations=points,
                        operation_function=hierarchical_upsert_operation,
                        operation_type="hierarchical_upsert"
                    )
                    
                    if result.success_rate >= 95.0:
                        logger.info(
                            f"Optimized hierarchical batch upsert completed: {result.processed_items}/{result.total_items} "
                            f"points ({result.success_rate:.1f}%) to '{collection_name}'"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Hierarchical batch upsert had low success rate ({result.success_rate:.1f}%), "
                            f"falling back to standard processing"
                        )
                        
                except Exception as e:
                    logger.warning(f"Optimized hierarchical batch upsert failed, falling back to standard: {e}")
            
            # Standard hierarchical upsert processing
            qdrant_points = []
            for point in points:
                qdrant_points.append(
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],  # Named vectors dict
                        payload=point.get("payload", {})
                    )
                )
            
            # Use connection pooling if available
            if self.use_connection_pooling and len(points) > 10:
                try:
                    with self.pool_manager.get_connection("qdrant") as pooled_client:
                        pooled_client.upsert(
                            collection_name=collection_name,
                            points=qdrant_points
                        )
                except Exception:
                    # Fall back to primary client
                    self.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points
                    )
            else:
                # Direct upsert for small batches
                self.client.upsert(
                    collection_name=collection_name,
                    points=qdrant_points
                )
            
            logger.debug(f"Upserted {len(points)} hierarchical points to '{collection_name}'")
            return True
            
        except Exception as e:
            raise StorageError(
                f"Failed to upsert hierarchical points to collection '{collection_name}': {e}",
                operation="upsert_hierarchical_points",
                component="vector_db_client",
                metadata={
                    "collection_name": collection_name,
                    "points_count": len(points),
                    "use_batch_optimization": use_batch_optimization,
                    "error_type": type(e).__name__
                }
            ) from e
    
    def search_hierarchical_named(
        self,
        collection_name: str,
        vector_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using a specific named vector in hierarchical collection.
        
        Args:
            collection_name: Hierarchical collection to search
            vector_name: Name of vector to search ("coarse", "medium", "fine")
            query_vector: Query vector for the specified scale
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional payload filters
            
        Returns:
            List of search results
        """
        try:
            # Validate vector name
            if vector_name not in ["coarse", "medium", "fine"]:
                raise ValidationError(
                    f"Invalid vector name: {vector_name}. Must be 'coarse', 'medium', or 'fine'",
                    operation="search_hierarchical_named",
                    component="vector_db_client"
                )
            
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, dict) and "gte" in value:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(gte=value["gte"])
                            )
                        )
                    elif isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                query_filter = models.Filter(must=must_conditions)
            
            # Perform search with named vector
            results = self.client.search(
                collection_name=collection_name,
                query_vector=(vector_name, query_vector),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert to our format
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                ))
            
            logger.debug(
                f"Found {len(search_results)} results using '{vector_name}' vector in '{collection_name}'"
            )
            return search_results
            
        except Exception as e:
            raise StorageError(
                f"Hierarchical named search failed in collection '{collection_name}': {e}",
                operation="search_hierarchical_named",
                component="vector_db_client",
                metadata={
                    "collection_name": collection_name,
                    "vector_name": vector_name,
                    "query_vector_dim": len(query_vector),
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "error_type": type(e).__name__
                }
            ) from e
    
    def dual_write_points(
        self,
        legacy_collection: str,
        hierarchical_collection: str,
        points: List[Dict[str, Any]],
        hierarchical_points: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Dual-write points to both legacy and hierarchical collections.
        
        Enables safe migration by writing to both collection types simultaneously.
        
        Args:
            legacy_collection: Legacy single-vector collection name
            hierarchical_collection: Hierarchical named-vector collection name
            points: Points for legacy collection (single vector per point)
            hierarchical_points: Points for hierarchical collection (named vectors)
            
        Returns:
            Dict with success status for each collection
        """
        results = {
            "legacy": False,
            "hierarchical": False
        }
        
        try:
            # Write to legacy collection first (safer fallback)
            try:
                results["legacy"] = self.upsert_points(legacy_collection, points)
                if results["legacy"]:
                    logger.debug(f"Successfully wrote {len(points)} points to legacy collection '{legacy_collection}'")
            except Exception as e:
                logger.error(f"Failed to write to legacy collection '{legacy_collection}': {e}")
                results["legacy"] = False
            
            # Write to hierarchical collection
            try:
                results["hierarchical"] = self.upsert_hierarchical_points(
                    hierarchical_collection, hierarchical_points
                )
                if results["hierarchical"]:
                    logger.debug(f"Successfully wrote {len(hierarchical_points)} points to hierarchical collection '{hierarchical_collection}'")
            except Exception as e:
                logger.error(f"Failed to write to hierarchical collection '{hierarchical_collection}': {e}")
                results["hierarchical"] = False
            
            # Log dual-write results
            if results["legacy"] and results["hierarchical"]:
                logger.info(f"Dual-write successful: {len(points)} points to both collections")
            elif results["legacy"]:
                logger.warning(f"Partial dual-write: only legacy collection '{legacy_collection}' succeeded")
            elif results["hierarchical"]:
                logger.warning(f"Partial dual-write: only hierarchical collection '{hierarchical_collection}' succeeded")
            else:
                logger.error("Dual-write failed: both collections failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Dual-write operation failed: {e}")
            return results
    
    def ensure_hierarchical_collection(
        self,
        name: str,
        coarse_size: int = 384,
        medium_size: int = 768,
        fine_size: int = 1536,
        distance: str = "cosine"
    ) -> bool:
        """
        Ensure hierarchical collection exists with proper configuration.
        
        Creates the collection if it doesn't exist, validates configuration if it does.
        
        Args:
            name: Collection name
            coarse_size: Coarse embedding dimension
            medium_size: Medium embedding dimension
            fine_size: Fine embedding dimension
            distance: Distance metric
            
        Returns:
            True if collection exists and is properly configured
        """
        try:
            # Check if collection exists
            if self.collection_exists(name):
                # Validate existing collection configuration
                try:
                    collection_info = self.client.get_collection(name)
                    
                    # Check if it has named vectors configuration
                    if hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'vectors'):
                        vectors_config = collection_info.config.params.vectors
                        
                        # Validate named vectors exist
                        if isinstance(vectors_config, dict):
                            required_vectors = ["coarse", "medium", "fine"]
                            existing_vectors = list(vectors_config.keys())
                            
                            if all(v in existing_vectors for v in required_vectors):
                                logger.debug(f"Hierarchical collection '{name}' already exists with proper configuration")
                                return True
                            else:
                                logger.warning(
                                    f"Collection '{name}' exists but missing named vectors. "
                                    f"Expected: {required_vectors}, Found: {existing_vectors}"
                                )
                                return False
                        else:
                            logger.warning(f"Collection '{name}' exists but doesn't have named vectors configuration")
                            return False
                    else:
                        logger.warning(f"Collection '{name}' exists but configuration is not accessible")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not validate existing collection '{name}': {e}")
                    # Assume it exists and is configured correctly
                    return True
            else:
                # Create new hierarchical collection
                success = self.create_hierarchical_collection(
                    name=name,
                    coarse_size=coarse_size,
                    medium_size=medium_size,
                    fine_size=fine_size,
                    distance=distance
                )
                
                if success:
                    logger.info(f"Created new hierarchical collection '{name}'")
                    return True
                else:
                    logger.error(f"Failed to create hierarchical collection '{name}'")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to ensure hierarchical collection '{name}': {e}")
            return False
    
    def get_hierarchical_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a hierarchical collection.
        
        Args:
            collection_name: Name of hierarchical collection
            
        Returns:
            Dictionary with collection information including named vector details
        """
        try:
            if not self.collection_exists(collection_name):
                return {
                    "exists": False,
                    "error": f"Collection '{collection_name}' does not exist"
                }
            
            # Get basic collection info
            collection_info = self.client.get_collection(collection_name)
            
            result = {
                "exists": True,
                "name": collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "disk_usage": getattr(collection_info, 'disk_usage', 0),
                "status": collection_info.status,
                "is_hierarchical": False,
                "named_vectors": {}
            }
            
            # Check for named vectors configuration
            if hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'vectors'):
                vectors_config = collection_info.config.params.vectors
                
                if isinstance(vectors_config, dict):
                    result["is_hierarchical"] = True
                    
                    for vector_name, vector_config in vectors_config.items():
                        result["named_vectors"][vector_name] = {
                            "size": vector_config.size,
                            "distance": vector_config.distance.name if hasattr(vector_config.distance, 'name') else str(vector_config.distance)
                        }
                    
                    # Check if it's a proper hierarchical collection
                    required_vectors = ["coarse", "medium", "fine"]
                    has_all_vectors = all(v in result["named_vectors"] for v in required_vectors)
                    result["is_complete_hierarchical"] = has_all_vectors
                    
                    if not has_all_vectors:
                        missing = [v for v in required_vectors if v not in result["named_vectors"]]
                        result["missing_vectors"] = missing
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get hierarchical collection info for '{collection_name}': {e}")
            return {
                "exists": False,
                "error": str(e)
            }

    def _upsert_with_retry(
        self,
        collection_name: str,
        points: List[models.PointStruct]
    ) -> bool:
        """
        Upsert points with retry logic.
        
        Args:
            collection_name: Target collection
            points: Points to upsert
            
        Returns:
            True if successful
        """
        for attempt in range(self.max_retries):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                return True
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Upsert attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All upsert attempts failed: {e}")
                    return False
        
        return False