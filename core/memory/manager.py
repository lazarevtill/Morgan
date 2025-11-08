"""
Memory Manager for Morgan AI Assistant
Handles persistent memory storage using PostgreSQL and Qdrant
"""
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np

from shared.utils.logging import setup_logging


class Memory:
    """Memory data class"""
    def __init__(
        self,
        id: str,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        category: Optional[str] = None,
        importance: int = 5,
        metadata: Optional[Dict] = None,
        created_at: Optional[datetime] = None,
        qdrant_point_id: Optional[str] = None
    ):
        self.id = id
        self.user_id = user_id
        self.content = content
        self.memory_type = memory_type
        self.category = category
        self.importance = importance
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.qdrant_point_id = qdrant_point_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "qdrant_point_id": self.qdrant_point_id
        }


class MemoryManager:
    """
    Manages memories using dual storage:
    - PostgreSQL for structured data
    - Qdrant for semantic vector search
    """

    def __init__(
        self,
        postgres_dsn: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_dimension: int = 384  # all-MiniLM-L6-v2 default
    ):
        self.postgres_dsn = postgres_dsn
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedding_dimension = embedding_dimension

        self.logger = setup_logging("memory_manager", "INFO", "logs/memory.log")

        self.pg_pool: Optional[asyncpg.Pool] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.embedding_model = None

        self.collection_name = "morgan_memories"

    async def start(self):
        """Initialize database connections"""
        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self.logger.info("PostgreSQL connection pool created")

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=30
            )

            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Qdrant collection exists: {self.collection_name}")

            # Initialize thread pool for CPU-bound operations
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

            # Initialize embedding service connection (no longer need thread pool for embeddings)
            # Embedding generation now happens via HTTP calls to LLM service
            self.embedding_service_ready = True

            self.logger.info("Memory Manager started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start Memory Manager: {e}", exc_info=True)
            raise

    async def stop(self):
        """Close database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
            self.logger.info("PostgreSQL connection pool closed")

        if self.qdrant_client:
            self.qdrant_client.close()
            self.logger.info("Qdrant client closed")

        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("Embedding executor shutdown")

    async def create_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        category: Optional[str] = None,
        importance: int = 5,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Memory:
        """
        Create a new memory

        Args:
            user_id: User identifier
            content: Memory content (text)
            memory_type: Type of memory (fact, preference, context, instruction)
            category: Optional category
            importance: Importance score 1-10
            metadata: Additional metadata
            embedding: Pre-computed embedding vector (if None, will be generated)

        Returns:
            Created Memory object
        """
        try:
            memory_id = str(uuid.uuid4())
            qdrant_point_id = str(uuid.uuid4())

            # Store in PostgreSQL
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memories (id, user_id, content, memory_type, category, importance, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    category,
                    importance,
                    metadata or {}
                )

                # Link to vector
                await conn.execute(
                    """
                    INSERT INTO vector_references (qdrant_point_id, qdrant_collection, entity_type, entity_id)
                    VALUES ($1, $2, $3, $4)
                    """,
                    qdrant_point_id,
                    self.collection_name,
                    "memory",
                    memory_id
                )

            # Store embedding in Qdrant
            if embedding is None:
                embedding = await self._generate_embedding(content)
                self.logger.debug(f"Generated embedding with dimension: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=qdrant_point_id,
                        vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        payload={
                            "memory_id": memory_id,
                            "user_id": user_id,
                            "content": content,
                            "memory_type": memory_type,
                            "category": category,
                            "importance": importance,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                ]
            )

            self.logger.info(f"Created memory: {memory_id} for user: {user_id}")

            return Memory(
                id=memory_id,
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                category=category,
                importance=importance,
                metadata=metadata,
                qdrant_point_id=qdrant_point_id
            )

        except Exception as e:
            self.logger.error(f"Failed to create memory: {e}", exc_info=True)
            raise

    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: int = 1
    ) -> List[Memory]:
        """
        Search memories semantically using vector similarity

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type
            min_importance: Minimum importance score

        Returns:
            List of matching Memory objects
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            self.logger.debug(f"Generated query embedding with dimension: {len(query_embedding) if hasattr(query_embedding, '__len__') else 'unknown'}")

            # Build filter
            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if memory_type:
                must_conditions.append(
                    FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
                )

            if min_importance > 1:
                must_conditions.append(
                    FieldCondition(key="importance", range={"gte": min_importance})
                )

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                query_filter=Filter(must=must_conditions) if must_conditions else None,
                limit=limit,
                with_payload=True
            )

            # Convert to Memory objects
            memories = []
            for result in search_results:
                payload = result.payload
                memories.append(Memory(
                    id=payload["memory_id"],
                    user_id=payload["user_id"],
                    content=payload["content"],
                    memory_type=payload.get("memory_type", "fact"),
                    category=payload.get("category"),
                    importance=payload.get("importance", 5),
                    metadata=payload.get("metadata", {}),
                    qdrant_point_id=result.id
                ))

            self.logger.info(f"Found {len(memories)} memories for query: {query[:50]}...")
            return memories

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}", exc_info=True)
            return []

    async def get_recent_memories(
        self,
        user_id: str,
        limit: int = 20,
        memory_type: Optional[str] = None
    ) -> List[Memory]:
        """Get most recent memories for a user"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT m.*, v.qdrant_point_id
                    FROM memories m
                    LEFT JOIN vector_references v ON v.entity_id = m.id AND v.entity_type = 'memory'
                    WHERE m.user_id = $1
                """

                params = [user_id]

                if memory_type:
                    query += " AND m.memory_type = $2"
                    params.append(memory_type)

                query += " ORDER BY m.created_at DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)

                rows = await conn.fetch(query, *params)

                memories = []
                for row in rows:
                    memories.append(Memory(
                        id=str(row["id"]),
                        user_id=row["user_id"],
                        content=row["content"],
                        memory_type=row["memory_type"],
                        category=row["category"],
                        importance=row["importance"],
                        metadata=row["metadata"],
                        created_at=row["created_at"],
                        qdrant_point_id=str(row["qdrant_point_id"]) if row.get("qdrant_point_id") else None
                    ))

                return memories

        except Exception as e:
            self.logger.error(f"Failed to get recent memories: {e}", exc_info=True)
            return []

    async def delete_memory(self, memory_id: str):
        """Delete a memory from both PostgreSQL and Qdrant"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Get vector reference first
                vector_ref = await conn.fetchrow(
                    "SELECT qdrant_point_id FROM vector_references WHERE entity_id = $1 AND entity_type = 'memory'",
                    memory_id
                )

                # Delete from PostgreSQL (cascade will delete vector_references)
                await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

                # Delete from Qdrant
                if vector_ref:
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=[str(vector_ref["qdrant_point_id"])]
                    )

            self.logger.info(f"Deleted memory: {memory_id}")

        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}", exc_info=True)
            raise

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text using LLM service

        Args:
            text: Text to generate embedding for

        Returns:
            Numpy array of embedding vector
        """
        try:
            # Import here to avoid circular imports
            from shared.utils.http_client import service_registry

            # Get LLM service client
            llm_client = await service_registry.get_service("llm")

            # Generate embedding using LLM service
            result = await llm_client.post("/embed", json_data={"text": text})

            if result.success and result.data.get("embedding"):
                embedding = result.data["embedding"]

                # Convert to numpy array and ensure correct dimension
                embedding_array = np.array(embedding, dtype=np.float32)

                # Truncate or pad to match expected dimension
                if len(embedding_array) > self.embedding_dimension:
                    embedding_array = embedding_array[:self.embedding_dimension]
                elif len(embedding_array) < self.embedding_dimension:
                    # Pad with zeros
                    padding = np.zeros(self.embedding_dimension - len(embedding_array), dtype=np.float32)
                    embedding_array = np.concatenate([embedding_array, padding])

                return embedding_array
            else:
                # Fallback to random vector if embedding fails
                self.logger.warning(f"Embedding generation failed: {result.error if result else 'Unknown error'}")
                return np.random.rand(self.embedding_dimension).astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            # Fallback to random vector
            return np.random.rand(self.embedding_dimension).astype(np.float32)

    async def update_memory_access(self, memory_id: str):
        """Update memory access statistics"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    memory_id
                )
        except Exception as e:
            self.logger.error(f"Failed to update memory access: {e}")

    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        try:
            async with self.pg_pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT memory_type) as unique_types,
                        COUNT(DISTINCT category) as unique_categories,
                        AVG(importance) as avg_importance,
                        MAX(created_at) as latest_memory
                    FROM memories
                    WHERE user_id = $1
                    """,
                    user_id
                )

                return {
                    "total_memories": stats["total_memories"],
                    "unique_types": stats["unique_types"],
                    "unique_categories": stats["unique_categories"],
                    "avg_importance": float(stats["avg_importance"]) if stats["avg_importance"] else 0,
                    "latest_memory": stats["latest_memory"].isoformat() if stats["latest_memory"] else None
                }

        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}
