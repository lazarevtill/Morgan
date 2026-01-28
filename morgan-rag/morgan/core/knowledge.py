"""
Knowledge Base - Document Storage and Retrieval

Simple, focused module for managing Morgan's knowledge.

KISS Principle: One responsibility - store and retrieve knowledge efficiently.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from morgan.caching.git_hash_tracker import GitHashTracker
from morgan.config import get_settings
from morgan.ingestion.document_processor import DocumentProcessor
from morgan.monitoring.metrics_collector import MetricsCollector
from morgan.services.embeddings import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient
from morgan.vectorization.hierarchical_embeddings import (
    get_hierarchical_embedding_service,
)
from morgan.core.domain.entities import KnowledgeChunk
from morgan.core.infrastructure.repositories import KnowledgeRepository

logger = get_logger(__name__)


class KnowledgeService:
    """
    Service for managing Morgan's knowledge.
    Orchestrates ingestion and search, delegating persistence to KnowledgeRepository.
    """

    def __init__(self):
        """Initialize knowledge service."""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.hierarchical_embedding_service = get_hierarchical_embedding_service()
        self.vector_db = VectorDBClient()
        self.document_processor = DocumentProcessor()
        self.git_tracker = GitHashTracker(cache_dir=Path.home() / ".morgan" / "cache")
        self.metrics_collector = MetricsCollector()

        # Knowledge collections
        self.main_collection = "morgan_knowledge"
        self.hierarchical_collection = "morgan_knowledge_hierarchical"
        self.memory_collection = (
            "morgan_memory"  # Memory collection for conversation context
        )

        # Repositories
        self.repository = KnowledgeRepository(self.vector_db, self.main_collection)
        self.hierarchical_repository = KnowledgeRepository(
            self.vector_db, self.hierarchical_collection
        )

        # Ensure collections exist
        self._ensure_collections()

        logger.info("Knowledge service initialized with repository-based persistence")

    def ingest_documents(
        self,
        source_path: str,
        document_type: str = "auto",
        show_progress: bool = True,
        collection: Optional[str] = None,
        use_hierarchical: bool = True,
    ) -> Dict[str, Any]:
        """
        Add documents to Morgan's knowledge base.

        Human-friendly interface for teaching Morgan new things.
        Now supports hierarchical embeddings for improved search performance.

        Args:
            source_path: Path to documents, directory, or URL
            document_type: Type of documents (auto, pdf, web, code, etc.)
            show_progress: Show progress to user
            collection: Optional collection name (uses default if None)
            use_hierarchical: Use hierarchical embeddings (default: True)

        Returns:
            Ingestion results with human-readable summary

        Example:
            >>> kb = KnowledgeBase()
            >>> result = kb.ingest_documents("./docs", show_progress=True)
            >>> print(f"Processed {result['documents_processed']} documents")
        """
        start_time = time.time()
        collection_name = collection or (
            self.hierarchical_collection if use_hierarchical else self.main_collection
        )

        logger.info(
            f"Starting document ingestion from: {source_path} "
            f"(hierarchical={use_hierarchical})"
        )

        # Initialize result tracking
        cache_hits = 0
        cache_misses = 0
        hierarchical_stats = {"coarse": 0, "medium": 0, "fine": 0}

        try:
            # Step 1: Check Git hash for cache reuse (implements R1.3)
            current_git_hash = None
            if Path(source_path).exists():
                current_git_hash = self.git_tracker.calculate_git_hash(source_path)
                if current_git_hash:
                    cache_status = self.git_tracker.check_cache_validity(
                        source_path, collection_name
                    )
                    if cache_status.is_valid:
                        # Cache hit - get collection info for metrics
                        collection_info = self.git_tracker.get_collection_info(
                            collection_name
                        )
                        document_count = (
                            collection_info.document_count if collection_info else 0
                        )

                        logger.info(
                            f"Cache hit for {source_path} "
                            f"(hash: {current_git_hash[:8]}..., {document_count} docs)"
                        )
                        cache_hits += 1

                        # Record cache hit metrics (implements R9.1)
                        cache_metrics = self.git_tracker.get_cache_metrics()

                        self.metrics_collector.record_git_cache_request(
                            cache_hit=True, source_type="git_hash"
                        )
                        self.metrics_collector.record_cache_hit_rate(
                            cache_type="git_hash",
                            hit_rate=cache_metrics["cache_performance"]["hit_rate"],
                        )

                        return {
                            "success": True,
                            "documents_processed": document_count,
                            "chunks_created": document_count,
                            "cache_hit": True,
                            "git_hash": current_git_hash,
                            "processing_time": time.time() - start_time,
                            "collection": collection_name,
                            "cache_metrics": {
                                "hit_rate": cache_metrics["cache_performance"][
                                    "hit_rate"
                                ],
                                "total_requests": cache_metrics["cache_performance"][
                                    "total_requests"
                                ],
                                "cache_hits": cache_metrics["cache_performance"][
                                    "cache_hits"
                                ],
                                "cache_misses": cache_metrics["cache_performance"][
                                    "cache_misses"
                                ],
                            },
                            "message": (
                                f"Using cached embeddings for {source_path} "
                                f"(unchanged content, {document_count} documents)"
                            ),
                        }
                    else:
                        cache_misses += 1
                        logger.debug(
                            f"Cache miss for {source_path} "
                            f"(stored: {cache_status.stored_hash[:8] if cache_status.stored_hash else 'none'}..., "
                            f"current: {current_git_hash[:8]}...)"
                        )

                        # Record cache miss metrics (implements R9.1)
                        self.metrics_collector.record_git_cache_request(
                            cache_hit=False, source_type="git_hash"
                        )

                        # Record hash calculation
                        self.metrics_collector.record_git_hash_calculation(
                            source_type=(
                                "git"
                                if self.git_tracker._is_git_repository(
                                    Path(source_path)
                                )
                                else "file"
                            )
                        )

            # Step 2: Process documents into chunks
            processing_result = self.document_processor.process_source(
                source_path=source_path,
                document_type=document_type,
                show_progress=show_progress,
            )

            chunks = processing_result["chunks"]
            if not chunks:
                return {
                    "success": False,
                    "message": "No documents found or processed",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                }

            # Step 3: Generate embeddings (hierarchical or legacy)
            if use_hierarchical:
                try:
                    logger.info(
                        f"Generating hierarchical embeddings for {len(chunks)} chunks..."
                    )
                    hierarchical_embeddings = self._create_hierarchical_embeddings(
                        chunks, show_progress, current_git_hash
                    )

                    # Step 4: Store hierarchical embeddings
                    success = self._store_hierarchical_embeddings(
                        hierarchical_embeddings, chunks, collection_name, document_type
                    )

                    if success:
                        # Update hierarchical stats
                        hierarchical_stats = {
                            "coarse": len(hierarchical_embeddings),
                            "medium": len(hierarchical_embeddings),
                            "fine": len(hierarchical_embeddings),
                        }

                        # Store Git hash for future cache hits (implements R1.3)
                        if current_git_hash:
                            # Calculate collection size for metrics
                            collection_size = sum(
                                len(chunk.content.encode("utf-8")) for chunk in chunks
                            )

                            self.git_tracker.store_git_hash(
                                source_path=source_path,
                                collection_name=collection_name,
                                git_hash=current_git_hash,
                                document_count=len(chunks),
                                size_bytes=collection_size,
                                metadata={
                                    "ingestion_type": "hierarchical",
                                    "processing_time": time.time() - start_time,
                                    "hierarchical_stats": hierarchical_stats,
                                },
                            )

                        logger.info("Successfully stored hierarchical embeddings")
                    else:
                        raise Exception("Failed to store hierarchical embeddings")

                except Exception as e:
                    logger.warning(
                        f"Hierarchical ingestion failed: {e}. "
                        f"Falling back to legacy ingestion."
                    )
                    # Fallback to legacy ingestion
                    return self._legacy_ingestion(
                        chunks,
                        collection_name,
                        document_type,
                        show_progress,
                        processing_result,
                        start_time,
                        cache_hits,
                        cache_misses,
                    )
            else:
                # Use legacy ingestion directly
                return self._legacy_ingestion(
                    chunks,
                    collection_name,
                    document_type,
                    show_progress,
                    processing_result,
                    start_time,
                    cache_hits,
                    cache_misses,
                )

            # Step 5: Extract knowledge areas/topics
            knowledge_areas = self._extract_knowledge_areas(chunks)

            # Step 6: Create human-friendly summary with cache metrics (implements R9.1)
            processing_time = time.time() - start_time

            # Get comprehensive cache metrics
            cache_metrics = self.git_tracker.get_cache_metrics()

            result = {
                "success": True,
                "documents_processed": processing_result["documents_processed"],
                "chunks_created": len(chunks),
                "knowledge_areas": knowledge_areas,
                "processing_time": processing_time,
                "collection": collection_name,
                "hierarchical": use_hierarchical,
                "hierarchical_stats": hierarchical_stats,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "git_hash": current_git_hash,
                "cache_metrics": {
                    "hit_rate": cache_metrics["cache_performance"]["hit_rate"],
                    "total_requests": cache_metrics["cache_performance"][
                        "total_requests"
                    ],
                    "cache_hits": cache_metrics["cache_performance"]["cache_hits"],
                    "cache_misses": cache_metrics["cache_performance"]["cache_misses"],
                    "hash_calculations": cache_metrics["cache_performance"][
                        "hash_calculations"
                    ],
                    "total_collections": cache_metrics["collection_stats"][
                        "total_collections"
                    ],
                    "total_documents": cache_metrics["collection_stats"][
                        "total_documents"
                    ],
                },
                "message": (
                    f"Successfully processed {processing_result['documents_processed']} documents "
                    f"into {len(chunks)} knowledge chunks in {processing_time:.1f} seconds "
                    f"({'hierarchical' if use_hierarchical else 'legacy'} mode). "
                    f"Cache hit rate: {cache_metrics['cache_performance']['hit_rate']:.1%}"
                ),
            }

            logger.info(f"Document ingestion completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "chunks_created": 0,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "message": f"Failed to process documents: {str(e)}",
            }

    def search_knowledge(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.7,
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Morgan's knowledge base.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query,
                instruction="query",
            )

            # Use repository for search
            repo = (
                self.hierarchical_repository
                if collection == self.hierarchical_collection
                else self.repository
            )
            chunks = repo.search(
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score,
            )

            # Convert to legacy format for compatibility (if needed) or keep as entities
            results = []
            for chunk in chunks:
                results.append(
                    {
                        "content": chunk.content,
                        "source": chunk.source,
                        "score": getattr(
                            chunk, "score", 0.0
                        ),  # Assuming repo adds score to entity or similar
                        "metadata": chunk.metadata,
                        "document_type": chunk.metadata.get("document_type", "unknown"),
                    }
                )

            logger.debug(f"Found {len(results)} relevant chunks for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []

    def get_document_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific document source.

        Args:
            source: Document source identifier

        Returns:
            List of chunks from that source
        """
        try:
            # Search by source filter
            results = self.vector_db.search_with_filter(
                collection_name=self.main_collection,
                filter_conditions={"source": source},
                limit=1000,  # Get all chunks from source
            )

            chunks = []
            for result in results:
                payload = result.payload
                chunks.append(
                    {
                        "content": payload.get("content", ""),
                        "source": payload.get("source", ""),
                        "metadata": payload.get("metadata", {}),
                        "ingested_at": payload.get("ingested_at", ""),
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Failed to get document by source: {e}")
            return []

    def delete_document(self, source: str) -> bool:
        """
        Delete all chunks from a specific document source.

        Args:
            source: Document source to delete

        Returns:
            True if deletion was successful
        """
        try:
            # Get all chunk IDs for this source
            chunks = self.get_document_by_source(source)
            if not chunks:
                logger.warning(f"No chunks found for source: {source}")
                return False

            # Delete chunks from vector database
            chunk_ids = [chunk.get("id") for chunk in chunks if chunk.get("id")]
            if chunk_ids:
                self.vector_db.delete_points(self.main_collection, chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks from source: {source}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Human-readable statistics about the knowledge base
        """
        try:
            # Get collection info
            collection_info = self.vector_db.get_collection_info(self.main_collection)

            # Get sample of documents to extract topics
            sample_results = self.vector_db.scroll_points(
                collection_name=self.main_collection, limit=100
            )

            # Extract topics and sources
            sources = set()
            topics = set()
            document_types = set()

            for result in sample_results:
                payload = result.payload
                source = payload.get("source", "")
                doc_type = payload.get("document_type", "unknown")
                metadata = payload.get("metadata", {})

                if source:
                    sources.add(source)

                if doc_type:
                    document_types.add(doc_type)

                # Extract topics from metadata
                if "topic" in metadata:
                    topics.add(metadata["topic"])
                elif "category" in metadata:
                    topics.add(metadata["category"])

            return {
                "document_count": len(sources),
                "chunk_count": collection_info.get("points_count", 0),
                "topics": list(topics)[:10],  # Top 10 topics
                "document_types": list(document_types),
                "sources": list(sources)[:20],  # Top 20 sources
                "storage_size_mb": collection_info.get("disk_usage", 0) / (1024 * 1024),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "collection_name": self.main_collection,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e), "document_count": 0, "chunk_count": 0}

    def clear_knowledge(self, confirm: bool = False) -> bool:
        """
        Clear all knowledge from the knowledge base.

        Args:
            confirm: Must be True to actually clear (safety measure)

        Returns:
            True if cleared successfully
        """
        if not confirm:
            logger.warning("Clear knowledge called without confirmation")
            return False

        try:
            # Delete the collection and recreate it
            self.vector_db.delete_collection(self.main_collection)
            self._ensure_collections()

            logger.info("Knowledge base cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False

    def _ensure_collections(self):
        """Ensure required collections exist."""
        try:
            # Get embedding dimension for legacy collections
            embedding_dim = self.embedding_service.get_dimension()

            # Create main knowledge collection (legacy)
            if not self.vector_db.collection_exists(self.main_collection):
                self.vector_db.create_collection(
                    name=self.main_collection,
                    vector_size=embedding_dim,
                    distance="cosine",
                )
                logger.info(
                    f"Created legacy knowledge collection: {self.main_collection}"
                )

            # Ensure hierarchical knowledge collection exists and is properly configured
            success = self.vector_db.ensure_hierarchical_collection(
                name=self.hierarchical_collection,
                coarse_size=384,  # Coarse embeddings
                medium_size=768,  # Medium embeddings
                fine_size=1536,  # Fine embeddings
                distance="cosine",
            )
            if success:
                logger.info(
                    f"Hierarchical knowledge collection ready: "
                    f"{self.hierarchical_collection}"
                )
            else:
                logger.warning(
                    "Failed to ensure hierarchical collection, "
                    "will use legacy fallback"
                )

            # Create memory collection
            if not self.vector_db.collection_exists(self.memory_collection):
                self.vector_db.create_collection(
                    name=self.memory_collection,
                    vector_size=embedding_dim,
                    distance="cosine",
                )
                logger.info(f"Created memory collection: {self.memory_collection}")

        except Exception as e:
            logger.error(f"Failed to ensure collections: {e}")
            raise

    def _create_hierarchical_embeddings(
        self,
        chunks: List[KnowledgeChunk],
        show_progress: bool,
        git_hash: Optional[str] = None,
    ) -> List:
        """
        Create hierarchical embeddings for chunks using HierarchicalEmbeddingService.

        Args:
            chunks: List of knowledge chunks
            show_progress: Show progress indicator
            git_hash: Git hash for metadata

        Returns:
            List of HierarchicalEmbedding objects
        """
        # Prepare data for batch processing
        contents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                **(chunk.metadata or {}),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "git_hash": git_hash,
            }
            metadatas.append(metadata)

        # Create hierarchical embeddings in batch
        hierarchical_embeddings = (
            self.hierarchical_embedding_service.create_batch_hierarchical_embeddings(
                contents=contents,
                metadatas=metadatas,
                show_progress=show_progress,
                use_cache=True,
            )
        )

        return hierarchical_embeddings

    def _store_hierarchical_embeddings(
        self,
        hierarchical_embeddings: List,
        chunks: List[KnowledgeChunk],
        collection_name: str,
        document_type: str,
    ) -> bool:
        """
        Store hierarchical embeddings in vector database with named vectors.

        Args:
            hierarchical_embeddings: List of HierarchicalEmbedding objects
            chunks: Original chunks for payload data
            collection_name: Target collection name
            document_type: Document type for metadata

        Returns:
            True if storage was successful
        """
        try:
            # Prepare points with named vectors
            points = []

            for hierarchical_emb, chunk in zip(hierarchical_embeddings, chunks):
                # Create point with named vectors for hierarchical collection
                point = {
                    "id": chunk.chunk_id,
                    "vector": {
                        "coarse": hierarchical_emb.get_embedding("coarse"),
                        "medium": hierarchical_emb.get_embedding("medium"),
                        "fine": hierarchical_emb.get_embedding("fine"),
                    },
                    "payload": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": {
                            **hierarchical_emb.metadata,
                            **(chunk.metadata or {}),
                        },
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                        "document_type": document_type,
                        "hierarchical_texts": hierarchical_emb.texts,
                        "embedding_type": "hierarchical",
                    },
                }
                points.append(point)

            # Use VectorDBClient's hierarchical upsert method
            success = self.vector_db.upsert_hierarchical_points(collection_name, points)

            if success:
                logger.info(
                    f"Successfully stored {len(points)} hierarchical embeddings"
                )
                return True
            else:
                logger.error("Failed to store hierarchical embeddings")
                return False

        except Exception as e:
            logger.error(f"Error storing hierarchical embeddings: {e}")
            return False

    def _legacy_ingestion(
        self,
        chunks: List[KnowledgeChunk],
        collection_name: str,
        document_type: str,
        show_progress: bool,
        processing_result: Dict[str, Any],
        start_time: float,
        cache_hits: int,
        cache_misses: int,
    ) -> Dict[str, Any]:
        """
        Fallback to legacy single-vector ingestion.

        Args:
            chunks: Knowledge chunks to process
            collection_name: Target collection (will use legacy collection)
            document_type: Document type
            show_progress: Show progress indicator
            processing_result: Original processing result
            start_time: Start time for timing
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses

        Returns:
            Ingestion result dictionary
        """
        try:
            logger.info(f"Using legacy ingestion for {len(chunks)} chunks...")

            # Use legacy collection
            legacy_collection = self.main_collection

            # Generate single embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(
                texts=chunk_texts, instruction="document", show_progress=show_progress
            )

            # Store in legacy format
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "payload": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                        "document_type": document_type,
                        "embedding_type": "legacy",
                    },
                }
                points.append(point)

            # Batch insert
            self.vector_db.upsert_points(legacy_collection, points)

            # Extract knowledge areas
            knowledge_areas = self._extract_knowledge_areas(chunks)

            # Create result
            processing_time = time.time() - start_time

            result = {
                "success": True,
                "documents_processed": processing_result["documents_processed"],
                "chunks_created": len(chunks),
                "knowledge_areas": knowledge_areas,
                "processing_time": processing_time,
                "collection": legacy_collection,
                "hierarchical": False,
                "fallback_used": True,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "message": (
                    f"Successfully processed {processing_result['documents_processed']} documents "
                    f"into {len(chunks)} knowledge chunks in {processing_time:.1f} seconds "
                    f"(legacy fallback mode)."
                ),
            }

            logger.info("Legacy ingestion completed: %s", result["message"])
            return result

        except Exception as e:
            logger.error(f"Legacy ingestion also failed: {e}")
            raise

    def _extract_knowledge_areas(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """
        Extract knowledge areas/topics from chunks.

        Simple approach: look at metadata and source patterns.
        """
        topics = set()

        for chunk in chunks:
            # From metadata
            metadata = chunk.metadata or {}
            if "topic" in metadata:
                topics.add(metadata["topic"])
            if "category" in metadata:
                topics.add(metadata["category"])

            # From source path
            source_path = Path(chunk.source)
            if len(source_path.parts) > 1:
                # Use directory name as topic
                topics.add(source_path.parts[-2])

        # Clean up topics
        cleaned_topics = []
        for topic in topics:
            if isinstance(topic, str) and len(topic) > 2:
                cleaned_topics.append(topic.replace("_", " ").title())

        return sorted(set(cleaned_topics))[:10]  # Top 10 unique topics


# Human-friendly helper functions
def quick_search(query: str, max_results: int = 5) -> List[str]:
    """
    Quick search of knowledge base.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of content snippets

    Example:
        >>> results = quick_search("Docker deployment")
        >>> for result in results:
        ...     print(result[:100] + "...")
    """
    kb = KnowledgeBase()
    search_results = kb.search_knowledge(query, max_results=max_results)
    return [result["content"] for result in search_results]


def add_knowledge_from_text(text: str, source: str = "manual_input") -> bool:
    """
    Add knowledge from a text string.

    Args:
        text: Text content to add
        source: Source identifier

    Returns:
        True if added successfully

    Example:
        >>> success = add_knowledge_from_text(
        ...     "Docker is a containerization platform...",
        ...     "docker_basics"
        ... )
    """
    try:
        from morgan.services.embeddings import get_embedding_service
        from morgan.vector_db.client import VectorDBClient
        from morgan.config import get_settings

        settings = get_settings()
        embedding_service = get_embedding_service()
        vector_db = VectorDBClient()
        collection_name = "morgan_knowledge"

        # Split text into chunks if it's long
        chunk_size = settings.morgan_chunk_size
        chunk_overlap = settings.morgan_chunk_overlap
        chunks = []
        if len(text) <= chunk_size:
            chunks = [text]
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start += chunk_size - chunk_overlap

        # Generate embeddings and store
        import uuid as _uuid
        from datetime import datetime as _dt

        points = []
        for chunk_text in chunks:
            chunk_id = str(_uuid.uuid4())
            embedding = embedding_service.encode(text=chunk_text, instruction="document")
            points.append({
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "content": chunk_text,
                    "source": source,
                    "metadata": {"source_type": "manual_text"},
                    "ingested_at": _dt.utcnow().isoformat(),
                    "document_type": "text",
                    "embedding_type": "legacy",
                },
            })

        if points:
            vector_db.upsert_points(collection_name, points)
            logger.info(f"Added {len(points)} chunks from text (source: {source})")
            return True

        return False

    except Exception as e:
        logger.error("Failed to add text knowledge: %s", e)
        return False


if __name__ == "__main__":
    # Demo knowledge base capabilities
    print("🧠 Morgan Knowledge Base Demo")
    print("=" * 40)

    kb = KnowledgeBase()

    # Get statistics
    stats = kb.get_statistics()
    print("Knowledge Base Stats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Topics: {', '.join(stats['topics'][:5])}")

    # Test search
    print("\nSearching for 'Docker'...")
    results = kb.search_knowledge("Docker", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.2f}")
        print(f"     Source: {result['source']}")
        print(f"     Content: {result['content'][:100]}...")
        print()
