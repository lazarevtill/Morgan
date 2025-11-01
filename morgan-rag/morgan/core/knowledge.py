"""
Knowledge Base - Document Storage and Retrieval

Simple, focused module for managing Morgan's knowledge.

KISS Principle: One responsibility - store and retrieve knowledge efficiently.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.embedding_service import get_embedding_service
from morgan.vector_db.client import VectorDBClient
from morgan.ingestion.document_processor import DocumentProcessor

logger = get_logger(__name__)


@dataclass
class KnowledgeChunk:
    """
    A piece of knowledge with metadata.
    
    Simple structure that's easy to understand and work with.
    """
    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}


class KnowledgeBase:
    """
    Morgan's Knowledge Base
    
    Stores and retrieves documents in a way that's:
    - Easy to understand
    - Fast to search
    - Simple to maintain
    
    KISS: Single responsibility - manage knowledge storage and retrieval.
    """
    
    def __init__(self):
        """Initialize knowledge base."""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBClient()
        self.document_processor = DocumentProcessor()
        
        # Knowledge collections
        self.main_collection = "morgan_knowledge"
        self.memory_collection = "morgan_memory"
        
        # Ensure collections exist
        self._ensure_collections()
        
        logger.info("Knowledge base initialized")
    
    def ingest_documents(
        self, 
        source_path: str, 
        document_type: str = "auto",
        show_progress: bool = True,
        collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to Morgan's knowledge base.
        
        Human-friendly interface for teaching Morgan new things.
        
        Args:
            source_path: Path to documents, directory, or URL
            document_type: Type of documents (auto, pdf, web, code, etc.)
            show_progress: Show progress to user
            collection: Optional collection name (uses default if None)
            
        Returns:
            Ingestion results with human-readable summary
            
        Example:
            >>> kb = KnowledgeBase()
            >>> result = kb.ingest_documents("./docs", show_progress=True)
            >>> print(f"Processed {result['documents_processed']} documents")
        """
        start_time = time.time()
        collection_name = collection or self.main_collection
        
        logger.info(f"Starting document ingestion from: {source_path}")
        
        try:
            # Step 1: Process documents into chunks
            processing_result = self.document_processor.process_source(
                source_path=source_path,
                document_type=document_type,
                show_progress=show_progress
            )
            
            chunks = processing_result["chunks"]
            if not chunks:
                return {
                    "success": False,
                    "message": "No documents found or processed",
                    "documents_processed": 0,
                    "chunks_created": 0
                }
            
            # Step 2: Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(
                texts=chunk_texts,
                instruction="document",  # Use document instruction for better retrieval
                show_progress=show_progress
            )
            
            # Step 3: Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in knowledge base...")
            
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "payload": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                        "ingested_at": datetime.utcnow().isoformat(),
                        "document_type": document_type
                    }
                }
                points.append(point)
            
            # Batch insert for efficiency
            self.vector_db.upsert_points(collection_name, points)
            
            # Step 4: Extract knowledge areas/topics
            knowledge_areas = self._extract_knowledge_areas(chunks)
            
            # Step 5: Create human-friendly summary
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "documents_processed": processing_result["documents_processed"],
                "chunks_created": len(chunks),
                "knowledge_areas": knowledge_areas,
                "processing_time": processing_time,
                "collection": collection_name,
                "message": f"Successfully processed {processing_result['documents_processed']} documents "
                          f"into {len(chunks)} knowledge chunks in {processing_time:.1f} seconds."
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
                "message": f"Failed to process documents: {str(e)}"
            }
    
    def search_knowledge(
        self, 
        query: str, 
        max_results: int = 10,
        min_score: float = 0.7,
        collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Morgan's knowledge base.
        
        Args:
            query: Search query in natural language
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0.0 to 1.0)
            collection: Optional collection to search (uses default if None)
            
        Returns:
            List of relevant knowledge chunks with scores
            
        Example:
            >>> kb = KnowledgeBase()
            >>> results = kb.search_knowledge("Docker deployment")
            >>> for result in results:
            ...     print(f"Score: {result['score']:.2f}")
            ...     print(f"Content: {result['content'][:100]}...")
        """
        collection_name = collection or self.main_collection
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query,
                instruction="query"  # Use query instruction for better matching
            )
            
            # Search vector database
            search_results = self.vector_db.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score
            )
            
            # Convert to human-friendly format
            results = []
            for result in search_results:
                payload = result.get("payload", {})
                results.append({
                    "content": payload.get("content", ""),
                    "source": payload.get("source", "Unknown"),
                    "score": result.get("score", 0.0),
                    "metadata": payload.get("metadata", {}),
                    "document_type": payload.get("document_type", "unknown")
                })
            
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
                limit=1000  # Get all chunks from source
            )
            
            chunks = []
            for result in results:
                payload = result.get("payload", {})
                chunks.append({
                    "content": payload.get("content", ""),
                    "source": payload.get("source", ""),
                    "metadata": payload.get("metadata", {}),
                    "ingested_at": payload.get("ingested_at", "")
                })
            
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
                collection_name=self.main_collection,
                limit=100
            )
            
            # Extract topics and sources
            sources = set()
            topics = set()
            document_types = set()
            
            for result in sample_results:
                payload = result.get("payload", {})
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
                "last_updated": datetime.utcnow().isoformat(),
                "collection_name": self.main_collection
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "error": str(e),
                "document_count": 0,
                "chunk_count": 0
            }
    
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
            # Get embedding dimension
            embedding_dim = self.embedding_service.get_embedding_dimension()
            
            # Create main knowledge collection
            if not self.vector_db.collection_exists(self.main_collection):
                self.vector_db.create_collection(
                    name=self.main_collection,
                    vector_size=embedding_dim,
                    distance="cosine"
                )
                logger.info(f"Created knowledge collection: {self.main_collection}")
            
            # Create memory collection
            if not self.vector_db.collection_exists(self.memory_collection):
                self.vector_db.create_collection(
                    name=self.memory_collection,
                    vector_size=embedding_dim,
                    distance="cosine"
                )
                logger.info(f"Created memory collection: {self.memory_collection}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collections: {e}")
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
        
        return sorted(list(set(cleaned_topics)))[:10]  # Top 10 unique topics


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
        kb = KnowledgeBase()
        
        # Create a simple chunk
        chunk = KnowledgeChunk(
            content=text,
            source=source,
            chunk_id=f"{source}_{hash(text)}",
            metadata={"type": "manual", "added_at": datetime.utcnow().isoformat()}
        )
        
        # Process as single document
        result = kb.ingest_documents(
            source_path="manual",  # Placeholder
            document_type="text",
            show_progress=False
        )
        
        return result.get("success", False)
        
    except Exception as e:
        logger.error(f"Failed to add text knowledge: {e}")
        return False


if __name__ == "__main__":
    # Demo knowledge base capabilities
    print("🧠 Morgan Knowledge Base Demo")
    print("=" * 40)
    
    kb = KnowledgeBase()
    
    # Get statistics
    stats = kb.get_statistics()
    print(f"Knowledge Base Stats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Topics: {', '.join(stats['topics'][:5])}")
    
    # Test search
    print(f"\nSearching for 'Docker'...")
    results = kb.search_knowledge("Docker", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.2f}")
        print(f"     Source: {result['source']}")
        print(f"     Content: {result['content'][:100]}...")
        print()