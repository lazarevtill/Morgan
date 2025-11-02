"""
Vector Storage - Vector database operations

Provides unified interface for vector database operations using existing VectorDBClient.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Simple vector document representation."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    collection: str
    timestamp: datetime


class VectorStorage:
    """
    Vector storage following KISS principles.
    
    Single responsibility: Manage vector database operations using existing client.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = None
        
        # Initialize vector database client
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize the existing vector database client."""
        try:
            # Use existing enhanced vector_db client
            from ..vector_db.client import VectorDBClient
            
            self.client = VectorDBClient()
            logger.info("Vector storage initialized with existing VectorDBClient")
            
        except Exception as e:
            logger.error("Failed to initialize vector storage: %s", e)
            raise
            
    def create_collection(self, collection_name: str, dimension: int,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new vector collection.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            metadata: Optional collection metadata
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            # Check if collection already exists
            if self.collection_exists(collection_name):
                logger.info("Collection %s already exists", collection_name)
                return True
                
            # Create collection using existing client
            success = self.client.create_collection(
                name=collection_name,
                vector_size=dimension,
                distance="cosine"
            )
            
            if success:
                logger.info("Created collection: %s", collection_name)
            else:
                logger.error("Failed to create collection: %s", collection_name)
                
            return success
            
        except Exception as e:
            logger.error("Error creating collection %s: %s", collection_name, e)
            return False
            
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            if not self.client:
                return False
                
            # Use existing client method if available
            if hasattr(self.client, 'collection_exists'):
                return self.client.collection_exists(collection_name)
            else:
                # Fallback: try to get collection info
                try:
                    self.client.get_collection_info(collection_name)
                    return True
                except Exception:
                    return False
                    
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            return False
            
    def store_documents(self, collection_name: str,
                       documents: List[VectorDocument]) -> bool:
        """
        Store multiple documents in a collection.
        
        Args:
            collection_name: Target collection name
            documents: List of documents to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            # Prepare documents for storage using existing client format
            points = []
            for doc in documents:
                point = {
                    'id': doc.id,
                    'vector': doc.embedding,
                    'payload': {
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'timestamp': doc.timestamp.isoformat(),
                        'collection': doc.collection
                    }
                }
                points.append(point)
                
            # Store using existing client
            success = self.client.upsert_points(collection_name, points)
            
            if success:
                logger.info("Stored %d documents in %s",
                           len(documents), collection_name)
            else:
                logger.error("Failed to store documents in %s", collection_name)
                
            return success
            
        except Exception as e:
            logger.error("Error storing documents: %s", e)
            return False
            
    def search_vectors(self, collection_name: str, query_vector: List[float],
                      limit: int = 10, score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict[str, Any]] = None
                      ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using existing client.
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results from existing client
        """
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            # Use existing client's search method
            if filter_conditions:
                # Use filter search if conditions provided
                results = self.client.search_with_filter(
                    collection_name=collection_name,
                    filter_conditions=filter_conditions,
                    limit=limit
                )
            else:
                # Use vector search
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
            
            # Convert to simple format
            search_results = []
            for result in results:
                search_results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
                
            logger.debug("Found %d results in %s",
                        len(search_results), collection_name)
            return search_results
            
        except Exception as e:
            logger.error("Error searching vectors: %s", e)
            return []
            
    def get_document(self, collection_name: str,
                    document_id: str) -> Optional[VectorDocument]:
        """
        Retrieve a specific document by ID.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            # Get document using existing client
            result = self.client.get_point(collection_name, document_id)
            
            if result:
                doc = VectorDocument(
                    id=result['id'],
                    content=result['payload']['content'],
                    embedding=result['vector'],
                    metadata=result['payload']['metadata'],
                    collection=collection_name,
                    timestamp=datetime.fromisoformat(
                        result['payload']['timestamp'])
                )
                return doc
            else:
                return None
                
        except Exception as e:
            logger.error("Error retrieving document %s: %s", document_id, e)
            return None
            
    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from a collection."""
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            success = self.client.delete_points(collection_name, [document_id])
            
            if success:
                logger.info("Deleted document %s from %s",
                           document_id, collection_name)
            else:
                logger.error("Failed to delete document %s", document_id)
                
            return success
            
        except Exception as e:
            logger.error("Error deleting document %s: %s", document_id, e)
            return False
            
    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection."""
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            success = self.client.delete_collection(collection_name)
            
            if success:
                logger.info("Deleted collection: %s", collection_name)
            else:
                logger.error("Failed to delete collection: %s", collection_name)
                
            return success
            
        except Exception as e:
            logger.error("Error deleting collection %s: %s", collection_name, e)
            return False
            
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            info = self.client.get_collection_info(collection_name)
            return info
            
        except Exception as e:
            logger.error("Error getting collection info: %s", e)
            return None
            
    def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            if not self.client:
                raise RuntimeError("Vector storage not initialized")
                
            collections = self.client.list_collections()
            return collections
            
        except Exception as e:
            logger.error("Error listing collections: %s", e)
            return []
            
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            collections = self.list_collections()
            stats = {
                'total_collections': len(collections),
                'collections': {}
            }
            
            for collection in collections:
                info = self.get_collection_info(collection)
                if info:
                    stats['collections'][collection] = info
                    
            return stats
            
        except Exception as e:
            logger.error("Error getting storage stats: %s", e)
            return {'error': str(e)}