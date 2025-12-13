"""
RAG (Retrieval-Augmented Generation) system for the Knowledge Engine.

This module implements semantic search using vector embeddings, context-aware
document retrieval, multi-stage ranking (vector similarity + reranking), and
source attribution with confidence scoring.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import structlog

from morgan_server.knowledge.vectordb import VectorDBClient, SearchResult
from morgan_server.knowledge.ingestion import DocumentChunk, DocumentProcessor


logger = structlog.get_logger(__name__)


class RAGError(Exception):
    """Base exception for RAG system errors."""


class EmbeddingError(RAGError):
    """Exception raised when embedding generation fails."""


class RetrievalError(RAGError):
    """Exception raised when document retrieval fails."""


@dataclass
class Source:
    """Represents a source document with attribution."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_score: Optional[float] = None


@dataclass
class RAGResult:
    """Result from RAG retrieval with sources and confidence."""

    query: str
    sources: List[Source]
    context: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingService:
    """Service for generating embeddings from text.
    
    This is a simple interface that can be implemented with different
    embedding providers (local models, Ollama, OpenAI-compatible APIs).
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the embedding service.
        
        Args:
            provider: Embedding provider ("local", "ollama", "openai-compatible")
            model: Model name/identifier
            device: Device for local models ("cpu", "cuda")
            endpoint: API endpoint for remote providers
            api_key: API key for remote providers
        """
        self.provider = provider
        self.model = model
        self.device = device
        self.endpoint = endpoint
        self.api_key = api_key
        self._model_instance = None
        
        logger.info(
            "embedding_service_initialized",
            provider=provider,
            model=model,
            device=device,
        )

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if self.provider == "local":
                return await self._embed_local(text)
            elif self.provider == "ollama":
                return await self._embed_ollama(text)
            elif self.provider == "openai-compatible":
                return await self._embed_openai_compatible(text)
            else:
                raise EmbeddingError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error("embedding_failed", text_length=len(text), error=str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if self.provider == "local":
                return await self._embed_batch_local(texts)
            elif self.provider == "ollama":
                return await self._embed_batch_ollama(texts)
            elif self.provider == "openai-compatible":
                return await self._embed_batch_openai_compatible(texts)
            else:
                raise EmbeddingError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error("batch_embedding_failed", batch_size=len(texts), error=str(e))
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")

    async def _embed_local(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        if self._model_instance is None:
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model, device=self.device)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model_instance.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model (batch)."""
        if self._model_instance is None:
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model, device=self.device)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model_instance.encode(texts, convert_to_numpy=True)
        )
        return [emb.tolist() for emb in embeddings]

    async def _embed_ollama(self, text: str) -> List[float]:
        """Generate embedding using Ollama API."""
        import aiohttp
        
        if not self.endpoint:
            raise EmbeddingError("Ollama endpoint not configured")
        
        url = f"{self.endpoint}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["embedding"]

    async def _embed_batch_ollama(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama API (batch)."""
        # Ollama doesn't have native batch support, so we do sequential requests
        embeddings = []
        for text in texts:
            embedding = await self._embed_ollama(text)
            embeddings.append(embedding)
        return embeddings

    async def _embed_openai_compatible(self, text: str) -> List[float]:
        """Generate embedding using OpenAI-compatible API."""
        import aiohttp
        
        if not self.endpoint:
            raise EmbeddingError("OpenAI-compatible endpoint not configured")
        
        url = f"{self.endpoint}/v1/embeddings"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {"model": self.model, "input": text}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data["data"][0]["embedding"]

    async def _embed_batch_openai_compatible(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI-compatible API (batch)."""
        import aiohttp
        
        if not self.endpoint:
            raise EmbeddingError("OpenAI-compatible endpoint not configured")
        
        url = f"{self.endpoint}/v1/embeddings"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {"model": self.model, "input": texts}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return [item["embedding"] for item in data["data"]]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        # Common dimensions for popular models
        dimension_map = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "nomic-embed-text": 768,
        }
        return dimension_map.get(self.model, 384)  # Default to 384


class RAGSystem:
    """RAG system for semantic search and context retrieval.
    
    Implements multi-stage retrieval:
    1. Vector similarity search
    2. Optional reranking for improved relevance
    3. Source attribution and confidence scoring
    """

    def __init__(
        self,
        vectordb_client: VectorDBClient,
        embedding_service: EmbeddingService,
        collection_name: str = "knowledge_base",
        top_k: int = 10,
        rerank_top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        """Initialize the RAG system.
        
        Args:
            vectordb_client: Vector database client
            embedding_service: Embedding service
            collection_name: Name of the collection to search
            top_k: Number of results to retrieve in initial search
            rerank_top_k: Number of results to keep after reranking
            score_threshold: Minimum similarity score threshold
        """
        self.vectordb_client = vectordb_client
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.score_threshold = score_threshold
        
        logger.info(
            "rag_system_initialized",
            collection_name=collection_name,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: bool = True,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides default)
            rerank: Whether to apply reranking
            filter_conditions: Optional metadata filters
            
        Returns:
            RAGResult with sources and context
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info("rag_retrieve_started", query=query[:100])
            
            # Step 1: Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Step 2: Vector similarity search
            k = top_k or self.top_k
            search_results = await self.vectordb_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=self.score_threshold,
                filter_conditions=filter_conditions,
            )
            
            if not search_results:
                logger.info("no_results_found", query=query[:100])
                return RAGResult(
                    query=query,
                    sources=[],
                    context="",
                    confidence=0.0,
                    metadata={"search_results": 0},
                )
            
            # Step 3: Convert to Source objects
            sources = [
                Source(
                    document_id=result.payload.get("document_id", "unknown"),
                    chunk_id=result.id,
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata=result.payload,
                )
                for result in search_results
            ]
            
            # Step 4: Optional reranking
            if rerank and len(sources) > 1:
                sources = await self._rerank_sources(query, sources)
                sources = sources[: self.rerank_top_k]
            
            # Step 5: Build context and calculate confidence
            context = self._build_context(sources)
            confidence = self._calculate_confidence(sources)
            
            logger.info(
                "rag_retrieve_completed",
                query=query[:100],
                num_sources=len(sources),
                confidence=confidence,
            )
            
            return RAGResult(
                query=query,
                sources=sources,
                context=context,
                confidence=confidence,
                metadata={
                    "search_results": len(search_results),
                    "reranked": rerank,
                    "top_score": sources[0].score if sources else 0.0,
                },
            )
            
        except Exception as e:
            logger.error("rag_retrieve_failed", query=query[:100], error=str(e))
            raise RetrievalError(f"Failed to retrieve documents: {e}")

    async def index_document(
        self,
        source: str,
        doc_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Index a document into the knowledge base.
        
        Args:
            source: Path to file or URL
            doc_type: Document type (auto-detect if "auto")
            metadata: Additional metadata
            
        Returns:
            Number of chunks indexed
            
        Raises:
            RAGError: If indexing fails
        """
        try:
            logger.info("indexing_document", source=source, doc_type=doc_type)
            
            # Process document into chunks
            processor = DocumentProcessor()
            chunks = processor.process(source, doc_type, metadata)
            
            if not chunks:
                logger.info("document_unchanged", source=source)
                return 0
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.embed_batch(chunk_texts)
            
            # Prepare payloads
            payloads = [
                {
                    "content": chunk.content,
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
                for chunk in chunks
            ]
            
            # Insert into vector database
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            await self.vectordb_client.insert_vectors(
                collection_name=self.collection_name,
                vectors=embeddings,
                payloads=payloads,
                ids=chunk_ids,
            )
            
            logger.info(
                "document_indexed",
                source=source,
                num_chunks=len(chunks),
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error("document_indexing_failed", source=source, error=str(e))
            raise RAGError(f"Failed to index document: {e}")

    async def _rerank_sources(
        self, query: str, sources: List[Source]
    ) -> List[Source]:
        """Rerank sources using cross-encoder or other reranking method.
        
        Args:
            query: Original query
            sources: List of sources to rerank
            
        Returns:
            Reranked list of sources
        """
        # Simple reranking based on query term overlap
        # In production, use a cross-encoder model for better results
        query_terms = set(query.lower().split())
        
        for source in sources:
            content_terms = set(source.content.lower().split())
            overlap = len(query_terms & content_terms)
            total = len(query_terms | content_terms)
            
            # Combine vector score with term overlap
            term_score = overlap / total if total > 0 else 0.0
            source.rerank_score = 0.7 * source.score + 0.3 * term_score
        
        # Sort by rerank score
        sources.sort(key=lambda s: s.rerank_score or s.score, reverse=True)
        
        return sources

    def _build_context(self, sources: List[Source]) -> str:
        """Build context string from sources.
        
        Args:
            sources: List of sources
            
        Returns:
            Formatted context string
        """
        if not sources:
            return ""
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}] (Score: {source.score:.3f})\n{source.content}"
            )
        
        return "\n\n".join(context_parts)

    def _calculate_confidence(self, sources: List[Source]) -> float:
        """Calculate confidence score based on source scores.
        
        Args:
            sources: List of sources
            
        Returns:
            Confidence score between 0 and 1
        """
        if not sources:
            return 0.0
        
        # Use weighted average of top scores
        weights = [1.0 / (i + 1) for i in range(len(sources))]
        total_weight = sum(weights)
        
        weighted_score = sum(
            source.score * weight
            for source, weight in zip(sources, weights)
        )
        
        confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Normalize to 0-1 range
        return min(max(confidence, 0.0), 1.0)

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Source]:
        """Simple semantic search without full RAG pipeline.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filter_conditions: Optional metadata filters
            
        Returns:
            List of sources
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search
            search_results = await self.vectordb_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=self.score_threshold,
                filter_conditions=filter_conditions,
            )
            
            # Convert to Source objects
            sources = [
                Source(
                    document_id=result.payload.get("document_id", "unknown"),
                    chunk_id=result.id,
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata=result.payload,
                )
                for result in search_results
            ]
            
            return sources
            
        except Exception as e:
            logger.error("semantic_search_failed", query=query[:100], error=str(e))
            raise RetrievalError(f"Failed to search: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = await self.vectordb_client.get_collection_stats(
                self.collection_name
            )
            
            if stats:
                return {
                    "collection_name": stats.name,
                    "total_chunks": stats.points_count,
                    "indexed_chunks": stats.indexed_vectors_count,
                    "status": stats.status,
                }
            else:
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": 0,
                    "indexed_chunks": 0,
                    "status": "not_found",
                }
        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }
