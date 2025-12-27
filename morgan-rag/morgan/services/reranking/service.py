"""
Unified Reranking Service for Morgan AI Assistant.

Consolidates infrastructure/local_reranking.py and jina/reranking/service.py
into a single implementation with multiple fallback strategies.
"""

import asyncio
import math
import os
import re
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.reranking.models import RerankResult, RerankStats
from morgan.utils.logger import get_logger
from morgan.utils.model_cache import setup_model_cache  # Use canonical implementation

logger = get_logger(__name__)

# Optional imports
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RerankingService:
    """
    Unified reranking service with multiple fallback strategies.

    This service consolidates the functionality from local_reranking.py
    and jina/reranking/service.py.

    Fallback hierarchy (in order of preference):
    1. Remote reranking endpoint (FastAPI service)
    2. Local CrossEncoder via sentence-transformers
    3. Embedding-based similarity (cosine similarity)
    4. BM25-style lexical matching (last resort)

    Self-hosted models (via sentence-transformers CrossEncoder):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English (recommended)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, English
    - BAAI/bge-reranker-base: Multilingual

    Example:
        >>> service = get_reranking_service()

        >>> # Rerank documents
        >>> results = await service.rerank(
        ...     query="What is Python?",
        ...     documents=["Python is a language", "Java is a language"]
        ... )

        >>> # Get top results
        >>> for result in results[:5]:
        ...     print(f"{result.score:.3f}: {result.text[:50]}...")
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_model: str = "all-MiniLM-L6-v2",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_device: str = "cpu",
        model_cache_dir: Optional[str] = None,
    ):
        """
        Initialize reranking service.

        Args:
            endpoint: Remote reranking endpoint URL
            model: CrossEncoder model name
            embedding_model: Embedding model for fallback
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_device: Device for local models ("cpu" or "cuda")
            model_cache_dir: Directory for model weights cache
        """
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._availability_lock = threading.Lock()

        # Setup model cache
        self.model_cache_path = setup_model_cache(model_cache_dir)

        # Configuration
        self.endpoint = endpoint or getattr(self.settings, "reranking_endpoint", None)
        self.model = model or getattr(
            self.settings, "reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.embedding_model_name = embedding_model
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_device = local_device

        # Lazy-loaded models
        self._cross_encoder = None
        self._cross_encoder_available: Optional[bool] = None
        self._embedding_model = None
        self._embedding_available: Optional[bool] = None

        # Statistics
        self.stats = RerankStats()

        logger.info(
            "RerankingService initialized: endpoint=%s, model=%s",
            self.endpoint,
            self.model,
        )

    # =========================================================================
    # Main Reranking Method
    # =========================================================================

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Uses multiple fallback strategies in order of preference:
        1. Remote reranking endpoint
        2. Local CrossEncoder model
        3. Embedding-based similarity
        4. BM25-style lexical matching

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return top K results (None = return all)

        Returns:
            List of RerankResult sorted by score (descending)
        """
        if not documents:
            return []

        self.stats.total_requests += 1
        self.stats.total_pairs += len(documents)

        start_time = time.time()
        results = None
        method_used = "none"

        try:
            # 1. Try remote endpoint first
            if self.endpoint and REQUESTS_AVAILABLE:
                try:
                    results = await self._rerank_remote(query, documents, top_k)
                    method_used = "remote"
                    self.stats.remote_calls += 1
                except Exception as e:
                    logger.warning("Remote reranking failed: %s", e)

            # 2. Try local CrossEncoder
            if results is None and self._check_cross_encoder_available():
                try:
                    results = self._rerank_cross_encoder(query, documents, top_k)
                    method_used = "cross_encoder"
                    self.stats.local_calls += 1
                except Exception as e:
                    logger.warning("CrossEncoder reranking failed: %s", e)

            # 3. Try embedding-based reranking
            if results is None and self._check_embedding_available():
                try:
                    results = await self._rerank_embedding(query, documents, top_k)
                    method_used = "embedding"
                    self.stats.embedding_calls += 1
                except Exception as e:
                    logger.warning("Embedding reranking failed: %s", e)

            # 4. Fallback to BM25-style lexical matching
            if results is None:
                results = self._rerank_bm25(query, documents, top_k)
                method_used = "bm25"
                self.stats.bm25_calls += 1

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_time += elapsed

            logger.debug(
                "Reranked %d documents in %.3fs using %s (%.1f docs/sec)",
                len(documents),
                elapsed,
                method_used,
                len(documents) / elapsed if elapsed > 0 else 0,
            )

            return results

        except Exception as e:
            self.stats.errors += 1
            logger.error("All reranking methods failed: %s", e)

            # Return documents in original order as last resort
            return [
                RerankResult(text=doc, score=1.0 - (i * 0.01), original_index=i)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]

    def rerank_sync(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Synchronous wrapper for rerank.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return top K results

        Returns:
            List of RerankResult sorted by score (descending)
        """
        try:
            # Try to use existing event loop if available
            loop = asyncio.get_running_loop()
            # Already in async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.rerank(query, documents, top_k))
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.rerank(query, documents, top_k))

    # =========================================================================
    # Reranking Strategies
    # =========================================================================

    async def _rerank_remote(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int],
    ) -> List[RerankResult]:
        """Rerank using remote endpoint."""
        payload = {"query": query, "documents": documents}
        if top_k is not None:
            payload["top_k"] = top_k

        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("results", []):
            results.append(
                RerankResult(
                    text=item["text"],
                    score=item["score"],
                    original_index=item["index"],
                )
            )

        return results

    def _check_cross_encoder_available(self) -> bool:
        """Check if CrossEncoder is available."""
        if self._cross_encoder_available is not None:
            return self._cross_encoder_available

        if not CROSS_ENCODER_AVAILABLE:
            logger.debug("CrossEncoder not installed")
            with self._availability_lock:
                self._cross_encoder_available = False
            return False

        try:
            logger.info("Loading CrossEncoder model (%s)...", self.model)
            self._cross_encoder = CrossEncoder(self.model, device=self.local_device)
            logger.info("CrossEncoder model loaded successfully")
            with self._availability_lock:
                self._cross_encoder_available = True
            return True
        except Exception as e:
            logger.error("Failed to load CrossEncoder: %s", e)
            with self._availability_lock:
                self._cross_encoder_available = False
            return False

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int],
    ) -> List[RerankResult]:
        """Rerank using local CrossEncoder."""
        if self._cross_encoder is None:
            raise RuntimeError("CrossEncoder not initialized")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self._cross_encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Normalize score to 0-1 range using sigmoid
            normalized_score = 1 / (1 + math.exp(-float(score)))
            results.append(
                RerankResult(text=doc, score=normalized_score, original_index=i)
            )

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def _check_embedding_available(self) -> bool:
        """Check if embedding model is available."""
        if self._embedding_available is not None:
            return self._embedding_available

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.debug("SentenceTransformer not installed")
            with self._availability_lock:
                self._embedding_available = False
            return False

        try:
            logger.info("Loading embedding model (%s)...", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=self.local_device
            )
            logger.info("Embedding model loaded successfully")
            with self._availability_lock:
                self._embedding_available = True
            return True
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            with self._availability_lock:
                self._embedding_available = False
            return False

    async def _rerank_embedding(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int],
    ) -> List[RerankResult]:
        """Rerank using embedding similarity (cosine similarity)."""
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        import numpy as np

        # Generate embeddings in thread pool
        loop = asyncio.get_event_loop()

        def encode_sync():
            query_embedding = self._embedding_model.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            doc_embeddings = self._embedding_model.encode(
                documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            return query_embedding, doc_embeddings

        query_embedding, doc_embeddings = await loop.run_in_executor(None, encode_sync)

        # Calculate cosine similarity (embeddings are already normalized)
        similarities = np.dot(doc_embeddings, query_embedding)

        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, similarities)):
            results.append(RerankResult(text=doc, score=float(score), original_index=i))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def _rerank_bm25(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int],
    ) -> List[RerankResult]:
        """
        Rerank using BM25-style lexical matching.

        This is the last resort fallback when no ML models are available.
        """

        def tokenize(text: str) -> List[str]:
            return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]

        query_terms = set(tokenize(query))

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Calculate average document length
        doc_tokens = [tokenize(doc) for doc in documents]
        avg_dl = (
            sum(len(tokens) for tokens in doc_tokens) / len(documents)
            if documents
            else 1
        )

        # Calculate document frequencies
        df = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1

        n_docs = len(documents)
        results = []

        for i, (doc, tokens) in enumerate(zip(documents, doc_tokens)):
            score = 0.0
            tf = Counter(tokens)
            dl = len(tokens)

            for term in query_terms:
                if term in tf:
                    # IDF component
                    idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
                    # TF component with length normalization
                    tf_score = (
                        tf[term]
                        * (k1 + 1)
                        / (tf[term] + k1 * (1 - b + b * dl / avg_dl))
                    )
                    score += idf * tf_score

            # Normalize to 0-1 range
            normalized_score = (
                min(1.0, score / (len(query_terms) * 3)) if query_terms else 0.0
            )
            results.append(
                RerankResult(text=doc, score=normalized_score, original_index=i)
            )

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    # =========================================================================
    # Status & Statistics
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if reranking service is available.

        Returns:
            True if any reranking method is available
        """
        # Check remote
        if self.endpoint and REQUESTS_AVAILABLE:
            try:
                health_url = self.endpoint.rstrip("/rerank") + "/health"
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return True
            except Exception as e:
                logger.debug("Remote reranking endpoint not available: %s", e)

        # Check local CrossEncoder
        if self._check_cross_encoder_available():
            return True

        # Check embedding model
        if self._check_embedding_available():
            return True

        # BM25 is always available
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.stats.to_dict()
        stats.update(
            {
                "endpoint": self.endpoint,
                "model": self.model,
                "embedding_model": self.embedding_model_name,
                "cross_encoder_available": self._cross_encoder_available,
                "embedding_available": self._embedding_available,
            }
        )
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = RerankStats()
        logger.info("Reranking stats reset")


# =============================================================================
# Singleton Management
# =============================================================================

from morgan.utils.singleton import SingletonFactory

# Create singleton factory (no cleanup needed for reranking service)
_reranking_service_factory = SingletonFactory(RerankingService)


def get_reranking_service(
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    force_new: bool = False,
    **kwargs,
) -> RerankingService:
    """
    Get singleton reranking service instance.

    Args:
        endpoint: Remote reranking endpoint URL
        model: CrossEncoder model name
        force_new: Force create new instance
        **kwargs: Additional service configuration

    Returns:
        Shared RerankingService instance
    """
    return _reranking_service_factory.get_instance(
        force_new=force_new,
        endpoint=endpoint,
        model=model,
        **kwargs,
    )


def reset_reranking_service():
    """Reset singleton instance (useful for testing)."""
    _reranking_service_factory.reset()
