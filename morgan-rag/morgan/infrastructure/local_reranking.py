"""
Local Reranking Service for Distributed Morgan Setup

Provides reranking via:
- Remote reranking endpoint (Host 6 - RTX 2060)
- Local CrossEncoder fallback
- Batch processing for efficiency
- Performance tracking

Designed for 6-host distributed architecture where reranking runs on
dedicated host (Host 6 with RTX 2060).

Model weights are cached locally to avoid re-downloading on each startup.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def setup_reranker_cache(cache_dir: Optional[str] = None):
    """
    Setup model cache directories and environment variables for rerankers.

    This ensures models are downloaded once and reused on subsequent starts.
    Also configures HF_TOKEN for downloading gated models if available.

    Args:
        cache_dir: Base directory for model cache. Defaults to ~/.morgan/models

    Environment variables loaded from .env:
        - HF_TOKEN: Hugging Face API token (for gated models)
        - HUGGING_FACE_HUB_TOKEN: Alternative HF token variable
        - MORGAN_MODEL_CACHE: Override default cache directory
    """
    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.debug("Loaded environment from .env file")
    except ImportError:
        pass  # dotenv not installed, use existing env vars

    if cache_dir is None:
        cache_dir = os.environ.get("MORGAN_MODEL_CACHE", "~/.morgan/models")

    cache_path = Path(cache_dir).expanduser()

    # Create subdirectories
    sentence_transformers_path = cache_path / "sentence-transformers"
    hf_path = cache_path / "huggingface"

    for path in [cache_path, sentence_transformers_path, hf_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Set environment variables for model caching
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_path)
    os.environ["HF_HOME"] = str(hf_path)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_path)

    # Configure HF_TOKEN for gated model downloads
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        logger.info("HF_TOKEN configured for authenticated model downloads")
    else:
        logger.debug("No HF_TOKEN found - some gated models may not be accessible")

    logger.info("Reranker model cache configured at %s", cache_path)
    return cache_path


@dataclass
class RerankResult:
    """Result from reranking operation."""

    text: str
    score: float
    original_index: int


@dataclass
class RerankStats:
    """Statistics for reranking operations."""

    total_requests: int = 0
    total_pairs: int = 0
    total_time: float = 0.0
    errors: int = 0

    @property
    def average_time(self) -> float:
        """Calculate average reranking time."""
        if self.total_requests > 0:
            return self.total_time / self.total_requests
        return 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (pairs/sec)."""
        if self.total_time > 0:
            return self.total_pairs / self.total_time
        return 0.0


class LocalRerankingService:
    """
    Local reranking service for distributed Morgan setup.

    100% Self-Hosted - No API Keys Required.

    Supports (in order of preference):
    1. Remote FastAPI reranking endpoint (primary)
    2. Local CrossEncoder via sentence-transformers (high quality fallback)
    3. Local embedding-based reranking (fast fallback)
    4. BM25-style lexical matching (last resort)

    Self-hosted models (via sentence-transformers CrossEncoder):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English (recommended)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, English
    - BAAI/bge-reranker-base: Multilingual

    Example:
        >>> service = LocalRerankingService(
        ...     endpoint="http://192.168.1.23:8080/rerank",
        ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ... )
        >>>
        >>> # Rerank search results
        >>> results = await service.rerank(
        ...     query="What is Python?",
        ...     documents=["Python is a language", "Java is a language", ...]
        ... )
        >>>
        >>> # Get top 5 results
        >>> top_5 = results[:5]
        >>>
        >>> # Get stats
        >>> stats = service.get_stats()
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_device: str = "cpu",
        model_cache_dir: Optional[str] = None,
        preload_model: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize local reranking service.

        Args:
            endpoint: Reranking endpoint URL
            model: Model name for local CrossEncoder
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_device: Device for local model ("cpu" or "cuda")
            model_cache_dir: Directory for model weights cache
            preload_model: If True, load model immediately
            embedding_model: Model for embedding-based fallback
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests package not available, remote reranking disabled")

        # Setup model cache directory before any model loading
        self.model_cache_path = setup_reranker_cache(model_cache_dir)

        self.endpoint = endpoint
        self.model = model
        self.embedding_model_name = embedding_model
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_device = local_device

        # Initialize local models (lazy by default)
        self._local_model = None
        self._local_available = None
        self._embedding_model = None
        self._embedding_available = None

        # Statistics
        self.stats = RerankStats()

        logger.info(
            "LocalRerankingService initialized: endpoint=%s, model=%s, " "cache_dir=%s",
            endpoint,
            model,
            self.model_cache_path,
        )

        # Preload model if requested (downloads weights on first run)
        if preload_model and CROSS_ENCODER_AVAILABLE:
            logger.info("Preloading reranking model %s...", model)
            self._check_local_available()

    def is_available(self) -> bool:
        """
        Check if reranking service is available.

        Returns:
            True if remote or local reranking available
        """
        # Check remote first
        if self.endpoint and REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    f"{self.endpoint.rstrip('/rerank')}/health", timeout=5.0
                )
                if response.status_code == 200:
                    logger.info("Remote reranking service available")
                    return True
            except Exception as e:
                logger.warning(f"Remote reranking service not available: {e}")

        # Check local fallback
        if self._check_local_available():
            logger.info("Local reranking service available")
            return True

        logger.error("No reranking service available")
        return False

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Uses multiple fallback strategies:
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
        self.stats.total_requests += 1
        self.stats.total_pairs += len(documents)

        if not documents:
            return []

        start_time = time.time()
        results = None
        method_used = "none"

        # Try each method in order of preference
        try:
            # 1. Try remote endpoint first
            if self.endpoint and REQUESTS_AVAILABLE:
                try:
                    results = await self._rerank_remote(query, documents, top_k)
                    method_used = "remote"
                except Exception as e:
                    logger.warning(f"Remote reranking failed, trying local: {e}")

            # 2. Try local CrossEncoder
            if results is None and self._check_local_available():
                try:
                    results = self._rerank_local(query, documents, top_k)
                    method_used = "cross_encoder"
                except Exception as e:
                    logger.warning(
                        f"CrossEncoder reranking failed, trying embedding: {e}"
                    )

            # 3. Try embedding-based reranking
            if results is None:
                try:
                    results = await self._rerank_embedding(query, documents, top_k)
                    method_used = "embedding"
                except Exception as e:
                    logger.warning(f"Embedding reranking failed, trying BM25: {e}")

            # 4. Fallback to BM25-style lexical matching
            if results is None:
                results = self._rerank_bm25(query, documents, top_k)
                method_used = "bm25"

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_time += elapsed

            logger.info(
                f"Reranked {len(documents)} documents in {elapsed:.3f}s "
                f"using {method_used} ({len(documents)/elapsed:.1f} docs/sec)"
            )

            return results

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"All reranking methods failed: {e}")
            # Return documents in original order as last resort
            return [
                RerankResult(text=doc, score=1.0 - (i * 0.01), original_index=i)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]

    async def _rerank_remote(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using remote endpoint."""
        payload = {"query": query, "documents": documents}

        if top_k is not None:
            payload["top_k"] = top_k

        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Parse results
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

        except Exception as e:
            logger.error(f"Remote reranking failed: {e}")
            raise

    def _check_local_available(self) -> bool:
        """Check if local CrossEncoder is available."""
        if self._local_available is not None:
            return self._local_available

        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers not installed")
            self._local_available = False
            return False

        try:
            logger.info(f"Loading local reranking model ({self.model})...")
            self._local_model = CrossEncoder(
                self.model,
                device=self.local_device,
            )
            logger.info("Local reranking model loaded successfully")
            self._local_available = True
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._local_available = False
            return False

    def _rerank_local(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using local CrossEncoder."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self._local_model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(text=doc, score=float(score), original_index=i))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def _check_embedding_available(self) -> bool:
        """Check if embedding model is available."""
        if self._embedding_available is not None:
            return self._embedding_available

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model ({self.embedding_model_name})...")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.local_device,
            )
            logger.info("Embedding model loaded successfully")
            self._embedding_available = True
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed for embedding fallback")
            self._embedding_available = False
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._embedding_available = False
            return False

    async def _rerank_embedding(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using embedding similarity (cosine similarity).

        This is a fallback when CrossEncoder is not available.
        Uses sentence-transformers for embedding generation.
        """
        if not self._check_embedding_available():
            raise RuntimeError("Embedding model not available")

        import numpy as np

        # Generate embeddings
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

        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def _rerank_bm25(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using BM25-style lexical matching.

        This is the last resort fallback when no ML models are available.
        Uses simple term frequency-based scoring.
        """
        import re
        from collections import Counter
        import math

        # Tokenize query and documents
        def tokenize(text: str) -> List[str]:
            # Simple tokenization: lowercase and split on non-alphanumeric
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

        # Calculate BM25 scores
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

            # Normalize score to 0-1 range (approximate)
            normalized_score = (
                min(1.0, score / (len(query_terms) * 3)) if query_terms else 0.0
            )
            results.append(
                RerankResult(text=doc, score=normalized_score, original_index=i)
            )

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self.stats.total_requests,
            "total_pairs": self.stats.total_pairs,
            "total_time": f"{self.stats.total_time:.2f}s",
            "average_time": f"{self.stats.average_time:.3f}s",
            "throughput": f"{self.stats.throughput:.1f} pairs/sec",
            "errors": self.stats.errors,
            "endpoint": self.endpoint,
            "model": self.model,
        }


# Global instance for singleton pattern
_service: Optional[LocalRerankingService] = None


def get_local_reranking_service(
    endpoint: Optional[str] = None, **kwargs
) -> LocalRerankingService:
    """
    Get global local reranking service instance (singleton).

    Args:
        endpoint: Endpoint URL (optional)
        **kwargs: Additional arguments for LocalRerankingService

    Returns:
        LocalRerankingService instance
    """
    global _service

    if _service is None:
        _service = LocalRerankingService(endpoint=endpoint, **kwargs)

    return _service
