"""Production-quality reranking service using Jina models."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class RerankingConfig:
    """Reranking service configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    compile_model: bool = False  # PyTorch 2.0+ compile
    max_retries: int = 3
    retry_min_wait: float = 0.5
    retry_max_wait: float = 5.0


@dataclass
class RerankResult:
    """Reranking result."""
    index: int
    text: str
    score: float
    original_score: float


class ModelManager:
    """Thread-safe model manager with lazy loading."""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self._model: Optional[CrossEncoder] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def get_model(self) -> CrossEncoder:
        """Get or initialize model.

        Returns:
            Loaded CrossEncoder model.
        """
        if self._initialized:
            return self._model

        async with self._lock:
            if self._initialized:  # Double-check
                return self._model

            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                self._load_model
            )

            self._initialized = True
            logger.info(
                "Model loaded successfully",
                extra={
                    "model": self.config.model_name,
                    "device": self.config.device,
                }
            )

        return self._model

    def _load_model(self) -> CrossEncoder:
        """Load model synchronously.

        Returns:
            Loaded CrossEncoder model.
        """
        start_time = time.time()

        model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=self.config.device,
        )

        # Move to device
        model.model = model.model.to(self.config.device)

        # Enable inference mode optimizations
        model.model.eval()
        if hasattr(torch, 'inference_mode'):
            # Use inference mode for better performance
            for param in model.model.parameters():
                param.requires_grad = False

        # Compile model if requested (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model.model = torch.compile(model.model)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        duration = time.time() - start_time
        logger.info(
            "Model loading completed",
            extra={
                "model": self.config.model_name,
                "device": self.config.device,
                "duration_s": round(duration, 2),
            }
        )

        return model

    async def unload_model(self) -> None:
        """Unload model and free resources."""
        async with self._lock:
            if self._model:
                # Move to CPU before deletion to free GPU memory
                if self.config.device == "cuda":
                    self._model.model = self._model.model.cpu()
                    torch.cuda.empty_cache()

                del self._model
                self._model = None
                self._initialized = False
                logger.info("Model unloaded")


class RerankingService:
    """Production-quality reranking service with async processing.

    Features:
    - Lazy model loading
    - Efficient batching with DataLoader
    - GPU acceleration when available
    - Async processing with thread pool
    - Proper resource cleanup
    - Retry logic for failures
    - Structured logging
    """

    def __init__(self, config: Optional[RerankingConfig] = None):
        """Initialize reranking service.

        Args:
            config: Service configuration.
        """
        self.config = config or RerankingConfig()
        self._model_manager = ModelManager(self.config)
        self._semaphore = asyncio.Semaphore(4)  # Limit concurrent reranking operations

        logger.info(
            "Initializing reranking service",
            extra={
                "model": self.config.model_name,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
            }
        )

    async def __aenter__(self):
        """Context manager entry."""
        # Preload model
        await self._model_manager.get_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self._model_manager.unload_model()
        return False

    async def _predict_batch(
        self,
        pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """Predict scores for query-document pairs.

        Args:
            pairs: List of (query, document) tuples.

        Returns:
            Array of relevance scores.
        """
        model = await self._model_manager.get_model()

        # Run prediction in thread pool
        loop = asyncio.get_event_loop()

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((RuntimeError, torch.cuda.OutOfMemoryError)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
        ):
            with attempt:
                async with self._semaphore:
                    start_time = time.time()

                    try:
                        scores = await loop.run_in_executor(
                            None,
                            lambda: model.predict(
                                pairs,
                                batch_size=self.config.batch_size,
                                show_progress_bar=False,
                            )
                        )

                        duration = time.time() - start_time
                        logger.debug(
                            "Batch prediction completed",
                            extra={
                                "pairs": len(pairs),
                                "duration_ms": round(duration * 1000, 2),
                            }
                        )

                        return scores

                    except torch.cuda.OutOfMemoryError as e:
                        # Clear cache and retry with smaller batch
                        if self.config.device == "cuda":
                            torch.cuda.empty_cache()
                        logger.warning(
                            "GPU OOM, clearing cache and retrying",
                            extra={"batch_size": len(pairs)}
                        )
                        raise

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: Query text.
            documents: List of document texts.
            top_k: Number of top results to return. If None, returns all.
            return_documents: Whether to include document text in results.

        Returns:
            Sorted list of rerank results.
        """
        if not documents:
            return []

        start_time = time.time()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get scores in batches
        all_scores = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = pairs[i:i + self.config.batch_size]
            batch_scores = await self._predict_batch(batch_pairs)
            all_scores.extend(batch_scores)

        # Create results with original indices
        results = [
            RerankResult(
                index=i,
                text=documents[i] if return_documents else "",
                score=float(score),
                original_score=float(score),  # Could be different if we had original scores
            )
            for i, score in enumerate(all_scores)
        ]

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit to top_k
        if top_k is not None:
            results = results[:top_k]

        duration = time.time() - start_time
        logger.info(
            "Reranking completed",
            extra={
                "query_length": len(query),
                "documents": len(documents),
                "top_k": top_k or len(documents),
                "duration_ms": round(duration * 1000, 2),
            }
        )

        return results

    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankResult]]:
        """Rerank multiple query-document sets concurrently.

        Args:
            queries: List of query texts.
            documents_list: List of document lists (one per query).
            top_k: Number of top results per query.

        Returns:
            List of reranked result lists.
        """
        if len(queries) != len(documents_list):
            raise ValueError("queries and documents_list must have same length")

        start_time = time.time()

        # Process all queries concurrently
        tasks = [
            self.rerank(query, docs, top_k=top_k, return_documents=True)
            for query, docs in zip(queries, documents_list)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch reranking failed for query",
                    extra={
                        "query_index": i,
                        "error": str(result),
                    }
                )
                results[i] = []  # Return empty list for failed queries

        duration = time.time() - start_time
        logger.info(
            "Batch reranking completed",
            extra={
                "queries": len(queries),
                "duration_s": round(duration, 2),
            }
        )

        return results

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[Tuple[str, float]],
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
    ) -> List[RerankResult]:
        """Rerank documents with existing scores.

        Combines reranking scores with original scores using weighted average.

        Args:
            query: Query text.
            documents: List of (text, original_score) tuples.
            top_k: Number of top results to return.
            score_weight: Weight for reranking score (0-1). Higher = more reranking.

        Returns:
            Sorted list of rerank results with combined scores.
        """
        if not documents:
            return []

        if not 0 <= score_weight <= 1:
            raise ValueError("score_weight must be between 0 and 1")

        start_time = time.time()

        # Extract texts and scores
        texts = [doc[0] for doc in documents]
        original_scores = [doc[1] for doc in documents]

        # Get reranking scores
        pairs = [(query, text) for text in texts]
        all_scores = []

        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = pairs[i:i + self.config.batch_size]
            batch_scores = await self._predict_batch(batch_pairs)
            all_scores.extend(batch_scores)

        # Normalize scores to [0, 1]
        rerank_scores = np.array(all_scores)
        if len(rerank_scores) > 1:
            rerank_min, rerank_max = rerank_scores.min(), rerank_scores.max()
            if rerank_max > rerank_min:
                rerank_scores = (rerank_scores - rerank_min) / (rerank_max - rerank_min)

        original_scores_arr = np.array(original_scores)
        if len(original_scores_arr) > 1:
            orig_min, orig_max = original_scores_arr.min(), original_scores_arr.max()
            if orig_max > orig_min:
                original_scores_arr = (original_scores_arr - orig_min) / (orig_max - orig_min)

        # Combine scores
        combined_scores = (
            score_weight * rerank_scores +
            (1 - score_weight) * original_scores_arr
        )

        # Create results
        results = [
            RerankResult(
                index=i,
                text=texts[i],
                score=float(combined_scores[i]),
                original_score=original_scores[i],
            )
            for i in range(len(texts))
        ]

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit to top_k
        if top_k is not None:
            results = results[:top_k]

        duration = time.time() - start_time
        logger.info(
            "Reranking with scores completed",
            extra={
                "query_length": len(query),
                "documents": len(documents),
                "top_k": top_k or len(documents),
                "score_weight": score_weight,
                "duration_ms": round(duration * 1000, 2),
            }
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "model": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "initialized": self._model_manager._initialized,
            "cuda_available": torch.cuda.is_available(),
        }
