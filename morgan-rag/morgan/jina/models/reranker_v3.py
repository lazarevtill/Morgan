"""
Jina Reranker V3 Model

Following official Jina AI example:
from sentence_transformers import CrossEncoder
model = CrossEncoder("jinaai/jina-reranker-v3", trust_remote_code=True)
query = "Which planet is known as the Red Planet?"
passages = ["Venus is often called Earth's twin...", "Mars, known for its reddish appearance..."]
scores = model.predict([(query, passage) for passage in passages])
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Result from reranking operation."""

    scores: List[float]
    query: str
    passages: List[str]
    model_name: str
    processing_time: float


class JinaRerankerV3:
    """
    Jina Reranker V3 model implementation following official examples.

    This model is optimized for English content and provides state-of-the-art
    reranking performance for search results.
    """

    MODEL_NAME = "jinaai/jina-reranker-v3"

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Jina Reranker V3.

        Args:
            cache_dir: Directory to cache the model
            token: Hugging Face token for authentication
        """
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.model = None
        self._is_loaded = False

        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> bool:
        """
        Load the Jina Reranker V3 model following official example.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded and self.model is not None:
            return True

        try:
            # Following official Jina AI example:
            # from sentence_transformers import CrossEncoder
            # model = CrossEncoder("jinaai/jina-reranker-v3", trust_remote_code=True)
            from sentence_transformers import CrossEncoder

            model_kwargs = {"trust_remote_code": True, "max_length": 512}

            if self.cache_dir:
                model_kwargs["cache_folder"] = self.cache_dir

            if self.token:
                model_kwargs["use_auth_token"] = self.token

            logger.info(f"Loading {self.MODEL_NAME}...")
            self.model = CrossEncoder(self.MODEL_NAME, **model_kwargs)

            # Fix padding token issue
            if hasattr(self.model, "tokenizer"):
                if self.model.tokenizer.pad_token is None:
                    if self.model.tokenizer.eos_token is not None:
                        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                    else:
                        # Add a padding token if none exists
                        self.model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    logger.info(
                        f"Set padding token to: {self.model.tokenizer.pad_token}"
                    )

                # Ensure the model knows about the padding token
                if hasattr(self.model, "model"):
                    self.model.model.resize_token_embeddings(len(self.model.tokenizer))

            self._is_loaded = True

            logger.info(f"Successfully loaded {self.MODEL_NAME}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            self._is_loaded = False
            return False

    def predict(self, query: str, passages: List[str]) -> RerankingResult:
        """
        Rerank passages for a given query following official example.

        Args:
            query: Search query
            passages: List of passages to rerank

        Returns:
            RerankingResult with scores and metadata
        """
        import time

        start_time = time.time()

        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load {self.MODEL_NAME}")

        try:
            # Following official Jina AI example but handle batch processing issues
            query_passage_pairs = [(query, passage) for passage in passages]

            # Process one by one if batch processing fails
            try:
                scores = self.model.predict(query_passage_pairs)
            except Exception as batch_error:
                logger.warning(
                    f"Batch processing failed: {batch_error}, processing individually"
                )
                scores = []
                for pair in query_passage_pairs:
                    try:
                        score = self.model.predict([pair])
                        scores.extend(score if isinstance(score, list) else [score])
                    except Exception as individual_error:
                        logger.error(
                            f"Individual prediction failed: {individual_error}"
                        )
                        scores.append(0.5)  # Default score

            # Convert to list if numpy array
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]

            processing_time = time.time() - start_time

            logger.debug(f"Reranked {len(passages)} passages in {processing_time:.3f}s")

            return RerankingResult(
                scores=scores,
                query=query,
                passages=passages,
                model_name=self.MODEL_NAME,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise

    def batch_predict(
        self, queries: List[str], passage_lists: List[List[str]]
    ) -> List[RerankingResult]:
        """
        Batch reranking for multiple queries.

        Args:
            queries: List of queries
            passage_lists: List of passage lists (one per query)

        Returns:
            List of RerankingResult objects
        """
        if len(queries) != len(passage_lists):
            raise ValueError("Number of queries must match number of passage lists")

        results = []
        for query, passages in zip(queries, passage_lists):
            result = self.predict(query, passages)
            results.append(result)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.MODEL_NAME,
            "language": "en",
            "max_query_length": 512,
            "max_document_length": 8192,
            "description": "Latest English reranker model with improved accuracy",
            "use_case": "Primary reranker for English content",
            "performance": "High accuracy, optimized for English queries",
            "is_loaded": self._is_loaded,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info(f"Unloaded {self.MODEL_NAME}")
