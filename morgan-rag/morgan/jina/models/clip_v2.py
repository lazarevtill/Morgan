"""
Jina CLIP V2 Model

Following official Jina AI example:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
embeddings = model.encode(sentences)
similarities = model.similarity(embeddings, embeddings)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEmbeddingResult:
    """Result from multimodal embedding operation."""

    embeddings: np.ndarray
    texts: List[str]
    images: Optional[List[Any]]
    model_name: str
    processing_time: float


@dataclass
class SimilarityResult:
    """Result from similarity computation."""

    similarities: np.ndarray
    query_embeddings: np.ndarray
    target_embeddings: np.ndarray
    processing_time: float


class JinaClipV2:
    """
    Jina CLIP V2 model implementation following official examples.

    This model handles both text and image content, providing unified
    embeddings for multimodal search and similarity computation.
    """

    MODEL_NAME = "jinaai/jina-clip-v2"

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Jina CLIP V2.

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
        Load the Jina CLIP V2 model following official example.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded and self.model is not None:
            return True

        try:
            # Following official Jina AI example:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"trust_remote_code": True}

            if self.cache_dir:
                model_kwargs["cache_folder"] = self.cache_dir

            if self.token:
                model_kwargs["token"] = self.token

            logger.info(f"Loading {self.MODEL_NAME}...")
            self.model = SentenceTransformer(self.MODEL_NAME, **model_kwargs)
            self._is_loaded = True

            logger.info(f"Successfully loaded {self.MODEL_NAME}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            self._is_loaded = False
            return False

    def encode(
        self, inputs: Union[List[str], List[Any]], batch_size: int = 32
    ) -> MultimodalEmbeddingResult:
        """
        Encode text or images into embeddings following official example.

        Args:
            inputs: List of texts or images to encode
            batch_size: Batch size for processing

        Returns:
            MultimodalEmbeddingResult with embeddings and metadata
        """
        import time

        start_time = time.time()

        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load {self.MODEL_NAME}")

        try:
            # Following official Jina AI example:
            # sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
            # embeddings = model.encode(sentences)

            embeddings = self.model.encode(inputs, batch_size=batch_size)

            processing_time = time.time() - start_time

            # Separate text and image inputs
            texts = [inp for inp in inputs if isinstance(inp, str)]
            images = [inp for inp in inputs if not isinstance(inp, str)]

            logger.debug(f"Encoded {len(inputs)} items in {processing_time:.3f}s")

            return MultimodalEmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                images=images if images else None,
                model_name=self.MODEL_NAME,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> SimilarityResult:
        """
        Compute similarity between embeddings following official example.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            SimilarityResult with similarity matrix
        """
        import time

        start_time = time.time()

        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load {self.MODEL_NAME}")

        try:
            # Following official Jina AI example:
            # similarities = model.similarity(embeddings, embeddings)

            similarities = self.model.similarity(embeddings1, embeddings2)

            processing_time = time.time() - start_time

            logger.debug(f"Computed similarities in {processing_time:.3f}s")

            return SimilarityResult(
                similarities=similarities,
                query_embeddings=embeddings1,
                target_embeddings=embeddings2,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise

    def encode_and_search(
        self, query: Union[str, Any], targets: List[Union[str, Any]], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Encode query and targets, then find most similar items.

        Args:
            query: Query text or image
            targets: List of target texts or images
            top_k: Number of top results to return

        Returns:
            Dictionary with search results and similarities
        """
        # Encode query and targets
        query_result = self.encode([query])
        target_result = self.encode(targets)

        # Compute similarities
        similarity_result = self.similarity(
            query_result.embeddings, target_result.embeddings
        )

        # Get top-k results
        similarities = similarity_result.similarities[
            0
        ]  # First row (query vs all targets)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "index": int(idx),
                    "content": targets[idx],
                    "similarity": float(similarities[idx]),
                }
            )

        return {
            "query": query,
            "results": results,
            "total_targets": len(targets),
            "processing_time": query_result.processing_time
            + target_result.processing_time
            + similarity_result.processing_time,
        }

    def extract_text_from_images(self, images: List[Any]) -> List[str]:
        """
        Extract text from images using DeepSeek-OCR via Ollama.

        Args:
            images: List of image objects (PIL Images, bytes, or file paths)

        Returns:
            List of extracted text strings
        """
        import asyncio

        try:
            from morgan.services.ocr_service import get_ocr_service, OCRMode

            service = get_ocr_service()

            async def extract_all():
                results = await service.extract_text_batch(images, OCRMode.FREE)
                return [r.text if r.success else "" for r in results]

            # Run async in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, extract_all())
                        return future.result(timeout=120)
                else:
                    return loop.run_until_complete(extract_all())
            except RuntimeError:
                return asyncio.run(extract_all())

        except ImportError as e:
            logger.warning("OCR service not available: %s", e)
            return ["" for _ in images]
        except Exception as e:
            logger.error("OCR extraction failed: %s", e)
            return ["" for _ in images]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.MODEL_NAME,
            "description": "Jina CLIP v2 for multimodal text and image embeddings",
            "use_case": "Multimodal search and similarity computation",
            "performance": "High quality unified embeddings for text and images",
            "embedding_dimension": 768,  # Typical CLIP embedding dimension
            "supports_text": True,
            "supports_images": True,
            "is_loaded": self._is_loaded,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info(f"Unloaded {self.MODEL_NAME}")
