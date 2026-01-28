"""
Local Jina Embedding Service

Self-hosted embedding generation using local Hugging Face models.
Single responsibility: embedding generation only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Local model dependencies
try:
    import torch
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[misc]
    logging.warning("sentence-transformers not available - local embeddings disabled")

logger = logging.getLogger(__name__)


class JinaEmbeddingService:
    """Local embedding generation service using Hugging Face models."""

    def __init__(
        self,
        max_workers: int = 4,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the local Jina embedding service.

        Args:
            max_workers: Maximum number of concurrent workers for batch processing
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.max_workers = max_workers
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/morgan/models")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._models = {}  # Cache for loaded models

        # Determine device
        if device is None:
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            elif (
                TRANSFORMERS_AVAILABLE
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(
            f"Initialized Local Jina Embedding Service - Device: {self.device}, Workers: {max_workers}"
        )

    def generate_embeddings(
        self, texts: List[str], model_name: str, batch_size: int = 32
    ) -> List[List[float]]:
        """
        Single responsibility: embedding generation only.

        Args:
            texts: List of texts to embed
            model_name: Jina model name to use
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return []

        logger.info(
            f"Generating embeddings for {len(texts)} texts using model '{model_name}'"
        )
        start_time = time.time()

        try:
            # Process in batches for better performance
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self._generate_batch_embeddings(batch, model_name)
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Generated {len(all_embeddings)} embeddings in {elapsed_time:.2f}s"
            )

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def generate_single_embedding(self, text: str, model_name: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            model_name: Jina model name to use

        Returns:
            Embedding vector
        """
        if not text.strip():
            logger.warning("Empty text provided for single embedding generation")
            return []

        embeddings = self.generate_embeddings([text], model_name, batch_size=1)
        return embeddings[0] if embeddings else []

    async def generate_embeddings_async(
        self, texts: List[str], model_name: str, batch_size: int = 32
    ) -> List[List[float]]:
        """
        Asynchronous embedding generation.

        Args:
            texts: List of texts to embed
            model_name: Jina model name to use
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.generate_embeddings, texts, model_name, batch_size
        )

    def _get_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """
        Load and cache a local model.

        Args:
            model_name: Jina model name

        Returns:
            Loaded SentenceTransformer model or None if unavailable
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available for local embeddings")
            return None

        if model_name in self._models:
            return self._models[model_name]

        # Map Jina model names to Hugging Face model IDs
        model_mapping = {
            "jina-embeddings-v4": "jinaai/jina-embeddings-v4",
            "jina-code-embeddings-1.5b": "jinaai/jina-code-embeddings-1.5b",
            "jina-clip-v2": "jinaai/jina-clip-v2",
            "jina-reranker-v3": "jinaai/jina-reranker-v3",
            "jina-reranker-v2-base-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
        }

        hf_model_id = model_mapping.get(model_name, model_name)

        try:
            logger.info(f"Loading local model: {hf_model_id}")
            model = SentenceTransformer(
                hf_model_id,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True,
            )

            self._models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    def _generate_batch_embeddings(
        self, texts: List[str], model_name: str
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using local models.

        Args:
            texts: Batch of texts to embed
            model_name: Jina model name to use

        Returns:
            List of embedding vectors
        """
        model = self._get_model(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not available, using fallback")
            return self._generate_fallback_embeddings(texts, model_name)

        try:
            # Generate embeddings using local model
            embeddings = model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Convert to list of lists
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            elif hasattr(embeddings, "numpy"):
                embeddings = embeddings.numpy().tolist()

            logger.debug(
                f"Generated {len(embeddings)} embeddings using local model {model_name}"
            )
            return embeddings

        except Exception as e:
            logger.error(
                f"Local embedding generation failed for {model_name}: {str(e)}"
            )
            return self._generate_fallback_embeddings(texts, model_name)

    def _generate_fallback_embeddings(
        self, texts: List[str], model_name: str
    ) -> List[List[float]]:
        """
        Generate fallback embeddings when local models are unavailable.

        Args:
            texts: Batch of texts to embed
            model_name: Model name for dimension reference

        Returns:
            List of simple hash-based embedding vectors
        """
        # Model dimensions for fallback
        model_dimensions = {
            "jina-embeddings-v4": 1024,
            "jina-code-embeddings-1.5b": 768,
            "jina-clip-v2": 512,
            "jina-reranker-v3": 1024,
            "jina-reranker-v2-base-multilingual": 768,
        }

        dim = model_dimensions.get(model_name, 768)

        embeddings = []
        for text in texts:
            # Simple hash-based embedding for fallback
            text_hash = hash(text.strip().lower())

            # Generate deterministic "embedding" from hash
            import random

            random.seed(text_hash)
            embedding = [random.gauss(0, 1) for _ in range(dim)]

            # Normalize to unit vector
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        logger.debug(
            f"Generated {len(embeddings)} fallback embeddings with dimension {dim}"
        )
        return embeddings

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Jina model name

        Returns:
            Dictionary with model information
        """
        model_info = {
            "jina-embeddings-v4": {
                "name": "jina-embeddings-v4",
                "hf_id": "jinaai/jina-embeddings-v4",
                "type": "text",
                "dimensions": 1024,
                "max_tokens": 8192,
                "description": "Latest general-purpose text embedding model",
                "local_available": TRANSFORMERS_AVAILABLE,
            },
            "jina-code-embeddings-1.5b": {
                "name": "jina-code-embeddings-1.5b",
                "hf_id": "jinaai/jina-code-embeddings-1.5b",
                "type": "code",
                "dimensions": 768,
                "max_tokens": 4096,
                "description": "Specialized model for code understanding",
                "local_available": TRANSFORMERS_AVAILABLE,
            },
            "jina-clip-v2": {
                "name": "jina-clip-v2",
                "hf_id": "jinaai/jina-clip-v2",
                "type": "multimodal",
                "dimensions": 512,
                "max_tokens": 2048,
                "description": "Multimodal model for text and image embeddings",
                "local_available": TRANSFORMERS_AVAILABLE,
            },
            "jina-reranker-v3": {
                "name": "jina-reranker-v3",
                "hf_id": "jinaai/jina-reranker-v3",
                "type": "reranker",
                "dimensions": 1024,
                "max_tokens": 8192,
                "description": "Latest reranking model for search relevance",
                "local_available": TRANSFORMERS_AVAILABLE,
            },
            "jina-reranker-v2-base-multilingual": {
                "name": "jina-reranker-v2-base-multilingual",
                "hf_id": "jinaai/jina-reranker-v2-base-multilingual",
                "type": "reranker",
                "dimensions": 768,
                "max_tokens": 4096,
                "description": "Multilingual reranking model",
                "local_available": TRANSFORMERS_AVAILABLE,
            },
        }

        return model_info.get(
            model_name,
            {
                "name": model_name,
                "hf_id": model_name,
                "type": "unknown",
                "dimensions": 768,
                "max_tokens": 2048,
                "description": "Unknown model",
                "local_available": False,
            },
        )

    def list_available_models(self) -> List[str]:
        """
        List all available local models.

        Returns:
            List of model names that can be used locally
        """
        if not TRANSFORMERS_AVAILABLE:
            return []

        return [
            "jina-embeddings-v4",
            "jina-code-embeddings-1.5b",
            "jina-clip-v2",
            "jina-reranker-v3",
            "jina-reranker-v2-base-multilingual",
        ]

    def preload_model(self, model_name: str) -> bool:
        """
        Preload a model to cache for faster subsequent use.

        Args:
            model_name: Model name to preload

        Returns:
            True if model was successfully loaded
        """
        model = self._get_model(model_name)
        return model is not None

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for local embedding service.

        Returns:
            Dictionary with system information
        """
        return {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "max_workers": self.max_workers,
            "loaded_models": list(self._models.keys()),
            "cuda_available": (
                TRANSFORMERS_AVAILABLE and torch.cuda.is_available()
                if TRANSFORMERS_AVAILABLE
                else False
            ),
            "mps_available": (
                (
                    TRANSFORMERS_AVAILABLE
                    and hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                )
                if TRANSFORMERS_AVAILABLE
                else False
            ),
        }

    def validate_inputs(self, texts: List[str], model_name: str) -> bool:
        """
        Validate inputs for embedding generation.

        Args:
            texts: List of texts to validate
            model_name: Model name to validate

        Returns:
            True if inputs are valid
        """
        if not texts:
            logger.error("Empty text list provided")
            return False

        if not model_name:
            logger.error("Empty model name provided")
            return False

        # Check for empty texts
        empty_texts = [i for i, text in enumerate(texts) if not text.strip()]
        if empty_texts:
            logger.warning(
                f"Found {len(empty_texts)} empty texts at indices: {empty_texts}"
            )

        return True

    def close(self):
        """Clean up resources."""
        # Clear model cache to free GPU memory
        for model_name, model in self._models.items():
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
                del model
                logger.debug(f"Cleared model from cache: {model_name}")
            except Exception as e:
                logger.warning(f"Error clearing model {model_name}: {str(e)}")

        self._models.clear()

        if self._executor:
            self._executor.shutdown(wait=True)

        # Clear CUDA cache if available
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Closed Local Jina Embedding Service")
