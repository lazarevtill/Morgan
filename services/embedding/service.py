"""
Production Embedding Service for Docker deployment.

Standalone embedding service optimized for containerized deployment.
Uses sentence-transformers for local embedding generation without
requiring external API calls.

Model weights are cached to avoid re-downloading on each startup.

Example:
    >>> from services.embedding import ProductionEmbeddingService
    >>>
    >>> service = ProductionEmbeddingService()
    >>> embeddings = service.encode(["Hello world", "Test text"])
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy load to speed up imports
_model = None
_model_name = None
_model_lock = threading.Lock()


def setup_model_cache() -> Path:
    """
    Setup model cache directories to avoid re-downloading on each startup.

    Models are downloaded once to the cache directory and reused later.

    Environment variables:
        - MODEL_CACHE_DIR: Cache directory (default: /app/models)
        - HF_TOKEN: Hugging Face API token (for gated models)

    Returns:
        Path to the cache directory
    """
    # Try to load .env file if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv not installed

    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    cache_path = Path(cache_dir)

    # Create subdirectories
    st_cache = cache_path / "sentence-transformers"
    hf_cache = cache_path / "huggingface"

    for path in [cache_path, st_cache, hf_cache]:
        path.mkdir(parents=True, exist_ok=True)

    # Set environment variables for model caching
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)

    # Configure HF_TOKEN for gated model downloads
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    return cache_path


# Setup cache on module import
_cache_path = setup_model_cache()


class ProductionEmbeddingService:
    """
    Production embedding service using sentence-transformers.

    Optimized for containerized deployment with:
    - Model caching to avoid re-downloads
    - Batch processing support
    - GPU acceleration when available
    - Thread-safe lazy loading

    Supported models (self-hosted, no API keys):
        - nomic-ai/nomic-embed-text-v1.5 (default, 768 dims)
        - BAAI/bge-base-en-v1.5 (768 dims)
        - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
        - intfloat/e5-base-v2 (768 dims)

    Example:
        >>> service = ProductionEmbeddingService()
        >>> embeddings = service.encode(["Hello", "World"])
        >>> print(f"Dimension: {service.get_dimension()}")
    """

    # Default models by use case
    DEFAULT_MODELS = {
        "general": "nomic-ai/nomic-embed-text-v1.5",
        "fast": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual": "intfloat/multilingual-e5-base",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress: bool = False,
    ):
        """
        Initialize production embedding service.

        Args:
            model_name: Model to use (default from EMBEDDING_MODEL env var)
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar during encoding
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", self.DEFAULT_MODELS["general"]
        )
        self.device = device or os.getenv("EMBEDDING_DEVICE", None)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress

        self._model = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_texts": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "errors": 0,
        }

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        print(f"Loading embedding model: {self.model_name}")
                        print(
                            "(Model downloaded on first run and cached for future use)"
                        )

                        # Load with trust_remote_code for models like nomic
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                            trust_remote_code=True,
                        )

                        self._dimension = self._model.get_sentence_embedding_dimension()
                        print(
                            f"Model loaded: {self.model_name} "
                            f"(dim={self._dimension}, device={self._model.device})"
                        )

                    except Exception as e:
                        self._stats["errors"] += 1
                        raise RuntimeError(
                            f"Failed to load embedding model '{self.model_name}': {e}"
                        ) from e

        return self._model

    def encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            instruction: Optional instruction prefix (for instruction-tuned models)
            batch_size: Override default batch size

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._load_model()
        batch_size = batch_size or self.batch_size

        start_time = time.time()

        try:
            # Prepare texts with instruction if provided
            if instruction:
                # For models that support instructions (like nomic)
                texts_to_encode = [f"{instruction}: {text}" for text in texts]
            else:
                texts_to_encode = texts

            # Encode with sentence-transformers
            embeddings = model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=self.show_progress,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )

            # Convert to list format
            result = embeddings.tolist()

            # Update stats
            elapsed = time.time() - start_time
            self._stats["total_texts"] += len(texts)
            self._stats["total_batches"] += (len(texts) + batch_size - 1) // batch_size
            self._stats["total_time"] += elapsed

            return result

        except Exception as e:
            self._stats["errors"] += 1
            raise RuntimeError(f"Embedding failed: {e}") from e

    def encode_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Encode a single text to embedding.

        Args:
            text: Text to encode
            instruction: Optional instruction prefix

        Returns:
            Embedding vector
        """
        result = self.encode([text], instruction=instruction)
        return result[0] if result else []

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding vector dimension
        """
        if self._dimension is None:
            self._load_model()
        return self._dimension or 768

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        model = self._load_model()
        return {
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "device": str(model.device) if model else "not_loaded",
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with usage statistics
        """
        avg_time = (
            self._stats["total_time"] / self._stats["total_batches"]
            if self._stats["total_batches"] > 0
            else 0
        )
        return {
            **self._stats,
            "avg_batch_time": avg_time,
            "model_loaded": self._model is not None,
        }

    def is_ready(self) -> bool:
        """
        Check if service is ready.

        Returns:
            True if model is loaded and ready
        """
        try:
            self._load_model()
            return True
        except Exception:
            return False


# Singleton instance
_service_instance: Optional[ProductionEmbeddingService] = None
_service_lock = threading.Lock()


def get_production_embedding_service(
    model_name: Optional[str] = None,
) -> ProductionEmbeddingService:
    """
    Get singleton production embedding service instance.

    Args:
        model_name: Optional model name override

    Returns:
        Shared ProductionEmbeddingService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ProductionEmbeddingService(model_name=model_name)

    return _service_instance
