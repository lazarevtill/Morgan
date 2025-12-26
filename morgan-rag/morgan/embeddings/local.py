"""
Local embedding provider implementation using sentence-transformers.
"""

from typing import List, Optional, Dict, Any

from .base import EmbeddingProvider
from morgan.config import get_settings
from morgan.utils.error_handling import EmbeddingError, ErrorSeverity
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Provider for local embeddings using sentence-transformers.
    """

    MODELS = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_tokens": 512,
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimensions": 768,
            "max_tokens": 514,
        },
    }

    def __init__(self, model_name: str, settings=None):
        self.settings = settings or get_settings()
        self._model_name = model_name
        self.model_config = self.MODELS.get(model_name, self.MODELS["all-MiniLM-L6-v2"])
        self._local_model = None
        self._local_available = None

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_dimension(self) -> int:
        return self.model_config["dimensions"]

    def is_available(self) -> bool:
        """Check if local sentence-transformers is available."""
        if self._local_available is not None:
            return self._local_available

        try:
            from sentence_transformers import SentenceTransformer

            device = getattr(self.settings, "embedding_device", "cpu")
            logger.info(f"Loading local embedding model ({self._model_name})...")
            self._local_model = SentenceTransformer(self._model_name, device=device)
            self._local_available = True
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, local embeddings unavailable"
            )
            self._local_available = False
            return False
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            self._local_available = False
            return False

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Encode text using local sentence-transformers."""
        if not self.is_available():
            raise RuntimeError("Local embedding service is not available")

        try:
            embedding = self._local_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(
                f"Local embedding encoding failed: {e}",
                operation="encode",
                component="local_embedding_provider",
                request_id=request_id,
                metadata={"model": self._model_name, "text_length": len(text)},
            ) from e

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        """Encode batch using local sentence-transformers."""
        if not self.is_available():
            raise RuntimeError("Local embedding service is not available")

        try:
            embeddings = self._local_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(
                f"Local batch embedding encoding failed: {e}",
                operation="encode_batch",
                component="local_embedding_provider",
                request_id=request_id,
                metadata={"model": self._model_name, "batch_size": len(texts)},
            ) from e
