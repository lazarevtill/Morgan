"""
Abstract base classes for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    """

    @abstractmethod
    def encode(
        self,
        text: str,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        """
        Encode single text to embedding vector.

        Args:
            text: Text to embed
            request_id: Optional request ID for tracing
            **kwargs: Provider-specific arguments

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def encode_batch(
        self,
        texts: List[str],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to embed
            request_id: Optional request ID for tracing
            **kwargs: Provider-specific arguments

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if available, False otherwise
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """
        Return the type of the provider (e.g., 'remote', 'local').
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the name of the model being used.
        """
        pass
