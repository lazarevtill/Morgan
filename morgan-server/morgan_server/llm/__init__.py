"""LLM Client - Self-hosted LLM integration."""

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM service fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Provides a common interface for different LLM providers
    (Ollama, OpenAI-compatible endpoints).
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content

        Raises:
            LLMConnectionError: If connection fails
            LLMTimeoutError: If request times out
            LLMRateLimitError: If rate limit exceeded
            LLMError: For other errors
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of generated text

        Raises:
            LLMConnectionError: If connection fails
            LLMTimeoutError: If request times out
            LLMRateLimitError: If rate limit exceeded
            LLMError: For other errors
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass


from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient


__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "OllamaClient",
    "OpenAICompatibleClient",
]
