"""
Middleware for embedding providers to handle cross-cutting concerns.
"""

import hashlib
from typing import List, Optional, Dict, Any

from .base import EmbeddingProvider
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingMiddleware(EmbeddingProvider):
    """
    Base class for embedding middleware.
    """

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    @property
    def provider_type(self) -> str:
        return self.provider.provider_type

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def get_dimension(self) -> int:
        return self.provider.get_dimension()

    def is_available(self) -> bool:
        return self.provider.is_available()

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        return self.provider.encode(text, request_id=request_id, **kwargs)

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        return self.provider.encode_batch(texts, request_id=request_id, **kwargs)


class CachingMiddleware(EmbeddingMiddleware):
    """
    Middleware that handles embedding caching.
    """

    def __init__(self, provider: EmbeddingProvider, cache):
        super().__init__(provider)
        self.cache = cache

    def _get_cache_key(self, text: str) -> str:
        """Get cache key for text."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb_{self.provider.model_name}_{content_hash}"

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        use_cache = kwargs.get("use_cache", True)
        if not use_cache:
            return super().encode(text, request_id=request_id, **kwargs)

        cache_key = self._get_cache_key(text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {self.provider.model_name}")
            return cached

        embedding = super().encode(text, request_id=request_id, **kwargs)
        self.cache.set(cache_key, embedding)
        return embedding

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        use_cache = kwargs.get("use_cache", True)
        if not use_cache:
            return super().encode_batch(texts, request_id=request_id, **kwargs)

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug(f"Cache miss: {len(uncached_texts)}/{len(texts)}")
            new_embeddings = super().encode_batch(
                uncached_texts, request_id=request_id, **kwargs
            )
            for i, idx in enumerate(uncached_indices):
                embedding = new_embeddings[i]
                results[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self.cache.set(cache_key, embedding)
        else:
            logger.debug(f"All {len(texts)} texts found in cache")

        return results


class InstructionMiddleware(EmbeddingMiddleware):
    """
    Middleware that handles instruction prefixes.
    """

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        instruction = kwargs.get("instruction")
        if instruction:
            text = self._apply_instruction_prefix(text, instruction)
        return super().encode(text, request_id=request_id, **kwargs)

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        instruction = kwargs.get("instruction")
        if instruction:
            texts = [self._apply_instruction_prefix(t, instruction) for t in texts]
        return super().encode_batch(texts, request_id=request_id, **kwargs)

    def _apply_instruction_prefix(self, text: str, instruction: str) -> str:
        """Apply instruction prefix."""
        if instruction == "query":
            return f"query: {text}"
        elif instruction == "document":
            return f"document: {text}"
        return f"{instruction}: {text}"


class RateLimitingMiddleware(EmbeddingMiddleware):
    """
    Middleware that handles rate limiting.
    """

    def __init__(self, provider: EmbeddingProvider, rate_limiter):
        super().__init__(provider)
        self.rate_limiter = rate_limiter

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        self.rate_limiter.acquire()
        return super().encode(text, request_id=request_id, **kwargs)

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        # For simplicity, we limit by the number of calls, not number of items.
        # This is a bit naive but matches how original code did it for remote.
        self.rate_limiter.acquire()
        return super().encode_batch(texts, request_id=request_id, **kwargs)


class RetryMiddleware(EmbeddingMiddleware):
    """
    Middleware that handles retries with exponential backoff.
    """

    def __init__(self, provider: EmbeddingProvider, max_retries: int = 3):
        super().__init__(provider)
        self.max_retries = max_retries

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return super().encode(text, request_id=request_id, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    base_delay = 2**attempt
                    jitter = random.uniform(0, base_delay * 0.3)
                    delay = base_delay + jitter
                    logger.warning(
                        f"Encoding failed (attempt {attempt+1}/{self.max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    import time

                    time.sleep(delay)
                else:
                    logger.error(
                        f"Encoding failed after {self.max_retries} attempts: {e}"
                    )

        if last_exception:
            raise last_exception
        return []

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return super().encode_batch(texts, request_id=request_id, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    base_delay = 2**attempt
                    jitter = random.uniform(0, base_delay * 0.3)
                    delay = base_delay + jitter
                    logger.warning(
                        f"Batch encoding failed (attempt {attempt+1}/{self.max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    import time

                    time.sleep(delay)
                else:
                    logger.error(
                        f"Batch encoding failed after {self.max_retries} attempts: {e}"
                    )

        if last_exception:
            raise last_exception
        return []
