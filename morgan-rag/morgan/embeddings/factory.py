"""
Factory for creating and configuring embedding providers.
"""

from pathlib import Path
from typing import Optional

from .base import EmbeddingProvider
from .remote import RemoteEmbeddingProvider
from .local import LocalEmbeddingProvider
from .middleware import CachingMiddleware, InstructionMiddleware, RateLimitingMiddleware, RetryMiddleware
from morgan.config import get_settings
from morgan.utils.cache import FileCache
from morgan.utils.rate_limiting import TokenBucketRateLimiter


def create_provider(model_name: str, settings=None) -> EmbeddingProvider:
    """
    Create a provider based on model name.
    """
    settings = settings or get_settings()
    
    # Determine if it's a remote or local model
    remote_models = RemoteEmbeddingProvider.MODELS.keys()
    
    if model_name in remote_models:
        return RemoteEmbeddingProvider(model_name, settings=settings)
    else:
        # Default to local if not recognized as remote, or explicitly local
        return LocalEmbeddingProvider(model_name, settings=settings)


def get_configured_provider(model_name: str, settings=None, cache=None) -> EmbeddingProvider:
    """
    Get a provider configured with standard middleware.
    """
    settings = settings or get_settings()
    provider = create_provider(model_name, settings)
    
    # Wrap with standard middleware
    
    # 1. Instructions (if supported)
    # The middleware handles checking if instruction is passed in kwargs
    provider = InstructionMiddleware(provider)
    
    # 2. Caching
    if cache is None:
        cache_dir = Path(settings.morgan_data_dir) / "cache" / "embeddings"
        cache = FileCache(cache_dir)
    provider = CachingMiddleware(provider, cache)
    
    # 3. Retries
    provider = RetryMiddleware(provider, max_retries=3)
    
    # 4. Rate Limiting (mostly for remote)
    if provider.provider_type == "remote":
        rate_limiter = TokenBucketRateLimiter(rate_limit=100, time_window=60.0)
        provider = RateLimitingMiddleware(provider, rate_limiter)
        
    return provider
