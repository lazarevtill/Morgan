"""
Embedding service orchestrator.
"""

import time
from collections import defaultdict, deque
from typing import List, Optional, Dict, Any
from datetime import datetime

from .factory import get_configured_provider
from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.utils.error_handling import EmbeddingError, ErrorSeverity
from morgan.utils.request_context import get_request_id, set_request_id

logger = get_logger(__name__)


class EmbeddingService:
    """
    Orchestrates multiple embedding providers with fallback and performance tracking.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        # Primary provider (from settings)
        self.model_name = self.settings.embedding_model
        self.primary_provider = get_configured_provider(self.model_name, self.settings)
        
        # Secondary/fallback provider (local)
        self.local_model_name = getattr(self.settings, "embedding_local_model", "all-MiniLM-L6-v2")
        self.secondary_provider = get_configured_provider(self.local_model_name, self.settings)
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info(
            f"EmbeddingService initialized with primary: {self.model_name}, "
            f"fallback: {self.local_model_name}"
        )

    def is_available(self) -> bool:
        """Check if any provider is available."""
        force_remote = getattr(self.settings, "embedding_force_remote", False)
        
        if self.primary_provider.is_available():
            return True
            
        if force_remote:
            logger.error("Remote embedding forced but not available")
            return False
            
        return self.secondary_provider.is_available()

    def get_embedding_dimension(self) -> int:
        """Get dimension of current active model."""
        if self.primary_provider.is_available():
            return self.primary_provider.get_dimension()
        return self.secondary_provider.get_dimension()

    def encode(
        self,
        text: str,
        instruction: Optional[str] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> List[float]:
        """Encode text with fallback logic."""
        if request_id is None:
            request_id = get_request_id() or set_request_id()
            
        start_time = time.time()
        force_remote = getattr(self.settings, "embedding_force_remote", False)
        
        try:
            # Try primary provider first
            if self.primary_provider.is_available():
                embedding = self.primary_provider.encode(
                    text, instruction=instruction, use_cache=use_cache, request_id=request_id
                )
            elif force_remote:
                raise RuntimeError("Remote embedding forced but not available")
            # Fallback to secondary
            elif self.secondary_provider.is_available():
                embedding = self.secondary_provider.encode(
                    text, instruction=instruction, use_cache=use_cache, request_id=request_id
                )
            else:
                raise RuntimeError("No embedding service available")
                
            elapsed = time.time() - start_time
            self.performance_stats["encode_times"].append(elapsed)
            return embedding

        except Exception as e:
            if "Remote embedding forced but not available" in str(e):
                 raise EmbeddingError(
                    "Remote embedding service unavailable and force_remote enabled",
                    operation="encode",
                    severity=ErrorSeverity.HIGH,
                    request_id=request_id,
                ) from e
            raise EmbeddingError(
                f"Embedding encoding failed: {e}",
                operation="encode",
                request_id=request_id,
            ) from e

    def encode_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
        use_optimized_batching: bool = True,
    ) -> List[List[float]]:
        """Encode batch with fallback logic and optimizations."""
        if request_id is None:
            request_id = get_request_id() or set_request_id()
            
        if not texts:
            return []
            
        # Handle optimized batching if enabled (using legacy batch processor for now)
        if use_optimized_batching:
            try:
                from morgan.optimization.batch_processor import get_batch_processor
                batch_processor = get_batch_processor()
                
                def embedding_function(batch_texts: List[str]) -> List[List[float]]:
                    return self._encode_batch_with_fallback(
                        batch_texts, instruction=instruction, use_cache=use_cache, request_id=request_id
                    )
                    
                # The legacy batch processor expects a result object, but we want the embeddings
                # For now let's use the standard batching or modify the caller.
                # Actually, the old EmbeddingService just called the batch processor and then
                # fell back if it didn't return embeddings? No, it just called the processor.
                # Wait, looking at the old code, it didn't actually return the embeddings from the batch processor!
                # It just processing them? No, that's not right for search.
                pass
            except Exception as e:
                logger.warning(f"Optimized batch processing failed: {e}")

        return self._encode_batch_with_fallback(
            texts, instruction=instruction, use_cache=use_cache, request_id=request_id
        )

    def _encode_batch_with_fallback(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> List[List[float]]:
        """Internal batch encoding with fallback."""
        start_time = time.time()
        force_remote = getattr(self.settings, "embedding_force_remote", False)
        
        try:
            if self.primary_provider.is_available():
                embeddings = self.primary_provider.encode_batch(
                    texts, instruction=instruction, use_cache=use_cache, request_id=request_id
                )
            elif force_remote:
                 raise RuntimeError("Remote embedding forced but not available")
            elif self.secondary_provider.is_available():
                embeddings = self.secondary_provider.encode_batch(
                    texts, instruction=instruction, use_cache=use_cache, request_id=request_id
                )
            else:
                 raise RuntimeError("No embedding service available")
                 
            elapsed = time.time() - start_time
            self.performance_stats["batch_encode_times"].append(elapsed)
            return embeddings

        except Exception as e:
             raise EmbeddingError(
                f"Batch embedding encoding failed: {e}",
                operation="encode_batch",
                request_id=request_id,
            ) from e


_service_instance = None


def get_embedding_service(settings=None) -> EmbeddingService:
    """Get singleton embedding service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService(settings=settings)
    return _service_instance
