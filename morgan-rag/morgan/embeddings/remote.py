"""
Remote embedding provider implementation.
"""

import time
import requests
import random
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base import EmbeddingProvider
from morgan.config import get_settings
from morgan.utils.error_handling import EmbeddingError, ErrorSeverity
from morgan.utils.logger import get_logger
from morgan.utils.rate_limiting import TokenBucketRateLimiter

logger = get_logger(__name__)


class RemoteEmbeddingProvider(EmbeddingProvider):
    """
    Provider for remote embedding services (Ollama, OpenAI-compatible).
    """

    MODELS = {
        "qwen3:latest": {
            "dimensions": 4096,
            "max_tokens": 8192,
            "supports_instructions": True,
        },
        "qwen3-embedding:latest": {
            "dimensions": 4096,
            "max_tokens": 8192,
            "supports_instructions": True,
        },
        "nomic-embed-text": {
            "dimensions": 768,
            "max_tokens": 8192,
            "supports_instructions": True,
        },
    }

    def __init__(self, model_name: str, settings=None):
        self.settings = settings or get_settings()
        self._model_name = model_name
        self.model_config = self.MODELS.get(
            model_name, self.MODELS["qwen3-embedding:latest"]
        )
        self._remote_available = None
        self._remote_base_url = None

        # Rate limiting (100 requests/minute)
        self.rate_limiter = TokenBucketRateLimiter(rate_limit=100, time_window=60.0)

    @property
    def provider_type(self) -> str:
        return "remote"

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_dimension(self) -> int:
        return self.model_config["dimensions"]

    def _get_remote_base_url(self) -> Optional[str]:
        """Get configured remote base URL for embeddings."""
        if self._remote_base_url is not None:
            return self._remote_base_url

        base_url = getattr(self.settings, "embedding_base_url", None)

        if not base_url:
            ollama_host = getattr(self.settings, "ollama_host", None)
            if ollama_host:
                if not ollama_host.startswith(("http://", "https://")):
                    base_url = f"http://{ollama_host}"
                else:
                    base_url = ollama_host

        if not base_url:
            base_url = getattr(self.settings, "llm_base_url", None)

        if not base_url:
            self._remote_base_url = None
            return None

        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        if base_url.endswith("/api"):
            base_url = base_url[:-4]

        self._remote_base_url = base_url
        return base_url

    def is_available(self) -> bool:
        """Check if remote embedding service is available."""
        if self._remote_available is not None:
            return self._remote_available

        base_url = self._get_remote_base_url()
        if not base_url:
            self._remote_available = False
            return False

        max_retries = 3
        delays = [1, 2, 4]

        for attempt in range(max_retries):
            try:
                model_urls = [f"{base_url}/api/models", f"{base_url}/api/tags"]
                headers = {}
                if hasattr(self.settings, "llm_api_key") and self.settings.llm_api_key:
                    headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

                response = None
                for url in model_urls:
                    try:
                        response = requests.get(url, headers=headers, timeout=5)
                        response.raise_for_status()
                        break
                    except Exception:
                        continue

                if response is None:
                    self._remote_available = False
                    return False

                try:
                    data = response.json()
                except ValueError:
                    self._remote_available = response.ok
                    return self._remote_available

                models = []
                if isinstance(data, dict) and "data" in data:
                    models = [
                        m.get("id") or m.get("name") or m.get("model", "")
                        for m in data.get("data", [])
                    ]
                elif isinstance(data, dict) and "models" in data:
                    models = [m.get("name", "") for m in data.get("models", [])]
                elif isinstance(data, list):
                    models = [
                        m.get("id") or m.get("name")
                        for m in data
                        if isinstance(m, dict)
                    ]

                if self._model_name not in models:
                    logger.warning(
                        f"Model {self._model_name} not found in remote service."
                    )
                    self._remote_available = False
                    return False

                self._remote_available = True
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delays[attempt])
                    continue
                logger.warning(f"Remote embedding service not available: {e}")
                self._remote_available = False
                return False

        self._remote_available = False
        return False

    def encode(
        self, text: str, request_id: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Encode text using remote embedding service."""
        try:
            self.rate_limiter.acquire(timeout=30.0)
        except TimeoutError:
            logger.warning(
                f"Rate limiter timeout, proceeding (request_id={request_id})"
            )

        base_url = self._get_remote_base_url()
        if not base_url:
            raise ValueError("Remote embedding base URL is not configured")

        url = f"{base_url}/api/embeddings"
        is_ollama_style = "ollama" in base_url or base_url.endswith(":11434")

        headers = {}
        if hasattr(self.settings, "llm_api_key") and self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        payload = {
            "model": self._model_name,
            "prompt" if is_ollama_style else "input": text,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            if is_ollama_style:
                return data["embedding"]
            else:
                return data["data"][0]["embedding"]

        except Exception as e:
            raise EmbeddingError(
                f"Remote embedding encoding failed: {e}",
                operation="encode",
                component="remote_embedding_provider",
                request_id=request_id,
                metadata={"model": self._model_name, "text_length": len(text)},
            ) from e

    def encode_batch(
        self, texts: List[str], request_id: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        """Encode batch using remote service with optimized API calls."""
        # For remote, we often process texts individually if the API doesn't support batching well
        # or we want to handle retries per item. Here we'll do a simple loop or use batch API if available.
        # This implementation follows the legacy _encode_batch_remote logic.

        results = []
        for text in texts:
            results.append(self.encode(text, request_id=request_id, **kwargs))
        return results
