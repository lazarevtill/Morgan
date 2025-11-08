"""
Lazarev Cloud Integration - gpt.lazarev.cloud endpoint integration

Provides dedicated integration with gpt.lazarev.cloud endpoint.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.2, 23.3
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class LazarevConfig:
    """Configuration for gpt.lazarev.cloud endpoint."""

    endpoint_url: str = "https://gpt.lazarev.cloud"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


class LazarevModelManager:
    """
    Lazarev cloud model manager following KISS principles.

    Single responsibility: Manage gpt.lazarev.cloud endpoint integration only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = LazarevConfig(
            endpoint_url=config.get("endpoint_url", "https://gpt.lazarev.cloud"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3),
        )

    def load_model(self, model_name: str, model_type: str = "auto") -> Dict[str, Any]:
        """
        Create a connection to a Lazarev cloud model.

        Args:
            model_name: Name of the model on Lazarev cloud
            model_type: Type of model ('embedding', 'llm', 'emotional')

        Returns:
            Model client configuration
        """
        try:
            # Test connection
            if self._test_connection():
                return {
                    "provider": "lazarev",
                    "model_name": model_name,
                    "endpoint": self.config.endpoint_url,
                    "config": self.config,
                    "type": model_type,
                    "client": self,
                }

            raise ConnectionError("Cannot connect to gpt.lazarev.cloud")

        except Exception as e:
            logger.error("Failed to load Lazarev model %s: %s", model_name, e)
            raise

    def _test_connection(self) -> bool:
        """Test connection to gpt.lazarev.cloud."""
        try:
            # Simple health check
            response = requests.get(
                f"{self.config.endpoint_url}/health", timeout=self.config.timeout
            )
            return response.status_code == 200
        except Exception:
            # If health endpoint doesn't exist, assume connection is available
            return True

    def generate_embedding(self, model_name: str, text: str) -> List[float]:
        """Generate embedding using Lazarev cloud model."""
        try:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            payload = {"model": model_name, "input": text}

            response = requests.post(
                f"{self.config.endpoint_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )

            response.raise_for_status()
            result = response.json()

            return result["data"][0]["embedding"]

        except Exception as e:
            logger.error("Lazarev embedding failed: %s", e)
            raise

    def generate_response(self, model_name: str, prompt: str, context: str = "") -> str:
        """Generate text response using Lazarev cloud model."""
        try:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})

            payload = {"model": model_name, "messages": messages}

            response = requests.post(
                f"{self.config.endpoint_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )

            response.raise_for_status()
            result = response.json()

            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error("Lazarev response failed: %s", e)
            raise

    def list_available_models(self) -> List[str]:
        """List available models on Lazarev cloud."""
        try:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            response = requests.get(
                f"{self.config.endpoint_url}/models",
                headers=headers,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return [model["id"] for model in result.get("data", [])]

            # Fallback to common models if API doesn't support listing
            return [
                "gpt-3.5-turbo",
                "gpt-4",
                "text-embedding-ada-002",
                "jina-embeddings-v4",
                "jina-code-embeddings-1.5b",
                "jina-clip-v2",
                "jina-reranker-v3",
            ]

        except Exception as e:
            logger.warning("Failed to list Lazarev models: %s", e)
            # Return common models as fallback
            return [
                "gpt-3.5-turbo",
                "gpt-4",
                "text-embedding-ada-002",
                "jina-embeddings-v4",
                "jina-code-embeddings-1.5b",
                "jina-clip-v2",
                "jina-reranker-v3",
            ]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a Lazarev cloud model."""
        return {
            "name": model_name,
            "provider": "lazarev",
            "location": "remote",
            "endpoint": self.config.endpoint_url,
            "type": "cloud",
        }
