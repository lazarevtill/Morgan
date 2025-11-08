"""
Local Model Manager - Local model integration

Handles local model loading and management for offline operation.
Supports Ollama, Transformers, and sentence-transformers.

Requirements addressed: 23.1, 23.2, 23.3
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LocalModelManager:
    """
    Local model manager following KISS principles.

    Single responsibility: Manage local AI models only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cache_dir = Path(config.get("cache_dir", "./data/models"))
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Ollama configuration (can be local or remote)
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")

        # Available local model providers
        self.providers = {
            "ollama": self._load_ollama_model,
            "transformers": self._load_transformers_model,
            "sentence_transformers": self._load_sentence_transformers_model,
        }

    def load_model(self, model_name: str, model_type: str = "auto") -> Any:
        """
        Load a local model.

        Args:
            model_name: Name of the model to load
            model_type: Type of model ('embedding', 'llm', 'emotional')

        Returns:
            Loaded model instance
        """
        try:
            # Determine provider based on model name or type
            provider = self._detect_provider(model_name, model_type)

            if provider not in self.providers:
                raise ValueError(f"Unsupported local provider: {provider}")

            # Load model using appropriate provider
            model = self.providers[provider](model_name, model_type)

            logger.info(
                "Successfully loaded local model: %s via %s", model_name, provider
            )
            return model

        except Exception as e:
            logger.error("Failed to load local model %s: %s", model_name, e)
            raise

    def _detect_provider(self, model_name: str, model_type: str) -> str:
        """Detect which local provider to use."""
        # Enhanced detection logic
        model_lower = model_name.lower()

        # Check for Ollama model patterns
        ollama_patterns = ["llama", "mistral", "codellama", "phi3", "gemma", "ollama"]
        if any(pattern in model_lower for pattern in ollama_patterns):
            return "ollama"

        # Check for sentence transformers patterns
        sentence_patterns = ["sentence", "all-", "paraphrase", "mpnet"]
        if model_type == "embedding" or any(
            pattern in model_lower for pattern in sentence_patterns
        ):
            return "sentence_transformers"

        # Default to transformers for other models
        return "transformers"

    def _load_ollama_model(self, model_name: str, model_type: str) -> Any:
        """Load model via Ollama (local or remote)."""
        try:
            # Import ollama only when needed
            import ollama

            # Configure client for local or remote Ollama
            client = ollama.Client(host=self.ollama_host)

            # Check if model is available
            models = client.list()
            available_models = [m["name"] for m in models.get("models", [])]

            if model_name not in available_models:
                logger.info(
                    "Pulling Ollama model: %s from %s", model_name, self.ollama_host
                )
                client.pull(model_name)

            return {
                "provider": "ollama",
                "model_name": model_name,
                "client": client,
                "host": self.ollama_host,
                "type": model_type,
            }

        except ImportError as exc:
            raise ImportError(
                "Ollama not installed. Install with: pip install ollama"
            ) from exc
        except Exception as e:
            logger.error(
                "Failed to load Ollama model %s from %s: %s",
                model_name,
                self.ollama_host,
                e,
            )
            raise

    def _load_transformers_model(self, model_name: str, model_type: str) -> Any:
        """Load model via Transformers."""
        try:
            from transformers import AutoModel, AutoTokenizer

            # Load model and tokenizer
            model = AutoModel.from_pretrained(
                model_name, cache_dir=str(self.model_cache_dir)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(self.model_cache_dir)
            )

            return {
                "provider": "transformers",
                "model_name": model_name,
                "model": model,
                "tokenizer": tokenizer,
                "type": model_type,
            }

        except ImportError as exc:
            raise ImportError(
                "Transformers not installed. " "Install with: pip install transformers"
            ) from exc
        except Exception as e:
            logger.error("Failed to load Transformers model %s: %s", model_name, e)
            raise

    def _load_sentence_transformers_model(
        self, model_name: str, model_type: str
    ) -> Any:
        """Load model via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_name, cache_folder=str(self.model_cache_dir)
            )

            return {
                "provider": "sentence_transformers",
                "model_name": model_name,
                "model": model,
                "type": model_type,
            }

        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc
        except Exception as e:
            logger.error(
                "Failed to load sentence-transformers model %s: %s", model_name, e
            )
            raise

    def list_models(self) -> List[str]:
        """List available local models."""
        models = []

        # Check Ollama models (local or remote)
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            ollama_models = client.list()
            models.extend([m["name"] for m in ollama_models.get("models", [])])
        except Exception:
            pass

        # Add common sentence-transformers models
        models.extend(
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"]
        )

        return models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a local model."""
        provider = self._detect_provider(model_name, "auto")

        info = {
            "name": model_name,
            "provider": provider,
            "location": "local",
            "cache_dir": str(self.model_cache_dir),
        }

        # Add Ollama host info if it's an Ollama model
        if provider == "ollama":
            info["ollama_host"] = self.ollama_host
            info["is_remote_ollama"] = not self.ollama_host.startswith(
                "http://localhost"
            )

        return info

    def generate_ollama_response(
        self, model_client: Dict[str, Any], prompt: str, context: str = ""
    ) -> str:
        """Generate response using Ollama model."""
        try:
            client = model_client["client"]
            model_name = model_client["model_name"]

            # Prepare messages
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})

            # Generate response
            response = client.chat(model=model_name, messages=messages)

            return response["message"]["content"]

        except Exception as e:
            logger.error("Failed to generate Ollama response: %s", e)
            raise

    def generate_ollama_embedding(
        self, model_client: Dict[str, Any], text: str
    ) -> List[float]:
        """Generate embedding using Ollama model."""
        try:
            client = model_client["client"]
            model_name = model_client["model_name"]

            # Generate embedding
            response = client.embeddings(model=model_name, prompt=text)

            return response["embedding"]

        except Exception as e:
            logger.error("Failed to generate Ollama embedding: %s", e)
            raise

    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            client.list()
            return True
        except Exception:
            return False

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        try:
            provider = self._detect_provider(model_name, "auto")

            if provider == "ollama":
                return self._is_ollama_model_available(model_name)
            elif provider == "sentence_transformers":
                return True  # Assume sentence transformers can be downloaded
            elif provider == "transformers":
                return True  # Assume transformers can be downloaded

            return False
        except Exception:
            return False

    def _is_ollama_model_available(self, model_name: str) -> bool:
        """Check if an Ollama model is available."""
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            models = client.list()
            available_models = [m["name"] for m in models.get("models", [])]
            return model_name in available_models
        except Exception:
            return False
