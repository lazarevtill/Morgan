"""
Model Manager - Central coordination for all model types

This module provides unified management for embedding models, LLMs, and
emotional models. Follows KISS principles with clear separation of concerns.

Requirements addressed: 23.1, 23.2, 23.3
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import ModelCache
from .lazarev import LazarevModelManager
from .local import LocalModelManager
from .selector import ModelSelector

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Simple model configuration."""

    model_type: str  # 'embedding', 'llm', 'emotional'
    model_name: str
    provider: str  # 'local', 'lazarev'
    config: Dict[str, Any]


class ModelManager:
    """
    Central model manager following KISS principles.

    Single responsibility: Coordinate model loading and management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_manager = LocalModelManager(config.get("local", {}))
        self.lazarev_manager = LazarevModelManager(config.get("lazarev", {}))
        self.cache = ModelCache(config.get("cache", {}))
        self.selector = ModelSelector(config.get("selector", {}))

        self._loaded_models: Dict[str, Any] = {}

    def load_model(self, model_name: str, model_type: str = "auto") -> Any:
        """
        Load a model with automatic provider selection.

        Args:
            model_name: Name of the model to load
            model_type: Type of model ('embedding', 'llm', 'emotional', 'auto')

        Returns:
            Loaded model instance
        """
        try:
            # Check cache first
            cached_model = self.cache.get_model(model_name)
            if cached_model:
                logger.info("Using cached model: %s", model_name)
                return cached_model

            # Select provider
            provider = self.selector.select_provider(model_name, model_type)

            # Load model from appropriate provider
            if provider == "local":
                model = self.local_manager.load_model(model_name, model_type)
            elif provider == "lazarev":
                model = self.lazarev_manager.load_model(model_name, model_type)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Cache the loaded model
            self.cache.store_model(model_name, model)
            self._loaded_models[model_name] = model

            logger.info("Successfully loaded model: %s from %s", model_name, provider)
            return model

        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, e)
            raise

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self._loaded_models.get(model_name)

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by provider."""
        return {
            "local": self.local_manager.list_models(),
            "lazarev": self.lazarev_manager.list_available_models(),
        }

    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        try:
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]
                self.cache.remove_model(model_name)
                logger.info("Unloaded model: %s", model_name)
                return True
            return False
        except Exception as e:
            logger.error("Failed to unload model %s: %s", model_name, e)
            return False

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        provider = self.selector.select_provider(model_name, "auto")

        if provider == "local":
            return self.local_manager.get_model_info(model_name)
        if provider == "lazarev":
            return self.lazarev_manager.get_model_info(model_name)

        return {"error": f"Unknown provider: {provider}"}

    def generate_response(self, model_name: str, prompt: str, context: str = "") -> str:
        """Generate text response using loaded model."""
        try:
            model = self.get_model(model_name)
            if not model:
                model = self.load_model(model_name, "llm")

            provider = model.get("provider")

            if provider == "ollama":
                return self.local_manager.generate_ollama_response(
                    model, prompt, context
                )
            elif provider == "lazarev":
                return self.lazarev_manager.generate_response(
                    model["model_name"], prompt, context
                )
            else:
                raise ValueError(
                    f"Response generation not supported for provider: " f"{provider}"
                )

        except Exception as e:
            logger.error("Failed to generate response with %s: %s", model_name, e)
            raise

    def generate_embedding(self, model_name: str, text: str) -> List[float]:
        """Generate embedding using loaded model."""
        try:
            model = self.get_model(model_name)
            if not model:
                model = self.load_model(model_name, "embedding")

            provider = model.get("provider")

            if provider == "ollama":
                return self.local_manager.generate_ollama_embedding(model, text)
            elif provider == "sentence_transformers":
                # Use sentence transformers model directly
                return model["model"].encode([text])[0].tolist()
            elif provider == "lazarev":
                return self.lazarev_manager.generate_embedding(
                    model["model_name"], text
                )
            else:
                raise ValueError(
                    f"Embedding generation not supported for provider: " f"{provider}"
                )

        except Exception as e:
            logger.error("Failed to generate embedding with %s: %s", model_name, e)
            raise

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available for loading."""
        try:
            provider = self.selector.select_provider(model_name, "auto")
            if provider == "local":
                return self.local_manager.is_model_available(model_name)
            elif provider == "lazarev":
                return True  # Assume lazarev models are available
            return False
        except Exception:
            return False

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return list(self._loaded_models.keys())

    def clear_cache(self) -> bool:
        """Clear all cached models."""
        try:
            self._loaded_models.clear()
            return self.cache.clear_cache()
        except Exception as e:
            logger.error("Failed to clear model cache: %s", e)
            return False
