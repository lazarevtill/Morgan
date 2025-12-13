"""
Model Selector - Local model selection and routing

Provides intelligent model selection and routing logic.
Follows KISS principles with simple, rule-based selection.

Requirements addressed: 23.1, 23.2, 23.3
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Model selector following KISS principles.

    Single responsibility: Select appropriate models and providers.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Default model preferences
        self.default_models = {
            "embedding": {
                "local": "all-MiniLM-L6-v2",
                "lazarev": "text-embedding-ada-002",
            },
            "llm": {"local": "ollama/gemma3", "lazarev": "gpt-3.5-turbo"},
            "emotional": {
                "local": "all-MiniLM-L6-v2",
                "lazarev": "text-embedding-ada-002",
            },
        }

        # Provider preferences (local-first by default)
        self.provider_preference = config.get(
            "provider_preference", ["local", "lazarev"]
        )

        # Model-specific routing rules
        self.routing_rules = {
            # Jina models - use lazarev endpoint
            "jina-embeddings-v4": "lazarev",
            "jina-code-embeddings-1.5b": "lazarev",
            "jina-clip-v2": "lazarev",
            "jina-reranker-v3": "lazarev",
            "jina-reranker-v2-base-multilingual": "lazarev",
            # Ollama models - use local manager (handles local/remote Ollama)
            "gemma3": "local",
            "llama3": "local",
            "llama3.1": "local",
            "mistral": "local",
            "codellama": "local",
            "phi3": "local",
            "gemma": "local",
            # Sentence transformers - prefer local
            "all-MiniLM-L6-v2": "local",
            "all-mpnet-base-v2": "local",
            "paraphrase-MiniLM-L6-v2": "local",
            # OpenAI models - use lazarev endpoint
            "gpt-3.5-turbo": "lazarev",
            "gpt-4": "lazarev",
            "gpt-4o": "lazarev",
            "text-embedding-ada-002": "lazarev",
        }

    def select_provider(self, model_name: str, model_type: str = "auto") -> str:
        """
        Select the appropriate provider for a model.

        Args:
            model_name: Name of the model
            model_type: Type of model ('embedding', 'llm', 'emotional', 'auto')

        Returns:
            Provider name ('local' or 'lazarev')
        """
        try:
            # Check explicit routing rules first
            if model_name in self.routing_rules:
                provider = self.routing_rules[model_name]
                logger.debug("Using routing rule for %s: %s", model_name, provider)
                return provider

            # Check for provider hints in model name
            model_lower = model_name.lower()
            ollama_models = ["llama", "mistral", "codellama", "phi3", "gemma"]

            if "ollama" in model_lower or any(
                ollama_model in model_lower for ollama_model in ollama_models
            ):
                # Local manager handles both local and remote Ollama
                return "local"

            if "jina" in model_lower:
                return "lazarev"

            if "gpt" in model_lower:
                return "lazarev"

            if "sentence" in model_lower or "all-" in model_lower:
                return "local"

            # Use provider preference order
            for provider in self.provider_preference:
                if self._is_model_available(model_name, provider):
                    logger.debug(
                        "Selected %s for %s based on availability", provider, model_name
                    )
                    return provider

            # Fallback to first preferred provider
            fallback = self.provider_preference[0]
            logger.warning(
                "No specific rule for %s, using fallback: %s", model_name, fallback
            )
            return fallback

        except Exception as e:
            logger.error("Failed to select provider for %s: %s", model_name, e)
            return "local"  # Safe fallback

    def select_model(self, model_type: str, provider: Optional[str] = None) -> str:
        """
        Select an appropriate model for a given type.

        Args:
            model_type: Type of model needed ('embedding', 'llm', 'emotional')
            provider: Preferred provider ('local', 'lazarev', or None for auto)

        Returns:
            Model name
        """
        try:
            if provider:
                # Use specific provider
                if (
                    model_type in self.default_models
                    and provider in self.default_models[model_type]
                ):
                    return self.default_models[model_type][provider]

                raise ValueError(
                    f"No default {model_type} model for provider {provider}"
                )

            # Auto-select based on preference
            for pref_provider in self.provider_preference:
                if (
                    model_type in self.default_models
                    and pref_provider in self.default_models[model_type]
                ):
                    return self.default_models[model_type][pref_provider]

            raise ValueError(f"No default {model_type} model available")

        except Exception as e:
            logger.error("Failed to select model for type %s: %s", model_type, e)
            # Ultimate fallback
            return "all-MiniLM-L6-v2"

    def get_model_recommendations(self, model_type: str) -> Dict[str, List[str]]:
        """
        Get model recommendations by provider.

        Args:
            model_type: Type of model ('embedding', 'llm', 'emotional')

        Returns:
            Dictionary of provider -> model list
        """
        recommendations = {
            "embedding": {
                "local": [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "paraphrase-MiniLM-L6-v2",
                ],
                "lazarev": [
                    "jina-embeddings-v4",
                    "text-embedding-ada-002",
                    "jina-code-embeddings-1.5b",
                ],
            },
            "llm": {
                "local": [
                    "gemma3",
                    "llama3",
                    "llama3.1",
                    "mistral",
                    "codellama",
                    "phi3",
                    "gemma",
                ],
                "lazarev": ["gpt-3.5-turbo", "gpt-4"],
            },
            "emotional": {
                "local": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                "lazarev": ["jina-embeddings-v4", "text-embedding-ada-002"],
            },
        }

        return recommendations.get(model_type, {"local": [], "lazarev": []})

    def _is_model_available(self, model_name: str, provider: str) -> bool:
        """
        Check if a model is available on a provider.

        This is a simplified check - in practice, you might want to
        actually query the providers to check availability.
        """
        try:
            model_lower = model_name.lower()

            if provider == "local":
                # Check common local model patterns
                local_patterns = ["ollama", "sentence", "all-", "paraphrase"]
                return any(pattern in model_lower for pattern in local_patterns)

            if provider == "lazarev":
                # Check common lazarev model patterns
                lazarev_patterns = ["jina", "gpt", "ada", "davinci"]
                return any(pattern in model_lower for pattern in lazarev_patterns)

            return False

        except Exception:
            return False

    def update_routing_rule(self, model_name: str, provider: str) -> None:
        """Add or update a routing rule."""
        self.routing_rules[model_name] = provider
        logger.info("Updated routing rule: %s -> %s", model_name, provider)

    def get_routing_rules(self) -> Dict[str, str]:
        """Get all current routing rules."""
        return self.routing_rules.copy()
