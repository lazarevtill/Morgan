"""
Model Selector for Jina AI Integration

Simple model selection based on content type following KISS principles.
Single responsibility: model selection only.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ModelSelector:
    """Simple model selection based on content type."""

    # Model mappings for different content types
    EMBEDDING_MODELS = {
        "text": "jina-embeddings-v4",
        "code": "jina-code-embeddings-1.5b",
        "multimodal": "jina-clip-v2",
        "default": "jina-embeddings-v4",
    }

    RERANKER_MODELS = {
        "en": "jina-reranker-v3",
        "multilingual": "jina-reranker-v2-base-multilingual",
        "default": "jina-reranker-v3",
    }

    def __init__(self):
        """Initialize the model selector."""
        logger.info("Initialized Jina AI Model Selector")

    def select_embedding_model(self, content_type: str) -> str:
        """
        Single responsibility: model selection only.

        Args:
            content_type: Type of content ('text', 'code', 'multimodal')

        Returns:
            Model name for the given content type
        """
        model = self.EMBEDDING_MODELS.get(
            content_type, self.EMBEDDING_MODELS["default"]
        )
        logger.debug(
            f"Selected embedding model '{model}' for content type '{content_type}'"
        )
        return model

    def select_reranker_model(self, language: str = "en") -> str:
        """
        Single responsibility: reranker selection only.

        Args:
            language: Language code ('en' for English, others use multilingual)

        Returns:
            Reranker model name for the given language
        """
        if language == "en":
            model = self.RERANKER_MODELS["en"]
        else:
            model = self.RERANKER_MODELS["multilingual"]

        logger.debug(f"Selected reranker model '{model}' for language '{language}'")
        return model

    def get_available_embedding_models(self) -> Dict[str, str]:
        """
        Get all available embedding models.

        Returns:
            Dictionary of content types to model names
        """
        return self.EMBEDDING_MODELS.copy()

    def get_available_reranker_models(self) -> Dict[str, str]:
        """
        Get all available reranker models.

        Returns:
            Dictionary of language types to model names
        """
        return self.RERANKER_MODELS.copy()

    def is_valid_content_type(self, content_type: str) -> bool:
        """
        Check if content type is supported.

        Args:
            content_type: Content type to validate

        Returns:
            True if content type is supported
        """
        return content_type in self.EMBEDDING_MODELS

    def is_valid_language(self, language: str) -> bool:
        """
        Check if language is supported for reranking.

        Args:
            language: Language code to validate

        Returns:
            True if language is supported
        """
        return (
            language in self.RERANKER_MODELS or language != "en"
        )  # Non-English uses multilingual
