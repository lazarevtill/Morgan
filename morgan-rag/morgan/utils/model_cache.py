"""
Unified Model Cache Setup for Morgan AI Assistant.

Provides a single function to setup model cache directories and environment
variables, eliminating duplicate cache setup code in local_embeddings.py,
local_reranking.py, and distributed_config.py.

Usage:
    from morgan.utils.model_cache import setup_model_cache

    # Setup with default directory
    cache_path = setup_model_cache()

    # Setup with custom directory
    cache_path = setup_model_cache("~/.custom/models")

    # Get current cache path
    from morgan.utils.model_cache import get_model_cache_path
    path = get_model_cache_path()
"""

import os
from pathlib import Path
from typing import Optional

from morgan.utils.logger import get_logger

logger = get_logger(__name__)

# Global cache path
_model_cache_path: Optional[Path] = None


def setup_model_cache(
    cache_dir: Optional[str] = None,
    load_dotenv: bool = True,
) -> Path:
    """
    Setup model cache directories and environment variables.

    This ensures models are downloaded once and reused on subsequent starts.
    Also configures HF_TOKEN for downloading gated models if available.

    Args:
        cache_dir: Base directory for model cache. Defaults to ~/.morgan/models
                  or MORGAN_MODEL_CACHE environment variable.
        load_dotenv: Whether to load .env file

    Returns:
        Path to the cache directory

    Environment variables set:
        - SENTENCE_TRANSFORMERS_HOME: For sentence-transformers models
        - HF_HOME: For Hugging Face models
        - TRANSFORMERS_CACHE: For transformers models
        - HF_DATASETS_CACHE: For datasets
        - HF_TOKEN: For authenticated downloads (if available)
    """
    global _model_cache_path

    # Load .env file if requested
    if load_dotenv:
        try:
            from dotenv import load_dotenv as _load_dotenv

            _load_dotenv()
            logger.debug("Loaded environment from .env file")
        except ImportError:
            pass

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get("MORGAN_MODEL_CACHE", "~/.morgan/models")

    cache_path = Path(cache_dir).expanduser()

    # Create subdirectories
    sentence_transformers_path = cache_path / "sentence-transformers"
    hf_path = cache_path / "huggingface"
    datasets_path = hf_path / "datasets"

    for path in [cache_path, sentence_transformers_path, hf_path, datasets_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Set environment variables for model caching
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_path)
    os.environ["HF_HOME"] = str(hf_path)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_path)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_path)

    # Configure HF_TOKEN for gated model downloads
    # Check multiple possible env var names
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    if hf_token:
        # Set all possible HF token env vars for compatibility
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        logger.info("HF_TOKEN configured for authenticated model downloads")
    else:
        logger.debug("No HF_TOKEN found - some gated models may not be accessible")

    # Store global cache path
    _model_cache_path = cache_path

    logger.info("Model cache configured at %s", cache_path)
    return cache_path


def get_model_cache_path() -> Optional[Path]:
    """
    Get the current model cache path.

    Returns:
        Path to model cache directory, or None if not yet configured
    """
    global _model_cache_path
    return _model_cache_path


def get_sentence_transformers_path() -> Optional[Path]:
    """Get path to sentence-transformers cache."""
    if _model_cache_path:
        return _model_cache_path / "sentence-transformers"
    return None


def get_huggingface_path() -> Optional[Path]:
    """Get path to Hugging Face cache."""
    if _model_cache_path:
        return _model_cache_path / "huggingface"
    return None


def clear_model_cache(confirm: bool = False) -> bool:
    """
    Clear all cached models.

    WARNING: This will delete all downloaded models!

    Args:
        confirm: Must be True to actually clear the cache

    Returns:
        True if cache was cleared, False otherwise
    """
    if not confirm:
        logger.warning("clear_model_cache called without confirm=True")
        return False

    if _model_cache_path is None:
        logger.warning("Model cache not configured")
        return False

    import shutil

    try:
        if _model_cache_path.exists():
            shutil.rmtree(_model_cache_path)
            _model_cache_path.mkdir(parents=True, exist_ok=True)
            logger.info("Model cache cleared at %s", _model_cache_path)
            return True
    except Exception as e:
        logger.error("Failed to clear model cache: %s", e)

    return False
