"""
OCR Service using DeepSeek-OCR via Ollama.

Provides token-efficient optical character recognition using the
DeepSeek-OCR vision-language model hosted locally via Ollama.

Reference: https://ollama.com/library/deepseek-ocr
Requires: Ollama v0.13.0+ with deepseek-ocr model

Example:
    >>> from morgan.services.ocr_service import get_ocr_service
    >>>
    >>> service = get_ocr_service()
    >>> text = await service.extract_text("path/to/image.png")
"""

import asyncio
import base64
import io
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)

# Check PIL availability
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image processing limited")


class OCRMode(str, Enum):
    """OCR extraction modes."""

    FREE = "free"  # Free-form OCR
    GROUNDING = "grounding"  # With layout awareness
    MARKDOWN = "markdown"  # Convert to markdown
    FIGURE = "figure"  # Parse figures/charts


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    text: str
    mode: OCRMode
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class OCRServiceError(Exception):
    """Raised when OCR service encounters an error."""


class OCRModelNotAvailableError(OCRServiceError):
    """Raised when DeepSeek-OCR model is not available."""


class OCRService:
    """
    OCR Service using DeepSeek-OCR via Ollama.

    DeepSeek-OCR is a vision-language model that performs token-efficient
    optical character recognition. It supports multiple modes:
    - Free OCR: General text extraction
    - Grounding: Layout-aware extraction
    - Markdown: Convert documents to markdown
    - Figure parsing: Extract text from charts/figures

    Example:
        >>> service = OCRService()
        >>> result = await service.extract_text("document.png")
        >>> print(result.text)
    """

    # DeepSeek-OCR prompts (model is sensitive to exact format)
    PROMPTS = {
        OCRMode.FREE: "Free OCR.",
        OCRMode.GROUNDING: "<|grounding|>Given the layout of the image.",
        OCRMode.MARKDOWN: "<|grounding|>Convert the document to markdown.",
        OCRMode.FIGURE: "Parse the figure.",
    }

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: str = "deepseek-ocr",
        timeout: float = 60.0,
    ):
        """
        Initialize OCR service.

        Args:
            ollama_url: Ollama API URL (default from settings)
            model: Model name (default: deepseek-ocr)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.ollama_url = (
            ollama_url
            or getattr(settings, "ollama_url", None)
            or "http://localhost:11434"
        )
        self.model = model
        self.timeout = timeout

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_characters_extracted": 0,
        }

        self._model_verified = False
        logger.info("OCRService initialized (model: %s)", model)

    async def verify_model(self) -> bool:
        """
        Verify that DeepSeek-OCR model is available.

        Returns:
            True if model is available

        Raises:
            OCRModelNotAvailableError: If model not found
        """
        if self._model_verified:
            return True

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]

                    # Check for deepseek-ocr variants
                    for model in models:
                        if "deepseek-ocr" in model.lower():
                            self._model_verified = True
                            logger.info("DeepSeek-OCR model verified: %s", model)
                            return True

                    raise OCRModelNotAvailableError(
                        f"deepseek-ocr not found. Available: {models}. "
                        "Install with: ollama pull deepseek-ocr"
                    )

                raise OCRModelNotAvailableError(
                    f"Failed to list Ollama models: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise OCRModelNotAvailableError(
                f"Cannot connect to Ollama at {self.ollama_url}: {e}"
            ) from e

    async def extract_text(
        self,
        image: Union[str, bytes, Path, "Image.Image"],
        mode: OCRMode = OCRMode.FREE,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Image as file path, bytes, Path, or PIL Image
            mode: OCR extraction mode

        Returns:
            OCRResult with extracted text
        """
        self._stats["total_requests"] += 1
        start_time = asyncio.get_event_loop().time()

        try:
            # Convert image to base64
            image_b64 = await self._prepare_image(image)

            # Get prompt for mode
            prompt = self.PROMPTS.get(mode, self.PROMPTS[OCRMode.FREE])

            # Call Ollama API
            text = await self._call_ollama(image_b64, prompt)

            elapsed = asyncio.get_event_loop().time() - start_time
            self._stats["successful_requests"] += 1
            self._stats["total_characters_extracted"] += len(text)

            logger.debug(
                "OCR extracted %d chars in %.2fs (mode: %s)",
                len(text),
                elapsed,
                mode.value,
            )

            return OCRResult(
                text=text,
                mode=mode,
                success=True,
                processing_time=elapsed,
                metadata={"model": self.model},
            )

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            self._stats["failed_requests"] += 1
            logger.error("OCR extraction failed: %s", e)

            return OCRResult(
                text="",
                mode=mode,
                success=False,
                error=str(e),
                processing_time=elapsed,
            )

    async def extract_text_batch(
        self,
        images: List[Union[str, bytes, Path]],
        mode: OCRMode = OCRMode.FREE,
    ) -> List[OCRResult]:
        """
        Extract text from multiple images.

        Args:
            images: List of images
            mode: OCR extraction mode

        Returns:
            List of OCRResults
        """
        tasks = [self.extract_text(img, mode) for img in images]
        return await asyncio.gather(*tasks)

    async def extract_to_markdown(
        self,
        image: Union[str, bytes, Path, "Image.Image"],
    ) -> OCRResult:
        """
        Extract document and convert to markdown format.

        Args:
            image: Document image

        Returns:
            OCRResult with markdown-formatted text
        """
        return await self.extract_text(image, OCRMode.MARKDOWN)

    async def extract_with_layout(
        self,
        image: Union[str, bytes, Path, "Image.Image"],
    ) -> OCRResult:
        """
        Extract text with layout/grounding awareness.

        Args:
            image: Image with structured layout

        Returns:
            OCRResult with layout-aware text
        """
        return await self.extract_text(image, OCRMode.GROUNDING)

    async def _prepare_image(
        self, image: Union[str, bytes, Path, "Image.Image"]
    ) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            if not image.exists():
                raise OCRServiceError(f"Image not found: {image}")

            with open(image, "rb") as f:
                image_bytes = f.read()

        elif isinstance(image, bytes):
            image_bytes = image

        elif PIL_AVAILABLE and isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        else:
            raise OCRServiceError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    async def _call_ollama(self, image_b64: str, prompt: str) -> str:
        """Call Ollama API with image."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 8192,
                        },
                    },
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise OCRServiceError(
                        f"Ollama API error ({response.status_code}): {error_text}"
                    )

                data = response.json()
                return data.get("response", "").strip()

        except httpx.TimeoutException as e:
            raise OCRServiceError(f"OCR request timed out: {e}") from e
        except httpx.RequestError as e:
            raise OCRServiceError(f"OCR request failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "model_verified": self._model_verified,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_service_instance: Optional[OCRService] = None
_service_lock = threading.Lock()


def get_ocr_service(
    ollama_url: Optional[str] = None,
    model: str = "deepseek-ocr",
) -> OCRService:
    """
    Get singleton OCR service instance.

    Args:
        ollama_url: Ollama API URL
        model: Model name

    Returns:
        Shared OCRService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = OCRService(
                    ollama_url=ollama_url,
                    model=model,
                )

    return _service_instance


# =============================================================================
# Convenience Functions
# =============================================================================


async def extract_text(
    image: Union[str, bytes, Path],
    mode: OCRMode = OCRMode.FREE,
) -> str:
    """
    Convenience function for text extraction.

    Args:
        image: Image to process
        mode: OCR mode

    Returns:
        Extracted text or empty string on error
    """
    service = get_ocr_service()
    result = await service.extract_text(image, mode)
    return result.text if result.success else ""


async def extract_to_markdown(image: Union[str, bytes, Path]) -> str:
    """
    Convenience function for markdown extraction.

    Args:
        image: Document image

    Returns:
        Markdown text or empty string on error
    """
    service = get_ocr_service()
    result = await service.extract_to_markdown(image)
    return result.text if result.success else ""


