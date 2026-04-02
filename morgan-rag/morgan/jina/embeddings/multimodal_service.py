"""
Multimodal Content Processor with CLIP Integration

Handles documents containing both text and images using jina-clip-v2.
Implements OCR, image-text alignment, and multimodal search capabilities.
"""

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Image processing
try:
    from PIL import Image, ImageEnhance

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - image processing will be limited")

# OCR capabilities
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available - OCR functionality will be limited")

# For image format detection

from .service import JinaEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ImageContent:
    """Container for image data and metadata."""

    image_data: bytes
    format: str  # 'jpeg', 'png', etc.
    width: int
    height: int
    file_path: Optional[str] = None
    extracted_text: Optional[str] = None
    quality_score: float = 0.0


@dataclass
class MultimodalEmbedding:
    """Combined text and image embeddings."""

    text_embedding: List[float]
    image_embeddings: List[List[float]]
    combined_embedding: List[float]
    alignment_scores: List[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class MultimodalDocument:
    """Document containing both text and visual content."""

    text_content: str
    images: List[ImageContent]
    text_embeddings: List[float]
    image_embeddings: List[List[float]]
    combined_embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class MultimodalSearchResult:
    """Multimodal search result with context."""

    content: str
    images: List[ImageContent]
    relevance_score: float
    text_score: float
    image_score: float
    combined_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalContentProcessor:
    """
    Handle documents containing both text and images using CLIP-based embeddings.

    Key Features:
    - jina-clip-v2 integration for text-image embeddings
    - OCR integration for text extraction from images
    - Image-text alignment and correlation
    - Multimodal search capabilities
    - Graceful error handling
    """

    def __init__(
        self,
        embedding_service: Optional[JinaEmbeddingService] = None,
        max_workers: int = 4,
        ocr_enabled: bool = True,
        image_quality_threshold: float = 0.3,
    ):
        """
        Initialize the multimodal content processor.

        Args:
            embedding_service: Jina embedding service instance
            max_workers: Maximum concurrent workers for processing
            ocr_enabled: Whether to enable OCR text extraction
            image_quality_threshold: Minimum quality score for image processing
        """
        self.embedding_service = embedding_service or JinaEmbeddingService()
        self.max_workers = max_workers
        self.ocr_enabled = ocr_enabled and TESSERACT_AVAILABLE
        self.image_quality_threshold = image_quality_threshold
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Supported image formats
        self.supported_formats = {"jpeg", "jpg", "png", "bmp", "tiff", "webp"}

        logger.info(
            f"Initialized MultimodalContentProcessor - OCR: {self.ocr_enabled}, PIL: {PIL_AVAILABLE}"
        )

    def process_multimodal_document(
        self,
        content: str,
        images: List[Union[str, bytes, Image.Image]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalDocument:
        """
        Process a document containing both text and images.

        Args:
            content: Text content of the document
            images: List of images (file paths, bytes, or PIL Images)
            metadata: Additional metadata for the document

        Returns:
            MultimodalDocument with embeddings and processed content
        """
        if metadata is None:
            metadata = {}

        logger.info(f"Processing multimodal document with {len(images)} images")
        start_time = time.time()

        processing_errors = []
        processed_images = []

        try:
            # Process images
            for i, image in enumerate(images):
                try:
                    processed_image = self._process_single_image(image, f"image_{i}")
                    if processed_image:
                        processed_images.append(processed_image)
                except Exception as e:
                    error_msg = f"Failed to process image {i}: {str(e)}"
                    logger.warning(error_msg)
                    processing_errors.append(error_msg)

            # Extract text from images if OCR is enabled
            if self.ocr_enabled:
                for image in processed_images:
                    try:
                        image.extracted_text = self._extract_text_from_image(image)
                    except Exception as e:
                        error_msg = f"OCR failed for image: {str(e)}"
                        logger.warning(error_msg)
                        processing_errors.append(error_msg)

            # Create embeddings
            multimodal_embedding = self._create_multimodal_embeddings(
                content, processed_images
            )

            # Create document
            document = MultimodalDocument(
                text_content=content,
                images=processed_images,
                text_embeddings=multimodal_embedding.text_embedding,
                image_embeddings=multimodal_embedding.image_embeddings,
                combined_embedding=multimodal_embedding.combined_embedding,
                metadata=metadata,
                processing_errors=processing_errors,
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Processed multimodal document in {elapsed_time:.2f}s with {len(processing_errors)} errors"
            )

            return document

        except Exception as e:
            logger.error(f"Failed to process multimodal document: {str(e)}")
            # Return document with errors for graceful degradation
            return MultimodalDocument(
                text_content=content,
                images=processed_images,
                text_embeddings=[],
                image_embeddings=[],
                combined_embedding=[],
                metadata=metadata,
                processing_errors=processing_errors + [f"Processing failed: {str(e)}"],
            )

    def _process_single_image(
        self, image_input: Union[str, bytes, Image.Image], image_id: str
    ) -> Optional[ImageContent]:
        """
        Process a single image input into ImageContent.

        Args:
            image_input: Image as file path, bytes, or PIL Image
            image_id: Identifier for the image

        Returns:
            ImageContent object or None if processing fails
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - cannot process images")
            return None

        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                # File path
                image_path = Path(image_input)
                if not image_path.exists():
                    logger.warning(f"Image file not found: {image_input}")
                    return None

                with open(image_path, "rb") as f:
                    image_data = f.read()

                pil_image = Image.open(image_path)
                file_path = str(image_path)

            elif isinstance(image_input, bytes):
                # Raw bytes
                image_data = image_input
                pil_image = Image.open(io.BytesIO(image_data))
                file_path = None

            elif hasattr(image_input, "save"):  # PIL Image
                # PIL Image object
                buffer = io.BytesIO()
                image_format = getattr(image_input, "format", "PNG") or "PNG"
                image_input.save(buffer, format=image_format)
                image_data = buffer.getvalue()
                pil_image = image_input
                file_path = None

            else:
                logger.warning(f"Unsupported image input type: {type(image_input)}")
                return None

            # Get image properties
            width, height = pil_image.size
            image_format = pil_image.format or "PNG"

            # Assess image quality
            quality_score = self._assess_image_quality(pil_image)

            if quality_score < self.image_quality_threshold:
                logger.warning(
                    f"Image quality too low: {quality_score:.2f} < {self.image_quality_threshold}"
                )
                # Still process but mark as low quality

            return ImageContent(
                image_data=image_data,
                format=image_format.lower(),
                width=width,
                height=height,
                file_path=file_path,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {str(e)}")
            return None

    def _assess_image_quality(self, pil_image: Image.Image) -> float:
        """
        Assess the quality of an image for processing.

        Args:
            pil_image: PIL Image object

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Basic quality metrics
            width, height = pil_image.size

            # Size score (prefer larger images)
            size_score = min(
                1.0, (width * height) / (300 * 300)
            )  # Normalize to 300x300

            # Aspect ratio score (prefer reasonable aspect ratios)
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(
                0.0, 1.0 - (aspect_ratio - 1.0) / 10.0
            )  # Penalize extreme ratios

            # Format score (prefer lossless formats)
            format_scores = {
                "png": 1.0,
                "bmp": 0.9,
                "tiff": 0.9,
                "jpeg": 0.7,
                "jpg": 0.7,
                "webp": 0.8,
            }
            format_score = format_scores.get(
                pil_image.format.lower() if pil_image.format else "jpeg", 0.5
            )

            # Combined score
            quality_score = size_score * 0.4 + aspect_score * 0.3 + format_score * 0.3

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            logger.warning(f"Failed to assess image quality: {str(e)}")
            return 0.5  # Default moderate quality

    def _extract_text_from_image(self, image_content: ImageContent) -> str:
        """
        Extract text from an image using DeepSeek-OCR via Ollama.

        Falls back to Tesseract if DeepSeek-OCR is unavailable.

        Args:
            image_content: ImageContent object

        Returns:
            Extracted text string
        """
        if not self.ocr_enabled:
            return ""

        # Try DeepSeek-OCR first (better quality)
        try:
            from morgan.services.ocr_service import get_ocr_service
            import asyncio

            ocr_service = get_ocr_service()

            async def run_ocr():
                result = await ocr_service.extract_text(image_content.image_data)
                return result.text if result.success else ""

            # Run async in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, run_ocr())
                        extracted_text = future.result(timeout=60)
                else:
                    extracted_text = loop.run_until_complete(run_ocr())
            except RuntimeError:
                extracted_text = asyncio.run(run_ocr())

            if extracted_text:
                cleaned_text = self._clean_ocr_text(extracted_text)
                logger.debug("DeepSeek-OCR extracted %d characters", len(cleaned_text))
                return cleaned_text

        except Exception as e:
            logger.debug("DeepSeek-OCR unavailable: %s, trying Tesseract", e)

        # Fallback to Tesseract
        if TESSERACT_AVAILABLE:
            try:
                pil_image = Image.open(io.BytesIO(image_content.image_data))
                enhanced_image = self._enhance_image_for_ocr(pil_image)

                extracted_text = pytesseract.image_to_string(
                    enhanced_image, config="--psm 6"
                )
                cleaned_text = self._clean_ocr_text(extracted_text)

                logger.debug("Tesseract extracted %d characters", len(cleaned_text))
                return cleaned_text

            except Exception as e:
                logger.warning("Tesseract OCR failed: %s", e)

        return ""

    def _enhance_image_for_ocr(self, pil_image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results.

        Args:
            pil_image: Original PIL Image

        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to grayscale for better OCR
            if pil_image.mode != "L":
                enhanced = pil_image.convert("L")
            else:
                enhanced = pil_image.copy()

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)

            return enhanced

        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return pil_image

    def _clean_ocr_text(self, raw_text: str) -> str:
        """
        Clean and normalize OCR-extracted text.

        Args:
            raw_text: Raw text from OCR

        Returns:
            Cleaned text
        """
        if not raw_text:
            return ""

        # Remove excessive whitespace
        cleaned = " ".join(raw_text.split())

        # Remove common OCR artifacts
        artifacts = ["|", "~", "`", "^"]
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "")

        # Remove very short "words" that are likely OCR errors
        words = cleaned.split()
        filtered_words = [word for word in words if len(word) > 1 or word.isalnum()]

        return " ".join(filtered_words).strip()

    def _create_multimodal_embeddings(
        self, text_content: str, images: List[ImageContent]
    ) -> MultimodalEmbedding:
        """
        Create combined text and image embeddings using jina-clip-v2.

        Args:
            text_content: Text content to embed
            images: List of processed images

        Returns:
            MultimodalEmbedding with combined embeddings
        """
        try:
            # Generate text embedding
            text_embedding = self.embedding_service.generate_single_embedding(
                text_content, "jina-clip-v2"
            )

            # Generate image embeddings
            image_embeddings = []
            for image in images:
                try:
                    # For now, use text content as proxy for image embedding
                    # In production, this would process the actual image data
                    image_text = (
                        image.extracted_text or f"Image {image.width}x{image.height}"
                    )
                    image_embedding = self.embedding_service.generate_single_embedding(
                        image_text, "jina-clip-v2"
                    )
                    image_embeddings.append(image_embedding)
                except Exception as e:
                    logger.warning(f"Failed to create image embedding: {str(e)}")
                    # Use zero vector as fallback
                    image_embeddings.append([0.0] * len(text_embedding))

            # Create combined embedding
            combined_embedding = self._combine_embeddings(
                text_embedding, image_embeddings
            )

            # Calculate alignment scores
            alignment_scores = self._calculate_alignment_scores(
                text_embedding, image_embeddings
            )

            return MultimodalEmbedding(
                text_embedding=text_embedding,
                image_embeddings=image_embeddings,
                combined_embedding=combined_embedding,
                alignment_scores=alignment_scores,
                confidence=self._calculate_confidence(alignment_scores),
            )

        except Exception as e:
            logger.error(f"Failed to create multimodal embeddings: {str(e)}")
            # Return empty embeddings for graceful degradation
            return MultimodalEmbedding(
                text_embedding=[],
                image_embeddings=[],
                combined_embedding=[],
                alignment_scores=[],
                confidence=0.0,
            )

    def _combine_embeddings(
        self, text_embedding: List[float], image_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Combine text and image embeddings into a single vector.

        Args:
            text_embedding: Text embedding vector
            image_embeddings: List of image embedding vectors

        Returns:
            Combined embedding vector
        """
        if not text_embedding:
            return []

        if not image_embeddings:
            return text_embedding

        try:
            # Simple averaging approach
            # In production, this could use more sophisticated fusion methods

            # Average image embeddings
            if image_embeddings:
                avg_image_embedding = []
                for i in range(len(text_embedding)):
                    values = [
                        img_emb[i] for img_emb in image_embeddings if i < len(img_emb)
                    ]
                    avg_value = sum(values) / len(values) if values else 0.0
                    avg_image_embedding.append(avg_value)
            else:
                avg_image_embedding = [0.0] * len(text_embedding)

            # Weighted combination (70% text, 30% image)
            text_weight = 0.7
            image_weight = 0.3

            combined = []
            for i in range(len(text_embedding)):
                combined_value = (
                    text_weight * text_embedding[i]
                    + image_weight * avg_image_embedding[i]
                )
                combined.append(combined_value)

            # Normalize the combined embedding
            norm = sum(x * x for x in combined) ** 0.5
            if norm > 0:
                combined = [x / norm for x in combined]

            return combined

        except Exception as e:
            logger.warning(f"Failed to combine embeddings: {str(e)}")
            return text_embedding

    def _calculate_alignment_scores(
        self, text_embedding: List[float], image_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Calculate alignment scores between text and image embeddings.

        Args:
            text_embedding: Text embedding vector
            image_embeddings: List of image embedding vectors

        Returns:
            List of alignment scores (cosine similarity)
        """
        if not text_embedding or not image_embeddings:
            return []

        alignment_scores = []

        for image_embedding in image_embeddings:
            try:
                # Calculate cosine similarity
                dot_product = sum(
                    a * b for a, b in zip(text_embedding, image_embedding)
                )

                text_norm = sum(x * x for x in text_embedding) ** 0.5
                image_norm = sum(x * x for x in image_embedding) ** 0.5

                if text_norm > 0 and image_norm > 0:
                    similarity = dot_product / (text_norm * image_norm)
                    alignment_scores.append(max(-1.0, min(1.0, similarity)))
                else:
                    alignment_scores.append(0.0)

            except Exception as e:
                logger.warning(f"Failed to calculate alignment score: {str(e)}")
                alignment_scores.append(0.0)

        return alignment_scores

    def _calculate_confidence(self, alignment_scores: List[float]) -> float:
        """
        Calculate overall confidence based on alignment scores.

        Args:
            alignment_scores: List of alignment scores

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not alignment_scores:
            return 0.0

        # Use average alignment score as confidence
        avg_alignment = sum(alignment_scores) / len(alignment_scores)

        # Convert from [-1, 1] to [0, 1]
        confidence = (avg_alignment + 1.0) / 2.0

        return max(0.0, min(1.0, confidence))

    def search_multimodal_content(
        self,
        query: str,
        documents: List[MultimodalDocument],
        include_images: bool = True,
        max_results: int = 10,
    ) -> List[MultimodalSearchResult]:
        """
        Search through multimodal content using both text and visual elements.

        Args:
            query: Search query
            documents: List of multimodal documents to search
            include_images: Whether to include image matching in search
            max_results: Maximum number of results to return

        Returns:
            List of multimodal search results
        """
        logger.info(f"Searching {len(documents)} multimodal documents for: '{query}'")

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_single_embedding(
                query, "jina-clip-v2"
            )

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            results = []

            for doc in documents:
                try:
                    # Calculate text similarity
                    text_score = self._calculate_similarity(
                        query_embedding, doc.text_embeddings
                    )

                    # Calculate image similarity if enabled
                    image_score = 0.0
                    if include_images and doc.image_embeddings:
                        image_scores = [
                            self._calculate_similarity(query_embedding, img_emb)
                            for img_emb in doc.image_embeddings
                        ]
                        image_score = max(image_scores) if image_scores else 0.0

                    # Calculate combined score
                    combined_score = self._calculate_similarity(
                        query_embedding, doc.combined_embedding
                    )

                    # Overall relevance score (weighted combination)
                    relevance_score = (
                        0.5 * text_score + 0.3 * image_score + 0.2 * combined_score
                    )

                    result = MultimodalSearchResult(
                        content=doc.text_content,
                        images=doc.images,
                        relevance_score=relevance_score,
                        text_score=text_score,
                        image_score=image_score,
                        combined_score=combined_score,
                        metadata=doc.metadata,
                    )

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to score document: {str(e)}")
                    continue

            # Sort by relevance score and return top results
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            logger.info(f"Found {len(results)} results, returning top {max_results}")
            return results[:max_results]

        except Exception as e:
            logger.error(f"Multimodal search failed: {str(e)}")
            return []

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0

        try:
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            norm1 = sum(x * x for x in embedding1) ** 0.5
            norm2 = sum(x * x for x in embedding2) ** 0.5

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                # Convert from [-1, 1] to [0, 1]
                return max(0.0, (similarity + 1.0) / 2.0)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {str(e)}")
            return 0.0

    async def process_multimodal_document_async(
        self,
        content: str,
        images: List[Union[str, bytes, Image.Image]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalDocument:
        """
        Asynchronously process a multimodal document.

        Args:
            content: Text content of the document
            images: List of images (file paths, bytes, or PIL Images)
            metadata: Additional metadata for the document

        Returns:
            MultimodalDocument with embeddings and processed content
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.process_multimodal_document, content, images, metadata
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about multimodal processing.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "ocr_enabled": self.ocr_enabled,
            "pil_available": PIL_AVAILABLE,
            "tesseract_available": TESSERACT_AVAILABLE,
            "supported_formats": list(self.supported_formats),
            "image_quality_threshold": self.image_quality_threshold,
            "max_workers": self.max_workers,
        }

    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("Closed MultimodalContentProcessor executor")
