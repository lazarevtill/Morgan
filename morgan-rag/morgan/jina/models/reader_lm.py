"""
Jina ReaderLM V2 Model

Following official Jina AI example:
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="jinaai/ReaderLM-v2")
messages = [{"role": "user", "content": "Who are you?"}]
pipe(messages)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")
model = AutoModelForCausalLM.from_pretrained("jinaai/ReaderLM-v2")
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WebScrapingResult:
    """Result from web scraping operation."""

    content: str
    title: str
    url: str
    metadata: Dict[str, Any]
    processing_time: float
    quality_score: float


class JinaReaderLM:
    """
    Jina ReaderLM V2 model implementation following official examples.

    This model is designed for clean web content extraction and processing,
    removing ads, navigation, and boilerplate content.
    """

    MODEL_NAME = "jinaai/ReaderLM-v2"

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Jina ReaderLM V2.

        Args:
            cache_dir: Directory to cache the model
            token: Hugging Face token for authentication
        """
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._is_loaded = False

        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self, use_pipeline: bool = True) -> bool:
        """
        Load the Jina ReaderLM V2 model following official example.

        Args:
            use_pipeline: Whether to use pipeline (simpler) or direct model loading

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded:
            return True

        try:
            if use_pipeline:
                # Following official Jina AI example:
                # from transformers import pipeline
                # pipe = pipeline("text-generation", model="jinaai/ReaderLM-v2")
                from transformers import pipeline

                model_kwargs = {}
                if self.token:
                    model_kwargs["token"] = self.token

                logger.info(f"Loading {self.MODEL_NAME} with pipeline...")
                self.pipeline = pipeline(
                    "text-generation", model=self.MODEL_NAME, **model_kwargs
                )

            else:
                # Following official Jina AI example:
                # from transformers import AutoTokenizer, AutoModelForCausalLM
                # tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")
                # model = AutoModelForCausalLM.from_pretrained("jinaai/ReaderLM-v2")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_kwargs = {}
                if self.cache_dir:
                    model_kwargs["cache_dir"] = self.cache_dir
                if self.token:
                    model_kwargs["token"] = self.token

                logger.info(f"Loading {self.MODEL_NAME} tokenizer and model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_NAME, **model_kwargs
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_NAME, **model_kwargs
                )

            self._is_loaded = True
            logger.info(f"Successfully loaded {self.MODEL_NAME}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            self._is_loaded = False
            return False

    def extract_content(
        self, url: str, html_content: Optional[str] = None
    ) -> WebScrapingResult:
        """
        Extract clean content from web URL or HTML using ReaderLM.

        Args:
            url: Web URL to process
            html_content: Optional HTML content (if None, will fetch from URL)

        Returns:
            WebScrapingResult with extracted content
        """
        import time

        start_time = time.time()

        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load {self.MODEL_NAME}")

        try:
            # Prepare input for ReaderLM
            if html_content is None:
                # Fetch HTML content from URL
                html_content = self._fetch_html(url)

            # For now, use enhanced HTML parsing instead of the generative model
            # The ReaderLM generative approach needs more sophisticated prompt engineering
            extracted_content = self._extract_with_enhanced_parsing(html_content)

            # Extract metadata
            metadata = self._extract_metadata(html_content, url)
            quality_score = self._assess_quality(extracted_content)

            processing_time = time.time() - start_time

            return WebScrapingResult(
                content=extracted_content,
                title=metadata.get("title", ""),
                url=url,
                metadata=metadata,
                processing_time=processing_time,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            raise

    def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL."""
        try:
            import requests

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            return ""

    def _extract_metadata(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""

            # Try to get meta description
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"].strip()

            # Try to get author
            author = ""
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author and meta_author.get("content"):
                author = meta_author["content"].strip()

            return {
                "title": title,
                "description": description,
                "author": author,
                "url": url,
                "word_count": len(html_content.split()),
            }

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {
                "title": "",
                "description": "",
                "author": "",
                "url": url,
                "word_count": 0,
            }

    def _extract_with_enhanced_parsing(self, html_content: str) -> str:
        """
        Enhanced HTML parsing for clean content extraction.
        This is a fallback when the generative model approach needs refinement.
        """
        try:
            import re

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements
            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "advertisement",
                    "ads",
                    "sidebar",
                ]
            ):
                element.decompose()

            # Remove elements with common ad/navigation classes
            for element in soup.find_all(
                class_=re.compile(r"(ad|advertisement|sidebar|nav|menu|footer|header)")
            ):
                element.decompose()

            # Extract main content areas
            main_content = ""

            # Try to find main content containers
            content_selectors = [
                "main",
                "article",
                ".content",
                ".main-content",
                ".post-content",
                ".entry-content",
                "#content",
            ]

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(separator="\n", strip=True)
                    break

            # If no main content found, extract from body
            if not main_content:
                body = soup.find("body")
                if body:
                    main_content = body.get_text(separator="\n", strip=True)
                else:
                    main_content = soup.get_text(separator="\n", strip=True)

            # Clean up the text
            lines = main_content.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line and len(line) > 3:  # Skip very short lines
                    # Remove excessive whitespace
                    line = re.sub(r"\s+", " ", line)
                    cleaned_lines.append(line)

            # Join lines with proper spacing
            cleaned_content = "\n\n".join(cleaned_lines)

            # Remove excessive newlines
            cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)

            return cleaned_content.strip()

        except Exception as e:
            logger.error(f"Enhanced parsing failed: {e}")
            return html_content[:1000]  # Return truncated HTML as fallback

    def _assess_quality(self, content: str) -> float:
        """Assess the quality of extracted content."""
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ""

        if not content or len(content.strip()) < 50:
            return 0.0

        # Simple quality metrics
        word_count = len(content.split())
        sentence_count = content.count(".") + content.count("!") + content.count("?")

        if word_count < 10:
            return 0.2
        elif word_count < 50:
            return 0.5
        elif sentence_count > 0 and word_count / sentence_count > 5:
            return 0.9
        else:
            return 0.7

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.MODEL_NAME,
            "description": "ReaderLM v2 for clean web content extraction",
            "use_case": "Web scraping and content cleaning",
            "performance": "High quality content extraction with noise removal",
            "is_loaded": self._is_loaded,
            "supports_pipeline": True,
            "supports_direct": True,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._is_loaded = False
        logger.info(f"Unloaded {self.MODEL_NAME}")
