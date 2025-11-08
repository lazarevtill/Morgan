"""
Hierarchical Embedding Service for Morgan RAG.

Implements three-scale hierarchical embeddings inspired by Matryoshka architecture:
- Coarse: Category + high-level topics (fast filtering, 90% candidate reduction)
- Medium: Section headers + key concepts (pattern matching)
- Fine: Full content with complete context (precise retrieval)

Features:
- Category-aware text construction for different scales
- Batch processing optimization
- Integration with existing embedding service
- Performance tracking and monitoring
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from morgan.services.embedding_service import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.utils.request_context import get_request_id, set_request_id
from morgan.vectorization.contrastive_clustering import (
    get_contrastive_clustering_engine,
)

logger = get_logger(__name__)


@dataclass
class HierarchicalEmbedding:
    """Multi-scale embedding representation."""

    coarse: List[float]
    medium: List[float]
    fine: List[float]
    texts: Dict[str, str]  # Original texts for each scale
    metadata: Dict[str, Any]

    def get_embedding(self, scale: str) -> List[float]:
        """Get embedding for specific scale."""
        if scale == "coarse":
            return self.coarse
        elif scale == "medium":
            return self.medium
        elif scale == "fine":
            return self.fine
        else:
            raise ValueError(
                f"Invalid scale: {scale}. Must be 'coarse', 'medium', or 'fine'"
            )

    def get_text(self, scale: str) -> str:
        """Get text used for specific scale."""
        return self.texts.get(scale, "")


class HierarchicalEmbeddingService:
    """
    Service for generating hierarchical embeddings with three scales.

    Implements Matryoshka-inspired architecture for efficient multi-stage search:
    - Coarse embeddings for fast category filtering (90% reduction)
    - Medium embeddings for section-level pattern matching
    - Fine embeddings for precise content retrieval
    """

    def __init__(self):
        """Initialize hierarchical embedding service."""
        self.embedding_service = get_embedding_service()
        self.clustering_engine = get_contrastive_clustering_engine()

        # Category mappings for text construction
        self.category_keywords = {
            "code": ["function", "class", "method", "variable", "implementation"],
            "documentation": [
                "guide",
                "tutorial",
                "reference",
                "manual",
                "documentation",
            ],
            "api": ["endpoint", "request", "response", "parameter", "authentication"],
            "configuration": [
                "config",
                "setting",
                "environment",
                "deployment",
                "setup",
            ],
            "troubleshooting": ["error", "issue", "problem", "solution", "debug"],
            "general": ["information", "content", "data", "text", "document"],
        }

        logger.info(
            "HierarchicalEmbeddingService initialized with contrastive clustering"
        )

    def create_hierarchical_embeddings(
        self,
        content: str,
        metadata: Dict[str, Any],
        category: Optional[str] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> HierarchicalEmbedding:
        """
        Create hierarchical embeddings for content.

        Args:
            content: Full text content to embed
            metadata: Document metadata (title, source, etc.)
            category: Content category for text construction
            use_cache: Whether to use embedding cache
            request_id: Optional request ID for tracing

        Returns:
            HierarchicalEmbedding with coarse, medium, and fine embeddings

        Example:
            >>> service = HierarchicalEmbeddingService()
            >>> metadata = {"title": "Docker Guide", "source": "docs/docker.md"}
            >>> embedding = service.create_hierarchical_embeddings(
            ...     content="Docker is a containerization platform...",
            ...     metadata=metadata,
            ...     category="documentation"
            ... )
            >>> coarse_emb = embedding.get_embedding("coarse")
        """
        # Get or generate request ID for tracing
        if request_id is None:
            request_id = get_request_id() or set_request_id()

        # Auto-detect category if not provided
        if category is None:
            category = self._detect_category(content, metadata)

        logger.debug(
            f"Creating hierarchical embeddings for content (length={len(content)}, "
            f"category={category}, request_id={request_id})"
        )

        start_time = time.time()

        # Build texts for each scale
        coarse_text = self.build_coarse_text(content, category, metadata)
        medium_text = self.build_medium_text(content, category, metadata)
        fine_text = self.build_fine_text(content, category, metadata)

        # Generate embeddings for each scale
        texts_to_embed = [coarse_text, medium_text, fine_text]
        embeddings = self.embedding_service.encode_batch(
            texts_to_embed,
            instruction="document",
            show_progress=False,
            use_cache=use_cache,
            request_id=request_id,
        )

        # Apply contrastive clustering to each embedding scale
        coarse_embedding = self.clustering_engine.apply_contrastive_bias(
            embeddings[0], category, "coarse"
        )
        medium_embedding = self.clustering_engine.apply_contrastive_bias(
            embeddings[1], category, "medium"
        )
        fine_embedding = self.clustering_engine.apply_contrastive_bias(
            embeddings[2], category, "fine"
        )

        # Create hierarchical embedding object
        hierarchical_embedding = HierarchicalEmbedding(
            coarse=coarse_embedding,
            medium=medium_embedding,
            fine=fine_embedding,
            texts={"coarse": coarse_text, "medium": medium_text, "fine": fine_text},
            metadata={
                **metadata,
                "category": category,
                "content_length": len(content),
                "processing_time": time.time() - start_time,
                "request_id": request_id,
            },
        )

        elapsed = time.time() - start_time
        logger.debug(
            f"Created hierarchical embeddings in {elapsed:.3f}s "
            f"(category={category}, request_id={request_id})"
        )

        return hierarchical_embedding

    def build_coarse_text(
        self, content: str, category: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Build coarse-level text for category and topic filtering.

        Focuses on:
        - Document category and type
        - High-level topics and themes
        - Title and section headers
        - Key terminology

        Args:
            content: Full content
            category: Content category
            metadata: Document metadata

        Returns:
            Coarse text for embedding
        """
        coarse_parts = []

        # Add category information
        coarse_parts.append(f"Category: {category}")

        # Add category-specific keywords
        if category in self.category_keywords:
            keywords = ", ".join(self.category_keywords[category])
            coarse_parts.append(f"Keywords: {keywords}")

        # Add title/filename if available
        title = metadata.get("title") or metadata.get("filename", "")
        if title:
            coarse_parts.append(f"Title: {title}")

        # Add source information
        source = metadata.get("source", "")
        if source:
            # Extract meaningful parts from file path
            source_path = Path(source)
            if len(source_path.parts) > 1:
                coarse_parts.append(f"Source: {'/'.join(source_path.parts[-2:])}")
            else:
                coarse_parts.append(f"Source: {source_path.name}")

        # Extract high-level topics from content (first 200 chars)
        content_preview = content[:200].strip()
        if content_preview:
            coarse_parts.append(f"Content: {content_preview}")

        # Add document type indicators
        doc_type = self._detect_document_type(content, metadata)
        if doc_type:
            coarse_parts.append(f"Type: {doc_type}")

        coarse_text = " | ".join(coarse_parts)

        # Limit coarse text length for efficiency
        if len(coarse_text) > 500:
            coarse_text = coarse_text[:500] + "..."

        return coarse_text

    def build_medium_text(
        self, content: str, category: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Build medium-level text for section and concept matching.

        Focuses on:
        - Section headers and structure
        - Key concepts and entities
        - Function/class names (for code)
        - Important terminology

        Args:
            content: Full content
            category: Content category
            metadata: Document metadata

        Returns:
            Medium text for embedding
        """
        medium_parts = []

        # Add title and category
        title = metadata.get("title") or metadata.get("filename", "")
        if title:
            medium_parts.append(f"Title: {title}")

        medium_parts.append(f"Category: {category}")

        # Extract structure based on content type
        if category == "code":
            # Extract function/class definitions
            code_elements = self._extract_code_elements(content)
            if code_elements:
                medium_parts.append(f"Code elements: {', '.join(code_elements[:10])}")

        elif category in ["documentation", "general"]:
            # Extract headers and key sections
            headers = self._extract_headers(content)
            if headers:
                medium_parts.append(f"Sections: {', '.join(headers[:8])}")

        # Add key concepts (extract important terms)
        key_concepts = self._extract_key_concepts(content, category)
        if key_concepts:
            medium_parts.append(f"Concepts: {', '.join(key_concepts[:10])}")

        # Add content summary (middle portion for context)
        content_length = len(content)
        if content_length > 1000:
            # Take middle section for better representation
            start_idx = content_length // 4
            end_idx = start_idx + 800
            content_sample = content[start_idx:end_idx].strip()
        else:
            content_sample = content[:800].strip()

        if content_sample:
            medium_parts.append(f"Content: {content_sample}")

        medium_text = " | ".join(medium_parts)

        # Limit medium text length
        if len(medium_text) > 1200:
            medium_text = medium_text[:1200] + "..."

        return medium_text

    def build_fine_text(
        self, content: str, category: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Build fine-level text for precise content matching.

        Uses full content with minimal processing for maximum precision.

        Args:
            content: Full content
            category: Content category
            metadata: Document metadata

        Returns:
            Fine text for embedding (full content with context)
        """
        fine_parts = []

        # Add minimal context
        title = metadata.get("title") or metadata.get("filename", "")
        if title:
            fine_parts.append(f"Document: {title}")

        # Add full content
        fine_parts.append(content.strip())

        fine_text = "\n".join(fine_parts)

        # Respect model token limits (approximate)
        max_chars = 8000  # Conservative limit for most models
        if len(fine_text) > max_chars:
            # Truncate but try to end at sentence boundary
            truncated = fine_text[:max_chars]
            last_sentence = truncated.rfind(".")
            if last_sentence > max_chars * 0.8:  # If we can find a sentence ending
                fine_text = truncated[: last_sentence + 1]
            else:
                fine_text = truncated + "..."

        return fine_text

    def create_batch_hierarchical_embeddings(
        self,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        categories: Optional[List[str]] = None,
        show_progress: bool = True,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> List[HierarchicalEmbedding]:
        """
        Create hierarchical embeddings for multiple contents in batch.

        More efficient than individual calls due to batch embedding processing.

        Args:
            contents: List of content strings
            metadatas: List of metadata dictionaries
            categories: Optional list of categories (auto-detected if None)
            show_progress: Show progress bar
            use_cache: Whether to use embedding cache
            request_id: Optional request ID for tracing

        Returns:
            List of HierarchicalEmbedding objects
        """
        if not contents:
            return []

        if len(contents) != len(metadatas):
            raise ValueError("Contents and metadatas must have same length")

        # Get or generate request ID for tracing
        if request_id is None:
            request_id = get_request_id() or set_request_id()

        logger.info(
            f"Creating batch hierarchical embeddings for {len(contents)} items "
            f"(request_id={request_id})"
        )

        start_time = time.time()

        # Auto-detect categories if not provided
        if categories is None:
            categories = [
                self._detect_category(content, metadata)
                for content, metadata in zip(contents, metadatas)
            ]

        # Build all texts for all scales
        all_texts = []
        text_mapping = []  # Track which text belongs to which item and scale

        for i, (content, metadata, category) in enumerate(
            zip(contents, metadatas, categories)
        ):
            coarse_text = self.build_coarse_text(content, category, metadata)
            medium_text = self.build_medium_text(content, category, metadata)
            fine_text = self.build_fine_text(content, category, metadata)

            all_texts.extend([coarse_text, medium_text, fine_text])
            text_mapping.extend(
                [
                    (i, "coarse", coarse_text),
                    (i, "medium", medium_text),
                    (i, "fine", fine_text),
                ]
            )

        # Generate all embeddings in one batch
        all_embeddings = self.embedding_service.encode_batch(
            all_texts,
            instruction="document",
            show_progress=show_progress,
            use_cache=use_cache,
            request_id=request_id,
        )

        # Organize embeddings back into hierarchical structure
        hierarchical_embeddings = []

        for i in range(len(contents)):
            # Extract embeddings for this item
            coarse_emb = all_embeddings[i * 3]
            medium_emb = all_embeddings[i * 3 + 1]
            fine_emb = all_embeddings[i * 3 + 2]

            # Apply contrastive clustering to each embedding scale
            category = categories[i]
            coarse_clustered = self.clustering_engine.apply_contrastive_bias(
                coarse_emb, category, "coarse"
            )
            medium_clustered = self.clustering_engine.apply_contrastive_bias(
                medium_emb, category, "medium"
            )
            fine_clustered = self.clustering_engine.apply_contrastive_bias(
                fine_emb, category, "fine"
            )

            # Extract texts for this item
            coarse_text = text_mapping[i * 3][2]
            medium_text = text_mapping[i * 3 + 1][2]
            fine_text = text_mapping[i * 3 + 2][2]

            hierarchical_embedding = HierarchicalEmbedding(
                coarse=coarse_clustered,
                medium=medium_clustered,
                fine=fine_clustered,
                texts={"coarse": coarse_text, "medium": medium_text, "fine": fine_text},
                metadata={
                    **metadatas[i],
                    "category": categories[i],
                    "content_length": len(contents[i]),
                    "request_id": request_id,
                },
            )

            hierarchical_embeddings.append(hierarchical_embedding)

        elapsed = time.time() - start_time
        logger.info(
            f"Created {len(hierarchical_embeddings)} hierarchical embeddings in {elapsed:.3f}s "
            f"({len(hierarchical_embeddings)/elapsed:.1f} items/sec, request_id={request_id})"
        )

        return hierarchical_embeddings

    def _detect_category(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Auto-detect content category based on content and metadata.

        Args:
            content: Content text
            metadata: Document metadata

        Returns:
            Detected category
        """
        # Check content patterns first (more accurate than file extensions)
        content_lower = content.lower()

        # Code indicators (but check if it's primarily code vs documentation with code snippets)
        code_patterns = [
            "def ",
            "function ",
            "class ",
            "import ",
            "from ",
            "const ",
            "var ",
            "let ",
        ]
        code_matches = sum(1 for pattern in code_patterns if pattern in content_lower)

        # If we have many code patterns and it's a code file, it's likely code
        if code_matches >= 2:
            source = metadata.get("source", "")
            if source:
                source_path = Path(source)
                ext = source_path.suffix.lower()
                if ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".java",
                    ".cpp",
                    ".c",
                    ".go",
                    ".rs",
                    ".rb",
                ]:
                    return "code"

        # Single code pattern might just be documentation with examples
        elif code_matches == 1:
            # Check if it's in a documentation context
            doc_indicators = [
                "guide",
                "tutorial",
                "reference",
                "manual",
                "documentation",
                "how to",
                "# ",
                "## ",
            ]
            if any(indicator in content_lower for indicator in doc_indicators):
                # Likely documentation with code examples, continue to other checks
                pass
            else:
                # Likely actual code
                return "code"

        # API documentation indicators
        api_patterns = [
            "endpoint",
            "api",
            "request",
            "response",
            "parameter",
            "authentication",
        ]
        if any(pattern in content_lower for pattern in api_patterns):
            return "api"

        # Troubleshooting indicators (check early to avoid conflicts)
        trouble_patterns = [
            "error:",
            "issue:",
            "problem:",
            "solution:",
            "debug:",
            "troubleshoot",
            "failed",
            "exception",
            "timeout",
        ]
        trouble_matches = sum(
            1 for pattern in trouble_patterns if pattern in content_lower
        )
        if trouble_matches >= 1:
            return "troubleshooting"

        # Configuration indicators (be more specific to avoid conflicts)
        config_patterns = ["config", "configuration file", "environment variable"]
        if any(pattern in content_lower for pattern in config_patterns):
            return "configuration"

        # Check for "setting" but not in error context
        if "setting" in content_lower and "error" not in content_lower:
            return "configuration"

        # Deployment could be configuration or documentation - check context
        if "deployment" in content_lower:
            # If it's a guide/tutorial about deployment, it's documentation
            if any(
                doc_word in content_lower
                for doc_word in ["guide", "tutorial", "how to", "# "]
            ):
                # Continue to documentation check
                pass
            else:
                return "configuration"

        # Check file extension as secondary indicator
        source = metadata.get("source", "")
        if source:
            source_path = Path(source)
            ext = source_path.suffix.lower()

            if ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".rb"]:
                return "code"
            elif ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".conf"]:
                return "configuration"
            elif ext in [".md", ".rst", ".txt", ".doc", ".docx"]:
                # For text files, check for documentation patterns
                doc_patterns = [
                    "guide",
                    "tutorial",
                    "reference",
                    "manual",
                    "documentation",
                    "how to",
                ]
                if any(pattern in content_lower for pattern in doc_patterns):
                    return "documentation"
                # If no specific doc patterns, could still be general documentation
                return "documentation"

        # Default to general
        return "general"

    def _detect_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Detect specific document type for coarse text."""
        source = metadata.get("source", "")
        if source:
            source_path = Path(source)
            ext = source_path.suffix.lower()

            type_mapping = {
                ".py": "Python code",
                ".js": "JavaScript code",
                ".ts": "TypeScript code",
                ".md": "Markdown document",
                ".json": "JSON configuration",
                ".yaml": "YAML configuration",
                ".yml": "YAML configuration",
                ".txt": "Text document",
                ".rst": "reStructuredText document",
            }

            return type_mapping.get(ext, "Document")

        return "Document"

    def _extract_code_elements(self, content: str) -> List[str]:
        """Extract function and class names from code content."""
        elements = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Python/JavaScript function definitions
            if line.startswith("def ") or line.startswith("function "):
                # Extract function name
                if "(" in line:
                    func_part = line.split("(")[0]
                    if "def " in func_part:
                        func_name = func_part.split("def ")[-1].strip()
                    elif "function " in func_part:
                        func_name = func_part.split("function ")[-1].strip()
                    else:
                        continue

                    if func_name and func_name.isidentifier():
                        elements.append(func_name)

            # Class definitions
            elif line.startswith("class "):
                if ":" in line or "{" in line:
                    class_part = line.split(":")[0].split("{")[0]
                    class_name = class_part.replace("class ", "").strip()
                    if "(" in class_name:
                        class_name = class_name.split("(")[0].strip()

                    if class_name and class_name.isidentifier():
                        elements.append(class_name)

        return elements[:20]  # Limit to prevent overflow

    def _extract_headers(self, content: str) -> List[str]:
        """Extract headers from markdown or structured text."""
        headers = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Markdown headers
            if line.startswith("#"):
                header = line.lstrip("#").strip()
                if header:
                    headers.append(header)

            # Other header patterns (underlined)
            elif len(line) > 0 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and all(c in "=-~^" for c in next_line)
                    and len(next_line) >= len(line) * 0.8
                ):
                    headers.append(line)

        return headers[:15]  # Limit to prevent overflow

    def _extract_key_concepts(self, content: str, category: str) -> List[str]:
        """Extract key concepts and important terms."""
        # Simple keyword extraction based on category
        concepts = []
        content_lower = content.lower()

        # Category-specific important terms
        if category == "code":
            code_terms = [
                "function",
                "class",
                "method",
                "variable",
                "parameter",
                "return",
                "import",
                "module",
            ]
            concepts.extend([term for term in code_terms if term in content_lower])

        elif category == "api":
            api_terms = [
                "endpoint",
                "request",
                "response",
                "authentication",
                "parameter",
                "header",
                "status",
            ]
            concepts.extend([term for term in api_terms if term in content_lower])

        elif category == "configuration":
            config_terms = [
                "setting",
                "environment",
                "variable",
                "parameter",
                "option",
                "value",
                "default",
            ]
            concepts.extend([term for term in config_terms if term in content_lower])

        # Extract capitalized words (likely important terms)
        words = content.split()
        for word in words:
            # Clean word
            clean_word = "".join(c for c in word if c.isalnum())
            if (
                len(clean_word) > 3
                and clean_word[0].isupper()
                and clean_word not in concepts
            ):
                concepts.append(clean_word.lower())

        return list(set(concepts))[:15]  # Remove duplicates and limit


# Singleton instance for global access
_hierarchical_embedding_service = None


def get_hierarchical_embedding_service() -> HierarchicalEmbeddingService:
    """
    Get singleton hierarchical embedding service instance.

    Returns:
        Shared HierarchicalEmbeddingService instance
    """
    global _hierarchical_embedding_service

    if _hierarchical_embedding_service is None:
        _hierarchical_embedding_service = HierarchicalEmbeddingService()

    return _hierarchical_embedding_service
