"""
Document ingestion and processing for the Knowledge Engine.

This module provides document loaders for various formats (PDF, markdown, text, web pages),
intelligent chunking with overlap, metadata extraction, and support for incremental updates.
"""

import hashlib
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import os

import structlog

# Change this root as appropriate for your deployment/environment.
SAFE_DOCUMENT_ROOT = Path("/data").resolve()

logger = structlog.get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""

    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None

    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None:
            # Generate deterministic ID based on content and metadata
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            doc_id = self.metadata.get("document_id", "unknown")
            self.chunk_id = f"{doc_id}_chunk_{self.chunk_index}_{content_hash}"


@dataclass
class Document:
    """Represents a processed document with metadata."""

    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamps and document ID."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.document_id is None:
            # Generate deterministic ID based on source
            source_hash = hashlib.sha256(self.source.encode()).hexdigest()[:16]
            self.document_id = f"doc_{source_hash}"


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: Union[str, Path]) -> Document:
        """Load a document from the given source."""
        pass

    @abstractmethod
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if this loader can handle the given source."""
        pass


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is a text file."""
        if isinstance(source, str) and not Path(source).exists():
            return False
        path = Path(source)
        return path.suffix.lower() in [".txt", ""]

    def load(self, source: Union[str, Path]) -> Document:
        """Load a text file."""
        # --- Path traversal protection ---
        path = Path(source)
        logger.info("loading_text_file", path=str(path))

        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            logger.error("file_not_found", path=str(path))
            raise FileNotFoundError(f"File not found: {path}")
        # Always re-resolve the SAFE_DOCUMENT_ROOT at runtime
        try:
            safe_root = SAFE_DOCUMENT_ROOT.resolve(strict=True)
        except FileNotFoundError:
            logger.error("safe_root_not_found", root=str(SAFE_DOCUMENT_ROOT))
            raise FileNotFoundError(f"SAFE_DOCUMENT_ROOT does not exist: {SAFE_DOCUMENT_ROOT}")
        try:
            resolved_path.relative_to(safe_root)
        except ValueError:
            logger.error("path_outside_safe_root", path=str(resolved_path), root=str(safe_root))
            raise PermissionError(f"Access to file outside allowed directory: {resolved_path}")

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {
                "filename": resolved_path.name,
                "file_size": resolved_path.stat().st_size,
                "file_modified": datetime.fromtimestamp(resolved_path.stat().st_mtime).isoformat(),
            }

            return Document(
                content=content,
                source=str(resolved_path),
                doc_type="text",
                metadata=metadata,
            )
        except Exception as e:
            logger.error("failed_to_load_text", path=str(resolved_path), error=str(e))
            raise


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is a markdown file."""
        if isinstance(source, str) and not Path(source).exists():
            return False
        path = Path(source)
        return path.suffix.lower() in [".md", ".markdown"]

    def load(self, source: Union[str, Path]) -> Document:
        """Load a markdown file."""
        path = Path(source)
        logger.info("loading_markdown_file", path=str(path))

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first heading if present
            title = None
            lines = content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            metadata = {
                "filename": path.name,
                "file_size": path.stat().st_size,
                "file_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "title": title,
            }

            return Document(
                content=content,
                source=str(path),
                doc_type="markdown",
                metadata=metadata,
            )
        except Exception as e:
            logger.error("failed_to_load_markdown", path=str(path), error=str(e))
            raise


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""

    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is a PDF file."""
        if isinstance(source, str) and not Path(source).exists():
            return False
        path = Path(source)
        return path.suffix.lower() == ".pdf"

    def load(self, source: Union[str, Path]) -> Document:
        """Load a PDF file."""
        path = Path(source)
        logger.info("loading_pdf_file", path=str(path))

        try:
            # Try pdfplumber first (better text extraction)
            try:
                import pdfplumber

                with pdfplumber.open(path) as pdf:
                    pages = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)

                    content = "\n\n".join(pages)
                    num_pages = len(pdf.pages)

            except ImportError:
                # Fall back to PyPDF2
                from PyPDF2 import PdfReader

                reader = PdfReader(path)
                pages = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)

                content = "\n\n".join(pages)
                num_pages = len(reader.pages)

            metadata = {
                "filename": path.name,
                "file_size": path.stat().st_size,
                "file_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "num_pages": num_pages,
            }

            return Document(
                content=content,
                source=str(path),
                doc_type="pdf",
                metadata=metadata,
            )
        except Exception as e:
            logger.error("failed_to_load_pdf", path=str(path), error=str(e))
            raise


class WebPageLoader(DocumentLoader):
    """Loader for web pages."""

    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is a URL."""
        if isinstance(source, Path):
            return False
        try:
            result = urlparse(str(source))
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def load(self, source: Union[str, Path]) -> Document:
        """Load a web page."""
        url = str(source)
        logger.info("loading_web_page", url=url)

        try:
            import html2text
            import requests
            from bs4 import BeautifulSoup

            # Fetch the page
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = soup.title.string if soup.title else None

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Convert to markdown for better readability
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            content = h.handle(str(soup))

            metadata = {
                "url": url,
                "title": title,
                "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "content_type": response.headers.get("content-type", ""),
            }

            return Document(
                content=content,
                source=url,
                doc_type="webpage",
                metadata=metadata,
            )
        except Exception as e:
            logger.error("failed_to_load_webpage", url=url, error=str(e))
            raise


class DocumentChunker:
    """Intelligent document chunker with overlap support."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Preferred separator for splitting (e.g., paragraph breaks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces with overlap.

        Args:
            document: The document to chunk

        Returns:
            List of document chunks
        """
        logger.info(
            "chunking_document",
            document_id=document.document_id,
            content_length=len(document.content),
            chunk_size=self.chunk_size,
        )

        chunks = []
        content = document.content

        # Split by separator first
        sections = content.split(self.separator)

        current_chunk = ""
        chunk_index = 0

        for section in sections:
            # If adding this section would exceed chunk size
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = {
                    **document.metadata,
                    "document_id": document.document_id,
                    "source": document.source,
                    "doc_type": document.doc_type,
                    "total_chunks": 0,  # Will be updated later
                }

                chunks.append(
                    DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        metadata=chunk_metadata,
                    )
                )

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last chunk_overlap characters as overlap
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + self.separator + section
                else:
                    current_chunk = section

                chunk_index += 1
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += self.separator + section
                else:
                    current_chunk = section

        # Add final chunk if there's content
        if current_chunk.strip():
            chunk_metadata = {
                **document.metadata,
                "document_id": document.document_id,
                "source": document.source,
                "doc_type": document.doc_type,
                "total_chunks": 0,
            }

            chunks.append(
                DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata=chunk_metadata,
                )
            )

        # Update total_chunks in all chunk metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total_chunks

        logger.info(
            "chunking_complete",
            document_id=document.document_id,
            num_chunks=total_chunks,
        )

        return chunks


class DocumentProcessor:
    """Main document processor that coordinates loading and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.loaders: List[DocumentLoader] = [
            TextLoader(),
            MarkdownLoader(),
            PDFLoader(),
            WebPageLoader(),
        ]
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._document_hashes: Dict[str, str] = {}  # For incremental updates

    def process(
        self,
        source: Union[str, Path],
        doc_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a document from source to chunks.

        Args:
            source: Path to file or URL
            doc_type: Document type (auto-detect if "auto")
            metadata: Additional metadata to attach

        Returns:
            List of document chunks ready for embedding
        """
        logger.info("processing_document", source=str(source), doc_type=doc_type)

        # Load document
        document = self._load_document(source, doc_type)

        # Add additional metadata if provided
        if metadata:
            document.metadata.update(metadata)

        # Check for incremental update
        content_hash = hashlib.sha256(document.content.encode()).hexdigest()
        if document.document_id in self._document_hashes:
            if self._document_hashes[document.document_id] == content_hash:
                logger.info(
                    "document_unchanged",
                    document_id=document.document_id,
                    source=str(source),
                )
                return []  # No changes, skip processing

        # Update hash
        self._document_hashes[document.document_id] = content_hash

        # Chunk document
        chunks = self.chunker.chunk(document)

        logger.info(
            "document_processed",
            document_id=document.document_id,
            num_chunks=len(chunks),
        )

        return chunks

    def _load_document(self, source: Union[str, Path], doc_type: str = "auto") -> Document:
        """
        Load a document using the appropriate loader.

        Args:
            source: Path to file or URL
            doc_type: Document type (auto-detect if "auto")

        Returns:
            Loaded document
        """
        # Auto-detect loader
        if doc_type == "auto":
            for loader in self.loaders:
                if loader.can_load(source):
                    return loader.load(source)

            # If no loader found, try text loader as fallback
            logger.warning("no_loader_found", source=str(source), using_fallback=True)
            return TextLoader().load(source)

        # Use specific loader based on doc_type
        loader_map = {
            "text": TextLoader(),
            "markdown": MarkdownLoader(),
            "pdf": PDFLoader(),
            "webpage": WebPageLoader(),
        }

        if doc_type not in loader_map:
            raise ValueError(f"Unknown document type: {doc_type}")

        return loader_map[doc_type].load(source)

    def get_document_hash(self, document_id: str) -> Optional[str]:
        """Get the stored hash for a document (for incremental updates)."""
        return self._document_hashes.get(document_id)

    def clear_document_hash(self, document_id: str) -> None:
        """Clear the stored hash for a document (force reprocessing)."""
        if document_id in self._document_hashes:
            del self._document_hashes[document_id]
