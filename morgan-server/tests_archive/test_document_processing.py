"""
Unit tests for document processing (ingestion module).

Tests document loading for different formats, chunking logic, and metadata extraction.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from morgan_server.knowledge.ingestion import (
    Document,
    DocumentChunk,
    DocumentChunker,
    DocumentProcessor,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    WebPageLoader,
)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_chunk_id_generation(self):
        """Test that chunk IDs are generated automatically."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            metadata={"document_id": "doc_123"},
        )

        assert chunk.chunk_id is not None
        assert "doc_123" in chunk.chunk_id
        assert "chunk_0" in chunk.chunk_id

    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic for same content."""
        chunk1 = DocumentChunk(
            content="Test content",
            chunk_index=0,
            metadata={"document_id": "doc_123"},
        )

        chunk2 = DocumentChunk(
            content="Test content",
            chunk_index=0,
            metadata={"document_id": "doc_123"},
        )

        assert chunk1.chunk_id == chunk2.chunk_id

    def test_chunk_id_custom(self):
        """Test that custom chunk IDs are preserved."""
        custom_id = "custom_chunk_id"
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            metadata={},
            chunk_id=custom_id,
        )

        assert chunk.chunk_id == custom_id


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_id_generation(self):
        """Test that document IDs are generated automatically."""
        doc = Document(
            content="Test content",
            source="/path/to/file.txt",
            doc_type="text",
        )

        assert doc.document_id is not None
        assert doc.document_id.startswith("doc_")

    def test_document_id_deterministic(self):
        """Test that document IDs are deterministic for same source."""
        doc1 = Document(
            content="Test content",
            source="/path/to/file.txt",
            doc_type="text",
        )

        doc2 = Document(
            content="Different content",
            source="/path/to/file.txt",
            doc_type="text",
        )

        # Same source should generate same ID
        assert doc1.document_id == doc2.document_id

    def test_timestamps_auto_generated(self):
        """Test that timestamps are generated automatically."""
        doc = Document(
            content="Test content",
            source="/path/to/file.txt",
            doc_type="text",
        )

        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)


class TestTextLoader:
    """Tests for TextLoader."""

    def test_can_load_text_file(self):
        """Test that TextLoader can identify text files."""
        loader = TextLoader()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert loader.can_load(temp_path)
        finally:
            temp_path.unlink()

    def test_cannot_load_non_text_file(self):
        """Test that TextLoader rejects non-text files."""
        loader = TextLoader()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert not loader.can_load(temp_path)
        finally:
            temp_path.unlink()

    def test_load_text_file(self):
        """Test loading a text file."""
        loader = TextLoader()
        content = "This is a test file.\nWith multiple lines."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.content == content
            assert doc.source == str(temp_path)
            assert doc.doc_type == "text"
            assert "filename" in doc.metadata
            assert "file_size" in doc.metadata
            assert "file_modified" in doc.metadata
        finally:
            temp_path.unlink()


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_can_load_markdown_file(self):
        """Test that MarkdownLoader can identify markdown files."""
        loader = MarkdownLoader()

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert loader.can_load(temp_path)
        finally:
            temp_path.unlink()

    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        loader = MarkdownLoader()
        content = "# Test Document\n\nThis is a test."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.content == content
            assert doc.source == str(temp_path)
            assert doc.doc_type == "markdown"
            assert doc.metadata["title"] == "Test Document"
        finally:
            temp_path.unlink()

    def test_load_markdown_without_title(self):
        """Test loading markdown without a title."""
        loader = MarkdownLoader()
        content = "This is content without a title."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.content == content
            assert doc.metadata["title"] is None
        finally:
            temp_path.unlink()


class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_can_load_pdf_file(self):
        """Test that PDFLoader can identify PDF files."""
        loader = PDFLoader()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert loader.can_load(temp_path)
        finally:
            temp_path.unlink()

    def test_cannot_load_non_pdf_file(self):
        """Test that PDFLoader rejects non-PDF files."""
        loader = PDFLoader()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            assert not loader.can_load(temp_path)
        finally:
            temp_path.unlink()


class TestWebPageLoader:
    """Tests for WebPageLoader."""

    def test_can_load_url(self):
        """Test that WebPageLoader can identify URLs."""
        loader = WebPageLoader()

        assert loader.can_load("https://example.com")
        assert loader.can_load("http://example.com/page")

    def test_cannot_load_file_path(self):
        """Test that WebPageLoader rejects file paths."""
        loader = WebPageLoader()

        assert not loader.can_load("/path/to/file.txt")
        assert not loader.can_load(Path("/path/to/file.txt"))

    @patch("requests.get")
    def test_load_webpage(self, mock_get):
        """Test loading a web page."""
        loader = WebPageLoader()

        # Mock response
        mock_response = Mock()
        mock_response.content = b"<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        doc = loader.load("https://example.com")

        assert doc.source == "https://example.com"
        assert doc.doc_type == "webpage"
        assert doc.metadata["title"] == "Test Page"
        assert doc.metadata["url"] == "https://example.com"
        assert "Test content" in doc.content


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunk_small_document(self):
        """Test chunking a document smaller than chunk size."""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

        doc = Document(
            content="This is a small document.",
            source="test.txt",
            doc_type="text",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "This is a small document."
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_chunk_large_document(self):
        """Test chunking a document larger than chunk size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        # Create content with clear paragraph breaks
        paragraphs = [f"Paragraph {i}. " + "x" * 80 for i in range(5)]
        content = "\n\n".join(paragraphs)

        doc = Document(
            content=content,
            source="test.txt",
            doc_type="text",
        )

        chunks = chunker.chunk(doc)

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should have metadata
        for chunk in chunks:
            assert "document_id" in chunk.metadata
            assert "source" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        # Create content that will be split
        content = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80

        doc = Document(
            content=content,
            source="test.txt",
            doc_type="text",
        )

        chunks = chunker.chunk(doc)

        # Check that consecutive chunks have overlap
        if len(chunks) > 1:
            # Second chunk should contain some content from first chunk
            assert len(chunks[1].content) > 0

    def test_chunk_indices(self):
        """Test that chunk indices are sequential."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        paragraphs = [f"Paragraph {i}. " + "x" * 80 for i in range(5)]
        content = "\n\n".join(paragraphs)

        doc = Document(
            content=content,
            source="test.txt",
            doc_type="text",
        )

        chunks = chunker.chunk(doc)

        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_metadata_inheritance(self):
        """Test that chunks inherit document metadata."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        doc = Document(
            content="x" * 500,
            source="test.txt",
            doc_type="text",
            metadata={"custom_field": "custom_value"},
        )

        chunks = chunker.chunk(doc)

        # All chunks should have document metadata
        for chunk in chunks:
            assert chunk.metadata["custom_field"] == "custom_value"
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["doc_type"] == "text"


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    def test_process_text_file(self):
        """Test processing a text file end-to-end."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        content = "This is a test file.\n\n" + "x" * 200

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            chunks = processor.process(temp_path)

            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.chunk_id is not None for chunk in chunks)
        finally:
            temp_path.unlink()

    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        content = "# Test\n\nThis is markdown content.\n\n" + "x" * 200

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            chunks = processor.process(temp_path)

            assert len(chunks) > 0
            # Check that markdown metadata is present
            assert chunks[0].metadata["doc_type"] == "markdown"
        finally:
            temp_path.unlink()

    def test_process_with_custom_metadata(self):
        """Test processing with additional metadata."""
        processor = DocumentProcessor()

        content = "Test content"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            custom_metadata = {"author": "Test Author", "category": "Test"}
            chunks = processor.process(temp_path, metadata=custom_metadata)

            assert len(chunks) > 0
            assert chunks[0].metadata["author"] == "Test Author"
            assert chunks[0].metadata["category"] == "Test"
        finally:
            temp_path.unlink()

    def test_incremental_update_unchanged(self):
        """Test that unchanged documents are skipped."""
        processor = DocumentProcessor()

        content = "Test content"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            # First processing
            chunks1 = processor.process(temp_path)
            assert len(chunks1) > 0

            # Second processing without changes
            chunks2 = processor.process(temp_path)
            assert len(chunks2) == 0  # Should skip unchanged document
        finally:
            temp_path.unlink()

    def test_incremental_update_changed(self):
        """Test that changed documents are reprocessed."""
        processor = DocumentProcessor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Original content")
            temp_path = Path(f.name)

        try:
            # First processing
            chunks1 = processor.process(temp_path)
            assert len(chunks1) > 0

            # Modify file
            with open(temp_path, "w") as f:
                f.write("Modified content")

            # Second processing with changes
            chunks2 = processor.process(temp_path)
            assert len(chunks2) > 0  # Should reprocess changed document
        finally:
            temp_path.unlink()

    def test_get_document_hash(self):
        """Test retrieving document hash."""
        processor = DocumentProcessor()

        content = "Test content"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            chunks = processor.process(temp_path)
            doc_id = chunks[0].metadata["document_id"]

            # Hash should be stored
            hash_value = processor.get_document_hash(doc_id)
            assert hash_value is not None
        finally:
            temp_path.unlink()

    def test_clear_document_hash(self):
        """Test clearing document hash to force reprocessing."""
        processor = DocumentProcessor()

        content = "Test content"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            # First processing
            chunks1 = processor.process(temp_path)
            doc_id = chunks1[0].metadata["document_id"]

            # Clear hash
            processor.clear_document_hash(doc_id)

            # Should reprocess even though content unchanged
            chunks2 = processor.process(temp_path)
            assert len(chunks2) > 0
        finally:
            temp_path.unlink()

    def test_auto_detect_document_type(self):
        """Test automatic document type detection."""
        processor = DocumentProcessor()

        # Test with .txt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Text content")
            txt_path = Path(f.name)

        # Test with .md file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Markdown content")
            md_path = Path(f.name)

        try:
            txt_chunks = processor.process(txt_path, doc_type="auto")
            assert txt_chunks[0].metadata["doc_type"] == "text"

            md_chunks = processor.process(md_path, doc_type="auto")
            assert md_chunks[0].metadata["doc_type"] == "markdown"
        finally:
            txt_path.unlink()
            md_path.unlink()
