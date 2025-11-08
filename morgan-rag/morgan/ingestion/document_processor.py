"""
Basic Document Processor for Morgan RAG

Simple wrapper around the enhanced processor for backward compatibility.
"""

from typing import Any, Dict, List

from morgan.utils.logger import get_logger

from .enhanced_processor import DocumentChunk, EnhancedDocumentProcessor

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Basic document processor - wrapper around enhanced processor.

    Provides simple interface for backward compatibility.
    """

    def __init__(self):
        """Initialize document processor."""
        self.enhanced_processor = EnhancedDocumentProcessor()
        logger.info("Document processor initialized")

    def process_source(
        self, source_path: str, document_type: str = "auto", show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process documents from source.

        Args:
            source_path: Path to file, directory, or URL
            document_type: Document type hint
            show_progress: Show progress

        Returns:
            Processing result dictionary
        """
        result = self.enhanced_processor.process_source(
            source_path=source_path,
            document_type=document_type,
            chunk_strategy="semantic",
            show_progress=show_progress,
        )

        # Convert to simple format for backward compatibility
        return {
            "success": result.success,
            "chunks": result.chunks,
            "documents_processed": result.documents_processed,
            "total_chunks": result.total_chunks,
            "processing_time": result.processing_time,
            "errors": result.errors,
        }

    def chunk_document(
        self,
        content: str,
        source: str = "unknown",
        max_chunk_size: int = None,
        overlap_size: int = None,
    ) -> List[DocumentChunk]:
        """
        Chunk document content.

        Args:
            content: Document content
            source: Source identifier
            max_chunk_size: Maximum chunk size
            overlap_size: Overlap between chunks

        Returns:
            List of document chunks
        """
        return self.enhanced_processor.chunk_document(
            content=content,
            document_type="text",
            source=source,
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
            strategy="semantic",
        )

    def extract_metadata(self, content: str, source_path: str) -> Dict[str, Any]:
        """
        Extract document metadata.

        Args:
            content: Document content
            source_path: Source path

        Returns:
            Metadata dictionary
        """
        return self.enhanced_processor.extract_metadata(
            content=content, source_path=source_path, document_type="auto"
        )
