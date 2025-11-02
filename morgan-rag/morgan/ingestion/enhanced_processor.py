"""
Enhanced Document Processor for Morgan RAG

Advanced document processing with semantic chunking capabilities.
Implements requirements 4.1, 4.2, 4.3, 4.4, 15.1, 15.2, 15.3, 15.4, 15.5.

Features:
- Multi-format support (PDF, Markdown, Code, Web) with boundary detection
- Intelligent chunking that respects paragraph, section, and code boundaries
- Document metadata extraction and enrichment system
- Semantic-aware chunking strategies
"""

import re
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import time

# Document processing libraries (with optional imports)
import requests
import ast
import tokenize
from io import StringIO

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    PyPDF2 = None

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    DocxDocument = None

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    markdown = None

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    openpyxl = None

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkBoundaries:
    """Information about chunk boundaries for intelligent splitting."""
    start_line: int
    end_line: int
    boundary_type: str  # paragraph, section, function, class, etc.
    context_before: str = ""
    context_after: str = ""


@dataclass
class DocumentChunk:
    """
    Enhanced document chunk with semantic information.
    
    Represents a semantically meaningful piece of a document.
    """
    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    boundaries: Optional[ChunkBoundaries] = None
    
    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    success: bool
    chunks: List[DocumentChunk]
    documents_processed: int
    total_chunks: int
    processing_time: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedDocumentProcessor:
    """
    Enhanced document processor with semantic chunking capabilities.
    
    Supports multiple document formats with intelligent boundary detection:
    - PDF documents with structure preservation
    - Markdown with header hierarchy
    - Code files with function/class boundaries
    - Web content with semantic structure
    - Office documents (Word, Excel)
    - Plain text with paragraph detection
    """
    
    # Supported file extensions by category
    SUPPORTED_FORMATS = {
        'pdf': ['.pdf'],
        'markdown': ['.md', '.markdown', '.mdown', '.mkd'],
        'code': [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.r', '.m', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat'
        ],
        'web': ['.html', '.htm', '.xml'],
        'office': ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'],
        'text': ['.txt', '.rtf', '.csv', '.json', '.yaml', '.yml', '.toml', '.ini'],
        'config': ['.conf', '.cfg', '.config', '.env', '.properties']
    }
    
    # Code language detection patterns
    CODE_PATTERNS = {
        'python': [r'def\s+\w+\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import'],
        'javascript': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'class\s+\w+', r'export\s+'],
        'java': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'package\s+'],
        'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'class\s+\w+', r'namespace\s+'],
        'go': [r'func\s+\w+\(', r'package\s+\w+', r'type\s+\w+\s+struct'],
        'rust': [r'fn\s+\w+\(', r'struct\s+\w+', r'impl\s+\w+', r'use\s+'],
    }
    
    def __init__(self):
        """Initialize enhanced document processor."""
        self.settings = get_settings()
        
        # Chunking configuration
        self.default_chunk_size = self.settings.morgan_chunk_size
        self.default_overlap = self.settings.morgan_chunk_overlap
        self.max_file_size = self.settings.morgan_max_file_size * 1024 * 1024  # Convert MB to bytes
        
        logger.info(
            f"Enhanced document processor initialized "
            f"(chunk_size={self.default_chunk_size}, overlap={self.default_overlap})"
        )
    
    def process_source(
        self,
        source_path: str,
        document_type: str = "auto",
        chunk_strategy: str = "semantic",
        show_progress: bool = True
    ) -> ProcessingResult:
        """
        Process documents from various sources.
        
        Args:
            source_path: Path to file, directory, or URL
            document_type: Type hint (auto, pdf, markdown, code, web, etc.)
            chunk_strategy: Chunking strategy (semantic, fixed, adaptive)
            show_progress: Show processing progress
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing source: {source_path} (type={document_type}, strategy={chunk_strategy})")
        
        try:
            # Determine source type
            if self._is_url(source_path):
                return self._process_url(source_path, document_type, chunk_strategy)
            elif Path(source_path).is_file():
                return self._process_file(Path(source_path), document_type, chunk_strategy)
            elif Path(source_path).is_dir():
                return self._process_directory(Path(source_path), document_type, chunk_strategy, show_progress)
            else:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    documents_processed=0,
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    errors=[f"Source not found: {source_path}"]
                )
                
        except Exception as e:
            logger.error(f"Failed to process source {source_path}: {e}")
            return ProcessingResult(
                success=False,
                chunks=[],
                documents_processed=0,
                total_chunks=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def chunk_document(
        self,
        content: str,
        document_type: str,
        source: str = "unknown",
        max_chunk_size: int = None,
        overlap_size: int = None,
        strategy: str = "semantic"
    ) -> List[DocumentChunk]:
        """
        Chunk document content using intelligent strategies.
        
        Args:
            content: Document content to chunk
            document_type: Type of document (pdf, markdown, code, etc.)
            source: Source identifier
            max_chunk_size: Maximum chunk size (uses default if None)
            overlap_size: Overlap between chunks (uses default if None)
            strategy: Chunking strategy (semantic, fixed, adaptive)
            
        Returns:
            List of document chunks with boundaries
        """
        if not content or not content.strip():
            return []
        
        max_size = max_chunk_size or self.default_chunk_size
        overlap = overlap_size or self.default_overlap
        
        logger.debug(f"Chunking {len(content)} chars using {strategy} strategy (max_size={max_size})")
        
        if strategy == "semantic":
            return self._chunk_semantic(content, document_type, source, max_size, overlap)
        elif strategy == "fixed":
            return self._chunk_fixed(content, source, max_size, overlap)
        elif strategy == "adaptive":
            return self._chunk_adaptive(content, document_type, source, max_size, overlap)
        else:
            logger.warning(f"Unknown chunking strategy '{strategy}', using semantic")
            return self._chunk_semantic(content, document_type, source, max_size, overlap)
    
    def extract_metadata(
        self,
        content: str,
        source_path: str,
        document_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extract and enrich document metadata.
        
        Args:
            content: Document content
            source_path: Source file path or URL
            document_type: Document type
            
        Returns:
            Enriched metadata dictionary
        """
        metadata = {
            "source": source_path,
            "document_type": document_type,
            "content_length": len(content),
            "processed_at": datetime.utcnow().isoformat(),
            "processor_version": "enhanced_v1.0"
        }
        
        # Basic content analysis
        lines = content.split('\n')
        metadata.update({
            "line_count": len(lines),
            "word_count": len(content.split()),
            "char_count": len(content),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        })
        
        # File-specific metadata
        if not self._is_url(source_path):
            file_path = Path(source_path)
            metadata.update({
                "filename": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            })
        
        # Content-specific metadata
        if document_type == "code" or self._detect_code_language(content):
            metadata.update(self._extract_code_metadata(content))
        elif document_type == "markdown":
            metadata.update(self._extract_markdown_metadata(content))
        elif document_type == "web":
            metadata.update(self._extract_web_metadata(content))
        
        return metadata
    
    def _process_file(
        self,
        file_path: Path,
        document_type: str,
        chunk_strategy: str
    ) -> ProcessingResult:
        """Process a single file."""
        start_time = time.time()
        
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    documents_processed=0,
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    errors=[f"File too large: {file_path} ({file_path.stat().st_size} bytes)"]
                )
            
            # Auto-detect document type
            if document_type == "auto":
                document_type = self._detect_document_type(file_path)
            
            # Extract content based on type
            content = self._extract_content(file_path, document_type)
            
            if not content:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    documents_processed=0,
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    errors=[f"No content extracted from: {file_path}"]
                )
            
            # Extract metadata
            metadata = self.extract_metadata(content, str(file_path), document_type)
            
            # Chunk content
            chunks = self.chunk_document(
                content=content,
                document_type=document_type,
                source=str(file_path),
                strategy=chunk_strategy
            )
            
            # Add file metadata to all chunks
            for chunk in chunks:
                chunk.metadata.update(metadata)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Processed {file_path.name}: {len(chunks)} chunks "
                f"in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                success=True,
                chunks=chunks,
                documents_processed=1,
                total_chunks=len(chunks),
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                chunks=[],
                documents_processed=0,
                total_chunks=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _process_directory(
        self,
        dir_path: Path,
        document_type: str,
        chunk_strategy: str,
        show_progress: bool
    ) -> ProcessingResult:
        """Process all supported files in a directory."""
        start_time = time.time()
        
        # Find all supported files
        supported_files = []
        for ext_list in self.SUPPORTED_FORMATS.values():
            for ext in ext_list:
                supported_files.extend(dir_path.rglob(f"*{ext}"))
        
        if not supported_files:
            return ProcessingResult(
                success=False,
                chunks=[],
                documents_processed=0,
                total_chunks=0,
                processing_time=time.time() - start_time,
                errors=[f"No supported files found in: {dir_path}"]
            )
        
        logger.info(f"Found {len(supported_files)} supported files in {dir_path}")
        
        all_chunks = []
        processed_count = 0
        errors = []
        
        # Process each file
        for file_path in supported_files:
            try:
                result = self._process_file(file_path, document_type, chunk_strategy)
                
                if result.success:
                    all_chunks.extend(result.chunks)
                    processed_count += 1
                else:
                    errors.extend(result.errors)
                    
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Processed directory {dir_path}: {processed_count}/{len(supported_files)} files, "
            f"{len(all_chunks)} total chunks in {processing_time:.2f}s"
        )
        
        return ProcessingResult(
            success=processed_count > 0,
            chunks=all_chunks,
            documents_processed=processed_count,
            total_chunks=len(all_chunks),
            processing_time=processing_time,
            errors=errors,
            metadata={
                "source_directory": str(dir_path),
                "total_files_found": len(supported_files),
                "files_processed": processed_count,
                "files_failed": len(supported_files) - processed_count
            }
        )
    
    def _process_url(
        self,
        url: str,
        document_type: str,
        chunk_strategy: str
    ) -> ProcessingResult:
        """Process content from a URL."""
        start_time = time.time()
        
        try:
            # Fetch content
            headers = {
                'User-Agent': 'Morgan RAG Document Processor 1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Auto-detect type if needed
            if document_type == "auto":
                content_type = response.headers.get('content-type', '').lower()
                if 'html' in content_type:
                    document_type = "web"
                elif 'pdf' in content_type:
                    document_type = "pdf"
                else:
                    document_type = "text"
            
            # Extract content
            if document_type == "web":
                content = self._extract_web_content(response.text, url)
            else:
                content = response.text
            
            if not content:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    documents_processed=0,
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    errors=[f"No content extracted from URL: {url}"]
                )
            
            # Extract metadata
            metadata = self.extract_metadata(content, url, document_type)
            metadata.update({
                "url": url,
                "content_type": response.headers.get('content-type'),
                "status_code": response.status_code
            })
            
            # Chunk content
            chunks = self.chunk_document(
                content=content,
                document_type=document_type,
                source=url,
                strategy=chunk_strategy
            )
            
            # Add URL metadata to all chunks
            for chunk in chunks:
                chunk.metadata.update(metadata)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Processed URL {url}: {len(chunks)} chunks "
                f"in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                success=True,
                chunks=chunks,
                documents_processed=1,
                total_chunks=len(chunks),
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return ProcessingResult(
                success=False,
                chunks=[],
                documents_processed=0,
                total_chunks=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _detect_document_type(self, file_path: Path) -> str:
        """Auto-detect document type from file extension and content."""
        extension = file_path.suffix.lower()
        
        # Check each format category
        for doc_type, extensions in self.SUPPORTED_FORMATS.items():
            if extension in extensions:
                return doc_type
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('text/'):
                return "text"
            elif 'pdf' in mime_type:
                return "pdf"
            elif 'html' in mime_type:
                return "web"
        
        # Default to text
        return "text"
    
    def _extract_content(self, file_path: Path, document_type: str) -> str:
        """Extract text content from file based on type."""
        try:
            if document_type == "pdf":
                return self._extract_pdf_content(file_path)
            elif document_type == "office":
                return self._extract_office_content(file_path)
            elif document_type == "web":
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                return self._extract_web_content(html_content)
            else:
                # Text-based files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return ""
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text from PDF with structure preservation."""
        if not HAS_PDFPLUMBER and not HAS_PYPDF2:
            logger.error(f"PDF processing not available - install pdfplumber or PyPDF2")
            return ""
        
        try:
            # Try pdfplumber first (better structure preservation)
            if HAS_PDFPLUMBER:
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    if text_parts:
                        return '\n\n'.join(text_parts)
        
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}, trying PyPDF2: {e}")
        
        try:
            # Fallback to PyPDF2
            if HAS_PYPDF2:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    return '\n\n'.join(text_parts)
                
        except Exception as e:
            logger.error(f"Failed to extract PDF content from {file_path}: {e}")
            return ""
        
        logger.error(f"No PDF processing libraries available")
        return ""
    
    def _extract_office_content(self, file_path: Path) -> str:
        """Extract text from Office documents."""
        extension = file_path.suffix.lower()
        
        try:
            if extension in ['.docx', '.doc']:
                if not HAS_DOCX:
                    logger.error(f"Word document processing not available - install python-docx")
                    return ""
                
                # Word documents
                doc = DocxDocument(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                return '\n\n'.join(paragraphs)
            
            elif extension in ['.xlsx', '.xls']:
                if not HAS_OPENPYXL:
                    logger.error(f"Excel document processing not available - install openpyxl")
                    return ""
                
                # Excel documents - basic text extraction
                wb = openpyxl.load_workbook(file_path, data_only=True)
                text_parts = []
                
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    sheet_text = f"Sheet: {sheet_name}\n"
                    
                    for row in sheet.iter_rows(values_only=True):
                        row_text = '\t'.join(str(cell) if cell is not None else '' for cell in row)
                        if row_text.strip():
                            sheet_text += row_text + '\n'
                    
                    text_parts.append(sheet_text)
                
                return '\n\n'.join(text_parts)
            
            else:
                logger.warning(f"Unsupported office format: {extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to extract office content from {file_path}: {e}")
            return ""
    
    def _extract_web_content(self, html_content: str, url: str = "") -> str:
        """Extract clean text from HTML content."""
        if not HAS_BS4:
            logger.warning(f"HTML processing not available - install beautifulsoup4. Returning raw HTML.")
            # Basic HTML tag removal as fallback
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Extract main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract web content: {e}")
            return html_content  # Return raw HTML as fallback
    
    def _chunk_semantic(
        self,
        content: str,
        document_type: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """
        Semantic chunking that respects document structure.
        
        Implements requirements 15.1, 15.2, 15.3, 15.4, 15.5.
        """
        if document_type == "code":
            return self._chunk_code_semantic(content, source, max_size, overlap)
        elif document_type == "markdown":
            return self._chunk_markdown_semantic(content, source, max_size, overlap)
        else:
            return self._chunk_text_semantic(content, source, max_size, overlap)
    
    def _chunk_code_semantic(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """
        Chunk code respecting function and class boundaries.
        
        Implements requirement 15.2: respect function and class boundaries.
        """
        chunks = []
        language = self._detect_code_language(content)
        
        try:
            if language == "python":
                chunks = self._chunk_python_code(content, source, max_size, overlap)
            else:
                # Generic code chunking by logical blocks
                chunks = self._chunk_generic_code(content, source, max_size, overlap)
                
        except Exception as e:
            logger.warning(f"Semantic code chunking failed, using text chunking: {e}")
            chunks = self._chunk_text_semantic(content, source, max_size, overlap)
        
        return chunks
    
    def _chunk_python_code(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Chunk Python code by AST nodes."""
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Extract top-level definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1  # AST is 1-indexed
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    # Extract function/class content
                    chunk_lines = lines[start_line:end_line]
                    chunk_content = '\n'.join(chunk_lines)
                    
                    if len(chunk_content) <= max_size:
                        # Single chunk for small functions/classes
                        chunk_id = self._generate_chunk_id(source, start_line)
                        
                        boundaries = ChunkBoundaries(
                            start_line=start_line,
                            end_line=end_line,
                            boundary_type=node.__class__.__name__.lower(),
                            context_before=lines[max(0, start_line-2):start_line] if start_line > 0 else [],
                            context_after=lines[end_line:end_line+2] if end_line < len(lines) else []
                        )
                        
                        chunk = DocumentChunk(
                            content=chunk_content,
                            source=source,
                            chunk_id=chunk_id,
                            boundaries=boundaries,
                            metadata={
                                "chunk_type": "code_block",
                                "language": "python",
                                "node_type": node.__class__.__name__,
                                "node_name": getattr(node, 'name', 'unknown'),
                                "start_line": start_line,
                                "end_line": end_line
                            }
                        )
                        
                        chunks.append(chunk)
                    else:
                        # Large function/class - split further
                        sub_chunks = self._split_large_code_block(
                            chunk_content, source, start_line, max_size, overlap
                        )
                        chunks.extend(sub_chunks)
            
            # Handle remaining content (imports, module-level code)
            if not chunks:
                # No functions/classes found, chunk as text
                return self._chunk_text_semantic(content, source, max_size, overlap)
                
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {source}, using generic code chunking: {e}")
            return self._chunk_generic_code(content, source, max_size, overlap)
        
        return chunks
    
    def _chunk_generic_code(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Generic code chunking by logical blocks."""
        chunks = []
        lines = content.split('\n')
        
        # Find logical boundaries (functions, classes, etc.)
        boundaries = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Common function/class patterns
            if (stripped.startswith(('def ', 'class ', 'function ', 'public class',
                                   'private class', 'protected class', 'struct ',
                                   'interface ', 'enum ', 'namespace '))):
                boundaries.append(i)
        
        if not boundaries:
            # No logical boundaries found, use paragraph-based chunking
            return self._chunk_text_semantic(content, source, max_size, overlap)
        
        # Create chunks based on boundaries
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
            
            chunk_lines = lines[start:end]
            chunk_content = '\n'.join(chunk_lines)
            
            if len(chunk_content) <= max_size:
                chunk_id = self._generate_chunk_id(source, start)
                
                boundaries_obj = ChunkBoundaries(
                    start_line=start,
                    end_line=end,
                    boundary_type="code_block",
                    context_before='\n'.join(lines[max(0, start-2):start]),
                    context_after='\n'.join(lines[end:end+2])
                )
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    source=source,
                    chunk_id=chunk_id,
                    boundaries=boundaries_obj,
                    metadata={
                        "chunk_type": "code_block",
                        "start_line": start,
                        "end_line": end
                    }
                )
                
                chunks.append(chunk)
            else:
                # Split large blocks
                sub_chunks = self._split_large_code_block(
                    chunk_content, source, start, max_size, overlap
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_markdown_semantic(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """
        Chunk Markdown respecting header hierarchy.
        
        Implements requirement 15.3: preserve headers and maintain hierarchical context.
        """
        chunks = []
        lines = content.split('\n')
        
        # Find header boundaries
        header_boundaries = []
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                header_boundaries.append((i, level, line.strip()))
        
        if not header_boundaries:
            # No headers found, use paragraph-based chunking
            return self._chunk_text_semantic(content, source, max_size, overlap)
        
        # Create chunks based on header hierarchy
        for i, (start_line, level, header) in enumerate(header_boundaries):
            # Find end of this section
            end_line = len(lines)
            for j in range(i + 1, len(header_boundaries)):
                next_start, next_level, _ = header_boundaries[j]
                if next_level <= level:  # Same or higher level header
                    end_line = next_start
                    break
            
            # Extract section content
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines)
            
            if len(section_content) <= max_size:
                # Single chunk for small sections
                chunk_id = self._generate_chunk_id(source, start_line)
                
                boundaries = ChunkBoundaries(
                    start_line=start_line,
                    end_line=end_line,
                    boundary_type=f"markdown_section_h{level}",
                    context_before='\n'.join(lines[max(0, start_line-2):start_line]),
                    context_after='\n'.join(lines[end_line:end_line+2])
                )
                
                chunk = DocumentChunk(
                    content=section_content,
                    source=source,
                    chunk_id=chunk_id,
                    boundaries=boundaries,
                    metadata={
                        "chunk_type": "markdown_section",
                        "header_level": level,
                        "header_text": header,
                        "start_line": start_line,
                        "end_line": end_line
                    }
                )
                
                chunks.append(chunk)
            else:
                # Large section - split by paragraphs while preserving header
                sub_chunks = self._split_large_markdown_section(
                    section_content, source, start_line, header, level, max_size, overlap
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_text_semantic(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """
        Chunk text respecting paragraph boundaries.
        
        Implements requirement 15.1: respect paragraph and section boundaries.
        """
        chunks = []
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        current_start = 0
        chunk_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed max size
            potential_chunk = current_chunk + ('\n\n' if current_chunk else '') + paragraph
            
            if len(potential_chunk) <= max_size:
                # Add paragraph to current chunk
                current_chunk = potential_chunk
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunk_id = self._generate_chunk_id(source, chunk_count)
                    
                    boundaries = ChunkBoundaries(
                        start_line=current_start,
                        end_line=i,
                        boundary_type="paragraph_group"
                    )
                    
                    chunk = DocumentChunk(
                        content=current_chunk,
                        source=source,
                        chunk_id=chunk_id,
                        boundaries=boundaries,
                        metadata={
                            "chunk_type": "text_paragraphs",
                            "paragraph_count": i - current_start,
                            "chunk_index": chunk_count
                        }
                    )
                    
                    chunks.append(chunk)
                    chunk_count += 1
                
                # Start new chunk with current paragraph
                current_chunk = paragraph
                current_start = i
        
        # Add final chunk
        if current_chunk:
            chunk_id = self._generate_chunk_id(source, chunk_count)
            
            boundaries = ChunkBoundaries(
                start_line=current_start,
                end_line=len(paragraphs),
                boundary_type="paragraph_group"
            )
            
            chunk = DocumentChunk(
                content=current_chunk,
                source=source,
                chunk_id=chunk_id,
                boundaries=boundaries,
                metadata={
                    "chunk_type": "text_paragraphs",
                    "paragraph_count": len(paragraphs) - current_start,
                    "chunk_index": chunk_count
                }
            )
            
            chunks.append(chunk)
        
        # Add overlap between chunks if requested
        if overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap_to_chunks(chunks, overlap)
        
        return chunks
    
    def _chunk_fixed(
        self,
        content: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        
        for i in range(0, len(content), max_size - overlap):
            chunk_content = content[i:i + max_size]
            
            if not chunk_content.strip():
                continue
            
            chunk_id = self._generate_chunk_id(source, i)
            
            chunk = DocumentChunk(
                content=chunk_content,
                source=source,
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "fixed_size",
                    "start_pos": i,
                    "end_pos": i + len(chunk_content),
                    "chunk_index": len(chunks)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_adaptive(
        self,
        content: str,
        document_type: str,
        source: str,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Adaptive chunking that adjusts based on content density."""
        # Start with semantic chunking
        semantic_chunks = self._chunk_semantic(content, document_type, source, max_size, overlap)
        
        # Analyze chunk sizes and adjust
        adjusted_chunks = []
        
        for chunk in semantic_chunks:
            if len(chunk.content) < max_size * 0.3:  # Very small chunk
                # Try to merge with next chunk if possible
                if adjusted_chunks and len(adjusted_chunks[-1].content) + len(chunk.content) <= max_size:
                    # Merge with previous chunk
                    prev_chunk = adjusted_chunks[-1]
                    merged_content = prev_chunk.content + '\n\n' + chunk.content
                    
                    prev_chunk.content = merged_content
                    prev_chunk.metadata["merged_chunks"] = prev_chunk.metadata.get("merged_chunks", 1) + 1
                else:
                    adjusted_chunks.append(chunk)
            else:
                adjusted_chunks.append(chunk)
        
        return adjusted_chunks
    
    def _split_large_code_block(
        self,
        content: str,
        source: str,
        start_line: int,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Split large code blocks while preserving structure."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size <= max_size:
                current_chunk_lines.append(line)
                current_size += line_size
            else:
                # Save current chunk
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk_id = self._generate_chunk_id(source, start_line + len(chunks))
                    
                    chunk = DocumentChunk(
                        content=chunk_content,
                        source=source,
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "code_block_split",
                            "part": len(chunks) + 1,
                            "start_line": start_line + i - len(current_chunk_lines),
                            "end_line": start_line + i
                        }
                    )
                    
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk_lines:
                    overlap_lines = current_chunk_lines[-overlap:]
                    current_chunk_lines = overlap_lines + [line]
                    current_size = sum(len(l) + 1 for l in current_chunk_lines)
                else:
                    current_chunk_lines = [line]
                    current_size = line_size
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_id = self._generate_chunk_id(source, start_line + len(chunks))
            
            chunk = DocumentChunk(
                content=chunk_content,
                source=source,
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "code_block_split",
                    "part": len(chunks) + 1,
                    "start_line": start_line + len(lines) - len(current_chunk_lines),
                    "end_line": start_line + len(lines)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_markdown_section(
        self,
        content: str,
        source: str,
        start_line: int,
        header: str,
        level: int,
        max_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Split large Markdown sections while preserving header context."""
        chunks = []
        
        # Split by paragraphs within the section
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = header + '\n\n'  # Always include header
        current_size = len(current_chunk)
        
        for paragraph in paragraphs[1:]:  # Skip header paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph) + 2  # +2 for newlines
            
            if current_size + paragraph_size <= max_size:
                current_chunk += paragraph + '\n\n'
                current_size += paragraph_size
            else:
                # Save current chunk
                if current_chunk.strip() != header.strip():
                    chunk_id = self._generate_chunk_id(source, start_line + len(chunks))
                    
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "markdown_section_split",
                            "header_level": level,
                            "header_text": header,
                            "part": len(chunks) + 1
                        }
                    )
                    
                    chunks.append(chunk)
                
                # Start new chunk with header
                current_chunk = header + '\n\n' + paragraph + '\n\n'
                current_size = len(current_chunk)
        
        # Add final chunk
        if current_chunk.strip() != header.strip():
            chunk_id = self._generate_chunk_id(source, start_line + len(chunks))
            
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "markdown_section_split",
                    "header_level": level,
                    "header_text": header,
                    "part": len(chunks) + 1
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _add_overlap_to_chunks(
        self,
        chunks: List[DocumentChunk],
        overlap: int
    ) -> List[DocumentChunk]:
        """
        Add overlap between chunks.
        
        Implements requirement 15.5: include 50-character overlap between adjacent chunks.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i - 1]
            
            # Get overlap from previous chunk
            overlap_text = previous_chunk.content[-overlap:] if len(previous_chunk.content) > overlap else previous_chunk.content
            
            # Add overlap to current chunk
            overlapped_content = overlap_text + '\n...\n' + current_chunk.content
            
            # Update chunk
            current_chunk.content = overlapped_content
            current_chunk.metadata["has_overlap"] = True
            current_chunk.metadata["overlap_size"] = len(overlap_text)
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _generate_chunk_id(self, source: str, index: Union[int, str]) -> str:
        """Generate unique chunk ID."""
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{index}"
    
    def _is_url(self, path: str) -> bool:
        """Check if path is a URL."""
        try:
            result = urlparse(path)
            return result.scheme in ['http', 'https']
        except:
            return False
    
    def _detect_code_language(self, content: str) -> Optional[str]:
        """Detect programming language from content."""
        for language, patterns in self.CODE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE):
                    return language
        return None
    
    def _extract_code_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from code content."""
        metadata = {"content_type": "code"}
        
        # Detect language
        language = self._detect_code_language(content)
        if language:
            metadata["language"] = language
        
        # Count functions, classes, etc.
        function_count = len(re.findall(r'^\s*(def|function|func)\s+\w+', content, re.MULTILINE))
        class_count = len(re.findall(r'^\s*(class|struct|interface)\s+\w+', content, re.MULTILINE))
        
        metadata.update({
            "function_count": function_count,
            "class_count": class_count,
            "import_count": len(re.findall(r'^\s*(import|from|#include|using)', content, re.MULTILINE))
        })
        
        return metadata
    
    def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from Markdown content."""
        metadata = {"content_type": "markdown"}
        
        # Count headers by level
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        header_counts = {}
        for level_marks, title in headers:
            level = len(level_marks)
            header_counts[f"h{level}_count"] = header_counts.get(f"h{level}_count", 0) + 1
        
        metadata.update(header_counts)
        
        # Extract first header as title
        if headers:
            metadata["title"] = headers[0][1]
        
        # Count links and images
        metadata.update({
            "link_count": len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content)),
            "image_count": len(re.findall(r'!\[([^\]]*)\]\([^)]+\)', content)),
            "code_block_count": len(re.findall(r'```', content)) // 2
        })
        
        return metadata
    
    def _extract_web_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from web content."""
        metadata = {"content_type": "web"}
        
        if not HAS_BS4:
            logger.warning(f"HTML metadata extraction not available - install beautifulsoup4")
            return metadata
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata["description"] = meta_desc.get('content', '').strip()
            
            # Count elements
            metadata.update({
                "heading_count": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "paragraph_count": len(soup.find_all('p')),
                "link_count": len(soup.find_all('a')),
                "image_count": len(soup.find_all('img'))
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract web metadata: {e}")
        
        return metadata

# Singleton instance for enhanced document processor
_enhanced_processor_instance = None


def get_enhanced_document_processor() -> EnhancedDocumentProcessor:
    """
    Get singleton enhanced document processor instance.
    
    Returns:
        Shared EnhancedDocumentProcessor instance
    """
    global _enhanced_processor_instance
    
    if _enhanced_processor_instance is None:
        _enhanced_processor_instance = EnhancedDocumentProcessor()
    
    return _enhanced_processor_instance