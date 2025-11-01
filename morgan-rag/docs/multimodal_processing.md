# Multimodal Content Processing

The Multimodal Content Processor enables Morgan RAG to handle documents containing both text and images using jina-clip-v2 integration.

## Features

- **jina-clip-v2 Integration**: Combined text and image embeddings for unified semantic understanding
- **OCR Text Extraction**: Automatic text extraction from images using Tesseract (when available)
- **Image-Text Alignment**: Correlation analysis between textual and visual content
- **Multimodal Search**: Search across both text and visual elements simultaneously
- **Graceful Error Handling**: Robust processing with fallback mechanisms for image failures
- **Performance Optimization**: Batch processing and efficient resource management

## Quick Start

### Basic Usage

```python
from morgan.jina.embeddings import (
    JinaEmbeddingService,
    MultimodalContentProcessor
)

# Initialize services
embedding_service = JinaEmbeddingService()
processor = MultimodalContentProcessor(
    embedding_service=embedding_service,
    ocr_enabled=True,
    image_quality_threshold=0.3
)

# Process a document with images
document = processor.process_multimodal_document(
    content="This document discusses AI trends with supporting charts.",
    images=[image_path1, image_bytes, pil_image],  # Various image formats supported
    metadata={"title": "AI Trends Report", "category": "research"}
)

# Search across multimodal content
results = processor.search_multimodal_content(
    query="artificial intelligence trends visualization",
    documents=[document],
    include_images=True,
    max_results=10
)
```

### Advanced Configuration

```python
# Configure processor with custom settings
processor = MultimodalContentProcessor(
    embedding_service=embedding_service,
    max_workers=8,                    # Concurrent processing threads
    ocr_enabled=True,                 # Enable OCR text extraction
    image_quality_threshold=0.5       # Minimum quality for processing
)

# Get processing statistics
stats = processor.get_processing_stats()
print(f"OCR Available: {stats['ocr_enabled']}")
print(f"Supported Formats: {stats['supported_formats']}")
```

## Data Models

### MultimodalDocument

Represents a processed document with both text and visual content:

```python
@dataclass
class MultimodalDocument:
    text_content: str                    # Original text content
    images: List[ImageContent]           # Processed images
    text_embeddings: List[float]         # Text embedding vector
    image_embeddings: List[List[float]]  # Image embedding vectors
    combined_embedding: List[float]      # Fused text+image embedding
    metadata: Dict[str, Any]             # Document metadata
    processing_errors: List[str]         # Any processing errors
```

### ImageContent

Container for processed image data:

```python
@dataclass
class ImageContent:
    image_data: bytes           # Raw image bytes
    format: str                 # Image format (jpeg, png, etc.)
    width: int                  # Image width in pixels
    height: int                 # Image height in pixels
    file_path: Optional[str]    # Original file path (if applicable)
    extracted_text: Optional[str]  # OCR-extracted text
    quality_score: float        # Quality assessment (0.0-1.0)
```

### MultimodalSearchResult

Search result with multimodal scoring:

```python
@dataclass
class MultimodalSearchResult:
    content: str                # Text content
    images: List[ImageContent]  # Associated images
    relevance_score: float      # Overall relevance (0.0-1.0)
    text_score: float          # Text-only similarity score
    image_score: float         # Image-only similarity score
    combined_score: float      # Combined embedding score
    metadata: Dict[str, Any]   # Result metadata
```

## Image Processing

### Supported Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP

### Quality Assessment

Images are automatically assessed for processing quality based on:

- **Size**: Larger images generally score higher
- **Aspect Ratio**: Reasonable ratios preferred over extreme ones
- **Format**: Lossless formats (PNG) score higher than lossy (JPEG)

### OCR Integration

When Tesseract is available, the processor automatically:

1. Enhances images for better OCR (contrast, sharpness)
2. Extracts text using configurable OCR settings
3. Cleans and normalizes extracted text
4. Includes extracted text in embeddings

## Search Capabilities

### Multimodal Search Strategies

The processor combines multiple search approaches:

1. **Text Similarity**: Semantic matching against document text
2. **Image Similarity**: Visual content matching (via text descriptions)
3. **Combined Similarity**: Unified text+image embedding matching

### Search Scoring

Results are scored using weighted combination:

- **50%** Text similarity score
- **30%** Image similarity score  
- **20%** Combined embedding score

### Image-Text Alignment

The processor calculates alignment scores between text and images using cosine similarity, providing insights into content coherence.

## Error Handling

### Graceful Degradation

The processor handles various failure scenarios:

- **Invalid Image Data**: Logs error, continues processing other images
- **Missing Files**: Warns about missing files, processes available content
- **OCR Failures**: Falls back to image-only processing
- **Embedding Failures**: Returns empty vectors, allows partial processing

### Error Reporting

Processing errors are collected in the `processing_errors` field of documents, allowing inspection of issues without blocking workflow.

## Performance Optimization

### Batch Processing

- Process multiple images concurrently using thread pools
- Configurable worker count for optimal resource usage
- Efficient memory management for large image collections

### Caching

- Image quality assessments are cached to avoid recomputation
- Embedding generation leverages existing Jina service optimizations
- Processed images can be reused across multiple documents

### Resource Management

- Automatic cleanup of temporary image data
- Configurable quality thresholds to skip low-quality images
- Memory-efficient processing of large image collections

## Integration Examples

### With Morgan Assistant

```python
from morgan.core.assistant import MorganAssistant
from morgan.jina.embeddings import MultimodalContentProcessor

# Extend Morgan with multimodal capabilities
class MultimodalMorgan(MorganAssistant):
    def __init__(self):
        super().__init__()
        self.multimodal_processor = MultimodalContentProcessor()
    
    def add_multimodal_document(self, content, images, metadata=None):
        # Process multimodal document
        doc = self.multimodal_processor.process_multimodal_document(
            content=content,
            images=images,
            metadata=metadata
        )
        
        # Add to knowledge base
        return self.add_document_to_collection(doc)
```

### With Vector Database

```python
from morgan.vector_db.client import QdrantClient

# Store multimodal embeddings
def store_multimodal_document(doc: MultimodalDocument, collection_name: str):
    client = QdrantClient()
    
    # Store with multimodal metadata
    client.upsert_points(
        collection_name=collection_name,
        points=[{
            "id": generate_id(),
            "vector": {
                "text": doc.text_embeddings,
                "combined": doc.combined_embedding
            },
            "payload": {
                "content": doc.text_content,
                "image_count": len(doc.images),
                "has_ocr_text": any(img.extracted_text for img in doc.images),
                "metadata": doc.metadata
            }
        }]
    )
```

## Dependencies

### Required

- `Pillow>=10.0.0` - Image processing
- `morgan.jina.embeddings.service` - Embedding generation

### Optional

- `pytesseract>=0.3.10` - OCR text extraction
- `tesseract` system binary - OCR engine

### Installation

```bash
# Install required dependencies
pip install Pillow>=10.0.0

# Install OCR dependencies (optional)
pip install pytesseract>=0.3.10

# Install Tesseract system binary
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from GitHub releases
```

## Troubleshooting

### Common Issues

1. **PIL Import Error**: Install Pillow with `pip install Pillow`
2. **OCR Not Working**: Install pytesseract and tesseract binary
3. **Low Quality Images**: Adjust `image_quality_threshold` parameter
4. **Memory Issues**: Reduce `max_workers` or process images in smaller batches

### Debug Information

```python
# Get detailed processing information
stats = processor.get_processing_stats()
print(f"Configuration: {stats}")

# Check document processing errors
for error in document.processing_errors:
    print(f"Processing error: {error}")

# Verify embedding dimensions
print(f"Text embedding dims: {len(document.text_embeddings)}")
print(f"Combined embedding dims: {len(document.combined_embedding)}")
```

## Best Practices

1. **Image Quality**: Use high-resolution images for better OCR results
2. **Batch Processing**: Process multiple documents together for efficiency
3. **Error Handling**: Always check `processing_errors` for issues
4. **Resource Management**: Call `processor.close()` when done
5. **Format Selection**: Prefer PNG over JPEG for text-heavy images
6. **Memory Usage**: Monitor memory usage with large image collections

## Requirements Satisfied

This implementation satisfies the following requirements:

- **R20.1**: jina-clip-v2 integration for multimodal embeddings
- **R20.2**: OCR integration for text extraction from images  
- **R20.3**: Image-text alignment and correlation analysis
- **R20.4**: Multimodal search capabilities across text and visual content
- **R20.5**: Graceful handling of image processing failures