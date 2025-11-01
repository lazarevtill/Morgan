#!/usr/bin/env python3
"""
Test script for Multimodal Content Processor

Tests the jina-clip-v2 integration, OCR capabilities, and multimodal search.
"""

import sys
import os
import logging
from pathlib import Path

# Add the morgan package to the path
sys.path.insert(0, str(Path(__file__).parent))

from morgan.jina.embeddings.multimodal_service import (
    MultimodalContentProcessor,
    ImageContent,
    MultimodalDocument
)
from morgan.jina.embeddings.service import JinaEmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_multimodal_processor():
    """Test the multimodal content processor functionality."""
    
    print("🔍 Testing Multimodal Content Processor")
    print("=" * 50)
    
    try:
        # Initialize the processor
        print("\n1. Initializing MultimodalContentProcessor...")
        embedding_service = JinaEmbeddingService()
        processor = MultimodalContentProcessor(
            embedding_service=embedding_service,
            ocr_enabled=True,
            image_quality_threshold=0.3
        )
        
        # Get processing stats
        stats = processor.get_processing_stats()
        print(f"   OCR Enabled: {stats['ocr_enabled']}")
        print(f"   PIL Available: {stats['pil_available']}")
        print(f"   Tesseract Available: {stats['tesseract_available']}")
        print(f"   Supported Formats: {stats['supported_formats']}")
        
        # Test with text-only content (no images)
        print("\n2. Testing text-only processing...")
        text_content = """
        This is a sample document about artificial intelligence and machine learning.
        It contains information about neural networks, deep learning, and computer vision.
        The document discusses various applications of AI in different industries.
        """
        
        doc_no_images = processor.process_multimodal_document(
            content=text_content.strip(),
            images=[],
            metadata={"source": "test_document", "type": "text"}
        )
        
        print(f"   Text Content Length: {len(doc_no_images.text_content)}")
        print(f"   Text Embeddings Dimension: {len(doc_no_images.text_embeddings)}")
        print(f"   Combined Embeddings Dimension: {len(doc_no_images.combined_embedding)}")
        print(f"   Processing Errors: {len(doc_no_images.processing_errors)}")
        
        # Test with mock image data (since we don't have actual image files)
        print("\n3. Testing multimodal processing with mock images...")
        
        # Create mock image data
        try:
            from PIL import Image
            import io
            
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='red')
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            mock_image_data = buffer.getvalue()
            
            # Process document with mock image
            doc_with_images = processor.process_multimodal_document(
                content=text_content.strip(),
                images=[mock_image_data],
                metadata={"source": "test_document_with_images", "type": "multimodal"}
            )
            
            print(f"   Text Content Length: {len(doc_with_images.text_content)}")
            print(f"   Number of Images: {len(doc_with_images.images)}")
            print(f"   Text Embeddings Dimension: {len(doc_with_images.text_embeddings)}")
            print(f"   Image Embeddings Count: {len(doc_with_images.image_embeddings)}")
            print(f"   Combined Embeddings Dimension: {len(doc_with_images.combined_embedding)}")
            print(f"   Processing Errors: {len(doc_with_images.processing_errors)}")
            
            if doc_with_images.images:
                image = doc_with_images.images[0]
                print(f"   Image Format: {image.format}")
                print(f"   Image Size: {image.width}x{image.height}")
                print(f"   Image Quality Score: {image.quality_score:.2f}")
                print(f"   Extracted Text: '{image.extracted_text}'")
            
        except ImportError:
            print("   PIL not available - skipping image processing test")
            doc_with_images = doc_no_images
        
        # Test multimodal search
        print("\n4. Testing multimodal search...")
        
        # Create a small collection of documents
        documents = [doc_no_images]
        if 'doc_with_images' in locals():
            documents.append(doc_with_images)
        
        # Add another test document
        doc2 = processor.process_multimodal_document(
            content="Computer vision and image processing are important fields in AI research.",
            images=[],
            metadata={"source": "test_document_2", "type": "text"}
        )
        documents.append(doc2)
        
        # Perform search
        search_results = processor.search_multimodal_content(
            query="artificial intelligence machine learning",
            documents=documents,
            include_images=True,
            max_results=5
        )
        
        print(f"   Search Results Count: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"   Result {i+1}:")
            print(f"     Relevance Score: {result.relevance_score:.3f}")
            print(f"     Text Score: {result.text_score:.3f}")
            print(f"     Image Score: {result.image_score:.3f}")
            print(f"     Combined Score: {result.combined_score:.3f}")
            print(f"     Content Preview: {result.content[:100]}...")
            print(f"     Images Count: {len(result.images)}")
        
        # Test error handling
        print("\n5. Testing error handling...")
        
        # Test with invalid image data
        doc_with_errors = processor.process_multimodal_document(
            content="Test document with invalid image",
            images=[b"invalid_image_data", "nonexistent_file.jpg"],
            metadata={"source": "error_test", "type": "test"}
        )
        
        print(f"   Processing Errors: {len(doc_with_errors.processing_errors)}")
        for error in doc_with_errors.processing_errors:
            print(f"     - {error}")
        
        # Clean up
        processor.close()
        embedding_service.close()
        
        print("\n✅ Multimodal Content Processor test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        return False


def test_image_content():
    """Test ImageContent functionality."""
    
    print("\n🖼️  Testing ImageContent")
    print("=" * 30)
    
    try:
        from PIL import Image
        import io
        
        # Create test image
        test_image = Image.new('RGB', (200, 150), color='blue')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        # Create ImageContent
        image_content = ImageContent(
            image_data=image_data,
            format='jpeg',
            width=200,
            height=150,
            file_path=None,
            extracted_text="Sample extracted text",
            quality_score=0.8
        )
        
        print(f"   Format: {image_content.format}")
        print(f"   Size: {image_content.width}x{image_content.height}")
        print(f"   Quality Score: {image_content.quality_score}")
        print(f"   Extracted Text: '{image_content.extracted_text}'")
        print(f"   Data Size: {len(image_content.image_data)} bytes")
        
        print("✅ ImageContent test completed successfully!")
        return True
        
    except ImportError:
        print("PIL not available - skipping ImageContent test")
        return True
    except Exception as e:
        print(f"❌ ImageContent test failed: {str(e)}")
        return False


def main():
    """Run all multimodal processor tests."""
    
    print("🚀 Starting Multimodal Content Processor Tests")
    print("=" * 60)
    
    success = True
    
    # Test ImageContent
    success &= test_image_content()
    
    # Test MultimodalContentProcessor
    success &= test_multimodal_processor()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())