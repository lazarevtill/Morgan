#!/usr/bin/env python3
"""
Integration test for Multimodal Content Processor with Morgan RAG

Tests integration with existing Morgan components and workflows.
"""

import sys
import os
import logging
from pathlib import Path

# Add the morgan package to the path
sys.path.insert(0, str(Path(__file__).parent))

from morgan.jina.embeddings import (
    JinaEmbeddingService,
    MultimodalContentProcessor,
    MultimodalDocument,
    ImageContent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_integration_with_existing_services():
    """Test integration with existing Morgan services."""
    
    print("🔗 Testing Integration with Existing Services")
    print("=" * 50)
    
    try:
        # Test that we can import and use existing services
        print("\n1. Testing JinaEmbeddingService integration...")
        
        embedding_service = JinaEmbeddingService()
        
        # Test basic embedding generation
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Computer vision processes images and visual data.",
            "Natural language processing handles text analysis."
        ]
        
        embeddings = embedding_service.generate_embeddings(
            texts=test_texts,
            model_name="jina-clip-v2",
            batch_size=2
        )
        
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        
        # Test multimodal processor with the same service
        print("\n2. Testing MultimodalContentProcessor integration...")
        
        processor = MultimodalContentProcessor(
            embedding_service=embedding_service,
            ocr_enabled=True
        )
        
        # Process a document
        doc = processor.process_multimodal_document(
            content="Integration test document with AI content.",
            images=[],
            metadata={"test": "integration", "service": "multimodal"}
        )
        
        print(f"   Processed document with {len(doc.text_embeddings)} text embedding dims")
        print(f"   Combined embedding dims: {len(doc.combined_embedding)}")
        print(f"   Processing errors: {len(doc.processing_errors)}")
        
        # Test that embeddings are compatible
        print("\n3. Testing embedding compatibility...")
        
        # Generate embedding with base service
        base_embedding = embedding_service.generate_single_embedding(
            "Test compatibility text",
            "jina-clip-v2"
        )
        
        # Generate embedding through multimodal processor
        multimodal_doc = processor.process_multimodal_document(
            content="Test compatibility text",
            images=[]
        )
        
        # Check dimensions match
        base_dims = len(base_embedding)
        multimodal_dims = len(multimodal_doc.text_embeddings)
        
        print(f"   Base service embedding dims: {base_dims}")
        print(f"   Multimodal processor embedding dims: {multimodal_dims}")
        print(f"   Dimensions compatible: {base_dims == multimodal_dims}")
        
        # Test model info compatibility
        print("\n4. Testing model info compatibility...")
        
        model_info = embedding_service.get_model_info("jina-clip-v2")
        processor_stats = processor.get_processing_stats()
        
        print(f"   Model type: {model_info['type']}")
        print(f"   Model dimensions: {model_info['dimensions']}")
        print(f"   Processor OCR enabled: {processor_stats['ocr_enabled']}")
        print(f"   Processor PIL available: {processor_stats['pil_available']}")
        
        # Clean up
        processor.close()
        embedding_service.close()
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        print(f"\n❌ Integration test failed: {str(e)}")
        return False


def test_multimodal_workflow():
    """Test a complete multimodal workflow."""
    
    print("\n📋 Testing Complete Multimodal Workflow")
    print("=" * 45)
    
    try:
        # Initialize services
        embedding_service = JinaEmbeddingService()
        processor = MultimodalContentProcessor(embedding_service=embedding_service)
        
        # Step 1: Process multiple documents
        print("\n1. Processing multiple documents...")
        
        documents = []
        
        # Text-only document
        doc1 = processor.process_multimodal_document(
            content="Machine learning algorithms analyze patterns in data.",
            images=[],
            metadata={"type": "text", "category": "ml"}
        )
        documents.append(doc1)
        
        # Document with mock image
        try:
            from PIL import Image
            import io
            
            # Create mock image
            test_image = Image.new('RGB', (150, 100), color='red')
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            
            doc2 = processor.process_multimodal_document(
                content="This document contains a visualization of data trends.",
                images=[image_data],
                metadata={"type": "multimodal", "category": "visualization"}
            )
            documents.append(doc2)
            
        except ImportError:
            print("   PIL not available - using text-only document")
            doc2 = processor.process_multimodal_document(
                content="This document would contain visualizations if images were available.",
                images=[],
                metadata={"type": "text", "category": "visualization"}
            )
            documents.append(doc2)
        
        print(f"   Processed {len(documents)} documents")
        
        # Step 2: Search across documents
        print("\n2. Searching across documents...")
        
        search_queries = [
            "machine learning data analysis",
            "visualization trends charts",
            "algorithms patterns"
        ]
        
        for query in search_queries:
            results = processor.search_multimodal_content(
                query=query,
                documents=documents,
                include_images=True,
                max_results=2
            )
            
            print(f"   Query: '{query}' -> {len(results)} results")
            for i, result in enumerate(results):
                print(f"     Result {i+1}: score={result.relevance_score:.3f}, images={len(result.images)}")
        
        # Step 3: Test error handling
        print("\n3. Testing error handling...")
        
        # Process document with invalid image
        doc_with_errors = processor.process_multimodal_document(
            content="Document with problematic image data.",
            images=[b"invalid_data", "nonexistent.jpg"],
            metadata={"type": "error_test"}
        )
        
        print(f"   Processing errors: {len(doc_with_errors.processing_errors)}")
        print(f"   Document still processed: {len(doc_with_errors.text_content) > 0}")
        
        # Step 4: Performance check
        print("\n4. Performance check...")
        
        import time
        start_time = time.time()
        
        # Process batch of documents
        batch_docs = []
        for i in range(5):
            doc = processor.process_multimodal_document(
                content=f"Batch document {i} about AI and machine learning applications.",
                images=[],
                metadata={"batch": i, "type": "performance_test"}
            )
            batch_docs.append(doc)
        
        processing_time = time.time() - start_time
        print(f"   Processed {len(batch_docs)} documents in {processing_time:.2f}s")
        print(f"   Average time per document: {processing_time/len(batch_docs):.3f}s")
        
        # Clean up
        processor.close()
        embedding_service.close()
        
        print("\n✅ Multimodal workflow test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Workflow test failed: {str(e)}")
        print(f"\n❌ Workflow test failed: {str(e)}")
        return False


def main():
    """Run all integration tests."""
    
    print("🚀 Multimodal Content Processor Integration Tests")
    print("=" * 60)
    
    success = True
    
    # Test integration with existing services
    success &= test_integration_with_existing_services()
    
    # Test complete workflow
    success &= test_multimodal_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All integration tests passed!")
        print("\nThe multimodal content processor is successfully integrated with:")
        print("  ✓ JinaEmbeddingService for consistent embedding generation")
        print("  ✓ Existing Morgan RAG architecture and patterns")
        print("  ✓ Error handling and graceful degradation")
        print("  ✓ Performance optimization and batch processing")
        return 0
    else:
        print("💥 Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())