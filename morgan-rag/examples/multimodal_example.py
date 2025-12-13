#!/usr/bin/env python3
"""
Multimodal Content Processing Example

Demonstrates how to use the MultimodalContentProcessor with Morgan RAG
for processing documents that contain both text and images.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the morgan package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.jina.embeddings.multimodal_service import (
    MultimodalContentProcessor,
    MultimodalDocument,
    MultimodalSearchResult,
    ImageContent
)
from morgan.jina.embeddings.service import JinaEmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalDocumentManager:
    """
    Example manager for multimodal documents in Morgan RAG.
    
    Demonstrates integration patterns for multimodal content processing.
    """
    
    def __init__(self):
        """Initialize the multimodal document manager."""
        self.embedding_service = JinaEmbeddingService()
        self.multimodal_processor = MultimodalContentProcessor(
            embedding_service=self.embedding_service,
            ocr_enabled=True,
            image_quality_threshold=0.3
        )
        self.documents: List[MultimodalDocument] = []
        
        logger.info("Initialized MultimodalDocumentManager")
    
    def add_document(
        self,
        content: str,
        images: List[Any] = None,
        metadata: Dict[str, Any] = None
    ) -> MultimodalDocument:
        """
        Add a multimodal document to the collection.
        
        Args:
            content: Text content of the document
            images: List of images (file paths, bytes, or PIL Images)
            metadata: Additional metadata for the document
            
        Returns:
            Processed MultimodalDocument
        """
        if images is None:
            images = []
        if metadata is None:
            metadata = {}
        
        logger.info(f"Adding document with {len(images)} images")
        
        # Process the document
        document = self.multimodal_processor.process_multimodal_document(
            content=content,
            images=images,
            metadata=metadata
        )
        
        # Add to collection
        self.documents.append(document)
        
        logger.info(f"Added document. Collection now has {len(self.documents)} documents")
        return document
    
    def search(
        self,
        query: str,
        include_images: bool = True,
        max_results: int = 10
    ) -> List[MultimodalSearchResult]:
        """
        Search through the multimodal document collection.
        
        Args:
            query: Search query
            include_images: Whether to include image matching
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        logger.info(f"Searching {len(self.documents)} documents for: '{query}'")
        
        results = self.multimodal_processor.search_multimodal_content(
            query=query,
            documents=self.documents,
            include_images=include_images,
            max_results=max_results
        )
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        total_docs = len(self.documents)
        docs_with_images = sum(1 for doc in self.documents if doc.images)
        total_images = sum(len(doc.images) for doc in self.documents)
        total_errors = sum(len(doc.processing_errors) for doc in self.documents)
        
        return {
            "total_documents": total_docs,
            "documents_with_images": docs_with_images,
            "total_images": total_images,
            "total_processing_errors": total_errors,
            "processor_stats": self.multimodal_processor.get_processing_stats()
        }
    
    def close(self):
        """Clean up resources."""
        self.multimodal_processor.close()
        self.embedding_service.close()
        logger.info("Closed MultimodalDocumentManager")


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for demonstration."""
    
    sample_docs = [
        {
            "content": """
            # Computer Vision in Healthcare
            
            Computer vision technology is revolutionizing healthcare by enabling 
            automated analysis of medical images. Deep learning models can now 
            detect diseases in X-rays, MRIs, and CT scans with accuracy matching 
            or exceeding human radiologists.
            
            Key applications include:
            - Cancer detection in mammograms
            - Diabetic retinopathy screening
            - Fracture detection in X-rays
            - Tumor segmentation in brain scans
            """,
            "images": [],
            "metadata": {
                "title": "Computer Vision in Healthcare",
                "category": "healthcare",
                "author": "Dr. Smith",
                "date": "2025-01-15"
            }
        },
        {
            "content": """
            # Natural Language Processing Advances
            
            Recent advances in natural language processing have led to more 
            sophisticated AI assistants and chatbots. Large language models 
            can now understand context, generate human-like text, and perform 
            complex reasoning tasks.
            
            Applications include:
            - Conversational AI assistants
            - Document summarization
            - Language translation
            - Code generation
            """,
            "images": [],
            "metadata": {
                "title": "NLP Advances",
                "category": "nlp",
                "author": "Prof. Johnson",
                "date": "2025-01-20"
            }
        },
        {
            "content": """
            # Machine Learning in Finance
            
            Machine learning algorithms are transforming the financial industry 
            by enabling better risk assessment, fraud detection, and algorithmic 
            trading. Banks and financial institutions are leveraging AI to 
            improve customer service and operational efficiency.
            
            Key use cases:
            - Credit scoring and risk assessment
            - Fraud detection and prevention
            - Algorithmic trading strategies
            - Customer service chatbots
            """,
            "images": [],
            "metadata": {
                "title": "ML in Finance",
                "category": "finance",
                "author": "Jane Doe",
                "date": "2025-01-25"
            }
        }
    ]
    
    return sample_docs


def demonstrate_multimodal_processing():
    """Demonstrate multimodal content processing capabilities."""
    
    print("🔍 Multimodal Content Processing Demonstration")
    print("=" * 60)
    
    # Initialize the document manager
    manager = MultimodalDocumentManager()
    
    try:
        # Add sample documents
        print("\n1. Adding sample documents...")
        sample_docs = create_sample_documents()
        
        for i, doc_data in enumerate(sample_docs):
            document = manager.add_document(
                content=doc_data["content"],
                images=doc_data["images"],
                metadata=doc_data["metadata"]
            )
            print(f"   Added document {i+1}: {doc_data['metadata']['title']}")
            print(f"     Text length: {len(document.text_content)} chars")
            print(f"     Images: {len(document.images)}")
            print(f"     Errors: {len(document.processing_errors)}")
        
        # Add a document with mock images
        print("\n2. Adding document with images...")
        try:
            from PIL import Image
            import io
            
            # Create mock images
            chart_image = Image.new('RGB', (300, 200), color='lightblue')
            diagram_image = Image.new('RGB', (250, 250), color='lightgreen')
            
            # Convert to bytes
            chart_buffer = io.BytesIO()
            chart_image.save(chart_buffer, format='PNG')
            chart_data = chart_buffer.getvalue()
            
            diagram_buffer = io.BytesIO()
            diagram_image.save(diagram_buffer, format='PNG')
            diagram_data = diagram_buffer.getvalue()
            
            # Add multimodal document
            multimodal_doc = manager.add_document(
                content="""
                # AI Performance Metrics Dashboard
                
                This document contains charts and diagrams showing the performance 
                of various AI models across different tasks. The visualizations 
                help understand model accuracy, training time, and resource usage.
                
                The first chart shows accuracy trends over time, while the second 
                diagram illustrates the model architecture and data flow.
                """,
                images=[chart_data, diagram_data],
                metadata={
                    "title": "AI Performance Dashboard",
                    "category": "analytics",
                    "author": "Data Team",
                    "date": "2025-01-30",
                    "has_visualizations": True
                }
            )
            
            print(f"   Added multimodal document: AI Performance Dashboard")
            print(f"     Text length: {len(multimodal_doc.text_content)} chars")
            print(f"     Images: {len(multimodal_doc.images)}")
            print(f"     Combined embedding dim: {len(multimodal_doc.combined_embedding)}")
            
        except ImportError:
            print("   PIL not available - skipping image document")
        
        # Display collection statistics
        print("\n3. Collection Statistics:")
        stats = manager.get_statistics()
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Documents with Images: {stats['documents_with_images']}")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Processing Errors: {stats['total_processing_errors']}")
        
        processor_stats = stats['processor_stats']
        print(f"   OCR Enabled: {processor_stats['ocr_enabled']}")
        print(f"   PIL Available: {processor_stats['pil_available']}")
        
        # Perform searches
        print("\n4. Performing multimodal searches...")
        
        search_queries = [
            "computer vision healthcare medical images",
            "natural language processing chatbots",
            "machine learning finance fraud detection",
            "AI performance metrics charts diagrams"
        ]
        
        for query in search_queries:
            print(f"\n   Query: '{query}'")
            results = manager.search(
                query=query,
                include_images=True,
                max_results=3
            )
            
            for i, result in enumerate(results):
                print(f"     Result {i+1}:")
                print(f"       Relevance: {result.relevance_score:.3f}")
                print(f"       Text Score: {result.text_score:.3f}")
                print(f"       Image Score: {result.image_score:.3f}")
                print(f"       Title: {result.metadata.get('title', 'Unknown')}")
                print(f"       Images: {len(result.images)}")
        
        # Test image-only search
        print("\n5. Testing image-focused search...")
        image_results = manager.search(
            query="charts diagrams visualizations performance",
            include_images=True,
            max_results=5
        )
        
        print(f"   Found {len(image_results)} results for image-focused query")
        for i, result in enumerate(image_results):
            print(f"     Result {i+1}: {result.metadata.get('title', 'Unknown')}")
            print(f"       Images: {len(result.images)}, Score: {result.relevance_score:.3f}")
        
        print("\n✅ Multimodal processing demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\n❌ Demonstration failed: {str(e)}")
    
    finally:
        # Clean up
        manager.close()


def main():
    """Run the multimodal processing demonstration."""
    
    print("🚀 Morgan RAG Multimodal Content Processing")
    print("=" * 70)
    print("This example demonstrates how to process and search documents")
    print("that contain both text and images using jina-clip-v2 integration.")
    print()
    
    demonstrate_multimodal_processing()
    
    print("\n" + "=" * 70)
    print("🎉 Demonstration completed!")
    print("\nKey features demonstrated:")
    print("  • Multimodal document processing with text and images")
    print("  • jina-clip-v2 integration for combined embeddings")
    print("  • OCR text extraction from images (when available)")
    print("  • Image-text alignment and correlation analysis")
    print("  • Multimodal search across text and visual content")
    print("  • Graceful error handling for image processing failures")


if __name__ == "__main__":
    main()