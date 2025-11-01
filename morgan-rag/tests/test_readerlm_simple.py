#!/usr/bin/env python3
"""
Simple test for ReaderLM functionality
"""

import sys
import os

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.jina.models.reader_lm import JinaReaderLM


def test_readerlm_simple():
    """Test ReaderLM with a simple HTML example."""
    print("Testing ReaderLM with simple HTML...")
    
    # Simple HTML content for testing
    test_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Welcome to Test Page</h1>
        <p>This is a simple test paragraph with some content.</p>
        <p>Another paragraph with more information about the topic.</p>
        <div class="sidebar">Advertisement content here</div>
        <footer>Footer content</footer>
    </body>
    </html>
    """
    
    try:
        # Initialize ReaderLM
        reader = JinaReaderLM()
        
        # Test model loading
        print("Loading ReaderLM model...")
        if reader.load_model(use_pipeline=True):
            print("✓ Model loaded successfully")
        else:
            print("✗ Model loading failed")
            return
        
        # Test content extraction with HTML content
        print("Extracting content from HTML...")
        result = reader.extract_content(
            url="http://test.example", 
            html_content=test_html
        )
        
        print(f"Title: {result.title}")
        print(f"Content Length: {len(result.content)} chars")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Content Preview: {result.content[:200]}...")
        
        # Clean up
        reader.unload_model()
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_readerlm_simple()