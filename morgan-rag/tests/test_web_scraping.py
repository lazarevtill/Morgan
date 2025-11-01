#!/usr/bin/env python3
"""
Test script for Jina Web Scraping Service

This script tests the intelligent web scraper implementation using real
Hugging Face pages.
"""

import sys
import os
import logging

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.jina.scraping.service import JinaWebScrapingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_huggingface_urls():
    """Test scraping real Hugging Face model pages."""
    print("\n=== Testing Real Hugging Face URLs ===")

    # Real Hugging Face URLs for testing
    huggingface_urls = [
        "https://huggingface.co/jinaai/jina-embeddings-v4",
        "https://huggingface.co/jinaai/jina-code-embeddings-1.5b",
        "https://huggingface.co/jinaai/jina-clip-v2",
        "https://huggingface.co/jinaai/jina-reranker-v3",
        "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual",
        "https://huggingface.co/jinaai/ReaderLM-v2"
    ]

    scraper = JinaWebScrapingService()

    print(f"Testing {len(huggingface_urls)} Hugging Face model pages...")

    for i, url in enumerate(huggingface_urls, 1):
        print(f"\n{i}. Testing: {url}")
        try:
            # Test URL validation first
            validation = scraper.validate_url(url)
            print(f"   Valid: {validation['is_valid']}")
            print(f"   Domain: {validation['domain']}")

            # Scrape the content
            content = scraper.scrape_url(url, extract_images=True)

            print(f"   Title: {content.title}")
            print(f"   Method: {content.extraction_method}")
            print(f"   Quality: {content.extraction_quality:.2f}")
            print(f"   Time: {content.processing_time:.2f}s")
            print(f"   Content Length: {len(content.content)} chars")
            print(f"   Images Found: {len(content.images)}")

            if content.author:
                print(f"   Author: {content.author}")

            # Show first 200 characters of content
            preview = content.content[:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")

            # Show quality assessment if available
            if 'quality_assessment' in content.metadata:
                qa = content.metadata['quality_assessment']
                print("   Quality Details:")
                print(f"     Readability: {qa['readability_score']:.2f}")
                print(f"     Completeness: {qa['completeness_score']:.2f}")
                print(f"     Structure: {qa['structure_score']:.2f}")
                print(f"     Noise Level: {qa['noise_level']:.2f}")

        except Exception as e:
            print(f"   ERROR: {str(e)}")
            logger.error("Failed to scrape %s: %s", url, str(e))

    scraper.close()


def test_batch_scraping_huggingface():
    """Test batch scraping with Hugging Face URLs."""
    print("\n=== Testing Batch Scraping with Hugging Face URLs ===")

    # Subset of URLs for batch testing
    batch_urls = [
        "https://huggingface.co/jinaai/jina-embeddings-v4",
        "https://huggingface.co/jinaai/jina-reranker-v3",
        "https://huggingface.co/jinaai/ReaderLM-v2"
    ]

    scraper = JinaWebScrapingService()

    try:
        print(f"Batch scraping {len(batch_urls)} URLs...")
        results = scraper.batch_scrape(
            batch_urls,
            max_concurrent=2,
            extract_images=True
        )

        print(f"\nBatch Results ({len(results)} URLs processed):")
        for i, content in enumerate(results, 1):
            print(f"\n{i}. {content.url}")
            print(f"   Title: {content.title}")
            print(f"   Method: {content.extraction_method}")
            print(f"   Quality: {content.extraction_quality:.2f}")
            print(f"   Time: {content.processing_time:.2f}s")
            print(f"   Content Length: {len(content.content)} chars")
            print(f"   Images: {len(content.images)}")

            # Show brief content preview
            preview = content.content[:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")

    except Exception as e:
        print(f"Error in batch scraping: {str(e)}")
        logger.error("Batch scraping failed: %s", str(e))

    scraper.close()


def test_extraction_methods():
    """Test different extraction methods on a single URL."""
    print("\n=== Testing Extraction Methods ===")

    test_url = "https://huggingface.co/jinaai/jina-embeddings-v4"

    scraper = JinaWebScrapingService()

    print(f"Testing extraction methods on: {test_url}")

    # Test: Local ReaderLM with HTML fallback
    print("\n1. Testing local ReaderLM extraction (with HTML fallback):")
    try:
        content = scraper.scrape_url(
            test_url,
            extract_images=True
        )
        print(f"   Method: {content.extraction_method}")
        print(f"   Quality: {content.extraction_quality:.2f}")
        print(f"   Time: {content.processing_time:.2f}s")
        print(f"   Content Length: {len(content.content)} chars")
        
        # Show which method was actually used
        if content.extraction_method == "readerlm-v2-local":
            print("   ✓ Successfully used local ReaderLM")
        elif content.extraction_method == "html_parsing":
            print("   ⚠ Fell back to HTML parsing")
        
    except Exception as e:
        print(f"   Extraction failed: {str(e)}")

    scraper.close()


def test_url_validation():
    """Test URL validation with Hugging Face URLs."""
    print("\n=== Testing URL Validation ===")

    scraper = JinaWebScrapingService()

    test_urls = [
        "https://huggingface.co/jinaai/jina-embeddings-v4",
        "https://huggingface.co/jinaai/jina-reranker-v3",
        "invalid-url",
        "http://example.com",
        "https://github.com/jina-ai/jina"
    ]

    for url in test_urls:
        result = scraper.validate_url(url)
        print(f"\nURL: {url}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Domain: {result['domain']}")
        print(f"  Supported: {result['is_supported']}")
        print(f"  Estimated Quality: {result['estimated_quality']}")
        if result['warnings']:
            print(f"  Warnings: {', '.join(result['warnings'])}")

    scraper.close()


def main():
    """Run all tests with real Hugging Face URLs."""
    print("Testing Jina Web Scraping Service with Real URLs")
    print("=" * 60)

    try:
        # Test URL validation
        test_url_validation()

        # Test individual URL scraping with real Hugging Face pages
        test_huggingface_urls()

        # Test different extraction methods
        test_extraction_methods()

        # Test batch scraping
        test_batch_scraping_huggingface()

        print("\n" + "=" * 60)
        print("All web scraping tests completed!")
        print("Note: Some tests may fail if no Jina API key is configured.")
        print("The service will automatically fall back to HTML parsing.")

    except Exception as e:
        logger.error("Test suite failed: %s", str(e))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())