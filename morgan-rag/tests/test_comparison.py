#!/usr/bin/env python3
"""
Compare ReaderLM vs HTML parsing results
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.jina.scraping.service import JinaWebScrapingService


def compare_methods():
    """Compare extraction methods."""
    test_url = "https://huggingface.co/jinaai/jina-reranker-v3"

    scraper = JinaWebScrapingService()

    print(f"Comparing extraction methods for: {test_url}")
    print("=" * 60)

    # Test the scraper (will try ReaderLM first, then fallback)
    content = scraper.scrape_url(test_url, extract_images=True)

    print(f"Method Used: {content.extraction_method}")
    print(f"Title: {content.title}")
    print(f"Quality: {content.extraction_quality:.2f}")
    print(f"Content Length: {len(content.content)} chars")
    print(f"Processing Time: {content.processing_time:.2f}s")

    if content.extraction_method == "readerlm-v2-local":
        print("✅ Successfully used local ReaderLM!")
        print("\nFirst 500 characters of extracted content:")
        print("-" * 40)
        print(content.content[:500])
        print("-" * 40)
    else:
        print("⚠️  Fell back to HTML parsing")

    scraper.close()


if __name__ == "__main__":
    compare_methods()
