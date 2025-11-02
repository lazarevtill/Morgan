#!/usr/bin/env python3
"""
Preview the actual content being extracted from Hugging Face pages
"""

import sys
import os

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.jina.scraping.service import JinaWebScrapingService


def preview_content():
    """Show the actual content being extracted."""
    test_url = "https://huggingface.co/jinaai/jina-embeddings-v4"
    
    scraper = JinaWebScrapingService()
    
    print(f"Extracting content from: {test_url}")
    print("=" * 60)
    
    content = scraper.scrape_url(test_url, extract_images=True)
    
    print(f"Title: {content.title}")
    print(f"Method: {content.extraction_method}")
    print(f"Quality: {content.extraction_quality:.2f}")
    print(f"Content Length: {len(content.content)} characters")
    print(f"Images Found: {len(content.images)}")
    
    print("\nFull Content:")
    print("-" * 40)
    print(content.content)
    print("-" * 40)
    
    print(f"\nMetadata:")
    for key, value in content.metadata.items():
        if key != 'quality_assessment':
            print(f"  {key}: {value}")
    
    if content.images:
        print(f"\nImages found:")
        for i, img in enumerate(content.images, 1):
            print(f"  {i}. {img}")
    
    scraper.close()


if __name__ == "__main__":
    preview_content()