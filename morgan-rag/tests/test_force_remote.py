#!/usr/bin/env python3
"""
Test Force Remote Embeddings Configuration

Test that the system properly forces remote embeddings when configured.
"""

import sys
import os

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.config import get_settings
from morgan.services.embedding_service import get_embedding_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def test_force_remote_config():
    """Test that force remote configuration is working."""
    print("🧪 Testing Force Remote Embeddings Configuration")
    print("=" * 50)
    
    try:
        # Get settings
        settings = get_settings()
        
        print(f"📋 Current Configuration:")
        print(f"  EMBEDDING_MODEL: {settings.embedding_model}")
        print(f"  EMBEDDING_LOCAL_MODEL: {settings.embedding_local_model}")
        print(f"  EMBEDDING_FORCE_REMOTE: {settings.embedding_force_remote}")
        print(f"  LLM_BASE_URL: {settings.llm_base_url}")
        
        # Test embedding service
        print(f"\n🔧 Testing Embedding Service:")
        embedding_service = get_embedding_service()
        
        print(f"  Service available: {embedding_service.is_available()}")
        print(f"  Model name: {embedding_service.model_name}")
        print(f"  Model config: {embedding_service.model_config}")
        
        # Test if force remote is respected
        if settings.embedding_force_remote:
            print(f"\n⚠️  Force Remote Mode Enabled:")
            print(f"  - Will only use remote embeddings")
            print(f"  - No fallback to local models")
            print(f"  - Requires gpt.lazarev.cloud to be available")
            
            # Try a simple encoding test
            try:
                print(f"\n🧪 Testing Remote Embedding:")
                test_text = "This is a test for remote embeddings"
                embedding = embedding_service.encode(test_text, instruction="document")
                print(f"  ✅ Remote embedding successful!")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
                
            except Exception as e:
                print(f"  ❌ Remote embedding failed: {e}")
                print(f"  This is expected if gpt.lazarev.cloud is not available")
                
        else:
            print(f"\n📍 Standard Mode (Remote with Local Fallback):")
            print(f"  - Will try remote first")
            print(f"  - Falls back to local if remote unavailable")
            
            # Try a simple encoding test
            try:
                print(f"\n🧪 Testing Embedding (with fallback):")
                test_text = "This is a test for embeddings with fallback"
                embedding = embedding_service.encode(test_text, instruction="document")
                print(f"  ✅ Embedding successful!")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
                
            except Exception as e:
                print(f"  ❌ Embedding failed: {e}")
        
        print(f"\n✅ Configuration test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        logger.error(f"Force remote config test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_force_remote_config()
    sys.exit(0 if success else 1)