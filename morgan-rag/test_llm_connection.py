#!/usr/bin/env python3
"""
Test LLM connection and verify Morgan setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from morgan.config import get_settings
from morgan.services.llm_service import LLMService
from morgan.embeddings.service import get_embedding_service

def test_settings():
    """Test settings loading."""
    print("=" * 60)
    print("Testing Settings Configuration")
    print("=" * 60)
    
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully")
        print(f"  - LLM Base URL: {settings.llm_base_url}")
        print(f"  - LLM Model: {settings.llm_model}")
        print(f"  - LLM API Key: {'***' if settings.llm_api_key else 'Not set'}")
        print(f"  - Embedding Model: {settings.embedding_model}")
        print(f"  - OLLAMA_HOST: {getattr(settings, 'ollama_host', 'Not set')}")
        print(f"  - Embedding Base URL: {settings.embedding_base_url or 'Not set (will use OLLAMA_HOST or LLM_BASE_URL)'}")
        print(f"  - Qdrant URL: {settings.qdrant_url}")
        return True
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False

def test_llm_connection():
    """Test LLM service connection."""
    print("\n" + "=" * 60)
    print("Testing LLM Connection")
    print("=" * 60)
    
    try:
        llm_service = LLMService()
        print("✅ LLM service initialized")
        
        # Test with a simple prompt
        print("\n📤 Sending test prompt...")
        response = llm_service.generate(
            prompt="Say 'Hello, Morgan is working!' in one sentence.",
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"✅ LLM response received:")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.content[:200]}...")
        print(f"  Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_service():
    """Test embedding service."""
    print("\n" + "=" * 60)
    print("Testing Embedding Service")
    print("=" * 60)
    
    try:
        embedding_service = get_embedding_service()
        print("✅ Embedding service initialized")
        
        # Check availability
        if embedding_service.is_available():
            print("✅ Embedding service is available")
            print(f"  Model: {embedding_service.model_name}")
            print(f"  Dimensions: {embedding_service.get_embedding_dimension()}")
            
            # Test encoding
            print("\n📤 Testing embedding generation...")
            embedding = embedding_service.encode("Test embedding", instruction="query")
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
            print(f"  First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ Embedding service is not available")
            return False
    except Exception as e:
        print(f"❌ Embedding service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Morgan RAG Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Test 1: Settings
    results.append(("Settings", test_settings()))
    
    # Test 2: LLM Connection
    if results[0][1]:  # Only test if settings loaded
        results.append(("LLM Connection", test_llm_connection()))
    
    # Test 3: Embedding Service
    if results[0][1]:  # Only test if settings loaded
        results.append(("Embedding Service", test_embedding_service()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Morgan is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


