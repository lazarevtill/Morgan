#!/usr/bin/env python3
"""
Test Morgan RAG setup - verify LLM and embedding services work.
"""

import sys
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_llm_direct():
    """Test LLM connection directly via API."""
    print("=" * 60)
    print("Testing LLM Connection (Direct API)")
    print("=" * 60)

    try:
        from morgan.config import get_settings

        settings = get_settings()

        print(f"LLM Base URL: {settings.llm_base_url}")
        print(f"LLM Model: {settings.llm_model}")
        print(f"API Key: {'***' if settings.llm_api_key else 'Not set'}")

        from openai import OpenAI

        client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key or "dummy-key"
        )

        print("\n📤 Sending test request to LLM...")
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, Morgan is working!' in one sentence.",
                }
            ],
            max_tokens=50,
            timeout=30.0,
        )

        print("✅ LLM Response received:")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.choices[0].message.content}")
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
        from morgan.services.embeddings import get_embedding_service
        from morgan.config import get_settings

        settings = get_settings()
        print(f"OLLAMA_HOST: {getattr(settings, 'ollama_host', 'Not set')}")
        print(f"Embedding Model: {settings.embedding_model}")
        print(
            f"Embedding Base URL: {settings.embedding_base_url or 'Not set (will use OLLAMA_HOST)'}"
        )

        embedding_service = get_embedding_service()

        if embedding_service.is_available():
            print("✅ Embedding service is available")
            print(f"  Model: {embedding_service.model_name}")
            print(f"  Dimensions: {embedding_service.get_embedding_dimension()}")

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


def test_web_service():
    """Test web service endpoints."""
    print("\n" + "=" * 60)
    print("Testing Web Service")
    print("=" * 60)

    endpoints = [
        "http://localhost:8080/health",
        "http://localhost:8000/health",
        "http://localhost:8080/api/health",
    ]

    for endpoint in endpoints:
        try:
            print(f"\nTesting {endpoint}...")
            response = requests.get(endpoint, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:100]}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection refused (service not running)")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

    return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Morgan RAG Setup Verification")
    print("=" * 60)

    results = []

    # Test 1: LLM Connection
    print("\n[1/3] Testing LLM Connection...")
    results.append(("LLM Connection", test_llm_direct()))

    # Test 2: Embedding Service
    print("\n[2/3] Testing Embedding Service...")
    results.append(("Embedding Service", test_embedding_service()))

    # Test 3: Web Service
    print("\n[3/3] Testing Web Service...")
    results.append(("Web Service", test_web_service()))

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
        print("\nTo check service logs:")
        print("  docker compose logs -f morgan")
        return 1


if __name__ == "__main__":
    sys.exit(main())
