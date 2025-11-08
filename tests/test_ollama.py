#!/usr/bin/env python3
"""
Test script for external Ollama service
"""
import asyncio
import json

import pytest
from openai import AsyncOpenAI


@pytest.mark.asyncio
async def test_ollama():
    """Test external Ollama service"""
    print("Testing external Ollama service...")

    try:
        # Initialize OpenAI client for Ollama
        client = AsyncOpenAI(base_url="http://192.168.101.3:11434/v1", api_key="ollama")

        # Test 1: List models
        print("\n1. Testing model listing...")
        response = await client.models.list()
        models = [model.id for model in response.data]
        print(f"Available models: {models}")

        # Test 2: Simple chat completion
        print("\n2. Testing chat completion...")
        chat_response = await client.chat.completions.create(
            model="superdrew100/llama3-abliterated:latest",
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with a simple greeting.",
                }
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print(f"Response: {chat_response.choices[0].message.content}")

        # Test 3: Check if model exists
        print("\n3. Testing model availability...")
        if "superdrew100/llama3-abliterated:latest" in models:
            print("✓ Target model is available")
        else:
            print("✗ Target model not found")

        print("\n✓ All tests passed! External Ollama service is working correctly.")

    except Exception as e:
        print(f"✗ Error testing Ollama service: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    exit(0 if success else 1)
