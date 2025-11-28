#!/usr/bin/env python3
"""
Morgan Setup and Test Script

This script verifies and tests all Morgan components:
1. LLM connection (ai.ishosting.com)
2. Embedding service (qwen3:latest)
3. Qdrant vector database
4. Jina reranker models
5. Complete RAG pipeline

Usage:
    python scripts/setup_and_test.py
"""

import json
import os
import sys
from pathlib import Path

# Add local libs to path if they exist
local_libs = Path(__file__).parent.parent / "local_libs"
if local_libs.exists():
    sys.path.insert(0, str(local_libs))

# Add morgan-rag to path
morgan_rag = Path(__file__).parent.parent
sys.path.insert(0, str(morgan_rag))

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def test_llm_connection():
    """Test LLM connection to ai.ishosting.com"""
    console.print("\n[bold blue]1. Testing LLM Connection (ai.ishosting.com)[/bold blue]")

    try:
        # Test model list
        response = requests.get(
            "https://ai.ishosting.com/api/models",
            headers={"Authorization": "Bearer sk-7903d76996ad42f7ade45455d92745c2"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        models = [m.get("id") for m in data.get("data", [])]

        console.print(f"  ✅ Connected successfully")
        console.print(f"  ✅ Found {len(models)} models")

        # Check for specific models
        required_models = ["gemma3:latest", "qwen3:latest"]
        for model in required_models:
            if model in models:
                console.print(f"  ✅ Model available: {model}")
            else:
                console.print(f"  ⚠️  Model not found: {model}")

        # Test chat completion
        console.print("\n  Testing chat completion...")
        chat_response = requests.post(
            "https://ai.ishosting.com/api/chat/completions",
            headers={
                "Authorization": "Bearer sk-7903d76996ad42f7ade45455d92745c2",
                "Content-Type": "application/json"
            },
            json={
                "model": "gemma3:latest",
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=30
        )
        chat_response.raise_for_status()
        chat_data = chat_response.json()

        reply = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        console.print(f"  ✅ Chat completion working: {reply[:50]}")

        return True

    except Exception as e:
        console.print(f"  ❌ LLM connection failed: {e}")
        return False


def test_embeddings():
    """Test embedding service"""
    console.print("\n[bold blue]2. Testing Embedding Service (qwen3:latest)[/bold blue]")

    try:
        response = requests.post(
            "https://ai.ishosting.com/api/embeddings",
            headers={
                "Authorization": "Bearer sk-7903d76996ad42f7ade45455d92745c2",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen3:latest",
                "input": "This is a test embedding"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("data", [{}])[0].get("embedding", [])

        if embedding:
            console.print(f"  ✅ Embeddings working")
            console.print(f"  ✅ Dimension: {len(embedding)}")
            console.print(f"  ✅ First 5 values: {embedding[:5]}")
            return True
        else:
            console.print(f"  ❌ No embedding returned")
            return False

    except Exception as e:
        console.print(f"  ❌ Embedding service failed: {e}")
        return False


def test_qdrant():
    """Test Qdrant connection"""
    console.print("\n[bold blue]3. Testing Qdrant Vector Database[/bold blue]")

    try:
        # Try local first
        qdrant_url = "http://localhost:6333"
        response = requests.get(f"{qdrant_url}/collections", timeout=5)

        if response.status_code != 200:
            # Try docker service name
            qdrant_url = "http://qdrant:6333"
            response = requests.get(f"{qdrant_url}/collections", timeout=5)

        response.raise_for_status()
        data = response.json()

        collections = data.get("result", {}).get("collections", [])

        console.print(f"  ✅ Connected to: {qdrant_url}")
        console.print(f"  ✅ Found {len(collections)} collections")

        for coll in collections:
            console.print(f"    - {coll.get('name')}")

        return True

    except Exception as e:
        console.print(f"  ❌ Qdrant connection failed: {e}")
        return False


def test_jina_reranker():
    """Test Jina reranker setup"""
    console.print("\n[bold blue]4. Testing Jina Reranker Models[/bold blue]")

    try:
        # Check if HF token is set
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            console.print(f"  ✅ HuggingFace token: {hf_token[:10]}...")
        else:
            console.print(f"  ⚠️  No HuggingFace token set")

        # Check model cache directory
        cache_dir = Path(os.getenv("MORGAN_MODEL_CACHE_DIR", "./data/cache/models"))
        console.print(f"  📁 Model cache: {cache_dir}")

        if not cache_dir.exists():
            console.print(f"  ℹ️  Creating cache directory...")
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to import reranking service
        try:
            from morgan.jina.reranking.service import JinaRerankingService

            service = JinaRerankingService(enable_background=False)
            console.print(f"  ✅ Reranking service initialized")
            console.print(f"  ✅ Reranking enabled: {service.reranking_enabled}")

            return True

        except ImportError as ie:
            console.print(f"  ⚠️  Reranking service import failed: {ie}")
            console.print(f"  ℹ️  Will use basic similarity scoring")
            return False

    except Exception as e:
        console.print(f"  ⚠️  Reranker check failed: {e}")
        return False


def test_morgan_imports():
    """Test Morgan module imports"""
    console.print("\n[bold blue]5. Testing Morgan Module Imports[/bold blue]")

    modules_to_test = [
        ("morgan.config", "Settings"),
        ("morgan.services.embedding_service", "EmbeddingService"),
        ("morgan.jina.reranking.service", "JinaRerankingService"),
        ("morgan.core.memory", "ConversationMemory"),
    ]

    success_count = 0

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            console.print(f"  ✅ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            console.print(f"  ❌ {module_name}.{class_name}: {e}")

    console.print(f"\n  Imported {success_count}/{len(modules_to_test)} modules")
    return success_count == len(modules_to_test)


def show_summary(results):
    """Show test summary"""
    console.print("\n[bold yellow]═══ Test Summary ═══[/bold yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")

    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        table.add_row(component, status_icon)

    console.print(table)

    # Overall status
    all_passed = all(results.values())
    if all_passed:
        console.print("\n[bold green]✅ All tests passed! Morgan is ready to use.[/bold green]")
    else:
        failed = [k for k, v in results.items() if not v]
        console.print(f"\n[bold red]❌ Some tests failed: {', '.join(failed)}[/bold red]")
        console.print("[yellow]ℹ️  Morgan may have limited functionality[/yellow]")


def main():
    """Main test runner"""
    console.print(Panel.fit(
        "[bold cyan]Morgan Setup & Test Script[/bold cyan]\n"
        "Testing all components...",
        border_style="blue"
    ))

    results = {}

    # Run tests
    results["LLM Connection"] = test_llm_connection()
    results["Embedding Service"] = test_embeddings()
    results["Qdrant Database"] = test_qdrant()
    results["Jina Reranker"] = test_jina_reranker()
    results["Module Imports"] = test_morgan_imports()

    # Show summary
    show_summary(results)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
