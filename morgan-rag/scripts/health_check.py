#!/usr/bin/env python3
"""
Morgan Health Check - Verify all services are working

Tests:
1. Configuration loaded
2. LLM service (ai.ishosting.com)
3. Embedding service
4. Qdrant vector database
5. Collections exist
6. Can query data
"""

import sys
from pathlib import Path

# Add paths
local_libs = Path(__file__).parent.parent / "local_libs"
if local_libs.exists():
    sys.path.insert(0, str(local_libs))
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from rich.console import Console
from rich.table import Table

console = Console()

def check_config():
    """Check configuration loads"""
    try:
        from morgan.config import get_settings
        settings = get_settings()
        console.print("✅ Configuration loaded")
        console.print(f"   LLM: {settings.llm_model} @ {settings.llm_base_url}")
        console.print(f"   Embedding: {settings.embedding_model}")
        console.print(f"   Qdrant: {settings.qdrant_url}")
        return True
    except Exception as e:
        console.print(f"❌ Configuration failed: {e}")
        return False


def check_llm():
    """Check LLM service"""
    try:
        from morgan.services.llm_service import get_llm_service
        llm = get_llm_service()
        response = llm.generate("Say 'OK'", max_tokens=5)
        console.print(f"✅ LLM service working")
        console.print(f"   Response: {response.content[:50]}")
        return True
    except Exception as e:
        console.print(f"❌ LLM service failed: {e}")
        return False


def check_embeddings():
    """Check embedding service"""
    try:
        from morgan.services.embedding_service import EmbeddingService
        service = EmbeddingService()
        emb = service.encode("test")
        console.print(f"✅ Embedding service working")
        console.print(f"   Dimensions: {len(emb)}")
        return True
    except Exception as e:
        console.print(f"❌ Embedding service failed: {e}")
        return False


def check_qdrant():
    """Check Qdrant database"""
    try:
        resp = requests.get("http://localhost:6333/collections")
        data = resp.json()
        collections = data["result"]["collections"]

        console.print(f"✅ Qdrant connected")
        console.print(f"   Collections: {len(collections)}")

        # Check each collection
        for coll in collections:
            name = coll["name"]
            detail = requests.get(f"http://localhost:6333/collections/{name}").json()
            points = detail["result"]["points_count"]
            console.print(f"   - {name}: {points} points")

        return True
    except Exception as e:
        console.print(f"❌ Qdrant failed: {e}")
        return False


def check_memory():
    """Check memory system"""
    try:
        from morgan.core.memory import ConversationMemory
        memory = ConversationMemory()
        console.print("✅ Memory system initialized")
        return True
    except Exception as e:
        console.print(f"❌ Memory system failed: {e}")
        return False


def check_assistant():
    """Check Morgan assistant"""
    try:
        from morgan.core.assistant import MorganAssistant
        assistant = MorganAssistant()
        console.print("✅ Morgan assistant ready")

        # Test simple query
        response = assistant.ask("Hello", user_id="health_check")
        console.print(f"   Test response: {response.answer[:50]}...")
        return True
    except Exception as e:
        console.print(f"❌ Assistant failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all health checks"""
    console.print("\n[bold cyan]Morgan Health Check[/bold cyan]")
    console.print("=" * 60)

    checks = {
        "Configuration": check_config,
        "LLM Service": check_llm,
        "Embedding Service": check_embeddings,
        "Qdrant Database": check_qdrant,
        "Memory System": check_memory,
        "Morgan Assistant": check_assistant,
    }

    results = {}
    for name, check_func in checks.items():
        console.print(f"\n[yellow]Checking {name}...[/yellow]")
        results[name] = check_func()

    # Summary
    console.print("\n" + "=" * 60)
    console.print("\n[bold]Health Check Summary:[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Component")
    table.add_column("Status", justify="center")

    for name, status in results.items():
        status_str = "[green]✅ PASS[/green]" if status else "[red]❌ FAIL[/red]"
        table.add_row(name, status_str)

    console.print(table)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    console.print(f"\n[bold]Overall: {passed}/{total} checks passed[/bold]")

    if passed == total:
        console.print("[green]✅ All systems operational![/green]")
        return 0
    else:
        console.print("[red]⚠️  Some systems need attention[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
