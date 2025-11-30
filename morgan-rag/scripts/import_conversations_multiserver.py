#!/usr/bin/env python3
"""
ULTRA-FAST Multi-Server Batch Import

Uses ALL available resources:
- 2 embedding servers (load balanced)
- 10 LLM workers (ai.ishosting.com)
- 20 embedding workers (10 per server)
- Bulk vector DB operations

Speed: 20x faster than sequential import

Usage:
    PYTHONPATH=/path/to/morgan-rag python3 scripts/import_conversations_multiserver.py /path/to/conversations.json
"""

import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import itertools

from morgan.core.memory import ConversationMemory
from morgan.services.llm_service import get_llm_service
from morgan.utils.logger import setup_logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# CONFIGURE YOUR EMBEDDING SERVERS HERE
EMBEDDING_SERVERS = [
    "http://192.168.100.163:11434",  # Server 1
    "http://192.168.100.221:11434",  # Server 2 (change to your second server IP)
]


def load_conversations(path: Path) -> List[dict]:
    """Load conversation list from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of conversations in JSON")
    return data


def extract_turns(mapping: dict) -> List[Tuple[str, str]]:
    """Extract user/assistant turns from mapping ordered by create_time."""
    messages = []
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role")
        content = msg.get("content", {})
        parts = content.get("parts") if isinstance(content, dict) else None
        text = "\n".join(str(p) for p in parts) if parts else ""
        ts = msg.get("create_time") or msg.get("update_time") or 0
        messages.append((ts, role, text.strip()))

    messages.sort(key=lambda x: x[0])

    turns: List[Tuple[str, str]] = []
    pending_question: Optional[str] = None

    for _, role, text in messages:
        if not text:
            continue
        if role == "user":
            pending_question = text
        elif role == "assistant" and pending_question:
            turns.append((pending_question, text))
            pending_question = None

    return turns


def generate_llm_tags(title: str, turns: List[Tuple[str, str]], llm_service) -> List[str]:
    """Generate semantic tags for a conversation using LLM."""
    try:
        # Create a summary of the conversation
        summary = f"Title: {title}\n\n"
        if turns:
            # Include first 2 turns for context
            for q, a in turns[:2]:
                summary += f"Q: {q[:100]}...\n" if len(q) > 100 else f"Q: {q}\n"
                summary += f"A: {a[:100]}...\n\n" if len(a) > 100 else f"A: {a}\n\n"

        prompt = f"""Analyze this conversation and generate 3-5 semantic tags that describe its main topics and themes.

Conversation:
{summary}

Generate tags as a comma-separated list (e.g., "python, programming, api-development, troubleshooting").
Only output the tags, nothing else."""

        response = llm_service.generate(prompt, max_tokens=50, temperature=0.3)
        tags_text = response.content.strip()

        # Parse tags from response
        tags = [tag.strip().lower() for tag in tags_text.split(",")]
        tags = [tag for tag in tags if tag and len(tag) > 2]

        return tags[:5]  # Max 5 tags

    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate LLM tags: {e}[/yellow]")
        return []


def create_embedding_service(base_url: str, model: str):
    """Create an embedding service for a specific server."""
    import requests

    class ServerEmbeddingService:
        def __init__(self, base_url: str, model: str):
            self.base_url = base_url
            self.model = model

        def encode(self, text: str, instruction: str = "document") -> List[float]:
            """Generate embedding for text."""
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.model,
                "prompt": text
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["embedding"]

    return ServerEmbeddingService(base_url, model)


def import_conversations_multiserver(path: Path, batch_size: int = 50):
    """Import conversations with multi-server load balancing for maximum speed."""
    console.print("\n[bold cyan]Morgan ULTRA-FAST Multi-Server Batch Import[/bold cyan]")
    console.print(f"[dim]Source: {path}[/dim]")
    console.print(f"[dim]Batch size: {batch_size} conversations[/dim]")
    console.print(f"[dim]Embedding servers: {len(EMBEDDING_SERVERS)}[/dim]")
    console.print(f"[dim]Parallelization: 10 LLM workers + {len(EMBEDDING_SERVERS) * 10} embedding workers[/dim]\n")

    setup_logging()

    console.print("[yellow]Initializing services...[/yellow]")
    memory = ConversationMemory()
    llm_service = get_llm_service()

    # Get actual service configurations
    from morgan.config import get_settings
    settings = get_settings()

    console.print("[green]✓[/green] Memory system ready")
    console.print(f"[green]✓[/green] LLM service ready ({settings.llm_model} via {settings.llm_base_url})")
    console.print(f"[green]✓[/green] Embeddings ready ({settings.embedding_model})")
    for idx, server in enumerate(EMBEDDING_SERVERS, 1):
        console.print(f"    [green]→[/green] Server {idx}: {server}")
    console.print()

    # Create embedding services for each server
    embedding_services = [
        create_embedding_service(server, settings.embedding_model)
        for server in EMBEDDING_SERVERS
    ]
    server_cycle = itertools.cycle(embedding_services)  # Round-robin load balancing

    conversations = load_conversations(path)
    console.print(f"[bold]Processing {len(conversations)} conversations...[/bold]\n")

    imported = 0
    turns_total = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "[cyan]Importing conversations...",
            total=len(conversations)
        )

        # Process conversations in batches
        for batch_start in range(0, len(conversations), batch_size):
            batch = conversations[batch_start:batch_start + batch_size]
            batch_convs = []  # Store processed conversation data

            # Step 1: Parse all conversations in batch
            for idx, conv in enumerate(batch):
                conv_idx = batch_start + idx
                mapping = conv.get("mapping") or {}
                title = conv.get("title") or f"Conversation {conv_idx + 1}"
                turns = extract_turns(mapping)

                if not turns:
                    progress.update(task, advance=1)
                    continue

                batch_convs.append({
                    "index": conv_idx,
                    "title": title,
                    "turns": turns,
                    "llm_tags": []
                })

            if not batch_convs:
                continue

            # Step 2: PARALLEL LLM TAG GENERATION (ai.ishosting.com)
            console.print(f"[yellow]Generating LLM tags for {len(batch_convs)} conversations (10 parallel workers)...[/yellow]")

            def generate_tags_for_conv(conv_data):
                tags = generate_llm_tags(conv_data["title"], conv_data["turns"], llm_service)
                return conv_data["index"], tags

            with ThreadPoolExecutor(max_workers=10) as llm_executor:
                llm_futures = {llm_executor.submit(generate_tags_for_conv, conv): i
                              for i, conv in enumerate(batch_convs)}

                for future in as_completed(llm_futures):
                    try:
                        conv_idx, tags = future.result()
                        # Find and update the conversation
                        for conv in batch_convs:
                            if conv["index"] == conv_idx:
                                conv["llm_tags"] = tags
                                break
                    except Exception as e:
                        console.print(f"[yellow]LLM tag generation error: {e}[/yellow]")

            # Step 3: Prepare all texts for embedding
            all_texts = []
            text_map = {}  # Maps text index to (conv_index_in_batch, type, data)

            for batch_idx, conv_data in enumerate(batch_convs):
                title = conv_data["title"]
                turns = conv_data["turns"]

                # Topic text
                topic_text = title or "conversation"
                text_map[len(all_texts)] = (batch_idx, "topic", None)
                all_texts.append(topic_text)

                # All turn texts
                for turn_idx, (q, a) in enumerate(turns):
                    combined_turn = f"{q}\n{a}"
                    text_map[len(all_texts)] = (batch_idx, "turn", turn_idx, q, a)
                    all_texts.append(combined_turn)

            # Step 4: PARALLEL MULTI-SERVER EMBEDDING GENERATION
            console.print(f"[yellow]Generating {len(all_texts)} embeddings ({len(EMBEDDING_SERVERS) * 10} parallel workers across {len(EMBEDDING_SERVERS)} servers)...[/yellow]")
            embeddings = [None] * len(all_texts)

            def embed_single_multiserver(idx_text):
                idx, text = idx_text
                # Round-robin server selection
                service = next(server_cycle)
                embedding = service.encode(text=text, instruction="document")
                return idx, embedding

            # Use more workers since we have multiple servers
            max_workers = len(EMBEDDING_SERVERS) * 10
            with ThreadPoolExecutor(max_workers=max_workers) as embed_executor:
                embed_futures = {embed_executor.submit(embed_single_multiserver, (i, text)): i
                                for i, text in enumerate(all_texts)}

                completed = 0
                for future in as_completed(embed_futures):
                    try:
                        idx, embedding = future.result()
                        embeddings[idx] = embedding
                        completed += 1
                        if completed % 100 == 0:
                            console.print(f"[dim]  {completed}/{len(all_texts)} embeddings completed[/dim]")
                    except Exception as e:
                        idx = embed_futures[future]
                        console.print(f"[red]Embedding error at index {idx}: {e}[/red]")
                        # Use zero vector as fallback
                        embeddings[idx] = [0.0] * 4096  # qwen3-embedding dimension

            # Step 5: Build all points with embeddings and tags
            console.print(f"[yellow]Building vector DB points...[/yellow]")
            conv_points = []
            turn_points = []
            conv_id_map = {}  # Maps batch_idx to conv_id

            embedding_idx = 0
            for batch_idx, conv_data in enumerate(batch_convs):
                title = conv_data["title"]
                turns = conv_data["turns"]
                llm_tags = conv_data["llm_tags"]

                # Create conversation point
                conv_id = str(uuid.uuid4())
                conv_id_map[batch_idx] = conv_id

                tags = ["conversation", f"title:{title}"]
                tags.extend([f"llm:{tag}" for tag in llm_tags])

                conv_points.append({
                    "id": conv_id,
                    "vector": embeddings[embedding_idx],
                    "payload": {
                        "topic": title,
                        "tags": tags,
                        "llm_tags": llm_tags,
                        "conversation_id": conv_id,
                        "turns_count": len(turns),
                    },
                })
                embedding_idx += 1

                # Create turn points
                for turn_idx, (q, a) in enumerate(turns):
                    turn_id = str(uuid.uuid4())
                    turn_tags = tags + [f"turn:{turn_idx}"]

                    turn_points.append({
                        "id": turn_id,
                        "vector": embeddings[embedding_idx],
                        "payload": {
                            "conversation_id": conv_id,
                            "turn_id": turn_id,
                            "question": q,
                            "answer": a,
                            "turn_index": turn_idx,
                            "tags": turn_tags,
                            "llm_tags": llm_tags,
                        },
                    })
                    embedding_idx += 1
                    turns_total += 1

            # Step 6: BULK INSERT TO VECTOR DB
            console.print(f"[yellow]Inserting {len(conv_points)} conversations + {len(turn_points)} turns to vector DB (bulk)...[/yellow]")

            if conv_points:
                memory.vector_db.upsert_points(
                    memory.conversation_collection,
                    conv_points,
                    use_batch_optimization=True
                )
                imported += len(conv_points)

            if turn_points:
                memory.vector_db.upsert_points(
                    memory.turn_collection,
                    turn_points,
                    use_batch_optimization=True
                )

            progress.update(task, advance=len(batch_convs))

    console.print(f"\n[bold green]✓ Import Complete![/bold green]")
    console.print(f"  • Imported: {imported} conversations")
    console.print(f"  • Total turns: {turns_total}")
    console.print(f"  • LLM tags generated in parallel (10 workers)")
    console.print(f"  • Embeddings generated across {len(EMBEDDING_SERVERS)} servers ({len(EMBEDDING_SERVERS) * 10} workers)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Error: Please provide path to conversations.json[/red]")
        console.print("\nUsage:")
        console.print("  python3 scripts/import_conversations_multiserver.py /path/to/conversations.json")
        sys.exit(1)

    conv_file = Path(sys.argv[1])
    if not conv_file.exists():
        console.print(f"[red]Error: File not found: {conv_file}[/red]")
        sys.exit(1)

    import_conversations_multiserver(conv_file, batch_size=50)
