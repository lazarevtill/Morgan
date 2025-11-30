#!/usr/bin/env python3
"""
Enhanced Import with LLM-generated Tags

Imports conversations with:
1. Embeddings from local Ollama (qwen3-embedding:8b via localhost Ollama)
2. LLM-generated tags using gemma3 (via ai.ishosting.com)

Usage:
    export PYTHONPATH=./local_libs:$PYTHONPATH
    python3 scripts/import_conversations_enhanced.py /path/to/conversations.json
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import uuid
import time

from morgan.core.memory import ConversationMemory
from morgan.services.llm_service import get_llm_service
from morgan.utils.logger import setup_logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


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


def import_conversations_enhanced(path: Path):
    """Import conversations with LLM-generated tags."""
    console.print("\n[bold cyan]Morgan Enhanced Conversation Import[/bold cyan]")
    console.print(f"[dim]Source: {path}[/dim]\n")

    setup_logging()

    console.print("[yellow]Initializing services...[/yellow]")
    memory = ConversationMemory()
    llm_service = get_llm_service()

    # Get actual service configurations
    from morgan.config import get_settings
    settings = get_settings()

    console.print("[green]✓[/green] Memory system ready")
    console.print(f"[green]✓[/green] LLM service ready ({settings.llm_model} via ai.ishosting.com)")
    console.print(f"[green]✓[/green] Embeddings ready ({settings.embedding_model} via local Ollama at {settings.embedding_base_url})\n")

    conversations = load_conversations(path)
    imported = 0
    turns_total = 0
    chunk_size = 4
    llm_tags_added = 0

    console.print(f"[bold]Processing {len(conversations)} conversations...[/bold]\n")

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

        for idx, conv in enumerate(conversations, 1):
            mapping = conv.get("mapping") or {}
            title = conv.get("title") or f"Conversation {idx}"

            # Extract turns
            turns = extract_turns(mapping)
            if not turns:
                progress.update(task, advance=1)
                continue

            # Generate LLM tags
            llm_tags = generate_llm_tags(title, turns, llm_service)
            if llm_tags:
                llm_tags_added += 1

            # Build complete tag set
            tags = ["conversation"]
            if title:
                tags.append(f"title:{title}")
            tags.extend([f"llm:{tag}" for tag in llm_tags])  # Prefix LLM tags

            # Create conversation in memory
            try:
                conv_id = memory.create_conversation(topic=title)

                # Upsert conversation topic with embedding
                topic_vector = memory.embedding_service.encode(
                    text=title or "conversation",
                    instruction="document"
                )

                memory.vector_db.upsert_points(
                    memory.conversation_collection,
                    [
                        {
                            "id": conv_id,
                            "vector": topic_vector,
                            "payload": {
                                "topic": title,
                                "tags": tags,
                                "llm_tags": llm_tags,  # Store LLM tags separately
                                "conversation_id": conv_id,
                                "turns_count": len(turns),
                            },
                        }
                    ],
                    use_batch_optimization=False,
                )

                # Add all turns
                for turn_idx, (q, a) in enumerate(turns):
                    turn_tags = tags + [f"turn:{turn_idx}"]
                    memory.add_turn(
                        conversation_id=conv_id,
                        question=q,
                        answer=a,
                        tags=turn_tags
                    )
                    turns_total += 1

                # Create chunked summaries for better retrieval
                chunk_points = []
                for i in range(0, len(turns), chunk_size):
                    chunk = turns[i : i + chunk_size]
                    if not chunk:
                        continue

                    text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chunk])
                    vec = memory.embedding_service.encode(
                        text=text,
                        instruction="document"
                    )

                    chunk_points.append(
                        {
                            "id": str(uuid.uuid4()),
                            "vector": vec,
                            "payload": {
                                "conversation_id": conv_id,
                                "chunk_index": i // chunk_size,
                                "tags": tags + [f"chunk:{i//chunk_size}"],
                                "llm_tags": llm_tags,
                                "text": text,
                            },
                        }
                    )

                if chunk_points:
                    memory.vector_db.upsert_points(
                        memory.turn_collection,
                        chunk_points,
                        use_batch_optimization=False
                    )

                imported += 1

            except Exception as e:
                console.print(f"\n[red]Error importing '{title}': {e}[/red]")
                continue

            progress.update(task, advance=1)

            # Rate limit for LLM API (avoid hitting rate limits)
            if idx % 5 == 0:
                time.sleep(0.5)

    # Final summary
    console.print("\n[bold green]✓ Import Complete![/bold green]\n")
    console.print(f"  [cyan]Conversations imported:[/cyan] {imported}/{len(conversations)}")
    console.print(f"  [cyan]Total turns processed:[/cyan] {turns_total}")
    console.print(f"  [cyan]LLM tags generated:[/cyan] {llm_tags_added} conversations")
    console.print(f"\n[bold]Data stored in:[/bold]")
    console.print(f"  • [dim]morgan_conversations:[/dim] {imported} conversation topics")
    console.print(f"  • [dim]morgan_turns:[/dim] ~{imported * chunk_size} chunked summaries")
    console.print(f"\n[green]🚀 Your conversations are now searchable with semantic tags![/green]\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("\n[red]Usage:[/red] python scripts/import_conversations_enhanced.py <conversations.json>\n")
        sys.exit(1)

    conv_file = Path(sys.argv[1])
    if not conv_file.exists():
        console.print(f"\n[red]Error: File not found:[/red] {conv_file}\n")
        sys.exit(1)

    import_conversations_enhanced(conv_file)
