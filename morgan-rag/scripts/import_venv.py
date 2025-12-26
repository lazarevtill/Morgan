#!/usr/bin/env python3
"""
Import conversations - Linux/WSL version (RECOMMENDED)

Uses local_libs packages installed on Linux.

Usage:
    export PYTHONPATH=./local_libs:$PYTHONPATH
    python3 scripts/import_conversations.py /path/to/conversations.json
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Add local_libs and morgan-rag to path
script_dir = Path(__file__).parent
morgan_rag_dir = script_dir.parent
local_libs = morgan_rag_dir / "local_libs"

if local_libs.exists():
    sys.path.insert(0, str(local_libs))

sys.path.insert(0, str(morgan_rag_dir))

import uuid

from morgan.core.memory import ConversationMemory
from morgan.utils.logger import setup_logging


def load_conversations(path: Path) -> List[dict]:
    """Load conversation list from JSON file."""
    print(f"Loading conversations from: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of conversations in JSON")
    print(f"Found {len(data)} conversations in file")
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


def import_conversations(path: Path):
    """Import conversations from file into Morgan memory."""
    print("\n" + "=" * 60)
    print("  Morgan - Conversation Import")
    print("=" * 60)

    setup_logging()

    print("\nInitializing Morgan memory system...")
    memory = ConversationMemory()
    print("✅ Memory system ready")

    conversations = load_conversations(path)
    imported = 0
    turns_total = 0
    chunk_size = 4

    print(f"\nStarting import of {len(conversations)} conversations...")
    print("=" * 60)

    for idx, conv in enumerate(conversations, 1):
        mapping = conv.get("mapping") or {}
        title = conv.get("title") or f"Conversation {idx}"
        tags = []
        if title:
            tags.append("title:" + title)
        tags.append("conversation")

        turns = extract_turns(mapping)
        if not turns:
            continue

        # Progress indicator every 10 conversations
        if idx % 10 == 0 or idx == 1:
            print(f"[{idx}/{len(conversations)}] Processing: {title[:50]}...")
            print(
                f"           Progress: {imported} imported, {turns_total} turns total"
            )

        try:
            conv_id = memory.create_conversation(topic=title)
            topic_vector = memory.embedding_service.encode(
                text=title or "conversation", instruction="document"
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
                            "conversation_id": conv_id,
                        },
                    }
                ],
                use_batch_optimization=False,
            )

            for turn_idx, (q, a) in enumerate(turns):
                turn_tags = tags + [f"turn:{turn_idx}"]
                memory.add_turn(
                    conversation_id=conv_id, question=q, answer=a, tags=turn_tags
                )
                turns_total += 1

            # Chunked summaries
            chunk_points = []
            for i in range(0, len(turns), chunk_size):
                chunk = turns[i : i + chunk_size]
                if not chunk:
                    continue
                text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chunk])
                vec = memory.embedding_service.encode(text=text, instruction="document")
                chunk_points.append(
                    {
                        "id": str(uuid.uuid4()),
                        "vector": vec,
                        "payload": {
                            "conversation_id": conv_id,
                            "chunk_index": i // chunk_size,
                            "tags": tags + [f"chunk:{i//chunk_size}"],
                            "text": text,
                        },
                    }
                )

            if chunk_points:
                memory.vector_db.upsert_points(
                    memory.turn_collection, chunk_points, use_batch_optimization=False
                )

            imported += 1

        except Exception as e:
            print(f"  ⚠️  Error importing conversation {idx}: {e}")
            continue

    print("=" * 60)
    print(f"\n✅ Import Complete!")
    print(f"   Conversations imported: {imported}/{len(conversations)}")
    print(f"   Total conversation turns: {turns_total}")
    print(f"\n🚀 Your conversations are now searchable in Morgan!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python scripts/import_venv.py <conversations.json>")
        print("\nExample:")
        print("  python scripts/import_venv.py conversations.json")
        print("  python scripts/import_venv.py ..\\conversations.json")
        print("\nMake sure your virtual environment is activated!")
        sys.exit(1)

    conv_file = Path(sys.argv[1])
    if not conv_file.exists():
        print(f"\n❌ Error: File not found: {conv_file}")
        sys.exit(1)

    import_conversations(conv_file)
