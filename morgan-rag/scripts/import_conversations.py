"""
Import exported conversations (OpenAI-style JSON) into Morgan conversation memory.

Usage:
    python scripts/import_conversations.py conversations.json
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import uuid

from morgan.core.memory import ConversationMemory
from morgan.utils.logger import setup_logging


def load_conversations(path: Path) -> List[dict]:
    """Load conversation list from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of conversations in JSON")
    return data


def extract_turns(mapping: dict) -> List[Tuple[str, str]]:
    """
    Extract user/assistant turns from mapping ordered by create_time.

    Pairs each user message with the next assistant message (if any).
    System/empty messages are skipped.
    """
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

    # sort by timestamp
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
    setup_logging()
    memory = ConversationMemory()

    conversations = load_conversations(path)
    imported = 0
    turns_total = 0
    chunk_size = 4

    for conv in conversations:
        mapping = conv.get("mapping") or {}
        title = conv.get("title")
        tags = []
        if title:
            tags.append("title:" + title)
        tags.append("conversation")
        turns = extract_turns(mapping)
        if not turns:
            continue

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

        for idx, (q, a) in enumerate(turns):
            turn_tags = tags + [f"turn:{idx}"]
            memory.add_turn(
                conversation_id=conv_id, question=q, answer=a, tags=turn_tags
            )
            turns_total += 1

        # Chunked summaries for better retrieval
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

    print(f"Imported {imported} conversations with {turns_total} turns.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_conversations.py <conversations.json>")
        sys.exit(1)

    import_conversations(Path(sys.argv[1]))
