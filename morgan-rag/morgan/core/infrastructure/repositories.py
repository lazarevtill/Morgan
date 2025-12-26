"""
Infrastructure repositories for Morgan Core.
Handles persistence using VectorDBClient.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.core.domain.entities import KnowledgeChunk, Conversation, ConversationTurn
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient

logger = get_logger(__name__)


class KnowledgeRepository:
    """
    Handles storage and retrieval of knowledge chunks.
    """

    def __init__(self, vector_db: VectorDBClient, collection_name: str):
        self.vector_db = vector_db
        self.collection_name = collection_name

    def store_chunks(self, chunks: List[KnowledgeChunk]):
        points = []
        for chunk in chunks:
            if chunk.embedding_type == "hierarchical":
                # This would need the specialized upsert for hierarchical
                # For now keeping it simple as a placeholder for the logic in knowledge.py
                pass
            else:
                point = {
                    "id": chunk.chunk_id,
                    "vector": chunk.embedding,
                    "payload": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                        "ingested_at": chunk.ingested_at.isoformat(),
                        "embedding_type": chunk.embedding_type,
                    },
                }
                points.append(point)

        if points:
            self.vector_db.upsert_points(self.collection_name, points)

    def search(
        self, query_vector: List[float], limit: int, score_threshold: float
    ) -> List[KnowledgeChunk]:
        search_results = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        chunks = []
        for res in search_results:
            payload = res.payload
            chunks.append(
                KnowledgeChunk(
                    content=payload.get("content", ""),
                    source=payload.get("source", "Unknown"),
                    chunk_id=str(res.id),
                    metadata=payload.get("metadata", {}),
                    embedding=None,  # We don't usually need to pull the embedding back
                    ingested_at=(
                        datetime.fromisoformat(payload.get("ingested_at"))
                        if payload.get("ingested_at")
                        else datetime.utcnow()
                    ),
                    embedding_type=payload.get("embedding_type", "legacy"),
                )
            )
        return chunks


class MemoryRepository:
    """
    Handles storage and retrieval of conversation history.
    """

    def __init__(
        self, vector_db: VectorDBClient, turn_collection: str, conv_collection: str
    ):
        self.vector_db = vector_db
        self.turn_collection = turn_collection
        self.conv_collection = conv_collection

    def store_turn(self, turn: ConversationTurn, embedding: List[float]):
        point = {
            "id": turn.turn_id,
            "vector": embedding,
            "payload": {
                "conversation_id": turn.conversation_id,
                "timestamp": turn.timestamp.isoformat(),
                "question": turn.question,
                "answer": turn.answer,
                "sources": turn.sources,
                "tags": turn.tags,
                "feedback_rating": turn.feedback_rating,
                "feedback_comment": turn.feedback_comment,
                "emotional_tone": turn.emotional_tone,
                "empathy_level": turn.empathy_level,
            },
        }
        self.vector_db.upsert_points(self.turn_collection, [point])

    def get_recent_turns(
        self, conversation_id: str, limit: int
    ) -> List[ConversationTurn]:
        results = self.vector_db.search_with_filter(
            collection_name=self.turn_collection,
            filter_conditions={"conversation_id": conversation_id},
            limit=limit,
            order_by="timestamp",
        )

        turns = []
        for res in results:
            payload = res.payload if hasattr(res, "payload") else res.get("payload", {})
            turns.append(
                ConversationTurn(
                    turn_id=(
                        str(res.id) if hasattr(res, "id") else payload.get("turn_id")
                    ),
                    conversation_id=payload.get("conversation_id"),
                    timestamp=datetime.fromisoformat(payload.get("timestamp")),
                    question=payload.get("question"),
                    answer=payload.get("answer"),
                    sources=payload.get("sources", []),
                    tags=payload.get("tags", []),
                    feedback_rating=payload.get("feedback_rating"),
                    feedback_comment=payload.get("feedback_comment"),
                    emotional_tone=payload.get("emotional_tone"),
                    empathy_level=payload.get("empathy_level", 0.0),
                )
            )

        turns.sort(key=lambda t: t.timestamp, reverse=True)
        return turns
