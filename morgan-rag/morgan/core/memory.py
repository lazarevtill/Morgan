"""
Conversation Memory - Learning from Human Interactions

Simple, focused module for remembering conversations and learning from them.

KISS Principle: One responsibility - remember conversations and learn from feedback.
Human-First: Make conversations feel natural and continuous.
"""

import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.embeddings import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient
from morgan.core.domain.entities import Conversation, ConversationTurn
from morgan.core.infrastructure.repositories import MemoryRepository

logger = get_logger(__name__)


class MemoryService:
    """
    Service for managing Morgan's conversation memory.
    Orchestrates turn storage and context retrieval, delegating persistence to MemoryRepository.
    """

    def __init__(self):
        """Initialize memory service."""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBClient()

        # Memory collections
        self.conversation_collection = "morgan_conversations"
        self.turn_collection = "morgan_turns"

        # Repositories
        self.repository = MemoryRepository(
            self.vector_db, self.turn_collection, self.conversation_collection
        )

        # Memory settings
        self.max_conversations = getattr(
            self.settings, "morgan_memory_max_conversations", 1000
        )
        self.max_turns_per_conversation = getattr(
            self.settings, "morgan_memory_max_turns_per_conversation", 100
        )
        self.context_window_turns = 5

        # Ensure collections exist
        self._ensure_collections()

        logger.info("Memory service initialized with repository-based persistence")

    def create_conversation(self, topic: Optional[str] = None) -> str:
        """
        Start a new conversation.

        Args:
            topic: Optional topic for the conversation

        Returns:
            Conversation ID for future reference

        Example:
            >>> memory = ConversationMemory()
            >>> conv_id = memory.create_conversation("Docker deployment")
            >>> print(f"Started conversation: {conv_id}")
        """
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        conversation = Conversation(
            conversation_id=conversation_id,
            started_at=timestamp,
            topic=topic,
            turns=[],
            last_activity=timestamp,
            tags=[],
        )

        # Store conversation metadata
        self._store_conversation_metadata(conversation)

        logger.info(f"Created new conversation: {conversation_id} (topic: {topic})")
        return conversation_id

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Add a turn to an existing conversation.
        """
        turn_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        turn = ConversationTurn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            timestamp=timestamp,
            question=question,
            answer=answer,
            sources=sources or [],
            tags=tags or [],
        )

        # Create embedding for the turn
        turn_text = f"Question: {question}\nAnswer: {answer}"
        turn_embedding = self.embedding_service.encode(
            text=turn_text, instruction="document"
        )

        # Store turn via repository
        self.repository.store_turn(turn, turn_embedding)

        # Update conversation metadata
        self._update_conversation_activity(conversation_id, timestamp.isoformat())

        logger.debug(f"Added turn to conversation {conversation_id}: {turn_id}")
        return turn_id

    def get_conversation_context(
        self, conversation_id: str, max_turns: Optional[int] = None
    ) -> str:
        """
        Get recent context from a conversation.

        Args:
            conversation_id: ID of the conversation
            max_turns: Maximum number of recent turns to include

        Returns:
            Formatted conversation context for LLM

        Example:
            >>> context = memory.get_conversation_context(conv_id, max_turns=3)
            >>> print(context)
        """
        if max_turns is None:
            max_turns = self.context_window_turns

        try:
            # Get recent turns from this conversation
            recent_turns = self._get_recent_turns(conversation_id, max_turns)

            if not recent_turns:
                return ""

            # Format as conversation context
            context_lines = []
            for turn in recent_turns:
                context_lines.append(f"Human: {turn.question}")
                context_lines.append(f"Morgan: {turn.answer}")

            return "\n".join(context_lines)

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get complete history of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of conversation turns in chronological order
        """
        try:
            turns = self._get_all_turns(conversation_id)

            # Convert to human-friendly format
            history = []
            for turn in turns:
                history.append(
                    {
                        "turn_id": turn.turn_id,
                        "timestamp": turn.timestamp,
                        "question": turn.question,
                        "answer": turn.answer,
                        "sources": turn.sources,
                        "feedback_rating": turn.feedback_rating,
                        "feedback_comment": turn.feedback_comment,
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def add_feedback(
        self,
        conversation_id: str,
        rating: int,
        comment: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> bool:
        """
        Add feedback to help Morgan learn and improve.

        Args:
            conversation_id: ID of the conversation
            rating: Rating from 1-5 (5 being excellent)
            comment: Optional feedback comment
            turn_id: Optional specific turn ID (uses last turn if None)

        Returns:
            True if feedback was recorded successfully

        Example:
            >>> success = memory.add_feedback(
            ...     conv_id,
            ...     5,
            ...     "Very helpful explanation!"
            ... )
        """
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                logger.error(f"Invalid rating: {rating}. Must be 1-5.")
                return False

            # Get the turn to update
            if turn_id is None:
                # Use the last turn in the conversation
                recent_turns = self._get_recent_turns(conversation_id, 1)
                if not recent_turns:
                    logger.error(f"No turns found in conversation: {conversation_id}")
                    return False
                turn_id = recent_turns[0].turn_id

            # Update turn with feedback
            success = self._update_turn_feedback(turn_id, rating, comment)

            if success:
                logger.info(
                    f"Recorded feedback for turn {turn_id}: {rating}/5 - {comment or 'No comment'}"
                )

                # Learn from feedback (simple approach for now)
                self._learn_from_feedback(turn_id, rating, comment)

                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False

    def search_conversations(
        self, query: str, max_results: int = 10, min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search through conversation history.

        Args:
            query: Search query
            max_results: Maximum number of results
            min_score: Minimum similarity score

        Returns:
            List of relevant conversation turns

        Example:
            >>> results = memory.search_conversations("Docker installation")
            >>> for result in results:
            ...     print(f"Q: {result['question']}")
            ...     print(f"A: {result['answer'][:100]}...")
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query"
            )

            # Search conversation turns
            search_results = self.vector_db.search(
                collection_name=self.turn_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score,
            )

            # Convert to human-friendly format
            results = []
            for result in search_results:
                payload = result.payload
                results.append(
                    {
                        "turn_id": payload.get("turn_id", ""),
                        "conversation_id": payload.get("conversation_id", ""),
                        "timestamp": payload.get("timestamp", ""),
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                        "sources": payload.get("sources", []),
                        "score": result.score,
                        "feedback_rating": payload.get("feedback_rating"),
                        "feedback_comment": payload.get("feedback_comment"),
                    }
                )

            logger.debug(
                f"Found {len(results)} relevant conversation turns for: '{query}'"
            )
            return results

        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []

    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from conversation history and feedback.

        Returns:
            Human-readable insights about Morgan's performance and learning
        """
        try:
            # Get sample of recent conversations
            recent_turns = self._get_recent_turns_all_conversations(limit=100)

            if not recent_turns:
                return {
                    "total_conversations": 0,
                    "total_turns": 0,
                    "average_rating": 0.0,
                    "message": "No conversation data available yet.",
                }

            # Calculate statistics
            total_turns = len(recent_turns)
            ratings = [
                turn.feedback_rating
                for turn in recent_turns
                if turn.feedback_rating is not None
            ]

            # Count conversations
            conversation_ids = {turn.conversation_id for turn in recent_turns}
            total_conversations = len(conversation_ids)

            # Calculate averages
            average_rating = sum(ratings) / len(ratings) if ratings else 0.0
            feedback_percentage = (
                (len(ratings) / total_turns * 100) if total_turns > 0 else 0.0
            )

            # Find common topics
            topics = []
            for turn in recent_turns:
                if len(turn.question) > 10:  # Skip very short questions
                    # Simple topic extraction from questions
                    words = turn.question.lower().split()
                    for word in words:
                        if len(word) > 4 and word.isalpha():
                            topics.append(word)

            # Count topic frequency
            from collections import Counter

            common_topics = [topic for topic, count in Counter(topics).most_common(10)]

            return {
                "total_conversations": total_conversations,
                "total_turns": total_turns,
                "average_rating": average_rating,
                "feedback_percentage": feedback_percentage,
                "common_topics": common_topics,
                "recent_activity": recent_turns[0].timestamp if recent_turns else None,
                "message": f"Morgan has had {total_conversations} conversations with {total_turns} turns. "
                f"Average rating: {average_rating:.1f}/5 ({feedback_percentage:.1f}% feedback rate).",
            }

        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {"error": str(e)}

    def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """
        Clean up old conversations to manage storage.

        Args:
            days_to_keep: Number of days of conversations to keep

        Returns:
            Number of conversations cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()

            # Find old conversations
            old_conversations = self.vector_db.search_with_filter(
                collection_name=self.conversation_collection,
                filter_conditions={"started_at": {"lt": cutoff_iso}},
                limit=1000,
            )

            # Delete old conversation turns
            deleted_count = 0
            for conv in old_conversations:
                conv_id = conv.get("payload", {}).get("conversation_id", "")
                if conv_id:
                    # Delete all turns for this conversation
                    self._delete_conversation_turns(conv_id)
                    deleted_count += 1

            # Delete conversation metadata
            conv_ids = [
                conv.get("payload", {}).get("conversation_id", "")
                for conv in old_conversations
            ]
            conv_ids = [cid for cid in conv_ids if cid]

            if conv_ids:
                self.vector_db.delete_points(self.conversation_collection, conv_ids)

            logger.info(
                f"Cleaned up {deleted_count} old conversations (older than {days_to_keep} days)"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
            return 0

    def _ensure_collections(self):
        """Ensure required collections exist."""
        try:
            # Get embedding dimension
            embedding_dim = self.embedding_service.get_dimension()

            # Create conversation metadata collection
            if self.vector_db.collection_exists(self.conversation_collection):
                current_dim = self.vector_db.get_collection_vector_size(
                    self.conversation_collection
                )
                if current_dim and current_dim != embedding_dim:
                    logger.warning(
                        "Recreating collection %s due to dimension mismatch (%s != %s)",
                        self.conversation_collection,
                        current_dim,
                        embedding_dim,
                    )
                    self.vector_db.delete_collection(self.conversation_collection)

            if not self.vector_db.collection_exists(self.conversation_collection):
                self.vector_db.create_collection(
                    name=self.conversation_collection,
                    vector_size=embedding_dim,
                    distance="cosine",
                )
                logger.info(
                    f"Created conversation collection: {self.conversation_collection}"
                )

            # Create turn collection for semantic search
            if self.vector_db.collection_exists(self.turn_collection):
                current_dim = self.vector_db.get_collection_vector_size(
                    self.turn_collection
                )
                if current_dim and current_dim != embedding_dim:
                    logger.warning(
                        "Recreating collection %s due to dimension mismatch (%s != %s)",
                        self.turn_collection,
                        current_dim,
                        embedding_dim,
                    )
                    self.vector_db.delete_collection(self.turn_collection)

            if not self.vector_db.collection_exists(self.turn_collection):
                self.vector_db.create_collection(
                    name=self.turn_collection,
                    vector_size=embedding_dim,
                    distance="cosine",
                )
                logger.info(f"Created turn collection: {self.turn_collection}")

        except Exception as e:
            logger.error(f"Failed to ensure collections: {e}")
            raise

    def _store_conversation_metadata(self, conversation: Conversation):
        """Store conversation metadata."""
        try:
            # Create a simple embedding for the conversation topic
            topic_text = conversation.topic or "general conversation"
            topic_embedding = self.embedding_service.encode(
                text=topic_text, instruction="document"
            )

            point = {
                "id": conversation.conversation_id,
                "vector": topic_embedding,
                "payload": asdict(conversation),
            }

            self.vector_db.upsert_points(self.conversation_collection, [point])

        except Exception as e:
            logger.error(f"Failed to store conversation metadata: {e}")

    def _store_turn(self, turn: ConversationTurn):
        """Store a conversation turn for semantic search."""
        try:
            # Create embedding from question and answer
            turn_text = f"Question: {turn.question}\nAnswer: {turn.answer}"
            turn_embedding = self.embedding_service.encode(
                text=turn_text, instruction="document"
            )

            point = {
                "id": turn.turn_id,
                "vector": turn_embedding,
                "payload": asdict(turn),
            }

            self.vector_db.upsert_points(self.turn_collection, [point])

        except Exception as e:
            logger.error(f"Failed to store turn: {e}")

    def _update_conversation_activity(self, conversation_id: str, timestamp: str):
        """Update last activity timestamp for a conversation."""
        try:
            # Get current conversation
            result = self.vector_db.get_point(
                self.conversation_collection, conversation_id
            )
            if result:
                payload = result.get("payload", {})
                payload["last_activity"] = timestamp
                payload["total_turns"] = payload.get("total_turns", 0) + 1

                # Update the point
                point = {
                    "id": conversation_id,
                    "vector": result.get("vector"),
                    "payload": payload,
                }
                self.vector_db.upsert_points(self.conversation_collection, [point])

        except Exception as e:
            logger.error(f"Failed to update conversation activity: {e}")

    def _get_recent_turns(
        self, conversation_id: str, max_turns: int
    ) -> List[ConversationTurn]:
        """Get recent turns from a specific conversation."""
        try:
            # Search for turns in this conversation
            results = self.vector_db.search_with_filter(
                collection_name=self.turn_collection,
                filter_conditions={"conversation_id": conversation_id},
                limit=max_turns,
                order_by="timestamp",
            )

            # Convert to ConversationTurn objects
            turns = []
            for result in results:
                if hasattr(result, "payload"):
                    payload = result.payload
                else:
                    payload = result.get("payload", {})
                turn = ConversationTurn(**payload)
                turns.append(turn)

            # Sort by timestamp (most recent first)
            turns.sort(key=lambda t: t.timestamp, reverse=True)
            return turns[:max_turns]

        except Exception as e:
            logger.error(f"Failed to get recent turns: {e}")
            return []

    def _get_all_turns(self, conversation_id: str) -> List[ConversationTurn]:
        """Get all turns from a specific conversation."""
        return self._get_recent_turns(conversation_id, self.max_turns_per_conversation)

    def _get_recent_turns_all_conversations(
        self, limit: int = 100
    ) -> List[ConversationTurn]:
        """Get recent turns from all conversations."""
        try:
            results = self.vector_db.scroll_points(
                collection_name=self.turn_collection, limit=limit
            )

            turns = []
            for result in results:
                if hasattr(result, "payload"):
                    payload = result.payload
                else:
                    payload = result.get("payload", {})
                turn = ConversationTurn(**payload)
                turns.append(turn)

            # Sort by timestamp (most recent first)
            turns.sort(key=lambda t: t.timestamp, reverse=True)
            return turns

        except Exception as e:
            logger.error(f"Failed to get recent turns from all conversations: {e}")
            return []

    def _update_turn_feedback(
        self, turn_id: str, rating: int, comment: Optional[str]
    ) -> bool:
        """Update a turn with feedback."""
        try:
            # Get current turn
            result = self.vector_db.get_point(self.turn_collection, turn_id)
            if not result:
                logger.error(f"Turn not found: {turn_id}")
                return False

            # Update payload with feedback
            payload = result.get("payload", {})
            payload["feedback_rating"] = rating
            payload["feedback_comment"] = comment

            # Update the point
            point = {"id": turn_id, "vector": result.get("vector"), "payload": payload}
            self.vector_db.upsert_points(self.turn_collection, [point])

            return True

        except Exception as e:
            logger.error(f"Failed to update turn feedback: {e}")
            return False

    def _learn_from_feedback(self, turn_id: str, rating: int, comment: Optional[str]):
        """
        Learn from user feedback.

        Simple learning approach for now - could be enhanced with ML later.
        """
        try:
            # For now, just log the feedback for analysis
            logger.info(
                f"Learning from feedback - Turn: {turn_id}, Rating: {rating}, Comment: {comment}"
            )

            # Future enhancements could include:
            # - Adjusting response patterns based on feedback
            # - Identifying common issues from low ratings
            # - Improving source selection based on feedback
            # - Fine-tuning retrieval based on successful responses

        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")

    def _delete_conversation_turns(self, conversation_id: str):
        """Delete all turns for a conversation."""
        try:
            # Get all turn IDs for this conversation
            turns = self._get_all_turns(conversation_id)
            turn_ids = [turn.turn_id for turn in turns]

            if turn_ids:
                self.vector_db.delete_points(self.turn_collection, turn_ids)

        except Exception as e:
            logger.error(f"Failed to delete conversation turns: {e}")


# Human-friendly helper functions
def quick_memory_search(query: str, max_results: int = 5) -> List[str]:
    """
    Quick search through conversation memory.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of relevant answers from past conversations

    Example:
        >>> answers = quick_memory_search("Docker installation")
        >>> for answer in answers:
        ...     print(answer[:100] + "...")
    """
    memory = ConversationMemory()
    results = memory.search_conversations(query, max_results=max_results)
    return [result["answer"] for result in results]


if __name__ == "__main__":
    # Demo conversation memory capabilities
    print("🧠 Morgan Conversation Memory Demo")
    print("=" * 40)

    memory = ConversationMemory()

    # Create a test conversation
    conv_id = memory.create_conversation("Docker Tutorial")
    print(f"Created conversation: {conv_id}")

    # Add some turns
    turn1 = memory.add_turn(
        conv_id,
        "What is Docker?",
        "Docker is a containerization platform that allows you to package applications...",
        ["docker-intro.md"],
    )

    turn2 = memory.add_turn(
        conv_id,
        "How do I install it?",
        "You can install Docker by downloading it from docker.com or using package managers...",
        ["docker-install.md"],
    )

    print(f"Added turns: {turn1}, {turn2}")

    # Add feedback
    memory.add_feedback(conv_id, 5, "Very helpful!")
    print("Added positive feedback")

    # Get conversation context
    context = memory.get_conversation_context(conv_id)
    print(f"\nConversation context:\n{context}")

    # Search conversations
    results = memory.search_conversations("Docker installation")
    print(f"\nSearch results for 'Docker installation': {len(results)} found")

    # Get learning insights
    insights = memory.get_learning_insights()
    print(f"\nLearning insights: {insights['message']}")
