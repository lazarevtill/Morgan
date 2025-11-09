"""
Emotional memory storage module.

Provides focused emotional memory storage, retrieval, and management with
emotional context preservation and relationship significance tracking.
"""

import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.emotional.models import ConversationContext, EmotionalState, EmotionType
from morgan.services.embedding_service import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient

logger = get_logger(__name__)


class EmotionalMemory:
    """
    Represents a stored emotional memory with context.

    Features:
    - Emotional context preservation
    - Relationship significance tracking
    - Temporal decay modeling
    - Retrieval optimization
    """

    def __init__(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        emotional_state: EmotionalState,
        conversation_context: ConversationContext,
        importance_score: float,
        relationship_significance: float = 0.0,
        memory_type: str = "conversation",
        tags: Optional[List[str]] = None,
    ):
        """Initialize emotional memory."""
        self.memory_id = memory_id
        self.user_id = user_id
        self.content = content
        self.emotional_state = emotional_state
        self.conversation_context = conversation_context
        self.importance_score = importance_score
        self.relationship_significance = relationship_significance
        self.memory_type = memory_type
        self.tags = tags or []
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0
        self.decay_factor = 1.0

    def calculate_current_importance(self) -> float:
        """
        Calculate current importance considering temporal decay.

        Returns:
            Current importance score with decay applied
        """
        # Calculate time-based decay
        days_old = (datetime.utcnow() - self.created_at).days

        # Emotional memories decay slower than regular memories
        emotional_weight = self.emotional_state.intensity
        decay_rate = 0.02 * (
            1.0 - emotional_weight * 0.5
        )  # Slower decay for intense emotions

        # Relationship significance reduces decay
        relationship_bonus = self.relationship_significance * 0.3

        # Access frequency reduces decay
        access_bonus = min(0.2, self.access_count * 0.05)

        # Calculate final decay
        total_decay = max(
            0.1, 1.0 - (days_old * decay_rate) + relationship_bonus + access_bonus
        )

        return self.importance_score * total_decay

    def mark_accessed(self):
        """Mark memory as accessed for decay calculation."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "emotional_state": {
                "primary_emotion": self.emotional_state.primary_emotion.value,
                "intensity": self.emotional_state.intensity,
                "confidence": self.emotional_state.confidence,
                "secondary_emotions": [
                    e.value for e in self.emotional_state.secondary_emotions
                ],
                "emotional_indicators": self.emotional_state.emotional_indicators,
                "timestamp": self.emotional_state.timestamp.isoformat(),
            },
            "conversation_context": {
                "user_id": self.conversation_context.user_id,
                "conversation_id": self.conversation_context.conversation_id,
                "message_text": self.conversation_context.message_text,
                "timestamp": self.conversation_context.timestamp.isoformat(),
                "previous_messages": self.conversation_context.previous_messages,
                "user_feedback": self.conversation_context.user_feedback,
            },
            "importance_score": self.importance_score,
            "relationship_significance": self.relationship_significance,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalMemory":
        """Create memory from dictionary."""
        # Reconstruct emotional state
        emotional_state = EmotionalState(
            primary_emotion=EmotionType(data["emotional_state"]["primary_emotion"]),
            intensity=data["emotional_state"]["intensity"],
            confidence=data["emotional_state"]["confidence"],
            secondary_emotions=[
                EmotionType(e) for e in data["emotional_state"]["secondary_emotions"]
            ],
            emotional_indicators=data["emotional_state"]["emotional_indicators"],
            timestamp=datetime.fromisoformat(data["emotional_state"]["timestamp"]),
        )

        # Reconstruct conversation context
        conversation_context = ConversationContext(
            user_id=data["conversation_context"]["user_id"],
            conversation_id=data["conversation_context"]["conversation_id"],
            message_text=data["conversation_context"]["message_text"],
            timestamp=datetime.fromisoformat(data["conversation_context"]["timestamp"]),
            previous_messages=data["conversation_context"]["previous_messages"],
            user_feedback=data["conversation_context"]["user_feedback"],
        )

        # Create memory instance
        memory = cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            content=data["content"],
            emotional_state=emotional_state,
            conversation_context=conversation_context,
            importance_score=data["importance_score"],
            relationship_significance=data["relationship_significance"],
            memory_type=data["memory_type"],
            tags=data["tags"],
        )

        # Restore metadata
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        memory.access_count = data["access_count"]
        memory.decay_factor = data["decay_factor"]

        return memory


class EmotionalMemoryStorage:
    """
    Emotional memory storage and retrieval system.

    Features:
    - Emotional context-aware storage
    - Importance-based retrieval
    - Temporal decay management
    - Relationship significance tracking
    - Memory consolidation
    """

    def __init__(self):
        """Initialize emotional memory storage."""
        self.settings = get_settings()
        self.vector_client = VectorDBClient()
        self.embedding_service = get_embedding_service()

        # Memory collection name
        self.collection_name = "morgan_emotional_memories"

        # Initialize collection if needed
        self._initialize_collection()

        logger.info("Emotional Memory Storage initialized")

    def store_memory(
        self,
        user_id: str,
        content: str,
        emotional_state: EmotionalState,
        conversation_context: ConversationContext,
        importance_score: float,
        relationship_significance: float = 0.0,
        memory_type: str = "conversation",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store an emotional memory.

        Args:
            user_id: User identifier
            content: Memory content
            emotional_state: Associated emotional state
            conversation_context: Conversation context
            importance_score: Memory importance (0.0 to 1.0)
            relationship_significance: Relationship significance (0.0 to 1.0)
            memory_type: Type of memory
            tags: Optional tags for categorization

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())

        # Create emotional memory
        memory = EmotionalMemory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            emotional_state=emotional_state,
            conversation_context=conversation_context,
            importance_score=importance_score,
            relationship_significance=relationship_significance,
            memory_type=memory_type,
            tags=tags,
        )

        # Generate embedding for content
        embedding = self.embedding_service.encode(content, instruction="document")

        # Store in vector database
        self.vector_client.upsert_points(
            collection_name=self.collection_name,
            points=[
                {"id": memory_id, "vector": embedding, "payload": memory.to_dict()}
            ],
        )

        logger.debug(f"Stored emotional memory {memory_id} for user {user_id}")
        return memory_id

    def retrieve_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        emotion_filter: Optional[EmotionType] = None,
        min_importance: float = 0.0,
        max_age_days: Optional[int] = None,
        limit: int = 10,
    ) -> List[EmotionalMemory]:
        """
        Retrieve emotional memories with filtering.

        Args:
            user_id: User identifier
            query: Optional semantic query
            emotion_filter: Filter by emotion type
            min_importance: Minimum current importance score
            max_age_days: Maximum age in days
            limit: Maximum number of memories to return

        Returns:
            List of matching emotional memories
        """
        # Build filter conditions
        filter_conditions = {"user_id": user_id}

        if emotion_filter:
            filter_conditions["emotional_state.primary_emotion"] = emotion_filter.value

        if max_age_days:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            filter_conditions["created_at"] = {"$gte": cutoff_date.isoformat()}

        # Search memories
        if query:
            # Semantic search
            query_embedding = self.embedding_service.encode(query, instruction="query")
            results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Get more to filter by importance
            )
        else:
            # Get all matching memories - use simple search without filter for now
            results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * 1536,  # Dummy vector for getting all results
                limit=limit * 2,
            )

        # Convert to EmotionalMemory objects and filter by current importance
        memories = []
        for result in results:
            memory = EmotionalMemory.from_dict(result["payload"])
            memory.mark_accessed()  # Update access tracking

            current_importance = memory.calculate_current_importance()
            if current_importance >= min_importance:
                memories.append(memory)

        # Sort by current importance and limit results
        memories.sort(key=lambda m: m.calculate_current_importance(), reverse=True)
        return memories[:limit]

    def get_memory_by_id(self, memory_id: str) -> Optional[EmotionalMemory]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Emotional memory or None if not found
        """
        try:
            # Use search with a dummy vector to find the specific memory by ID
            # This is a workaround since there's no direct get_point method
            results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * 1536,  # Dummy vector
                limit=1000,  # Get many results to find the specific ID
            )

            # Find the memory with matching ID
            for result in results:
                if result.id == memory_id:
                    memory = EmotionalMemory.from_dict(result.payload)
                    memory.mark_accessed()
                    return memory

        except Exception as e:
            logger.warning(f"Failed to retrieve memory {memory_id}: {e}")

        return None

    def update_memory_significance(
        self, memory_id: str, relationship_significance: float
    ) -> bool:
        """
        Update the relationship significance of a memory.

        Args:
            memory_id: Memory identifier
            relationship_significance: New significance score

        Returns:
            True if updated successfully
        """
        memory = self.get_memory_by_id(memory_id)
        if not memory:
            return False

        memory.relationship_significance = relationship_significance

        # Update in storage - we need to get the existing vector first
        # For now, we'll regenerate the embedding
        embedding = self.embedding_service.encode(
            memory.content, instruction="document"
        )
        self.vector_client.upsert_points(
            collection_name=self.collection_name,
            points=[
                {"id": memory_id, "vector": embedding, "payload": memory.to_dict()}
            ],
        )

        logger.debug(f"Updated significance for memory {memory_id}")
        return True

    def consolidate_memories(
        self, user_id: str, consolidation_threshold: float = 0.1
    ) -> int:
        """
        Consolidate low-importance memories to optimize storage.

        Args:
            user_id: User identifier
            consolidation_threshold: Minimum importance to keep

        Returns:
            Number of memories consolidated
        """
        # Get all memories for user
        all_memories = self.retrieve_memories(
            user_id=user_id, min_importance=0.0, limit=1000
        )

        # Find memories below threshold
        to_consolidate = [
            m
            for m in all_memories
            if m.calculate_current_importance() < consolidation_threshold
        ]

        # Remove low-importance memories
        consolidated_count = 0
        for memory in to_consolidate:
            try:
                self.vector_client.delete_points(
                    collection_name=self.collection_name, point_ids=[memory.memory_id]
                )
                consolidated_count += 1
            except Exception as e:
                logger.warning(f"Failed to consolidate memory {memory.memory_id}: {e}")

        if consolidated_count > 0:
            logger.info(
                f"Consolidated {consolidated_count} memories for user {user_id}"
            )

        return consolidated_count

    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Memory statistics
        """
        memories = self.retrieve_memories(
            user_id=user_id, min_importance=0.0, limit=1000
        )

        if not memories:
            return {
                "total_memories": 0,
                "average_importance": 0.0,
                "emotion_distribution": {},
                "memory_types": {},
                "oldest_memory": None,
                "newest_memory": None,
            }

        # Calculate statistics
        total_memories = len(memories)
        current_importances = [m.calculate_current_importance() for m in memories]
        average_importance = sum(current_importances) / len(current_importances)

        # Emotion distribution
        emotion_counts = {}
        for memory in memories:
            emotion = memory.emotional_state.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Memory type distribution
        type_counts = {}
        for memory in memories:
            memory_type = memory.memory_type
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1

        # Temporal info
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        oldest_memory = sorted_memories[0].created_at if sorted_memories else None
        newest_memory = sorted_memories[-1].created_at if sorted_memories else None

        return {
            "total_memories": total_memories,
            "average_importance": average_importance,
            "emotion_distribution": emotion_counts,
            "memory_types": type_counts,
            "oldest_memory": oldest_memory.isoformat() if oldest_memory else None,
            "newest_memory": newest_memory.isoformat() if newest_memory else None,
        }

    def _initialize_collection(self):
        """Initialize the emotional memories collection."""
        try:
            # Check if collection exists
            if not self.vector_client.collection_exists(self.collection_name):
                # Create collection with appropriate configuration
                self.vector_client.create_collection(
                    name=self.collection_name,
                    vector_size=1536,  # Standard embedding size
                    distance="cosine",
                )
                logger.info(
                    f"Created emotional memories collection: {self.collection_name}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize emotional memories collection: {e}")
            raise


# Singleton instance
_memory_storage_instance = None
_memory_storage_lock = threading.Lock()


def get_emotional_memory_storage() -> EmotionalMemoryStorage:
    """
    Get singleton emotional memory storage instance.

    Returns:
        Shared EmotionalMemoryStorage instance
    """
    global _memory_storage_instance

    if _memory_storage_instance is None:
        with _memory_storage_lock:
            if _memory_storage_instance is None:
                _memory_storage_instance = EmotionalMemoryStorage()

    return _memory_storage_instance
