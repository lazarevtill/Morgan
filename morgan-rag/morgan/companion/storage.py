"""
Companion data storage and vector database schema management.

This module provides storage functionality for companion profiles, emotional data,
relationship milestones, and empathetic responses using Qdrant vector database.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from morgan.vector_db.client import VectorDBClient
from morgan.emotional.models import (
    CompanionProfile, EmotionalState, UserPreferences, RelationshipMilestone
)
from morgan.services.embedding_service import EmbeddingService
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class CompanionStorage:
    """
    Storage manager for companion profiles and emotional data.
    
    Handles vector database operations for companion relationships,
    emotional intelligence, and user preferences.
    """
    
    # Collection names
    COMPANIONS_COLLECTION = "morgan_companions"
    MEMORIES_COLLECTION = "morgan_memories"
    EMOTIONS_COLLECTION = "morgan_emotions"
    MILESTONES_COLLECTION = "morgan_milestones"
    
    def __init__(self, vector_client: Optional[VectorDBClient] = None,
                 embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize companion storage.
        
        Args:
            vector_client: Vector database client
            embedding_service: Embedding service for vectorization
        """
        self.vector_client = vector_client or VectorDBClient()
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize collections
        self._ensure_collections_exist()
    
    def _ensure_collections_exist(self):
        """Ensure all required collections exist with proper schema."""
        collections_config = {
            self.COMPANIONS_COLLECTION: {
                "vector_size": 1536,  # User profile embedding dimension
                "description": "Companion profiles with relationship data"
            },
            self.MEMORIES_COLLECTION: {
                "vector_size": 1536,  # Memory content embedding dimension
                "description": "Conversation memories with emotional context"
            },
            self.EMOTIONS_COLLECTION: {
                "vector_size": 768,   # Emotional state embedding dimension
                "description": "Emotional states and mood patterns"
            },
            self.MILESTONES_COLLECTION: {
                "vector_size": 512,   # Milestone description embedding dimension
                "description": "Relationship milestones and achievements"
            }
        }
        
        for collection_name, config in collections_config.items():
            if not self.vector_client.collection_exists(collection_name):
                success = self.vector_client.create_collection(
                    name=collection_name,
                    vector_size=config["vector_size"],
                    distance="cosine"
                )
                if success:
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.error(f"Failed to create collection: {collection_name}")
    
    def store_companion_profile(self, profile: CompanionProfile) -> bool:
        """
        Store or update a companion profile.
        
        Args:
            profile: CompanionProfile to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Create profile embedding from user preferences and interaction patterns
            profile_text = self._build_profile_text(profile)
            profile_embedding = self.embedding_service.embed_text(profile_text)
            
            # Prepare point data
            point_data = {
                "id": profile.user_id,
                "vector": profile_embedding,
                "payload": {
                    "user_id": profile.user_id,
                    "preferred_name": profile.preferred_name,
                    "relationship_duration_days": profile.get_relationship_age_days(),
                    "interaction_count": profile.interaction_count,
                    "trust_level": profile.trust_level,
                    "engagement_score": profile.engagement_score,
                    "last_interaction": profile.last_interaction.isoformat(),
                    "profile_created": profile.profile_created.isoformat(),
                    
                    # Communication preferences
                    "communication_style": profile.communication_preferences.communication_style.value,
                    "preferred_response_length": profile.communication_preferences.preferred_response_length.value,
                    "topics_of_interest": profile.communication_preferences.topics_of_interest,
                    "learning_goals": profile.communication_preferences.learning_goals,
                    
                    # Emotional and relationship data
                    "emotional_patterns": profile.emotional_patterns,
                    "shared_memories": profile.shared_memories,
                    "milestone_count": len(profile.relationship_milestones),
                    
                    # Metadata
                    "updated_at": datetime.utcnow().isoformat(),
                    "profile_version": "1.0"
                }
            }
            
            return self.vector_client.upsert_points(
                collection_name=self.COMPANIONS_COLLECTION,
                points=[point_data]
            )
            
        except Exception as e:
            logger.error(f"Failed to store companion profile for {profile.user_id}: {e}")
            return False
    
    def get_companion_profile(self, user_id: str) -> Optional[CompanionProfile]:
        """
        Retrieve a companion profile by user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            CompanionProfile if found, None otherwise
        """
        try:
            results = self.vector_client.search_with_filter(
                collection_name=self.COMPANIONS_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=1
            )
            
            if not results:
                return None
            
            return self._build_companion_profile_from_payload(results[0].payload)
            
        except Exception as e:
            logger.error(f"Failed to get companion profile for {user_id}: {e}")
            return None
    
    def store_emotional_state(self, user_id: str, emotional_state: EmotionalState) -> bool:
        """
        Store an emotional state record.
        
        Args:
            user_id: User identifier
            emotional_state: EmotionalState to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Create embedding from emotional indicators and context
            emotion_text = f"{emotional_state.primary_emotion.value} {' '.join(emotional_state.emotional_indicators)}"
            emotion_embedding = self.embedding_service.embed_text(emotion_text)
            
            # Generate unique ID for this emotional state
            emotion_id = f"{user_id}_{int(emotional_state.timestamp.timestamp())}"
            
            point_data = {
                "id": emotion_id,
                "vector": emotion_embedding,
                "payload": {
                    "user_id": user_id,
                    "primary_emotion": emotional_state.primary_emotion.value,
                    "intensity": emotional_state.intensity,
                    "confidence": emotional_state.confidence,
                    "secondary_emotions": [e.value for e in emotional_state.secondary_emotions],
                    "emotional_indicators": emotional_state.emotional_indicators,
                    "timestamp": emotional_state.timestamp.isoformat(),
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            return self.vector_client.upsert_points(
                collection_name=self.EMOTIONS_COLLECTION,
                points=[point_data]
            )
            
        except Exception as e:
            logger.error(f"Failed to store emotional state for {user_id}: {e}")
            return False
    
    def store_relationship_milestone(self, user_id: str, milestone: RelationshipMilestone) -> bool:
        """
        Store a relationship milestone.
        
        Args:
            user_id: User identifier
            milestone: RelationshipMilestone to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Create embedding from milestone description
            milestone_embedding = self.embedding_service.embed_text(milestone.description)
            
            point_data = {
                "id": milestone.milestone_id,
                "vector": milestone_embedding,
                "payload": {
                    "user_id": user_id,
                    "milestone_id": milestone.milestone_id,
                    "milestone_type": milestone.milestone_type.value,
                    "description": milestone.description,
                    "timestamp": milestone.timestamp.isoformat(),
                    "emotional_significance": milestone.emotional_significance,
                    "related_memories": milestone.related_memories,
                    "user_feedback": milestone.user_feedback,
                    "celebration_acknowledged": milestone.celebration_acknowledged,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            return self.vector_client.upsert_points(
                collection_name=self.MILESTONES_COLLECTION,
                points=[point_data]
            )
            
        except Exception as e:
            logger.error(f"Failed to store milestone for {user_id}: {e}")
            return False
    
    def get_user_emotional_history(self, user_id: str, days: int = 30) -> List[EmotionalState]:
        """
        Get user's emotional history for the specified period.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of EmotionalState records
        """
        try:
            # Get all emotional states for user
            results = self.vector_client.search_with_filter(
                collection_name=self.EMOTIONS_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=1000  # Large limit to get comprehensive history
            )
            
            # Filter by date and convert to EmotionalState objects
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            emotional_states = []
            
            for result in results:
                payload = result.payload
                timestamp = datetime.fromisoformat(payload["timestamp"])
                
                if timestamp >= cutoff_date:
                    emotional_states.append(self._build_emotional_state_from_payload(payload))
            
            # Sort by timestamp (newest first)
            emotional_states.sort(key=lambda x: x.timestamp, reverse=True)
            return emotional_states
            
        except Exception as e:
            logger.error(f"Failed to get emotional history for {user_id}: {e}")
            return []
    
    def get_user_milestones(self, user_id: str) -> List[RelationshipMilestone]:
        """
        Get all relationship milestones for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of RelationshipMilestone records
        """
        try:
            results = self.vector_client.search_with_filter(
                collection_name=self.MILESTONES_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=100
            )
            
            milestones = []
            for result in results:
                milestones.append(self._build_milestone_from_payload(result.payload))
            
            # Sort by timestamp (oldest first)
            milestones.sort(key=lambda x: x.timestamp)
            return milestones
            
        except Exception as e:
            logger.error(f"Failed to get milestones for {user_id}: {e}")
            return []
    
    def search_similar_emotional_states(self, emotional_state: EmotionalState, 
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar emotional states across all users.
        
        Args:
            emotional_state: EmotionalState to find similar states for
            limit: Maximum results to return
            
        Returns:
            List of similar emotional state records
        """
        try:
            # Create query embedding
            emotion_text = f"{emotional_state.primary_emotion.value} {' '.join(emotional_state.emotional_indicators)}"
            query_embedding = self.embedding_service.embed_text(emotion_text)
            
            results = self.vector_client.search(
                collection_name=self.EMOTIONS_COLLECTION,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.7  # Only return reasonably similar states
            )
            
            return [
                {
                    "emotional_state": self._build_emotional_state_from_payload(result.payload),
                    "similarity_score": result.score,
                    "user_id": result.payload["user_id"]
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to search similar emotional states: {e}")
            return []
    
    def _build_profile_text(self, profile: CompanionProfile) -> str:
        """Build text representation of profile for embedding."""
        prefs = profile.communication_preferences
        
        text_parts = [
            f"Communication style: {prefs.communication_style.value}",
            f"Response length: {prefs.preferred_response_length.value}",
            f"Topics of interest: {', '.join(prefs.topics_of_interest)}",
            f"Learning goals: {', '.join(prefs.learning_goals)}",
            f"Interaction count: {profile.interaction_count}",
            f"Trust level: {profile.trust_level}",
            f"Engagement score: {profile.engagement_score}"
        ]
        
        return " | ".join(text_parts)
    
    def _build_companion_profile_from_payload(self, payload: Dict[str, Any]) -> CompanionProfile:
        """Build CompanionProfile from vector database payload."""
        from morgan.emotional.models import CommunicationStyle, ResponseLength
        
        # Parse communication preferences
        communication_preferences = UserPreferences(
            topics_of_interest=payload.get("topics_of_interest", []),
            communication_style=CommunicationStyle(payload.get("communication_style", "friendly")),
            preferred_response_length=ResponseLength(payload.get("preferred_response_length", "detailed")),
            learning_goals=payload.get("learning_goals", []),
            personal_context=payload.get("personal_context", {}),
            last_updated=datetime.fromisoformat(payload.get("updated_at", datetime.utcnow().isoformat()))
        )
        
        # Calculate relationship duration
        profile_created = datetime.fromisoformat(payload["profile_created"])
        relationship_duration = datetime.utcnow() - profile_created
        
        return CompanionProfile(
            user_id=payload["user_id"],
            preferred_name=payload.get("preferred_name", ""),
            relationship_duration=relationship_duration,
            interaction_count=payload.get("interaction_count", 0),
            communication_preferences=communication_preferences,
            emotional_patterns=payload.get("emotional_patterns", {}),
            shared_memories=payload.get("shared_memories", []),
            relationship_milestones=[],  # Load separately if needed
            last_interaction=datetime.fromisoformat(payload["last_interaction"]),
            trust_level=payload.get("trust_level", 0.0),
            engagement_score=payload.get("engagement_score", 0.0),
            profile_created=profile_created
        )
    
    def _build_emotional_state_from_payload(self, payload: Dict[str, Any]) -> EmotionalState:
        """Build EmotionalState from vector database payload."""
        from morgan.emotional.models import EmotionType
        
        return EmotionalState(
            primary_emotion=EmotionType(payload["primary_emotion"]),
            intensity=payload["intensity"],
            confidence=payload["confidence"],
            secondary_emotions=[EmotionType(e) for e in payload.get("secondary_emotions", [])],
            emotional_indicators=payload.get("emotional_indicators", []),
            timestamp=datetime.fromisoformat(payload["timestamp"])
        )
    
    def _build_milestone_from_payload(self, payload: Dict[str, Any]) -> RelationshipMilestone:
        """Build RelationshipMilestone from vector database payload."""
        from morgan.emotional.models import MilestoneType
        
        return RelationshipMilestone(
            milestone_id=payload["milestone_id"],
            milestone_type=MilestoneType(payload["milestone_type"]),
            description=payload["description"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            emotional_significance=payload["emotional_significance"],
            related_memories=payload.get("related_memories", []),
            user_feedback=payload.get("user_feedback"),
            celebration_acknowledged=payload.get("celebration_acknowledged", False)
        )
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user (GDPR compliance).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if all data deleted successfully
        """
        try:
            success = True
            
            # Delete companion profile
            success &= self.vector_client.delete_points(
                collection_name=self.COMPANIONS_COLLECTION,
                point_ids=[user_id]
            )
            
            # Get and delete emotional states
            emotion_results = self.vector_client.search_with_filter(
                collection_name=self.EMOTIONS_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=1000
            )
            if emotion_results:
                emotion_ids = [result.id for result in emotion_results]
                success &= self.vector_client.delete_points(
                    collection_name=self.EMOTIONS_COLLECTION,
                    point_ids=emotion_ids
                )
            
            # Get and delete milestones
            milestone_results = self.vector_client.search_with_filter(
                collection_name=self.MILESTONES_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=100
            )
            if milestone_results:
                milestone_ids = [result.id for result in milestone_results]
                success &= self.vector_client.delete_points(
                    collection_name=self.MILESTONES_COLLECTION,
                    point_ids=milestone_ids
                )
            
            if success:
                logger.info(f"Successfully deleted all data for user {user_id}")
            else:
                logger.error(f"Some data deletion failed for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete user data for {user_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for all companion collections.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {}
        
        collections = [
            self.COMPANIONS_COLLECTION,
            self.MEMORIES_COLLECTION,
            self.EMOTIONS_COLLECTION,
            self.MILESTONES_COLLECTION
        ]
        
        for collection_name in collections:
            try:
                info = self.vector_client.get_collection_info(collection_name)
                stats[collection_name] = info
            except Exception as e:
                stats[collection_name] = {"error": str(e)}
        
        return stats