"""
Vector database schema definitions for companion and emotional data.

This module defines the schema structure for all companion-related collections
in the Qdrant vector database, including field types, indexing strategies,
and data organization patterns.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class CollectionSchema:
    """Schema definition for a vector database collection."""
    name: str
    vector_size: int
    distance_metric: str
    description: str
    payload_schema: Dict[str, str]
    indexes: List[str]


class CompanionDatabaseSchema:
    """
    Complete schema definitions for companion and emotional intelligence data.
    
    Defines four main collections:
    1. morgan_companions - User companion profiles with relationship data
    2. morgan_memories - Conversation memories with emotional context  
    3. morgan_emotions - Emotional states and mood patterns
    4. morgan_milestones - Relationship milestones and achievements
    """
    
    @staticmethod
    def get_companions_schema() -> CollectionSchema:
        """
        Schema for companion profiles collection.
        
        Stores complete user relationship profiles with embeddings based on
        communication preferences, interests, and interaction patterns.
        """
        return CollectionSchema(
            name="morgan_companions",
            vector_size=1536,  # OpenAI text-embedding-3-large dimension
            distance_metric="cosine",
            description="Companion profiles with relationship and preference data",
            payload_schema={
                # Core identity
                "user_id": "keyword",           # Primary identifier
                "preferred_name": "text",       # What user likes to be called
                
                # Relationship metrics
                "relationship_duration_days": "integer",  # Days since first interaction
                "interaction_count": "integer",           # Total interactions
                "trust_level": "float",                   # 0.0 to 1.0
                "engagement_score": "float",              # 0.0 to 1.0
                
                # Timestamps
                "last_interaction": "datetime",   # ISO format
                "profile_created": "datetime",    # ISO format
                "updated_at": "datetime",         # ISO format
                
                # Communication preferences
                "communication_style": "keyword",        # formal, casual, technical, friendly
                "preferred_response_length": "keyword",  # brief, detailed, comprehensive
                "topics_of_interest": "keyword[]",       # Array of interest topics
                "learning_goals": "keyword[]",           # Array of learning objectives
                
                # Emotional and relationship data
                "emotional_patterns": "json",     # Complex emotional pattern data
                "shared_memories": "keyword[]",   # Array of memory IDs
                "milestone_count": "integer",     # Number of relationship milestones
                
                # Metadata
                "profile_version": "keyword"      # Schema version for migrations
            },
            indexes=[
                "user_id",                    # Primary lookup
                "communication_style",       # Filter by communication preference
                "last_interaction",          # Sort by recency
                "trust_level",              # Filter by relationship depth
                "engagement_score"          # Filter by engagement level
            ]
        )
    
    @staticmethod
    def get_memories_schema() -> CollectionSchema:
        """
        Schema for conversation memories collection.
        
        Stores important conversation segments with emotional context and
        relationship significance for personalized interactions.
        """
        return CollectionSchema(
            name="morgan_memories",
            vector_size=1536,  # Memory content embedding dimension
            distance_metric="cosine", 
            description="Conversation memories with emotional and relationship context",
            payload_schema={
                # Core memory data
                "user_id": "keyword",           # User this memory belongs to
                "memory_id": "keyword",         # Unique memory identifier
                "content": "text",              # Memory content text
                "importance_score": "float",    # 0.0 to 1.0 importance rating
                
                # Context and categorization
                "entities": "keyword[]",        # Named entities in memory
                "concepts": "keyword[]",        # Key concepts discussed
                "conversation_id": "keyword",   # Source conversation ID
                
                # Emotional context
                "emotional_context": "json",    # Emotional state during memory
                "user_mood": "keyword",         # Primary user mood
                "relationship_significance": "float",  # 0.0 to 1.0 relationship impact
                
                # Feedback and validation
                "feedback_rating": "integer",   # 1-5 user feedback rating
                "user_feedback": "text",        # Optional user feedback text
                
                # Timestamps
                "timestamp": "datetime",        # When memory was created
                "created_at": "datetime",       # When stored in database
                "last_accessed": "datetime"     # When memory was last retrieved
            },
            indexes=[
                "user_id",                  # Primary user lookup
                "importance_score",         # Filter by importance
                "conversation_id",          # Group by conversation
                "user_mood",               # Filter by emotional context
                "timestamp",               # Sort by chronology
                "relationship_significance" # Filter by relationship impact
            ]
        )
    
    @staticmethod
    def get_emotions_schema() -> CollectionSchema:
        """
        Schema for emotional states collection.
        
        Stores individual emotional state records for mood tracking,
        pattern analysis, and empathetic response generation.
        """
        return CollectionSchema(
            name="morgan_emotions",
            vector_size=768,   # Emotional embedding dimension (smaller, focused)
            distance_metric="cosine",
            description="Emotional states and mood patterns for empathetic AI",
            payload_schema={
                # Core emotional data
                "user_id": "keyword",              # User identifier
                "emotion_id": "keyword",           # Unique emotion record ID
                "primary_emotion": "keyword",      # joy, sadness, anger, fear, surprise, disgust, neutral
                "intensity": "float",              # 0.0 to 1.0 emotion intensity
                "confidence": "float",             # 0.0 to 1.0 detection confidence
                
                # Secondary emotions and indicators
                "secondary_emotions": "keyword[]", # Additional detected emotions
                "emotional_indicators": "text[]",  # Text patterns that indicated emotion
                
                # Context and triggers
                "conversation_context": "text",    # What was being discussed
                "emotional_triggers": "keyword[]", # What triggered this emotion
                "response_generated": "text",      # How Morgan responded
                
                # Temporal data
                "timestamp": "datetime",           # When emotion was detected
                "session_id": "keyword",           # Conversation session ID
                "duration_minutes": "integer",     # How long emotion lasted
                
                # Metadata
                "detection_method": "keyword",     # How emotion was detected
                "created_at": "datetime"           # Database storage time
            },
            indexes=[
                "user_id",              # Primary user lookup
                "primary_emotion",      # Filter by emotion type
                "intensity",           # Filter by intensity level
                "timestamp",           # Chronological sorting
                "session_id",          # Group by conversation session
                "confidence"           # Filter by detection confidence
            ]
        )
    
    @staticmethod
    def get_milestones_schema() -> CollectionSchema:
        """
        Schema for relationship milestones collection.
        
        Stores significant moments and achievements in the user-Morgan
        relationship for celebration and relationship building.
        """
        return CollectionSchema(
            name="morgan_milestones",
            vector_size=512,   # Milestone description embedding (smaller, descriptive)
            distance_metric="cosine",
            description="Relationship milestones and achievements",
            payload_schema={
                # Core milestone data
                "user_id": "keyword",                    # User identifier
                "milestone_id": "keyword",               # Unique milestone ID
                "milestone_type": "keyword",             # first_conversation, breakthrough_moment, etc.
                "description": "text",                   # Human-readable description
                "emotional_significance": "float",       # 0.0 to 1.0 emotional impact
                
                # Related data
                "related_memories": "keyword[]",         # Associated memory IDs
                "conversation_context": "text",          # What led to this milestone
                "achievement_category": "keyword",       # learning, relationship, personal, etc.
                
                # User interaction
                "user_feedback": "text",                 # User's response to milestone
                "celebration_acknowledged": "boolean",   # Whether user acknowledged celebration
                "user_satisfaction": "float",            # 0.0 to 1.0 user satisfaction with milestone
                
                # Temporal data
                "timestamp": "datetime",                 # When milestone occurred
                "detected_at": "datetime",               # When Morgan detected it
                "celebrated_at": "datetime",             # When milestone was celebrated
                
                # Metadata
                "auto_detected": "boolean",              # Whether automatically detected
                "created_at": "datetime"                 # Database storage time
            },
            indexes=[
                "user_id",                    # Primary user lookup
                "milestone_type",             # Filter by milestone category
                "emotional_significance",     # Filter by significance level
                "timestamp",                  # Chronological sorting
                "achievement_category",       # Group by achievement type
                "celebration_acknowledged"    # Filter by acknowledgment status
            ]
        )
    
    @staticmethod
    def get_all_schemas() -> List[CollectionSchema]:
        """Get all companion database schemas."""
        return [
            CompanionDatabaseSchema.get_companions_schema(),
            CompanionDatabaseSchema.get_memories_schema(),
            CompanionDatabaseSchema.get_emotions_schema(),
            CompanionDatabaseSchema.get_milestones_schema()
        ]
    
    @staticmethod
    def get_schema_summary() -> Dict[str, Any]:
        """
        Get a summary of all schemas for documentation and validation.
        
        Returns:
            Dictionary with schema information and statistics
        """
        schemas = CompanionDatabaseSchema.get_all_schemas()
        
        return {
            "total_collections": len(schemas),
            "total_vector_dimensions": sum(s.vector_size for s in schemas),
            "collections": {
                schema.name: {
                    "vector_size": schema.vector_size,
                    "distance_metric": schema.distance_metric,
                    "description": schema.description,
                    "field_count": len(schema.payload_schema),
                    "index_count": len(schema.indexes)
                }
                for schema in schemas
            },
            "schema_version": "1.0",
            "last_updated": "2025-01-01"
        }


# Schema validation utilities
def validate_companion_payload(payload: Dict[str, Any]) -> List[str]:
    """
    Validate a companion profile payload against schema.
    
    Args:
        payload: Payload dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    schema = CompanionDatabaseSchema.get_companions_schema()
    
    # Check required fields
    required_fields = ["user_id", "preferred_name", "communication_style"]
    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types and ranges
    if "trust_level" in payload:
        trust = payload["trust_level"]
        if not isinstance(trust, (int, float)) or not 0.0 <= trust <= 1.0:
            errors.append("trust_level must be float between 0.0 and 1.0")
    
    if "engagement_score" in payload:
        engagement = payload["engagement_score"]
        if not isinstance(engagement, (int, float)) or not 0.0 <= engagement <= 1.0:
            errors.append("engagement_score must be float between 0.0 and 1.0")
    
    return errors


def validate_emotion_payload(payload: Dict[str, Any]) -> List[str]:
    """
    Validate an emotional state payload against schema.
    
    Args:
        payload: Payload dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["user_id", "primary_emotion", "intensity", "confidence"]
    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
    
    # Validate emotion values
    valid_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
    if "primary_emotion" in payload and payload["primary_emotion"] not in valid_emotions:
        errors.append(f"Invalid primary_emotion: {payload['primary_emotion']}")
    
    # Validate ranges
    for field in ["intensity", "confidence"]:
        if field in payload:
            value = payload[field]
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                errors.append(f"{field} must be float between 0.0 and 1.0")
    
    return errors