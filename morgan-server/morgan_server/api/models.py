"""
API Models for Morgan Server

This module defines all Pydantic models for API requests and responses.
These models provide validation, serialization, and documentation for the API.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Chat API Models
# ============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    user_id: Optional[str] = Field(None, description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()


class Source(BaseModel):
    """Source information for RAG responses."""

    content: str = Field(..., description="Source content snippet")
    document_id: Optional[str] = Field(None, description="Document identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source metadata"
    )


class MilestoneCelebration(BaseModel):
    """Milestone celebration information."""

    milestone_type: str = Field(..., description="Type of milestone")
    message: str = Field(..., description="Celebration message")
    achieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Achievement timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional milestone data"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    emotional_tone: Optional[str] = Field(None, description="Detected emotional tone")
    empathy_level: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Empathy level"
    )
    personalization_elements: List[str] = Field(
        default_factory=list, description="Applied personalization"
    )
    milestone_celebration: Optional[MilestoneCelebration] = Field(
        None, description="Milestone info"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    sources: List[Source] = Field(default_factory=list, description="RAG sources")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# ============================================================================
# Memory API Models
# ============================================================================


class MemoryStats(BaseModel):
    """Statistics about conversation memory."""

    total_conversations: int = Field(
        ..., ge=0, description="Total number of conversations"
    )
    active_conversations: int = Field(..., ge=0, description="Active conversations")
    total_messages: int = Field(..., ge=0, description="Total messages")
    oldest_conversation: Optional[datetime] = Field(
        None, description="Oldest conversation timestamp"
    )
    newest_conversation: Optional[datetime] = Field(
        None, description="Newest conversation timestamp"
    )


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""

    query: str = Field(..., min_length=1, description="Search query")
    user_id: Optional[str] = Field(None, description="User identifier")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class MemorySearchResult(BaseModel):
    """Result from memory search."""

    conversation_id: str = Field(..., description="Conversation identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    message: str = Field(..., description="User message")
    response: str = Field(..., description="Assistant response")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


# ============================================================================
# Knowledge API Models
# ============================================================================


class LearnRequest(BaseModel):
    """Request model for learning from documents."""

    source: Optional[str] = Field(None, description="File path to document")
    url: Optional[str] = Field(None, description="URL to fetch document from")
    content: Optional[str] = Field(None, description="Direct content to learn")
    doc_type: str = Field(
        "auto", description="Document type (auto, pdf, markdown, text, html)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, v):
        """Validate document type."""
        valid_types = ["auto", "pdf", "markdown", "text", "html", "docx"]
        if v not in valid_types:
            raise ValueError(f"doc_type must be one of {valid_types}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure at least one source is provided
        if not any([self.source, self.url, self.content]):
            raise ValueError("At least one of source, url, or content must be provided")


class LearnResponse(BaseModel):
    """Response model for learning operation."""

    status: str = Field(..., description="Operation status")
    documents_processed: int = Field(
        ..., ge=0, description="Number of documents processed"
    )
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    processing_time_seconds: float = Field(..., ge=0.0, description="Processing time")


class KnowledgeStats(BaseModel):
    """Statistics about knowledge base."""

    total_documents: int = Field(..., ge=0, description="Total documents")
    total_chunks: int = Field(..., ge=0, description="Total chunks")
    total_size_bytes: int = Field(..., ge=0, description="Total size in bytes")
    collections: List[str] = Field(
        default_factory=list, description="Available collections"
    )


# ============================================================================
# Profile API Models
# ============================================================================


class ProfileResponse(BaseModel):
    """Response model for user profile."""

    user_id: str = Field(..., description="User identifier")
    preferred_name: Optional[str] = Field(None, description="User's preferred name")
    relationship_age_days: int = Field(
        ..., ge=0, description="Days since first interaction"
    )
    interaction_count: int = Field(..., ge=0, description="Total interactions")
    trust_level: float = Field(..., ge=0.0, le=1.0, description="Trust level")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Engagement score")
    communication_style: str = Field(..., description="Preferred communication style")
    response_length: str = Field(..., description="Preferred response length")
    topics_of_interest: List[str] = Field(
        default_factory=list, description="Topics of interest"
    )


class PreferenceUpdate(BaseModel):
    """Request model for updating user preferences."""

    communication_style: Optional[str] = Field(None, description="Communication style")
    response_length: Optional[str] = Field(
        None, description="Response length preference"
    )
    topics_of_interest: Optional[List[str]] = Field(
        None, description="Topics of interest"
    )
    preferred_name: Optional[str] = Field(None, description="Preferred name")

    @field_validator("communication_style")
    @classmethod
    def validate_communication_style(cls, v):
        """Validate communication style."""
        if v is not None:
            valid_styles = [
                "casual",
                "professional",
                "friendly",
                "technical",
                "playful",
            ]
            if v not in valid_styles:
                raise ValueError(f"communication_style must be one of {valid_styles}")
        return v

    @field_validator("response_length")
    @classmethod
    def validate_response_length(cls, v):
        """Validate response length."""
        if v is not None:
            valid_lengths = ["brief", "moderate", "detailed"]
            if v not in valid_lengths:
                raise ValueError(f"response_length must be one of {valid_lengths}")
        return v


# ============================================================================
# Health and Status API Models
# ============================================================================


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp",
    )
    version: str = Field(..., description="Server version")
    uptime_seconds: float = Field(..., ge=0.0, description="Server uptime")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status value."""
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v


class ComponentStatus(BaseModel):
    """Status of a single component."""

    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status (up, down, degraded)")
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Component latency")
    error: Optional[str] = Field(None, description="Error message if down")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status value."""
        valid_statuses = ["up", "down", "degraded"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v


class SystemMetrics(BaseModel):
    """System-wide metrics."""

    requests_total: int = Field(..., ge=0, description="Total requests")
    requests_per_second: float = Field(..., ge=0.0, description="Current request rate")
    average_response_time_ms: float = Field(
        ..., ge=0.0, description="Average response time"
    )
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate")
    active_sessions: int = Field(..., ge=0, description="Active sessions")


class StatusResponse(BaseModel):
    """Response model for detailed status."""

    status: str = Field(..., description="Overall status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp",
    )
    components: Dict[str, ComponentStatus] = Field(
        ..., description="Component statuses"
    )
    metrics: SystemMetrics = Field(..., description="System metrics")


# ============================================================================
# Error Response Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )

    @field_validator("error")
    @classmethod
    def validate_error_code(cls, v):
        """Ensure error code is uppercase and uses underscores."""
        if not v.isupper() or " " in v:
            raise ValueError(
                "Error code must be uppercase with underscores (e.g., INVALID_REQUEST)"
            )
        return v


# ============================================================================
# Timeline API Models
# ============================================================================


class TimelineEvent(BaseModel):
    """A single event in the user's timeline."""

    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    description: str = Field(..., description="Event description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")


class TimelineResponse(BaseModel):
    """Response model for user timeline."""

    user_id: str = Field(..., description="User identifier")
    events: List[TimelineEvent] = Field(
        default_factory=list, description="Timeline events"
    )
    total_events: int = Field(..., ge=0, description="Total number of events")
