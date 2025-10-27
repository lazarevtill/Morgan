"""
PostgreSQL database models for Morgan AI Assistant
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConversationModel(BaseModel):
    """Conversation database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    conversation_id: str
    user_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

    class Config:
        from_attributes = True


class MessageModel(BaseModel):
    """Message database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sequence_number: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[int] = None

    class Config:
        from_attributes = True


class StreamingSessionModel(BaseModel):
    """Streaming session database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    session_id: str
    user_id: str
    conversation_id: Optional[UUID] = None
    status: str = "active"  # active, paused, ended, error
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    session_type: str  # audio, text, websocket
    metadata: Dict[str, Any] = Field(default_factory=dict)
    audio_chunks_count: int = 0
    total_duration_ms: int = 0

    class Config:
        from_attributes = True


class AudioTranscriptionModel(BaseModel):
    """Audio transcription database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    session_id: UUID
    message_id: Optional[UUID] = None
    transcription: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    audio_format: Optional[str] = None
    sample_rate: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class TTSGenerationModel(BaseModel):
    """TTS generation cache database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    message_id: UUID
    text_hash: str
    voice: str
    speed: float = 1.0
    audio_data: Optional[bytes] = None
    audio_format: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class UserPreferencesModel(BaseModel):
    """User preferences database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: str
    preferred_language: str = "en"
    preferred_voice: str = "af_heart"
    tts_speed: float = 1.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class SystemMetricModel(BaseModel):
    """System metric database model"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    metric_type: str
    metric_value: float
    service_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

