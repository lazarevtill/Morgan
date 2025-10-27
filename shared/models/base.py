"""
Base models and data structures for Morgan AI Assistant
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel as PydanticBaseModel, Field
import json


class BaseModel(PydanticBaseModel):
    """Base class for all data models using Pydantic"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert model to JSON string"""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str):
        """Create model from JSON string"""
        return cls.model_validate_json(json_str)


class Message(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ConversationContext(BaseModel):
    """Conversation context model"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: str = Field(..., description="User identifier")
    messages: List[Message] = Field(default_factory=list, description="List of conversation messages")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get last n messages from conversation"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages

    def get_messages_by_role(self, role: str) -> List[Message]:
        """Get all messages by a specific role"""
        return [msg for msg in self.messages if msg.role == role]


class AudioChunk(BaseModel):
    """Audio data chunk model"""
    data: bytes = Field(..., description="Audio data bytes")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    channels: int = Field(default=1, description="Number of audio channels")
    format: str = Field(default="wav", description="Audio format")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Audio timestamp")


class ProcessingResult(BaseModel):
    """Generic processing result model"""
    success: bool = Field(..., description="Processing success status")
    data: Any = Field(..., description="Processing result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class ServiceStatus(BaseModel):
    """Service health status model"""
    service_name: str = Field(..., description="Name of the service")
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: Optional[str] = Field(default=None, description="Service version")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")
    last_check: Optional[datetime] = Field(default_factory=datetime.now, description="Last health check timestamp")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Service metrics")


class LLMRequest(BaseModel):
    """LLM service request model"""
    prompt: str = Field(..., description="Text prompt for generation")
    model: Optional[str] = Field(default=None, description="Model to use")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, le=8192, description="Maximum tokens to generate")
    context: Optional[List[Message]] = Field(default=None, description="Conversation context")
    stream: bool = Field(default=False, description="Enable streaming response")


class LLMResponse(BaseModel):
    """LLM service response model"""
    text: str = Field(..., description="Generated text")
    model: Optional[str] = Field(default=None, description="Model used")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class TTSRequest(BaseModel):
    """TTS service request model"""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice to use")
    speed: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Speech speed")
    language: Optional[str] = Field(default=None, description="Language code")
    format: Optional[str] = Field(default=None, description="Output format")
    sample_rate: Optional[int] = Field(default=None, gt=0, description="Sample rate")


class TTSResponse(BaseModel):
    """TTS service response model"""
    audio_data: bytes = Field(..., description="Generated audio data")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class STTRequest(BaseModel):
    """STT service request model"""
    audio_data: bytes = Field(..., description="Audio data to transcribe")
    language: Optional[str] = Field(default=None, description="Language code")
    model: Optional[str] = Field(default=None, description="Model to use")
    prompt: Optional[str] = Field(default=None, description="Transcription prompt")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature")


class STTResponse(BaseModel):
    """STT service response model"""
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(default=None, description="Detected language")
    confidence: Optional[float] = Field(default=None, description="Transcription confidence (0.0-1.0, where 1.0 is highest confidence)")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed transcription segments")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class VADRequest(BaseModel):
    """Voice Activity Detection request model"""
    audio_data: bytes = Field(..., description="Audio data for VAD")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection threshold")
    sample_rate: Optional[int] = Field(default=None, gt=0, description="Audio sample rate")


class VADResponse(BaseModel):
    """Voice Activity Detection response model"""
    speech_detected: bool = Field(..., description="Whether speech was detected")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence")
    speech_segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Speech segments")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class Intent(BaseModel):
    """Intent recognition model"""
    intent: str = Field(..., description="Detected intent")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Intent confidence")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Intent parameters")
    entities: Optional[List[Dict[str, Any]]] = Field(default=None, description="Named entities")


class Command(BaseModel):
    """Command execution model"""
    action: str = Field(..., description="Action to execute")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Command parameters")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Command context")
    priority: int = Field(default=1, ge=1, le=10, description="Command priority")


class Response(BaseModel):
    """Assistant response model"""
    text: str = Field(..., description="Response text")
    audio_data: Optional[bytes] = Field(default=None, description="Response audio data")
    actions: Optional[List[Command]] = Field(default=None, description="Actions to execute")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Response confidence")
