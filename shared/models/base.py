"""
Base models and data structures for Morgan AI Assistant
"""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class BaseModel:
    """Base class for all data models"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert model to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str):
        """Create model from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Message(BaseModel):
    """Chat message model"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ConversationContext(BaseModel):
    """Conversation context model"""
    conversation_id: str
    user_id: str
    messages: List[Message]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

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


@dataclass
class AudioChunk(BaseModel):
    """Audio data chunk model"""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ProcessingResult(BaseModel):
    """Generic processing result model"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None


@dataclass
class ServiceStatus(BaseModel):
    """Service health status model"""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    version: Optional[str] = None
    uptime: Optional[float] = None
    last_check: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest(BaseModel):
    """LLM service request model"""
    prompt: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context: Optional[List[Message]] = None
    stream: bool = False


@dataclass
class LLMResponse(BaseModel):
    """LLM service response model"""
    text: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TTSRequest(BaseModel):
    """TTS service request model"""
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    language: Optional[str] = None
    format: Optional[str] = None
    sample_rate: Optional[int] = None


@dataclass
class TTSResponse(BaseModel):
    """TTS service response model"""
    audio_data: bytes
    format: str = "wav"
    sample_rate: int = 22050
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class STTRequest(BaseModel):
    """STT service request model"""
    audio_data: bytes
    language: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None


@dataclass
class STTResponse(BaseModel):
    """STT service response model"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None


@dataclass
class VADRequest(BaseModel):
    """Voice Activity Detection request model"""
    audio_data: bytes
    threshold: Optional[float] = None
    sample_rate: Optional[int] = None


@dataclass
class VADResponse(BaseModel):
    """Voice Activity Detection response model"""
    speech_detected: bool
    confidence: Optional[float] = None
    speech_segments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Intent(BaseModel):
    """Intent recognition model"""
    intent: str
    confidence: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None


@dataclass
class Command(BaseModel):
    """Command execution model"""
    action: str
    parameters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    priority: int = 1


@dataclass
class Response(BaseModel):
    """Assistant response model"""
    text: str
    audio_data: Optional[bytes] = None
    actions: Optional[List[Command]] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
