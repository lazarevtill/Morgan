"""
VAD Service implementation using Silero VAD
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from pydantic import BaseModel

from shared.config.base import ServiceConfig
from shared.models.base import VADRequest, VADResponse, ProcessingResult, AudioChunk
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, AudioError, ErrorCode
from shared.utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class VADConfig(BaseModel):
    """VAD service configuration"""
    host: str = "0.0.0.0"
    port: int = 8004
    model: str = "silero_vad"
    threshold: float = 0.5
    min_speech_duration: float = 0.25
    max_speech_duration: float = 30.0
    window_size: int = 512
    sample_rate: int = 16000
    device: str = "cpu"
    log_level: str = "INFO"


class VADService:
    """Voice Activity Detection Service using Silero VAD"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("vad")
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "vad_service",
            self.config.get("log_level", "INFO"),
            "logs/vad_service.log"
        )

        # Load configuration
        self.vad_config = VADConfig(**self.config.all())

        # Model management
        self.vad_model = None
        self.vad_utils = None
        self.device = torch.device(self.vad_config.device)

        # VAD state management
        self.active_states = {}  # Track states per session/conversation
        self.audio_utils = AudioUtils()

        self.logger.info(f"VAD Service initialized with device: {self.device}")

    async def start(self):
        """Start the VAD service"""
        try:
            await self._load_vad_model()
            self.logger.info("VAD Service started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start VAD service: {e}")
            raise

    async def stop(self):
        """Stop the VAD service"""
        self.logger.info("VAD Service stopping...")

        # Clean up model
        if self.vad_model:
            del self.vad_model
            self.vad_model = None

        # Clear states
        self.active_states.clear()

        self.logger.info("VAD Service stopped")

    async def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            self.logger.info("Loading Silero VAD model...")

            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )

            self.vad_model = model
            self.vad_utils = utils

            self.logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load VAD model: {e}")
            raise AudioError(f"VAD model load failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def detect_speech(self, request: VADRequest) -> VADResponse:
        """Detect speech in audio data"""
        with Timer(self.logger, f"VAD detection for audio length {len(request.audio_data)} bytes"):
            try:
                # Convert audio bytes to numpy array
                audio_array = self.audio_utils.bytes_to_numpy(
                    request.audio_data,
                    sample_rate=request.sample_rate or self.vad_config.sample_rate
                )

                # Apply VAD
                return await self._process_vad(audio_array, request)

            except Exception as e:
                self.logger.error(f"Error detecting speech: {e}")
                raise AudioError(f"Speech detection failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _process_vad(self, audio_array: np.ndarray, request: VADRequest) -> VADResponse:
        """Process audio with VAD"""
        try:
            # Get or create VAD state for this session
            state_key = getattr(request, 'session_id', 'default')
            if state_key not in self.active_states:
                self.active_states[state_key] = self.vad_utils[0](reload=False)

            vad_state = self.active_states[state_key]

            # Process audio in chunks for better accuracy
            chunk_size = int(self.vad_config.sample_rate * 1.0)  # 1 second chunks
            speech_segments = []
            confidences = []

            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]

                if len(chunk) == 0:
                    continue

                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    padding = np.zeros(chunk_size - len(chunk))
                    chunk = np.concatenate([chunk, padding])

                # Run VAD
                confidence = self.vad_model(chunk, self.vad_config.sample_rate, state=vad_state)
                confidences.append(confidence)

                # Check if speech is detected
                if confidence > (request.threshold or self.vad_config.threshold):
                    speech_segments.append({
                        "start": i / self.vad_config.sample_rate,
                        "end": (i + len(chunk)) / self.vad_config.sample_rate,
                        "confidence": float(confidence),
                        "chunk_start": i,
                        "chunk_end": i + len(chunk)
                    })

            # Clean up inactive states periodically
            await self._cleanup_states()

            # Determine if overall speech is detected
            speech_detected = len(speech_segments) > 0
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return VADResponse(
                speech_detected=speech_detected,
                confidence=float(avg_confidence),
                speech_segments=speech_segments,
                metadata={
                    "model": self.vad_config.model,
                    "device": str(self.device),
                    "sample_rate": self.vad_config.sample_rate,
                    "threshold": request.threshold or self.vad_config.threshold,
                    "total_chunks": len(confidences),
                    "speech_chunks": len(speech_segments)
                }
            )

        except Exception as e:
            self.logger.error(f"VAD processing failed: {e}")
            raise

    async def _cleanup_states(self):
        """Clean up inactive VAD states"""
        # Simple cleanup: remove states older than 5 minutes
        import time
        current_time = time.time()
        timeout = 300  # 5 minutes

        to_remove = []
        for key, state_info in self.active_states.items():
            if hasattr(state_info, 'last_used'):
                if current_time - state_info.last_used > timeout:
                    to_remove.append(key)
            else:
                # Mark state as used
                state_info.last_used = current_time

        for key in to_remove:
            del self.active_states[key]

        if to_remove:
            self.logger.debug(f"Cleaned up {len(to_remove)} inactive VAD states")

    async def process_stream(self, audio_chunks: List[AudioChunk]) -> List[VADResponse]:
        """Process streaming audio chunks"""
        try:
            responses = []

            for chunk in audio_chunks:
                # Create VAD request
                vad_request = VADRequest(
                    audio_data=chunk.data,
                    threshold=self.vad_config.threshold,
                    sample_rate=chunk.sample_rate
                )

                response = await self.detect_speech(vad_request)
                responses.append(response)

            return responses

        except Exception as e:
            self.logger.error(f"Stream processing failed: {e}")
            raise AudioError(f"Stream processing failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def get_vad_info(self) -> Dict[str, Any]:
        """Get VAD model and configuration info"""
        return {
            "model": self.vad_config.model,
            "threshold": self.vad_config.threshold,
            "min_speech_duration": self.vad_config.min_speech_duration,
            "max_speech_duration": self.vad_config.max_speech_duration,
            "window_size": self.vad_config.window_size,
            "sample_rate": self.vad_config.sample_rate,
            "device": str(self.device),
            "active_states": len(self.active_states)
        }

    async def reset_state(self, session_id: str = "default"):
        """Reset VAD state for a session"""
        try:
            if session_id in self.active_states:
                del self.active_states[session_id]
                self.logger.info(f"Reset VAD state for session: {session_id}")
            else:
                self.logger.warning(f"No VAD state found for session: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to reset VAD state: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test basic VAD functionality
            test_audio = np.random.randn(self.vad_config.sample_rate).astype(np.float32)
            test_request = VADRequest(
                audio_data=self.audio_utils.numpy_to_bytes(test_audio, self.vad_config.sample_rate),
                threshold=self.vad_config.threshold,
                sample_rate=self.vad_config.sample_rate
            )

            result = await self.detect_speech(test_request)

            return {
                "status": "healthy",
                "model": self.vad_config.model,
                "device": str(self.device),
                "active_states": len(self.active_states),
                "threshold": self.vad_config.threshold,
                "test_detection": result.speech_detected or not result.speech_detected  # Just check it works
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device),
                "active_states": len(self.active_states)
            }
