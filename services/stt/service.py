"""
STT Service implementation with Silero VAD integration
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import io
import wave
import numpy as np
from pathlib import Path

import torch
import torchaudio
from pydantic import BaseModel

from shared.config.base import ServiceConfig
from shared.models.base import STTRequest, STTResponse, ProcessingResult, AudioChunk
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, AudioError, ErrorCode
from shared.utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class STTConfig(BaseModel):
    """STT service configuration"""
    host: str = "0.0.0.0"
    port: int = 8003
    model: str = "whisper-large-v3"
    device: str = "cuda"
    language: str = "auto"
    sample_rate: int = 16000
    chunk_size: int = 1024
    threshold: float = 0.5
    min_silence_duration: float = 0.5
    log_level: str = "INFO"


class STTService:
    """STT Service with Silero VAD integration"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("stt")
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "stt_service",
            self.config.get("log_level", "INFO"),
            "logs/stt_service.log"
        )

        # Load configuration
        self.stt_config = STTConfig(**self.config.all())

        # Model management
        self.whisper_model = None
        self.vad_model = None
        self.device = torch.device(self.stt_config.device if torch.cuda.is_available() else "cpu")

        # Audio processing
        self.audio_utils = AudioUtils()

        # VAD state
        self.vad_state = None
        self.speech_buffer = []
        self.silence_threshold = self.stt_config.min_silence_duration * self.stt_config.sample_rate

        self.logger.info(f"STT Service initialized with device: {self.device}")

    async def start(self):
        """Start the STT service"""
        try:
            await self._load_whisper_model()
            await self._load_vad_model()
            self.logger.info("STT Service started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start STT service: {e}")
            raise

    async def stop(self):
        """Stop the STT service"""
        self.logger.info("STT Service stopping...")

        # Clean up models
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None

        if self.vad_model:
            del self.vad_model
            self.vad_model = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("STT Service stopped")

    async def _load_whisper_model(self):
        """Load Whisper model"""
        try:
            from faster_whisper import WhisperModel

            self.logger.info(f"Loading Whisper model: {self.stt_config.model}")

            # Load model with optimized settings for CUDA 13
            self.whisper_model = WhisperModel(
                self.stt_config.model,
                device=str(self.device),
                compute_type="float16" if self.device.type == "cuda" else "int8",
                cpu_threads=4,
                num_workers=1,
                download_root="data/models/whisper",
                local_files_only=False  # Allow downloading if not present
            )

            self.logger.info("Whisper model loaded successfully")

        except ImportError:
            self.logger.error("faster-whisper not available, falling back to openai-whisper")
            await self._load_openai_whisper()
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            await self._load_openai_whisper()

    async def _load_openai_whisper(self):
        """Load OpenAI Whisper model as fallback"""
        try:
            import whisper

            self.logger.info(f"Loading OpenAI Whisper model: {self.stt_config.model}")

            self.whisper_model = whisper.load_model(
                self.stt_config.model,
                device=str(self.device),
                download_root="data/models/whisper"
            )

            self.logger.info("OpenAI Whisper model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load OpenAI Whisper model: {e}")
            raise AudioError(f"No STT model available: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )

            self.vad_model = model
            self.vad_utils = utils

            # Initialize VAD state
            self.vad_state = self.vad_utils[0](reload=False)

            self.logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load VAD model: {e}")
            # VAD is optional, continue without it
            self.vad_model = None

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe audio to text"""
        with Timer(self.logger, f"STT transcription for audio length {len(request.audio_data)} bytes"):
            try:
                # Convert audio bytes to numpy array
                audio_array = self.audio_utils.bytes_to_numpy(
                    request.audio_data,
                    sample_rate=self.stt_config.sample_rate
                )

                # Apply VAD if available
                if self.vad_model:
                    audio_array = await self._apply_vad(audio_array)

                # Transcribe with Whisper
                return await self._transcribe_with_whisper(audio_array, request)

            except Exception as e:
                self.logger.error(f"Error transcribing audio: {e}")
                raise AudioError(f"Transcription failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _apply_vad(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply Voice Activity Detection to audio"""
        try:
            # Reset VAD state if needed
            if not hasattr(self, '_vad_state_initialized') or not self._vad_state_initialized:
                self.vad_state = self.vad_utils[0](reload=False)
                self._vad_state_initialized = True

            # Process audio in chunks for VAD
            chunk_size = int(self.stt_config.sample_rate * 0.5)  # 0.5 second chunks
            speech_segments = []

            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]

                if len(chunk) < chunk_size:
                    # Pad last chunk
                    padding = np.zeros(chunk_size - len(chunk))
                    chunk = np.concatenate([chunk, padding])

                # Run VAD
                confidence = self.vad_model(chunk, self.stt_config.sample_rate, state=self.vad_state)

                if confidence > self.stt_config.threshold:
                    speech_segments.append(chunk)
                else:
                    # Keep some silence for context
                    if speech_segments:
                        speech_segments.append(chunk[:int(chunk_size * 0.1)])

            if speech_segments:
                return np.concatenate(speech_segments)
            else:
                # If no speech detected, return original with warning
                self.logger.warning("No speech detected by VAD")
                return audio_array

        except Exception as e:
            self.logger.error(f"VAD processing failed: {e}")
            # Fallback to original audio
            return audio_array

    async def _transcribe_with_whisper(self, audio_array: np.ndarray, request: STTRequest) -> STTResponse:
        """Transcribe audio using Whisper"""
        try:
            # Prepare transcription options
            trans_options = {
                "language": request.language or self.stt_config.language,
                "temperature": request.temperature or 0.0,
                "initial_prompt": request.prompt,
                "suppress_tokens": [-1],  # Suppress timestamps
                "without_timestamps": True
            }

            if self.whisper_model.__class__.__module__.startswith('faster_whisper'):
                # Faster Whisper API
                segments, info = self.whisper_model.transcribe(
                    audio_array,
                    **trans_options
                )

                # Extract text from segments
                text = " ".join([segment.text.strip() for segment in segments])

                # Calculate confidence (average of all segments)
                confidence = sum(segment.avg_logprob for segment in segments) / len(segments) if segments else 0.0

                # Extract detailed segments
                detailed_segments = [{
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.avg_logprob
                } for segment in segments]

            else:
                # OpenAI Whisper API
                audio_tensor = torch.from_numpy(audio_array).float()
                if self.device.type == "cuda":
                    audio_tensor = audio_tensor.cuda()

                result = self.whisper_model.transcribe(audio_tensor, **trans_options)
                text = result["text"].strip()
                confidence = 0.0  # OpenAI Whisper doesn't provide confidence scores
                detailed_segments = result.get("segments", [])

            return STTResponse(
                text=text,
                language=request.language or self.stt_config.language,
                confidence=confidence,
                duration=len(audio_array) / self.stt_config.sample_rate,
                segments=detailed_segments,
                metadata={
                    "model": self.stt_config.model,
                    "vad_enabled": self.vad_model is not None,
                    "device": str(self.device)
                }
            )

        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            raise

    async def transcribe_stream(self, audio_chunks: List[AudioChunk]) -> STTResponse:
        """Transcribe streaming audio chunks"""
        try:
            # Combine all audio chunks
            combined_audio = []
            total_duration = 0.0

            for chunk in audio_chunks:
                audio_array = self.audio_utils.bytes_to_numpy(
                    chunk.data,
                    sample_rate=chunk.sample_rate
                )
                combined_audio.append(audio_array)
                total_duration += len(audio_array) / chunk.sample_rate

            if not combined_audio:
                return STTResponse(
                    text="",
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio data"}
                )

            # Concatenate audio
            final_audio = np.concatenate(combined_audio)

            # Create a simple STT request
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(final_audio, self.stt_config.sample_rate),
                language=self.stt_config.language
            )

            return await self.transcribe(request)

        except Exception as e:
            self.logger.error(f"Stream transcription failed: {e}")
            raise AudioError(f"Stream transcription failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def detect_language(self, audio_bytes: bytes) -> str:
        """Detect language of audio"""
        try:
            # Convert to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(audio_bytes, self.stt_config.sample_rate)

            # Use Whisper for language detection
            if self.whisper_model.__class__.__module__.startswith('faster_whisper'):
                _, info = self.whisper_model.transcribe(audio_array, language=None)
                detected_language = info.language
            else:
                # OpenAI Whisper language detection
                audio_tensor = torch.from_numpy(audio_array).float()
                if self.device.type == "cuda":
                    audio_tensor = audio_tensor.cuda()

                result = self.whisper_model.transcribe(audio_tensor, language=None)
                detected_language = result.get("language", "unknown")

            return detected_language

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown"

    async def list_models(self) -> List[str]:
        """List available Whisper models"""
        return [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3",
            "distil-large-v2", "distil-large-v3"
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test basic transcription
            test_audio = np.random.randn(self.stt_config.sample_rate).astype(np.float32)
            test_request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(test_audio, self.stt_config.sample_rate),
                language="en"
            )

            result = await self.transcribe(test_request)

            return {
                "status": "healthy",
                "model": self.stt_config.model,
                "device": str(self.device),
                "vad_enabled": self.vad_model is not None,
                "sample_rate": self.stt_config.sample_rate,
                "test_transcription": len(result.text) > 0
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device),
                "vad_enabled": self.vad_model is not None
            }
