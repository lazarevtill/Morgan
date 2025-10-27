"""
STT Service implementation with Silero VAD integration
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import io
import wave
import numpy as np
import time
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
    model: str = "large-v3"
    device: str = "cuda"
    language: str = "auto"
    sample_rate: int = 16000
    chunk_size: int = 1024
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_duration: float = 0.5
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
        self.silence_threshold = self.stt_config.vad_min_silence_duration * self.stt_config.sample_rate

        # Streaming sessions
        self.streaming_sessions: Dict[str, Dict[str, Any]] = {}

        self.logger.info(f"STT Service initialized with device: {self.device}")

    async def start(self):
        """Start the STT service"""
        try:
            await self._load_whisper_model()
            if self.stt_config.vad_enabled:
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
                download_root="data/models",
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
                download_root="data/models"
            )

            self.logger.info("OpenAI Whisper model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load OpenAI Whisper model: {e}")
            raise AudioError(f"No STT model available: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            # Set torch hub cache directory to use mounted volume
            import os
            torch.hub.set_dir("data/models/torch_hub")

            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )

            # Unpack utilities
            # utils tuple order: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
            self.get_speech_timestamps = utils[0]
            self.collect_chunks = utils[4]

            self.vad_model = model
            self.vad_utils = utils
            self.vad_state = None  # not needed for batch API

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

                # Apply VAD if enabled and available
                if self.stt_config.vad_enabled and self.vad_model:
                    audio_array = await self._apply_vad(audio_array)

                # Transcribe with Whisper
                return await self._transcribe_with_whisper(audio_array, request)

            except Exception as e:
                self.logger.error(f"Error transcribing audio: {e}")
                raise AudioError(f"Transcription failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _apply_vad(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply Voice Activity Detection to audio"""
        try:
            # Convert numpy array to torch tensor as expected by silero-vad
            audio_tensor = torch.from_numpy(audio_array).float()

            # Detect speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.stt_config.sample_rate,
                threshold=self.stt_config.vad_threshold
            )

            if not speech_timestamps:
                self.logger.warning("No speech detected by VAD")
                return audio_array

            # Collect only speech chunks and return as numpy
            speech_audio = self.collect_chunks(speech_timestamps, audio_tensor)
            return speech_audio.numpy()

        except Exception as e:
            self.logger.error(f"VAD processing failed: {e}")
            # Fallback to original audio
            return audio_array

    async def _transcribe_with_whisper(self, audio_array: np.ndarray, request: STTRequest) -> STTResponse:
        """Transcribe audio using Whisper"""
        try:
            # Determine language (None means auto-detect)
            language = request.language or self.stt_config.language
            if language == "auto":
                language = None  # Whisper uses None for auto-detection

            # Prepare transcription options
            trans_options = {
                "language": language,
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

                # Convert segments generator to list
                segments_list = list(segments)

                # Extract text from segments
                text = " ".join([segment.text.strip() for segment in segments_list])

                # Get detected language from info
                detected_language = info.language if hasattr(info, 'language') else (language or "en")

                # Calculate confidence (average of all segments)
                confidence = sum(segment.avg_logprob for segment in segments_list) / len(segments_list) if segments_list else 0.0

                # Extract detailed segments
                detailed_segments = [{
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.avg_logprob
                } for segment in segments_list]

            else:
                # OpenAI Whisper API
                audio_tensor = torch.from_numpy(audio_array).float()
                if self.device.type == "cuda":
                    audio_tensor = audio_tensor.cuda()

                result = self.whisper_model.transcribe(audio_tensor, **trans_options)
                text = result["text"].strip()
                detected_language = result.get("language", language or "en")
                confidence = 0.0  # OpenAI Whisper doesn't provide confidence scores
                detailed_segments = result.get("segments", [])

            return STTResponse(
                text=text,
                language=detected_language,
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

    async def transcribe_chunk(self, audio_bytes: bytes, language: Optional[str] = None) -> STTResponse:
        """Transcribe a single audio chunk (for streaming)"""
        try:
            # Create STT request
            request = STTRequest(
                audio_data=audio_bytes,
                language=language or self.stt_config.language
            )

            # Transcribe without VAD for real-time processing
            return await self._transcribe_with_whisper_direct(audio_bytes, request)

        except Exception as e:
            self.logger.error(f"Chunk transcription failed: {e}")
            raise AudioError(f"Chunk transcription failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def transcribe_streaming(self, audio_chunks: List[bytes],
                                 language: Optional[str] = None) -> STTResponse:
        """Transcribe streaming audio chunks in real-time"""
        try:
            # Combine all audio chunks
            combined_audio = []

            for chunk in audio_chunks:
                audio_array = self.audio_utils.bytes_to_numpy(
                    chunk,
                    sample_rate=self.stt_config.sample_rate
                )
                combined_audio.append(audio_array)

            if not combined_audio:
                return STTResponse(
                    text="",
                    language=language or self.stt_config.language,
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio data", "streaming": True}
                )

            # Concatenate all chunks
            final_audio = np.concatenate(combined_audio)

            # Create request and transcribe
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(final_audio, self.stt_config.sample_rate),
                language=language or self.stt_config.language
            )

            return await self.transcribe(request)

        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")
            raise AudioError(f"Streaming transcription failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def start_audio_stream(self, session_id: str, language: str = "auto") -> Dict[str, Any]:
        """Start a new audio streaming session"""
        try:
            # Initialize streaming session
            self.streaming_sessions[session_id] = {
                "chunks": [],
                "start_time": time.time(),
                "language": language,
                "is_active": True
            }

            self.logger.info(f"Started audio streaming session: {session_id}")
            return {
                "session_id": session_id,
                "status": "active",
                "language": language,
                "sample_rate": self.stt_config.sample_rate,
                "chunk_size": self.stt_config.chunk_size
            }

        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            raise AudioError(f"Failed to start audio stream: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def add_audio_chunk(self, session_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """Add audio chunk to streaming session"""
        try:
            if session_id not in self.streaming_sessions:
                raise AudioError(f"Session {session_id} not found", ErrorCode.AUDIO_PROCESSING_ERROR)

            session = self.streaming_sessions[session_id]
            if not session["is_active"]:
                raise AudioError(f"Session {session_id} is not active", ErrorCode.AUDIO_PROCESSING_ERROR)

            # Add chunk to session
            session["chunks"].append(audio_bytes)

            # Check if we have enough audio for transcription (at least 1 second)
            total_samples = sum(len(self.audio_utils.bytes_to_numpy(chunk, self.stt_config.sample_rate))
                               for chunk in session["chunks"])

            if total_samples >= self.stt_config.sample_rate:  # At least 1 second
                # Transcribe current buffer
                result = await self.transcribe_streaming(session["chunks"], session["language"])

                # Clear processed chunks (keep last 0.5 seconds for context)
                if len(session["chunks"]) > 2:
                    session["chunks"] = session["chunks"][-1:]

                return {
                    "session_id": session_id,
                    "transcription": result.text,
                    "confidence": result.confidence,
                    "is_final": False,
                    "duration": result.duration,
                    "metadata": result.metadata
                }
            else:
                return {
                    "session_id": session_id,
                    "transcription": "",
                    "confidence": 0.0,
                    "is_final": False,
                    "duration": total_samples / self.stt_config.sample_rate,
                    "metadata": {"buffering": True}
                }

        except Exception as e:
            self.logger.error(f"Failed to add audio chunk: {e}")
            raise AudioError(f"Failed to add audio chunk: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def end_audio_stream(self, session_id: str) -> STTResponse:
        """End audio streaming session and return final transcription"""
        try:
            if session_id not in self.streaming_sessions:
                raise AudioError(f"Session {session_id} not found", ErrorCode.AUDIO_PROCESSING_ERROR)

            session = self.streaming_sessions[session_id]
            session["is_active"] = False

            # Transcribe all remaining chunks
            if session["chunks"]:
                result = await self.transcribe_streaming(session["chunks"], session["language"])
            else:
                result = STTResponse(
                    text="",
                    language=session["language"],
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio received"}
                )

            # Clean up session
            del self.streaming_sessions[session_id]

            self.logger.info(f"Ended audio streaming session: {session_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to end audio stream: {e}")
            raise AudioError(f"Failed to end audio stream: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)


    async def _transcribe_with_whisper_direct(self, audio_bytes: bytes, request: STTRequest) -> STTResponse:
        """Direct transcription without VAD for streaming"""
        try:
            # Convert audio bytes to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes,
                sample_rate=self.stt_config.sample_rate
            )

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
                    "vad_enabled": False,  # Direct transcription
                    "device": str(self.device)
                }
            )

        except Exception as e:
            self.logger.error(f"Direct Whisper transcription failed: {e}")
            raise

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

    async def process_realtime_audio(self, audio_bytes: bytes, language: str = "auto") -> Dict[str, Any]:
        """Process audio with real-time VAD and transcription"""
        try:
            # Convert to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(audio_bytes, self.stt_config.sample_rate)

            # Apply real-time VAD if enabled
            if self.stt_config.vad_enabled and self.vad_model:
                processed_audio = await self._apply_realtime_vad(audio_array)

                # If no speech detected, return empty result
                if len(processed_audio) == 0:
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "vad_result": "no_speech",
                        "segments": []
                    }
                audio_array = processed_audio

            # Transcribe the processed audio
            request = STTRequest(audio_data=audio_bytes, language=language)
            response = await self._transcribe_with_whisper_direct(audio_bytes, request)

            return {
                "text": response.text,
                "confidence": response.confidence,
                "language": response.language,
                "duration": response.duration,
                "segments": response.segments,
                "vad_result": "speech_detected" if self.vad_model else "vad_disabled"
            }

        except Exception as e:
            self.logger.error(f"Real-time audio processing failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "segments": []
            }

    async def _apply_realtime_vad(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply real-time VAD processing to audio stream"""
        try:
            # Convert to tensor for VAD
            audio_tensor = torch.from_numpy(audio_array).float()

            # Use collect_chunks for real-time processing
            if hasattr(self, 'get_speech_timestamps') and hasattr(self, 'collect_chunks'):
                # Detect speech segments
                speech_timestamps = self.get_speech_timestamps(
                    audio_tensor,
                    self.vad_model,
                    sampling_rate=self.stt_config.sample_rate,
                    threshold=self.stt_config.vad_threshold
                )

                if speech_timestamps:
                    # Collect only speech chunks
                    speech_audio = self.collect_chunks(speech_timestamps, audio_tensor)
                    return speech_audio.numpy()
                else:
                    # No speech detected
                    return np.array([], dtype=np.float32)

            # Fallback to simple VAD model call
            elif self.vad_model:
                # Process in smaller chunks for real-time
                chunk_size = int(self.stt_config.sample_rate * 0.5)  # 0.5 second chunks
                speech_chunks = []

                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]

                    if len(chunk) < chunk_size:
                        # Pad last chunk
                        padding = np.zeros(chunk_size - len(chunk))
                        chunk = np.concatenate([chunk, padding])

                    # Convert chunk to tensor
                    chunk_tensor = torch.from_numpy(chunk).float()

                    # Run VAD on chunk
                    confidence = self.vad_model(chunk_tensor, self.stt_config.sample_rate)

                    if confidence > self.stt_config.vad_threshold:
                        speech_chunks.append(chunk)

                if speech_chunks:
                    return np.concatenate(speech_chunks)
                else:
                    return np.array([], dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Real-time VAD processing failed: {e}")
            # Return original audio if VAD fails
            return audio_array

        return audio_array

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
            # Lightweight checks only to avoid heavy GPU inference here
            whisper_loaded = self.whisper_model is not None

            # Optional: quick VAD run on small silent buffer (CPU-friendly)
            if self.vad_model is not None and hasattr(self, 'get_speech_timestamps'):
                silent = np.zeros(int(self.stt_config.sample_rate * 0.25), dtype=np.float32)
                _ = self.get_speech_timestamps(
                    torch.from_numpy(silent).float(),
                    self.vad_model,
                    sampling_rate=self.stt_config.sample_rate,
                    threshold=self.stt_config.vad_threshold
                )

            return {
                "status": "healthy" if whisper_loaded else "degraded",
                "model": self.stt_config.model,
                "device": str(self.device),
                "vad_enabled": self.vad_model is not None,
                "sample_rate": self.stt_config.sample_rate,
                "test_transcription": False
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device),
                "vad_enabled": self.vad_model is not None
            }

    async def list_active_sessions(self) -> Dict[str, Any]:
        """List active streaming sessions"""
        try:
            sessions = list(self.streaming_sessions.keys())
            return {
                "active_sessions": sessions,
                "count": len(sessions)
            }
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return {
                "active_sessions": [],
                "count": 0,
                "error": str(e)
            }
