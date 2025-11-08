"""
STT Service implementation with Silero VAD integration
"""

import asyncio
import io
import logging
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel

from shared.config.base import ServiceConfig
from shared.models.base import AudioChunk, ProcessingResult, STTRequest, STTResponse
from shared.utils.audio import AudioUtils
from shared.utils.exceptions import (
    AudioException,
    AudioProcessingError,
    ErrorCategory,
    MorganException,
)
from shared.utils.logging import Timer, setup_logging

logger = logging.getLogger(__name__)


class STTConfig(BaseModel):
    """STT service configuration"""

    host: str = "0.0.0.0"
    port: int = 8003
    model: str = "distil-large-v3.5 "
    device: str = "cuda"
    language: str = "en"
    sample_rate: int = 16000
    chunk_size: int = 1024
    real_time_chunk_size: int = 512
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.1
    vad_max_speech_duration: float = 30.0
    vad_min_silence_duration: float = 0.5
    vad_speech_pad_ms: int = 200
    real_time_enabled: bool = True
    log_level: str = "INFO"


class STTService:
    """STT Service with faster-whisper built-in VAD integration"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("stt")
        self.logger = setup_logging(
            "stt_service", self.config.get("log_level", "INFO"), "logs/stt_service.log"
        )

        # Load configuration
        self.stt_config = STTConfig(**self.config.all())

        # Require CUDA for STT service - no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for STT service but not available. "
                "Please ensure you have a CUDA-compatible GPU and drivers installed."
            )

        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")

        # Model management
        self.whisper_model = None
        self.device = torch.device(self.stt_config.device)
        self.vad_available = False  # Will be set after whisper model loads

        # Audio processing
        self.audio_utils = AudioUtils()

        # VAD state (using faster-whisper built-in)

        # Real-time processing state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription_time = 0
        self.min_transcription_interval = 0.5  # Minimum 500ms between transcriptions

        # Streaming sessions
        self.streaming_sessions: Dict[str, Dict[str, Any]] = {}

        self.logger.info(f"STT Service initialized with device: {self.device}")

    async def start(self):
        """Start the STT service"""
        try:
            await self._load_whisper_model()
            if self.stt_config.vad_enabled:
                self._setup_vad_filter()
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

        self.vad_available = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("STT Service stopped")

    async def _load_whisper_model(self):
        """Load faster-whisper model with proper VAD support"""
        try:
            from faster_whisper import WhisperModel

            self.logger.info(f"Loading faster-whisper model: {self.stt_config.model}")

            # Load model with optimized settings for CUDA
            self.whisper_model = WhisperModel(
                self.stt_config.model,
                device=str(self.device),
                compute_type="float16" if self.device.type == "cuda" else "int8",
                cpu_threads=4 if self.device.type == "cpu" else 1,
                num_workers=1,
                download_root="data/models/whisper",
                local_files_only=False,  # Allow downloading if not present
            )

            self.logger.info(
                f"faster-whisper model loaded successfully on {self.device}"
            )
            self.logger.info(f"VAD enabled: {self.stt_config.vad_enabled}")

        except Exception as e:
            self.logger.error(f"Failed to load faster-whisper model: {e}")
            raise AudioProcessingError(
                f"Whisper model loading failed: {e}", operation="model_loading"
            )

    def _setup_vad_filter(self):
        """Setup VAD filter availability check"""
        try:
            # VAD is handled through transcribe parameters in faster-whisper
            self.vad_available = self.stt_config.vad_enabled
            self.logger.info(f"VAD configured: {self.vad_available}")

        except Exception as e:
            self.logger.warning(f"VAD setup failed: {e}")
            self.vad_available = False

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe audio to text"""
        with Timer(
            self.logger,
            f"STT transcription for audio length {len(request.audio_data)} bytes",
        ):
            try:
                # Convert audio bytes to numpy array
                audio_array = self.audio_utils.bytes_to_numpy(
                    request.audio_data, sample_rate=self.stt_config.sample_rate
                )

                # Apply VAD if enabled and available
                if self.stt_config.vad_enabled and self.vad_available:
                    audio_array = await self._apply_vad_filter(audio_array)

                # Transcribe with Whisper
                return await self._transcribe_with_whisper(audio_array, request)

            except Exception as e:
                self.logger.error(f"Error transcribing audio: {e}")
                raise AudioProcessingError(
                    f"Transcription failed: {e}", operation="transcription"
                )

    async def _apply_vad_filter(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply VAD filter using faster-whisper's built-in functionality"""
        try:
            # Since we're now using VAD parameters directly in the transcribe method,
            # we don't need to pre-process the audio here. The VAD filtering
            # happens during transcription, so we can return the original audio.
            # The actual VAD filtering is handled by faster-whisper internally.

            if self.stt_config.vad_enabled and self.vad_available:
                self.logger.debug(
                    "VAD will be applied during transcription by faster-whisper"
                )
                # Return original audio - VAD filtering happens in transcribe()
                return audio_array
            else:
                # Fallback to simple energy-based VAD if needed
                return await self._apply_simple_energy_vad(audio_array)

        except Exception as e:
            self.logger.error(f"VAD filter processing failed: {e}")
            # Return original audio as fallback
            return audio_array

    async def _apply_simple_energy_vad(self, audio_array: np.ndarray) -> np.ndarray:
        """Simple energy-based VAD as fallback"""
        try:
            # Calculate frame energy
            frame_length = int(self.stt_config.sample_rate * 0.025)  # 25ms frames
            hop_length = int(self.stt_config.sample_rate * 0.010)  # 10ms hop

            energy_threshold = self.stt_config.vad_threshold * np.max(
                np.abs(audio_array)
            )

            speech_frames = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i : i + frame_length]
                energy = np.sqrt(np.mean(frame**2))

                if energy > energy_threshold:
                    speech_frames.extend(
                        range(i, min(i + frame_length, len(audio_array)))
                    )

            if speech_frames:
                # Get unique indices and create continuous speech segment
                speech_indices = sorted(list(set(speech_frames)))
                start_idx = speech_indices[0]
                end_idx = speech_indices[-1]

                # Add some padding
                padding = int(self.stt_config.sample_rate * 0.1)  # 100ms padding
                start_idx = max(0, start_idx - padding)
                end_idx = min(len(audio_array), end_idx + padding)

                self.logger.debug(
                    f"Energy VAD detected speech from {start_idx} to {end_idx}"
                )
                return audio_array[start_idx:end_idx]
            else:
                self.logger.debug("Energy VAD: No speech detected")
                return np.array([], dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Simple energy VAD failed: {e}")
            return audio_array

    async def _transcribe_with_whisper(
        self, audio_array: np.ndarray, request: STTRequest
    ) -> STTResponse:
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
                "without_timestamps": True,
            }

            if self.whisper_model.__class__.__module__.startswith("faster_whisper"):
                # Faster Whisper API with VAD integration
                trans_options_with_vad = trans_options.copy()

                # Add VAD parameters for real-time processing
                if self.stt_config.vad_enabled:
                    trans_options_with_vad.update(
                        {
                            "vad_filter": True,
                            "vad_parameters": {
                                "threshold": self.stt_config.vad_threshold,
                                "min_speech_duration_ms": int(
                                    self.stt_config.vad_min_speech_duration * 1000
                                ),
                                "max_speech_duration_s": self.stt_config.vad_max_speech_duration,
                                "min_silence_duration_ms": int(
                                    self.stt_config.vad_min_silence_duration * 1000
                                ),
                                "speech_pad_ms": self.stt_config.vad_speech_pad_ms,
                            },
                        }
                    )
                    self.logger.debug(
                        f"Using VAD parameters: {trans_options_with_vad['vad_parameters']}"
                    )

                segments, info = self.whisper_model.transcribe(
                    audio_array, **trans_options_with_vad
                )

                # Convert segments generator to list
                segments_list = list(segments)

                # Extract text from segments
                text = " ".join([segment.text.strip() for segment in segments_list])

                # Get detected language from info
                detected_language = (
                    info.language if hasattr(info, "language") else (language or "en")
                )

                # Calculate confidence (convert negative log probabilities to 0-1 range)
                if segments_list:
                    # Convert avg_logprob (negative) to confidence (0-1)
                    # Using sigmoid-like transformation: confidence = 1 / (1 + exp(-avg_logprob))
                    # Clamp to ensure valid range [0, 1]
                    avg_logprob = sum(
                        segment.avg_logprob for segment in segments_list
                    ) / len(segments_list)
                    try:
                        confidence = 1.0 / (
                            1.0 + np.exp(-avg_logprob)
                        )  # Sigmoid transformation
                        confidence = max(
                            0.0, min(1.0, float(confidence))
                        )  # Clamp to [0, 1]
                    except (OverflowError, ValueError):
                        # Handle extreme values
                        confidence = 0.5 if avg_logprob > -2.0 else 0.1
                else:
                    confidence = 0.0

                # Extract detailed segments
                detailed_segments = [
                    {
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": max(
                            0.0,
                            min(1.0, float(1.0 / (1.0 + np.exp(-segment.avg_logprob)))),
                        ),  # Clamp to [0, 1]
                    }
                    for segment in segments_list
                ]

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
                    "vad_enabled": self.vad_available,
                    "vad_type": (
                        "faster_whisper_builtin"
                        if self.vad_available
                        else "energy_based"
                    ),
                    "device": str(self.device),
                },
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
                    chunk.data, sample_rate=chunk.sample_rate
                )
                combined_audio.append(audio_array)
                total_duration += len(audio_array) / chunk.sample_rate

            if not combined_audio:
                return STTResponse(
                    text="",
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio data"},
                )

            # Concatenate audio
            final_audio = np.concatenate(combined_audio)

            # Create a simple STT request
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(
                    final_audio, self.stt_config.sample_rate
                ),
                language=self.stt_config.language,
            )

            return await self.transcribe(request)

        except Exception as e:
            self.logger.error(f"Stream transcription failed: {e}")
            raise AudioProcessingError(
                f"Stream transcription failed: {e}", operation="stream_transcription"
            )

    async def transcribe_chunk(
        self, audio_bytes: bytes, language: Optional[str] = None
    ) -> STTResponse:
        """Transcribe a single audio chunk (for streaming)"""
        try:
            # Convert audio bytes to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes, self.stt_config.sample_rate
            )

            # Apply real-time VAD if enabled
            if self.stt_config.vad_enabled and self.vad_available:
                processed_audio = await self._apply_vad_filter(audio_array)

                # If no speech detected, return empty result
                if len(processed_audio) == 0:
                    return STTResponse(
                        text="",
                        language=language or self.stt_config.language,
                        confidence=0.0,
                        duration=len(audio_array) / self.stt_config.sample_rate,
                        segments=[],
                        metadata={
                            "model": self.stt_config.model,
                            "vad_enabled": self.vad_available,
                            "vad_type": (
                                "faster_whisper_builtin"
                                if self.vad_available
                                else "energy_based"
                            ),
                            "vad_result": "no_speech",
                            "device": str(self.device),
                            "real_time": True,
                        },
                    )

                audio_array = processed_audio

            # Create STT request
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(
                    audio_array, self.stt_config.sample_rate
                ),
                language=language or self.stt_config.language,
            )

            # Transcribe with Whisper
            return await self._transcribe_with_whisper_direct(audio_bytes, request)

        except Exception as e:
            self.logger.error(f"Chunk transcription failed: {e}")
            raise AudioProcessingError(
                f"Chunk transcription failed: {e}", operation="chunk_transcription"
            )

    async def transcribe_streaming(
        self, audio_chunks: List[bytes], language: Optional[str] = None
    ) -> STTResponse:
        """Transcribe streaming audio chunks in real-time"""
        try:
            # Combine all audio chunks
            combined_audio = []

            for chunk in audio_chunks:
                audio_array = self.audio_utils.bytes_to_numpy(
                    chunk, sample_rate=self.stt_config.sample_rate
                )
                combined_audio.append(audio_array)

            if not combined_audio:
                return STTResponse(
                    text="",
                    language=language or self.stt_config.language,
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio data", "streaming": True},
                )

            # Concatenate all chunks
            final_audio = np.concatenate(combined_audio)

            # Create request and transcribe
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(
                    final_audio, self.stt_config.sample_rate
                ),
                language=language or self.stt_config.language,
            )

            return await self.transcribe(request)

        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")
            raise AudioProcessingError(
                f"Streaming transcription failed: {e}",
                operation="streaming_transcription",
            )

    async def start_audio_stream(
        self, session_id: str, language: str = "auto"
    ) -> Dict[str, Any]:
        """Start a new audio streaming session"""
        try:
            # Initialize streaming session
            self.streaming_sessions[session_id] = {
                "chunks": [],
                "start_time": time.time(),
                "language": language,
                "is_active": True,
            }

            self.logger.info(f"Started audio streaming session: {session_id}")
            return {
                "session_id": session_id,
                "status": "active",
                "language": language,
                "sample_rate": self.stt_config.sample_rate,
                "chunk_size": self.stt_config.chunk_size,
            }

        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            raise AudioProcessingError(
                f"Failed to start audio stream: {e}", operation="start_stream"
            )

    async def add_audio_chunk(
        self, session_id: str, audio_bytes: bytes
    ) -> Dict[str, Any]:
        """Add audio chunk to streaming session"""
        try:
            if session_id not in self.streaming_sessions:
                raise AudioProcessingError(
                    f"Session {session_id} not found", operation="add_chunk"
                )

            session = self.streaming_sessions[session_id]
            if not session["is_active"]:
                raise AudioProcessingError(
                    f"Session {session_id} is not active", operation="add_chunk"
                )

            # Add chunk to session
            session["chunks"].append(audio_bytes)

            # Check if we have enough audio for transcription (at least 1 second)
            total_samples = sum(
                len(self.audio_utils.bytes_to_numpy(chunk, self.stt_config.sample_rate))
                for chunk in session["chunks"]
            )

            if total_samples >= self.stt_config.sample_rate:  # At least 1 second
                # Transcribe current buffer
                result = await self.transcribe_streaming(
                    session["chunks"], session["language"]
                )

                # Clear processed chunks (keep last 0.5 seconds for context)
                if len(session["chunks"]) > 2:
                    session["chunks"] = session["chunks"][-1:]

                return {
                    "session_id": session_id,
                    "transcription": result.text,
                    "confidence": result.confidence,
                    "is_final": False,
                    "duration": result.duration,
                    "metadata": result.metadata,
                }
            else:
                return {
                    "session_id": session_id,
                    "transcription": "",
                    "confidence": 0.0,
                    "is_final": False,
                    "duration": total_samples / self.stt_config.sample_rate,
                    "metadata": {"buffering": True},
                }

        except Exception as e:
            self.logger.error(f"Failed to add audio chunk: {e}")
            raise AudioProcessingError(
                f"Failed to add audio chunk: {e}", operation="add_chunk"
            )

    async def end_audio_stream(self, session_id: str) -> STTResponse:
        """End audio streaming session and return final transcription"""
        try:
            if session_id not in self.streaming_sessions:
                raise AudioProcessingError(
                    f"Session {session_id} not found", operation="end_stream"
                )

            session = self.streaming_sessions[session_id]
            session["is_active"] = False

            # Transcribe all remaining chunks
            if session["chunks"]:
                result = await self.transcribe_streaming(
                    session["chunks"], session["language"]
                )
            else:
                result = STTResponse(
                    text="",
                    language=session["language"],
                    confidence=0.0,
                    duration=0.0,
                    metadata={"error": "No audio received"},
                )

            # Clean up session
            del self.streaming_sessions[session_id]

            self.logger.info(f"Ended audio streaming session: {session_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to end audio stream: {e}")
            raise AudioProcessingError(
                f"Failed to end audio stream: {e}", operation="end_stream"
            )

    async def process_realtime_chunk(
        self,
        audio_bytes: bytes,
        session_id: str = "default",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process audio chunk in real-time with VAD and transcription"""
        try:
            current_time = time.time()

            # Convert audio bytes to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes, self.stt_config.sample_rate
            )

            # Apply real-time VAD if enabled
            vad_result = "vad_disabled"
            if self.stt_config.vad_enabled and self.vad_available:
                processed_audio = await self._apply_vad_filter(audio_array)

                if len(processed_audio) == 0:
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "vad_result": "no_speech",
                        "segments": [],
                        "is_final": False,
                        "session_id": session_id,
                    }

                audio_array = processed_audio
                vad_result = "speech_detected"

            # Throttle transcription requests
            if (
                current_time - self.last_transcription_time
                < self.min_transcription_interval
            ):
                return {
                    "text": "",
                    "confidence": 0.0,
                    "vad_result": vad_result,
                    "segments": [],
                    "is_final": False,
                    "session_id": session_id,
                    "throttled": True,
                }

            # Transcribe the audio
            request = STTRequest(
                audio_data=self.audio_utils.numpy_to_bytes(
                    audio_array, self.stt_config.sample_rate
                ),
                language=language or self.stt_config.language,
            )

            response = await self._transcribe_with_whisper_direct(audio_bytes, request)
            self.last_transcription_time = current_time

            return {
                "text": response.text,
                "confidence": response.confidence,
                "language": response.language,
                "duration": response.duration,
                "segments": response.segments,
                "vad_result": vad_result,
                "is_final": response.confidence
                > 0.8,  # Consider high confidence as final
                "session_id": session_id,
                "metadata": {
                    **response.metadata,
                    "vad_type": (
                        "faster_whisper_builtin"
                        if self.vad_available
                        else "energy_based"
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Real-time chunk processing failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "segments": [],
                "is_final": False,
                "session_id": session_id,
            }

    async def _transcribe_with_whisper_direct(
        self, audio_bytes: bytes, request: STTRequest
    ) -> STTResponse:
        """Direct transcription without VAD for streaming"""
        try:
            # Convert audio bytes to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes, sample_rate=self.stt_config.sample_rate
            )

            # Prepare transcription options
            # Convert "auto" to None for faster-whisper compatibility
            language = request.language or self.stt_config.language
            if language == "auto":
                language = None

            trans_options = {
                "language": language,
                "temperature": request.temperature or 0.0,
                "initial_prompt": request.prompt,
                "suppress_tokens": [-1],  # Suppress timestamps
                "without_timestamps": True,
            }

            if self.whisper_model.__class__.__module__.startswith("faster_whisper"):
                # Faster Whisper API
                segments, info = self.whisper_model.transcribe(
                    audio_array, **trans_options
                )

                # Extract text from segments
                text = " ".join([segment.text.strip() for segment in segments])

                # Calculate confidence (convert negative log probabilities to 0-1 range)
                if segments:
                    # Convert avg_logprob (negative) to confidence (0-1)
                    # Using sigmoid-like transformation: confidence = 1 / (1 + exp(-avg_logprob))
                    # Clamp to ensure valid range [0, 1]
                    avg_logprob = sum(
                        segment.avg_logprob for segment in segments
                    ) / len(segments)
                    try:
                        confidence = 1.0 / (
                            1.0 + np.exp(-avg_logprob)
                        )  # Sigmoid transformation
                        confidence = max(
                            0.0, min(1.0, float(confidence))
                        )  # Clamp to [0, 1]
                    except (OverflowError, ValueError):
                        # Handle extreme values
                        confidence = 0.5 if avg_logprob > -2.0 else 0.1
                else:
                    confidence = 0.0

                # Extract detailed segments
                detailed_segments = [
                    {
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": max(
                            0.0,
                            min(1.0, float(1.0 / (1.0 + np.exp(-segment.avg_logprob)))),
                        ),  # Clamp to [0, 1]
                    }
                    for segment in segments
                ]

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
                    "vad_enabled": self.vad_available,
                    "vad_type": (
                        "faster_whisper_builtin"
                        if self.vad_available
                        else "energy_based"
                    ),
                    "device": str(self.device),
                },
            )

        except Exception as e:
            self.logger.error(f"Direct Whisper transcription failed: {e}")
            raise

    async def detect_language(self, audio_bytes: bytes) -> str:
        """Detect language of audio"""
        try:
            # Convert to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes, self.stt_config.sample_rate
            )

            # Use Whisper for language detection
            if self.whisper_model.__class__.__module__.startswith("faster_whisper"):
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

    async def process_realtime_audio(
        self, audio_bytes: bytes, language: str = "auto"
    ) -> Dict[str, Any]:
        """Process audio with real-time VAD and transcription"""
        try:
            # Convert to numpy array
            audio_array = self.audio_utils.bytes_to_numpy(
                audio_bytes, self.stt_config.sample_rate
            )

            # Apply real-time VAD if enabled
            if self.stt_config.vad_enabled and self.vad_available:
                processed_audio = await self._apply_vad_filter(audio_array)

                # If no speech detected, return empty result
                if len(processed_audio) == 0:
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "vad_result": "no_speech",
                        "segments": [],
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
                "vad_result": (
                    "speech_detected" if self.vad_available else "vad_disabled"
                ),
                "vad_type": (
                    "faster_whisper_builtin" if self.vad_available else "energy_based"
                ),
            }

        except Exception as e:
            self.logger.error(f"Real-time audio processing failed: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e), "segments": []}

    async def list_models(self) -> List[str]:
        """List available faster-whisper models"""
        return [
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
            "distil-large-v3",
            "large-v3-turbo",
            "turbo",
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Lightweight checks only to avoid heavy GPU inference here
            whisper_loaded = self.whisper_model is not None

            # VAD is now handled by faster-whisper internally, no separate VAD model needed

            return {
                "status": "healthy" if whisper_loaded else "degraded",
                "model": self.stt_config.model,
                "device": str(self.device),
                "vad_enabled": self.vad_available,
                "vad_type": (
                    "faster_whisper_builtin" if self.vad_available else "energy_based"
                ),
                "sample_rate": self.stt_config.sample_rate,
                "test_transcription": False,
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device),
                "vad_enabled": self.vad_available,
                "vad_type": (
                    "faster_whisper_builtin" if self.vad_available else "energy_based"
                ),
            }

    async def list_active_sessions(self) -> Dict[str, Any]:
        """List active streaming sessions"""
        try:
            sessions = list(self.streaming_sessions.keys())
            return {"active_sessions": sessions, "count": len(sessions)}
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return {"active_sessions": [], "count": 0, "error": str(e)}
