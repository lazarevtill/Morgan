"""
TTS Service implementation for Morgan AI Assistant using csm-streaming
"""

import asyncio
import io
import logging
import os
import wave
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel

# Enable CUDA memory management optimizations but suppress compile errors
# This allows torch.compile to work but gracefully falls back on errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress torch.compile errors and fall back to eager mode gracefully
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from shared.config.base import ServiceConfig
from shared.models.base import ProcessingResult, TTSRequest, TTSResponse
from shared.utils.exceptions import (
    AudioException,
    AudioProcessingError,
    ErrorCategory,
    MorganException,
)
from shared.utils.logging import Timer, setup_logging

logger = logging.getLogger(__name__)


class TTSConfig(BaseModel):
    """TTS service configuration"""

    host: str = "0.0.0.0"
    port: int = 8002
    model: str = "csm"
    device: str = "cuda"
    language: str = "en"
    voice: str = "default"
    speed: float = 1.0
    output_format: str = "wav"
    sample_rate: int = 24000
    streaming_enabled: bool = True
    chunk_size: int = 512
    buffer_size: int = 2048
    log_level: str = "INFO"


class TTSService:
    """High-performance TTS Service using csm-streaming"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("tts")
        self.logger = setup_logging(
            "tts_service", self.config.get("log_level", "INFO"), "logs/tts_service.log"
        )

        # Load configuration
        self.tts_config = TTSConfig(**self.config.all())

        # Require CUDA for TTS service - no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for TTS service but not available. "
                "Please ensure you have a CUDA-compatible GPU and drivers installed."
            )

        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")

        # Model management (CSM only)
        self.device = torch.device(self.tts_config.device)
        self.csm_model = None
        self.csm_processor = None
        self.csm_sample_rate = 24000

        # Streaming state
        self.streaming_sessions = {}

        # Performance optimizations for real-time streaming
        self.stream_chunk_size = self.tts_config.chunk_size
        self.stream_buffer_size = self.tts_config.buffer_size

        self.logger.info(
            f"TTS Service initialized with csm-streaming on device: {self.device}"
        )

    async def start(self):
        """Start the TTS service"""
        try:
            await self._load_model()
            self.logger.info("TTS Service started successfully with csm-streaming")
        except Exception as e:
            self.logger.error(f"Failed to start TTS service: {e}")
            raise

    async def stop(self):
        """Stop the TTS service"""
        self.logger.info("TTS Service stopping...")

        # Clean up CSM models
        if hasattr(self, "csm_model") and self.csm_model:
            if hasattr(self.csm_model, "cpu"):
                self.csm_model.cpu()
            del self.csm_model
            self.csm_model = None

        if hasattr(self, "csm_processor"):
            del self.csm_processor
            self.csm_processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("TTS Service stopped")

    async def _load_model(self):
        """Load CSM streaming model (only option)"""
        self.logger.info("Loading CSM streaming model...")
        await self._load_csm_model()
        self.model_type = "csm"
        self.logger.info("CSM streaming model loaded successfully")

    async def _load_csm_model(self):
        """Load CSM streaming model using csm-streaming-tf Generator"""
        try:
            self.logger.info("Loading CSM-1B model using csm-streaming-tf Generator...")

            # Check if required modules are available
            try:
                from transformers import AutoProcessor, CsmForConditionalGeneration

                self.logger.info("Transformers CSM modules imported successfully")
            except ImportError as ie:
                self.logger.error(f"Failed to import CSM transformers: {ie}")
                raise

            # Get HuggingFace token from environment
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                self.logger.warning(
                    "HF_TOKEN not set, attempting to load without authentication"
                )

            # Model ID - try the correct one
            model_id = "sesame/csm-1b"  # This should be the correct model ID

            self.logger.info(f"Loading model from: {model_id}")

            # Load CSM model directly (not using the generator.py wrapper for now)
            try:
                self.csm_model = CsmForConditionalGeneration.from_pretrained(
                    model_id, token=hf_token, cache_dir="data/models/tts"
                ).to(self.device)

                self.csm_processor = AutoProcessor.from_pretrained(
                    model_id, token=hf_token, cache_dir="data/models/tts"
                )

                # Set generation config
                self.csm_model.generation_config.max_length = 250
                self.csm_model.generation_config.max_new_tokens = None
                self.csm_model.generation_config.cache_implementation = "static"
                self.csm_model.depth_decoder.generation_config.cache_implementation = (
                    "static"
                )

                self.csm_sample_rate = 24000

                self.logger.info(f"CSM model loaded successfully on {self.device}")
                self.logger.info(f"Model class: {type(self.csm_model).__name__}")
                self.logger.info(f"Sample rate: {self.csm_sample_rate}Hz")

                # Create generator instance
                from services.tts.generator import Generator

                self.csm_generator = Generator(
                    self.csm_model, self.csm_processor, self.device.type
                )

            except Exception as model_error:
                self.logger.error(f"Failed to load CSM model: {model_error}")
                # Try fallback model path
                try:
                    fallback_model_id = "eustlb/csm-1b"
                    self.logger.info(f"Trying fallback model: {fallback_model_id}")

                    self.csm_model = CsmForConditionalGeneration.from_pretrained(
                        fallback_model_id, token=hf_token, cache_dir="data/models/tts"
                    ).to(self.device)

                    self.csm_processor = AutoProcessor.from_pretrained(
                        fallback_model_id, token=hf_token, cache_dir="data/models/tts"
                    )

                    # Set generation config
                    self.csm_model.generation_config.max_length = 250
                    self.csm_model.generation_config.max_new_tokens = None
                    self.csm_model.generation_config.cache_implementation = "static"
                    self.csm_model.depth_decoder.generation_config.cache_implementation = (
                        "static"
                    )

                    self.csm_sample_rate = 24000

                    # Create generator instance
                    from services.tts.generator import Generator

                    self.csm_generator = Generator(
                        self.csm_model, self.csm_processor, self.device.type
                    )

                    self.logger.info("CSM model loaded successfully with fallback")

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback model loading also failed: {fallback_error}"
                    )
                    raise model_error

        except ImportError as ie:
            self.logger.error(
                f"Failed to import CSM model: {ie}. "
                "Please ensure transformers from GitHub main is installed."
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to load CSM model: {e}")
            raise

    async def _process_text(self, text: str) -> str:
        """Process text for CSM model (async wrapper for preprocessing)"""
        # CSM accepts text directly, but we apply basic preprocessing
        return self._preprocess_text_for_tts(text)

    async def _generate_csm_speech(self, request: TTSRequest, text: str) -> TTSResponse:
        """Generate speech using CSM with csm-streaming-tf Generator"""
        try:
            if not hasattr(self, "csm_generator"):
                raise AudioProcessingError(
                    "CSM generator not loaded", operation="speech_setup"
                )

            # Process text for CSM
            processed_text = await self._process_text(text)

            self.logger.debug(
                f"Generating speech with CSM: text_length={len(processed_text)}"
            )

            # Format text with speaker ID using conversation template format
            speaker_id = (
                request.voice if request.voice and request.voice.isdigit() else "0"
            )

            # Use apply_chat_template format (as per csm-streaming-tf)
            # First prepare the conversation
            conversation = [
                {
                    "role": speaker_id,
                    "content": [{"type": "text", "text": processed_text}],
                }
            ]

            # Apply chat template and tokenize
            inputs = self.csm_processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True
            )

            # Move to device (handle if it's a dict or tensor)
            if isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
            elif hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                self.logger.warning(f"Unexpected inputs type: {type(inputs)}")

            # Generate audio using csm-streaming-tf Generator
            audio_chunks = []
            for chunk in self.csm_generator.generate_stream(
                inputs, chunk_token_size=20
            ):
                audio_chunks.append(chunk.cpu().numpy())

            # Concatenate all audio chunks
            if len(audio_chunks) == 0:
                raise AudioProcessingError(
                    "No audio generated", operation="csm_synthesis"
                )

            audio_array = np.concatenate(audio_chunks)

            # Calculate duration
            duration = len(audio_array) / self.csm_sample_rate

            # Convert to WAV bytes
            audio_bytes = self._numpy_to_wav_bytes(audio_array, self.csm_sample_rate)

            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=self.csm_sample_rate,
                duration=duration,
                metadata={
                    "model": "csm-1b",
                    "voice": speaker_id,
                    "text_length": len(processed_text),
                },
            )

        except Exception as e:
            self.logger.error(f"CSM generation failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"CSM speech generation failed: {e}", operation="csm_generation"
            )

    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text for TTS synthesis.
        Handles special characters and normalizes text.
        """
        import re

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Handle repeated punctuation
        text = re.sub(r"!{3,}", "!", text)
        text = re.sub(r"\?{3,}", "?", text)
        text = re.sub(r"\.{4,}", "...", text)

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        # Trim whitespace
        text = text.strip()

        return text

    async def generate_speech(self, request: TTSRequest) -> TTSResponse:
        """Generate speech from text"""
        with Timer(self.logger, f"TTS generation for text length {len(request.text)}"):
            try:
                # Preprocess text
                processed_text = self._preprocess_text_for_tts(request.text)

                if not processed_text:
                    processed_text = "No text to speak"

                self.logger.debug(f"Original text: {request.text[:100]}...")
                self.logger.debug(f"Processed text: {processed_text[:100]}...")

                # Only use CSM streaming
                if not hasattr(self, "csm_model"):
                    raise AudioProcessingError(
                        "CSM model not loaded", operation="model_check"
                    )

                return await self._generate_csm_speech(request, processed_text)

            except Exception as e:
                self.logger.error(f"Error generating speech: {e}", exc_info=True)
                raise AudioProcessingError(
                    f"Speech generation failed: {e}", operation="generate_speech"
                )

    async def generate_speech_stream(
        self, request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate streaming speech from text using CSM streaming"""
        try:
            processed_text = self._preprocess_text_for_tts(request.text)

            if not processed_text:
                processed_text = "No text to speak"

            # Use CSM streaming generation directly
            async for audio_chunk in self._generate_csm_speech_stream(
                request, processed_text
            ):
                yield audio_chunk

        except Exception as e:
            self.logger.error(f"Error generating streaming speech: {e}")
            raise AudioProcessingError(
                f"Streaming speech generation failed: {e}",
                operation="stream_generation",
            )

    async def _generate_csm_speech_stream(
        self, request: TTSRequest, text: str
    ) -> AsyncGenerator[bytes, None]:
        """Generate streaming speech using CSM with real-time chunking"""
        try:
            if not hasattr(self, "csm_generator"):
                raise AudioProcessingError(
                    "CSM generator not loaded", operation="stream_setup"
                )

            # Process text for CSM
            processed_text = await self._process_text(text)

            self.logger.debug(
                f"Generating streaming speech with CSM: text_length={len(processed_text)}"
            )

            # Format text with speaker ID using conversation template format
            speaker_id = (
                request.voice if request.voice and request.voice.isdigit() else "0"
            )

            # Use apply_chat_template format (as per csm-streaming-tf)
            # First prepare the conversation
            conversation = [
                {
                    "role": speaker_id,
                    "content": [{"type": "text", "text": processed_text}],
                }
            ]

            # Apply chat template and tokenize
            inputs = self.csm_processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True
            )

            # Move to device (handle if it's a dict or tensor)
            if isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
            elif hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                self.logger.warning(f"Unexpected inputs type: {type(inputs)}")

            # Generate audio using csm-streaming-tf Generator in streaming mode
            chunk_count = 0
            chunk_token_size = max(
                10, min(50, self.stream_chunk_size // 20)
            )  # Adaptive chunk size
            async for chunk in self._stream_csm_audio_chunks(
                inputs, request, chunk_token_size
            ):
                # Convert chunk to WAV bytes
                wav_chunk = self._numpy_to_wav_bytes(
                    chunk.cpu().numpy(), self.csm_sample_rate
                )
                yield wav_chunk
                chunk_count += 1

                self.logger.debug(
                    f"Yielded chunk {chunk_count}, size: {len(wav_chunk)} bytes"
                )

        except Exception as e:
            self.logger.error(f"CSM streaming generation failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"CSM speech streaming failed: {e}", operation="csm_streaming"
            )

    async def _stream_csm_audio_chunks(
        self, inputs, request: TTSRequest, chunk_token_size: int = 20
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Stream audio chunks from CSM generator"""
        try:
            # Run CSM generation in a separate thread to avoid blocking
            def run_generation():
                """Run CSM generation in sync context"""
                try:
                    # Use the CSM generator's streaming method with optimized chunk size
                    for chunk in self.csm_generator.generate_stream(
                        inputs, chunk_token_size=chunk_token_size
                    ):
                        # chunk is a torch tensor with audio data
                        yield chunk
                except Exception as e:
                    self.logger.error(f"CSM generation error in thread: {e}")
                    return

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Execute generation in a thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit the generation task
                future = executor.submit(lambda: list(run_generation()))

                # Wait for chunks and yield them
                chunks = await loop.run_in_executor(executor, future.result)

                for chunk in chunks:
                    yield chunk

        except Exception as e:
            self.logger.error(f"Error in CSM audio chunk streaming: {e}")
            raise

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes"""
        # Normalize audio data
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    async def list_voices(self) -> Dict[str, Any]:
        """List available voices for CSM streaming"""
        voices = ["default", "speaker_0", "speaker_1"]  # CSM supported voices

        return {
            "voices": voices,
            "current_voice": self.tts_config.voice,
            "current_model": "csm-streaming",
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for CSM streaming service"""
        try:
            # Test basic functionality
            test_response = await self.generate_speech(
                TTSRequest(text="test", voice=self.tts_config.voice)
            )

            return {
                "status": "healthy",
                "model": "csm-streaming",
                "device": str(self.device),
                "sample_rate": (
                    self.csm_sample_rate
                    if hasattr(self, "csm_sample_rate")
                    else self.tts_config.sample_rate
                ),
                "streaming_enabled": self.tts_config.streaming_enabled,
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "device": str(self.device)}
