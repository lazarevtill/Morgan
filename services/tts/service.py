"""
TTS Service implementation for Morgan AI Assistant using csm-streaming
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import io
import wave
from pathlib import Path

import torch
import torchaudio
import numpy as np
from pydantic import BaseModel

from shared.config.base import ServiceConfig
from shared.models.base import TTSRequest, TTSResponse, ProcessingResult
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, AudioError, ErrorCode

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
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "tts_service",
            self.config.get("log_level", "INFO"),
            "logs/tts_service.log"
        )

        # Load configuration
        self.tts_config = TTSConfig(**self.config.all())

        # Model management
        self.model = None
        self.device = torch.device(self.tts_config.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.vocoder = None
        
        # Streaming state
        self.streaming_sessions = {}

        self.logger.info(f"TTS Service initialized with csm-streaming on device: {self.device}")

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

        # Clean up model
        if self.model:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None

        if self.vocoder:
            if hasattr(self.vocoder, 'cpu'):
                self.vocoder.cpu()
            del self.vocoder
            self.vocoder = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("TTS Service stopped")

    async def _load_model(self):
        """Load csm-streaming TTS model"""
        try:
            # Import csm-streaming
            try:
                from csm.tts import CSMTextToSpeech
                
                self.logger.info("Loading csm-streaming model...")
                
                # Initialize csm-streaming model
                self.model = CSMTextToSpeech(
                    device=str(self.device),
                    language=self.tts_config.language
                )
                
                self.logger.info("csm-streaming model loaded successfully")
                
            except ImportError as e:
                self.logger.error(f"csm-streaming not available: {e}")
                self.logger.info("Falling back to Coqui TTS")
                await self._load_fallback_tts()
                
        except Exception as e:
            self.logger.error(f"Failed to load csm-streaming model: {e}")
            await self._load_fallback_tts()

    async def _load_fallback_tts(self):
        """Load fallback TTS (Coqui TTS or pyttsx3)"""
        try:
            from TTS.api import TTS
            
            # Set TTS cache directory
            import os
            os.environ["TTS_CACHE_DIR"] = "data/models/tts"
            
            model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
            self.logger.info(f"Loading fallback TTS model: {model_name}")
            
            self.model = TTS(model_name, progress_bar=False, gpu=torch.cuda.is_available())
            self.model_type = "coqui_tts"
            self.logger.info("Coqui TTS model loaded as fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to load Coqui TTS: {e}")
            await self._load_pyttsx3_engine()

    async def _load_pyttsx3_engine(self):
        """Load pyttsx3 engine as last resort fallback"""
        try:
            import pyttsx3

            self.model = pyttsx3.init()
            self.model_type = "pyttsx3"

            # Configure voice
            voices = self.model.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.model.setProperty('voice', voice.id)
                        break

            self.model.setProperty('rate', 200)
            self.model.setProperty('volume', 0.9)

            self.logger.info("pyttsx3 engine loaded as fallback")

        except Exception as e:
            self.logger.error(f"Failed to load pyttsx3 engine: {e}")
            raise AudioError(f"No TTS engine available: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text for TTS synthesis.
        Handles special characters and normalizes text.
        """
        import re

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Handle repeated punctuation
        text = re.sub(r'!{3,}', '!', text)
        text = re.sub(r'\?{3,}', '?', text)
        text = re.sub(r'\.{4,}', '...', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
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

                # Check model type and generate accordingly
                model_type = getattr(self, 'model_type', 'csm')
                
                if model_type == 'pyttsx3':
                    return await self._generate_pyttsx3_speech(request, processed_text)
                elif model_type == 'coqui_tts':
                    return await self._generate_coqui_speech(request, processed_text)
                else:
                    return await self._generate_csm_speech(request, processed_text)

            except Exception as e:
                self.logger.error(f"Error generating speech: {e}", exc_info=True)
                raise AudioError(f"Speech generation failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    async def _generate_csm_speech(self, request: TTSRequest, text: str) -> TTSResponse:
        """Generate speech using csm-streaming"""
        try:
            # Generate audio using csm-streaming
            audio_tensor = await asyncio.to_thread(
                self.model.synthesize,
                text=text,
                speed=request.speed or self.tts_config.speed,
                voice=request.voice or self.tts_config.voice
            )
            
            # Convert tensor to numpy
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            # Flatten if needed
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            
            # Convert to WAV bytes
            audio_bytes = self._numpy_to_wav_bytes(audio_np, self.tts_config.sample_rate)
            
            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=self.tts_config.sample_rate,
                duration=len(audio_np) / self.tts_config.sample_rate,
                metadata={
                    "model": "csm-streaming",
                    "voice": request.voice or self.tts_config.voice,
                    "text_length": len(text),
                    "original_text": request.text[:100],
                    "processed_text": text[:100]
                }
            )

        except Exception as e:
            self.logger.error(f"csm-streaming generation failed: {e}", exc_info=True)
            raise

    async def _generate_coqui_speech(self, request: TTSRequest, text: str) -> TTSResponse:
        """Generate speech using Coqui TTS"""
        try:
            # Generate audio file
            temp_file = f"/tmp/tts_output_{id(request)}.wav"
            
            await asyncio.to_thread(
                self.model.tts_to_file,
                text=text,
                file_path=temp_file,
                speed=request.speed or self.tts_config.speed
            )

            # Load audio data
            audio_data, sample_rate = torchaudio.load(temp_file)

            # Convert to numpy and then bytes
            audio_numpy = audio_data.numpy().flatten()
            audio_bytes = self._numpy_to_wav_bytes(audio_numpy, int(sample_rate))

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=int(sample_rate),
                duration=len(audio_numpy) / sample_rate,
                metadata={"model": "coqui_tts", "voice": request.voice or "default"}
            )

        except Exception as e:
            self.logger.error(f"Coqui TTS generation failed: {e}")
            raise

    async def _generate_pyttsx3_speech(self, request: TTSRequest, text: str) -> TTSResponse:
        """Generate speech using pyttsx3"""
        try:
            # Generate audio to file first
            temp_file = f"/tmp/pyttsx3_output_{id(request)}.wav"
            
            await asyncio.to_thread(self.model.save_to_file, text, temp_file)
            await asyncio.to_thread(self.model.runAndWait)

            # Load the generated file
            audio_data, sample_rate = torchaudio.load(temp_file)

            # Convert to numpy and then bytes
            audio_numpy = audio_data.numpy().flatten()
            audio_bytes = self._numpy_to_wav_bytes(audio_numpy, int(sample_rate))

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=int(sample_rate),
                duration=len(audio_numpy) / sample_rate,
                metadata={"model": "pyttsx3", "voice": request.voice or "default"}
            )

        except Exception as e:
            self.logger.error(f"pyttsx3 generation failed: {e}")
            raise

    async def generate_speech_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """Generate streaming speech from text"""
        try:
            processed_text = self._preprocess_text_for_tts(request.text)
            
            if not processed_text:
                processed_text = "No text to speak"

            # For csm-streaming, generate chunks
            if hasattr(self.model, 'synthesize_stream'):
                async for audio_chunk in self.model.synthesize_stream(
                    text=processed_text,
                    speed=request.speed or self.tts_config.speed,
                    chunk_size=self.tts_config.chunk_size
                ):
                    # Convert chunk to bytes and yield
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy()
                    else:
                        audio_np = np.array(audio_chunk)
                    
                    chunk_bytes = audio_np.tobytes()
                    yield chunk_bytes
            else:
                # Fallback: generate full audio and chunk it
                response = await self.generate_speech(request)
                
                # Chunk the audio data
                chunk_size = self.tts_config.chunk_size * 2  # 16-bit samples
                audio_data = response.audio_data
                
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]

        except Exception as e:
            self.logger.error(f"Error generating streaming speech: {e}")
            raise AudioError(f"Streaming speech generation failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

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
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    async def list_voices(self) -> Dict[str, Any]:
        """List available voices"""
        voices = ["default"]  # csm-streaming default voice
        
        return {
            "voices": voices,
            "current_voice": self.tts_config.voice,
            "current_model": "csm-streaming" if not hasattr(self, 'model_type') else self.model_type
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test basic functionality
            test_response = await self.generate_speech(TTSRequest(
                text="test",
                voice=self.tts_config.voice
            ))

            return {
                "status": "healthy",
                "model": "csm-streaming" if not hasattr(self, 'model_type') else self.model_type,
                "device": str(self.device),
                "sample_rate": self.tts_config.sample_rate,
                "streaming_enabled": self.tts_config.streaming_enabled
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device)
            }
