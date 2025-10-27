"""
TTS Service implementation for Morgan AI Assistant
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
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
    model: str = "kokoro"
    device: str = "cuda"
    language: str = "en-us"
    voice: str = "af_heart"
    speed: float = 1.0
    output_format: str = "wav"
    sample_rate: int = 22050
    log_level: str = "INFO"


class TTSService:
    """High-performance TTS Service"""

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
        self.voice_presets = {}
        self.available_voices = []

        self.logger.info(f"TTS Service initialized with device: {self.device}")

    async def start(self):
        """Start the TTS service"""
        try:
            await self._load_model()
            await self._load_voices()
            self.logger.info("TTS Service started successfully")
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

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("TTS Service stopped")

    async def _load_model(self):
        """Load TTS model"""
        try:
            if self.tts_config.model == "kokoro":
                await self._load_kokoro_model()
            elif self.tts_config.model.startswith("tts-"):
                await self._load_tts_model()
            else:
                # Fallback to pyttsx3
                await self._load_pyttsx3_engine()

        except Exception as e:
            self.logger.error(f"Failed to load model {self.tts_config.model}: {e}")
            # Fallback to pyttsx3
            await self._load_pyttsx3_engine()

    async def _load_kokoro_model(self):
        """Load Kokoro TTS model"""
        try:
            # Import here to avoid dependency issues if not available
            from kokoro import KModel, KPipeline

            # Create Kokoro model and pipeline
            self.kokoro_model = KModel()
            self.kokoro_pipeline = KPipeline(lang_code='a', model=self.kokoro_model)  # 'a' for American English
            self.logger.info("Kokoro model loaded successfully")

        except ImportError:
            self.logger.warning("Kokoro not available, falling back to pyttsx3")
            await self._load_pyttsx3_engine()
        except Exception as e:
            self.logger.error(f"Failed to load Kokoro model: {e}")
            await self._load_pyttsx3_engine()

    async def _load_tts_model(self):
        """Load Coqui TTS model"""
        try:
            from TTS.api import TTS

            # Set TTS cache directory to use mounted volume
            import os
            os.environ["TTS_CACHE_DIR"] = "data/models/tts"

            # Map model names to TTS models
            model_map = {
                "tts-1": "tts_models/en/ljspeech/tacotron2-DDC_ph",
                "tts-2": "tts_models/en/ljspeech/tacotron2-DDC",
                "tts-3": "tts_models/en/ljspeech/neural_hmm"
            }

            model_name = model_map.get(self.tts_config.model, "tts_models/en/ljspeech/tacotron2-DDC_ph")

            self.logger.info(f"Loading TTS model: {model_name}")

            self.model = TTS(model_name, progress_bar=False)
            self.logger.info("TTS model loaded successfully")

        except ImportError:
            self.logger.warning("Coqui TTS not available, falling back to pyttsx3")
            await self._load_pyttsx3_engine()
        except Exception as e:
            self.logger.error(f"Failed to load TTS model: {e}")
            await self._load_pyttsx3_engine()

    async def _load_pyttsx3_engine(self):
        """Load pyttsx3 engine as fallback"""
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

    async def _load_voices(self):
        """Load available voices"""
        try:
            if self.tts_config.model == "kokoro":
                # Kokoro voice presets
                self.voice_presets = {
                    "af_heart": {"lang": "en-us", "speed": 1.0},
                    "am_michael": {"lang": "en-us", "speed": 1.0},
                    "bf_emma": {"lang": "en-gb", "speed": 1.0},
                    "bm_george": {"lang": "en-gb", "speed": 1.0},
                }
            elif hasattr(self.model, 'get_voices'):
                # TTS model voices
                voices = self.model.get_voices()
                for voice in voices:
                    self.voice_presets[voice.name] = {"lang": voice.language}

            self.available_voices = list(self.voice_presets.keys())
            self.logger.info(f"Loaded {len(self.available_voices)} voices")

        except Exception as e:
            self.logger.error(f"Failed to load voices: {e}")

    async def generate_speech(self, request: TTSRequest) -> TTSResponse:
        """Generate speech from text"""
        with Timer(self.logger, f"TTS generation for text length {len(request.text)}"):
            try:
                # Select voice
                voice = request.voice or self.tts_config.voice

                if self.tts_config.model == "kokoro":
                    return await self._generate_kokoro_speech(request, voice)
                elif hasattr(self.model, 'tts'):
                    return await self._generate_tts_speech(request, voice)
                else:
                    return await self._generate_pyttsx3_speech(request, voice)

            except Exception as e:
                self.logger.error(f"Error generating speech: {e}")
                raise AudioError(f"Speech generation failed: {e}", ErrorCode.AUDIO_PROCESSING_ERROR)

    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text for TTS to handle special characters that cause silence or issues.

        Kokoro and other TTS engines often have trouble with certain punctuation:
        - Hyphens/dashes can cause silence
        - Colons can cause pauses or silence
        - Ellipsis can cause long pauses
        - Multiple punctuation marks can confuse the model
        """
        import re

        # Replace hyphens/dashes with natural pauses (commas)
        text = text.replace(' - ', ', ')     # Spaced hyphen
        text = text.replace(' – ', ', ')     # En dash
        text = text.replace(' — ', ', ')     # Em dash
        text = text.replace('—', ', ')       # Em dash without spaces

        # Replace standalone dashes at start/end of phrases
        text = text.replace('- ', '').replace(' -', '')

        # Handle colons - replace with comma for natural pause
        # But keep colons in time formats (e.g., "3:00")
        text = re.sub(r'(?<!\d):(?!\d)', ',', text)

        # Handle ellipsis and multiple periods
        text = text.replace('...', '. ')
        text = text.replace('..', '. ')

        # Handle multiple exclamation/question marks
        text = re.sub(r'!+', '!', text)  # Multiple ! to single !
        text = re.sub(r'\?+', '?', text)  # Multiple ? to single ?

        # Handle semicolons (replace with comma)
        text = text.replace(';', ',')

        # Remove quotes that might cause issues
        text = text.replace('"', '').replace("'", '')

        # Handle parentheses - remove them but keep content
        text = text.replace('(', ', ').replace(')', ', ')
        text = text.replace('[', ', ').replace(']', ', ')
        text = text.replace('{', ', ').replace('}', ', ')

        # Remove multiple commas
        text = re.sub(r',+', ',', text)

        # Remove comma before period
        text = re.sub(r',\s*\.', '.', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Clean up spaces around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1 ', text)  # Ensure single space after punctuation

        # Trim leading/trailing whitespace and punctuation
        text = text.strip(' ,.!?')

        return text

    async def _generate_kokoro_speech(self, request: TTSRequest, voice: str) -> TTSResponse:
        """Generate speech using Kokoro"""
        try:
            # Preprocess text to handle special characters
            processed_text = self._preprocess_text_for_tts(request.text)

            if not processed_text:
                # If preprocessing resulted in empty text, use a default message
                processed_text = "No text to speak"

            self.logger.debug(f"Original text: {request.text[:100]}...")
            self.logger.debug(f"Processed text: {processed_text[:100]}...")

            # Generate audio using KPipeline
            results = list(self.kokoro_pipeline(
                text=processed_text,
                voice=voice,
                speed=request.speed or self.tts_config.speed
            ))

            # Get the first result (should be only one for single text input)
            if results:
                result = results[0]
                audio_data = result.output.audio  # Get audio tensor
                sample_rate = 24000  # Kokoro typically uses 24kHz

                # Convert tensor to numpy array
                audio_np = audio_data.cpu().numpy() if hasattr(audio_data, 'cpu') else audio_data

                # Convert to bytes
                audio_bytes = self._numpy_to_wav_bytes(audio_np, sample_rate)

                return TTSResponse(
                    audio_data=audio_bytes,
                    format=self.tts_config.output_format,
                    sample_rate=sample_rate,
                    duration=len(audio_np) / sample_rate,
                    metadata={"model": "kokoro", "voice": voice, "original_text": request.text, "processed_text": processed_text}
                )
            else:
                raise AudioError("No audio generated from Kokoro", ErrorCode.AUDIO_PROCESSING_ERROR)

        except Exception as e:
            self.logger.error(f"Kokoro generation failed: {e}", exc_info=True)
            raise

    async def _generate_tts_speech(self, request: TTSRequest, voice: str) -> TTSResponse:
        """Generate speech using Coqui TTS"""
        try:
            # Generate audio file
            temp_file = f"/tmp/tts_output_{id(request)}.wav"
            self.model.tts_to_file(
                text=request.text,
                file_path=temp_file,
                speed=request.speed or self.tts_config.speed
            )

            # Load audio data
            audio_data, sample_rate = torchaudio.load(temp_file)

            # Convert to numpy and then bytes
            audio_numpy = audio_data.numpy().flatten()
            audio_bytes = self._numpy_to_wav_bytes(audio_numpy, sample_rate)

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=sample_rate,
                duration=len(audio_numpy) / sample_rate,
                metadata={"model": "tts", "voice": voice}
            )

        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            raise

    async def _generate_pyttsx3_speech(self, request: TTSRequest, voice: str) -> TTSResponse:
        """Generate speech using pyttsx3"""
        try:
            # Generate audio to file first
            temp_file = f"/tmp/pyttsx3_output_{id(request)}.wav"
            self.model.save_to_file(request.text, temp_file)
            self.model.runAndWait()

            # Load the generated file
            audio_data, sample_rate = torchaudio.load(temp_file)

            # Convert to numpy and then bytes
            audio_numpy = audio_data.numpy().flatten()
            audio_bytes = self._numpy_to_wav_bytes(audio_numpy, sample_rate)

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            return TTSResponse(
                audio_data=audio_bytes,
                format=self.tts_config.output_format,
                sample_rate=sample_rate,
                duration=len(audio_numpy) / sample_rate,
                metadata={"model": "pyttsx3", "voice": voice}
            )

        except Exception as e:
            self.logger.error(f"pyttsx3 generation failed: {e}")
            raise

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes"""
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data

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
        return {
            "voices": self.available_voices,
            "presets": self.voice_presets,
            "current_voice": self.tts_config.voice,
            "current_model": self.tts_config.model
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test basic functionality with a valid voice
            test_response = await self.generate_speech(TTSRequest(text="test", voice=self.tts_config.voice))

            return {
                "status": "healthy",
                "model": self.tts_config.model,
                "device": str(self.device),
                "available_voices": len(self.available_voices),
                "current_voice": self.tts_config.voice
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device)
            }
