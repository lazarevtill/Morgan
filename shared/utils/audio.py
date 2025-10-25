"""
Audio processing utilities for Morgan AI Assistant
"""
import numpy as np
import wave
import io
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AudioUtils:
    """Audio processing utilities"""

    @staticmethod
    def bytes_to_numpy(audio_bytes: bytes, sample_rate: int = 16000,
                      channels: int = 1) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Try to read as WAV first
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

                # Convert to float32 and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Convert to mono if stereo
                if wav_file.getnchannels() == 2:
                    audio_float = audio_float.reshape((-1, 2)).mean(axis=1)

                # Resample if necessary
                if wav_file.getframerate() != sample_rate:
                    audio_float = AudioUtils._resample(audio_float, wav_file.getframerate(), sample_rate)

                return audio_float

        except Exception as e:
            logger.error(f"Failed to convert audio bytes to numpy: {e}")
            raise ValueError(f"Invalid audio format: {e}")

    @staticmethod
    def numpy_to_bytes(audio_array: np.ndarray, sample_rate: int = 16000,
                      format: str = "wav") -> bytes:
        """Convert numpy array to audio bytes"""
        if format.lower() != "wav":
            raise ValueError(f"Unsupported format: {format}")

        # Ensure audio is in valid range
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    @staticmethod
    def _resample(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Simple resampling using linear interpolation"""
        if original_rate == target_rate:
            return audio

        # Calculate resampling ratio
        ratio = target_rate / original_rate
        new_length = int(len(audio) * ratio)

        # Create new time array
        old_times = np.arange(len(audio)) / original_rate
        new_times = np.arange(new_length) / target_rate

        # Linear interpolation
        return np.interp(new_times, old_times, audio)

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        # Find first and last non-silent samples
        mask = np.abs(audio) > threshold

        if not np.any(mask):
            return np.array([])

        first_sample = np.argmax(mask)
        last_sample = len(mask) - np.argmax(mask[::-1]) - 1

        return audio[first_sample:last_sample + 1]

    @staticmethod
    def split_audio_chunks(audio: np.ndarray, chunk_size: int = 16000,
                          overlap: int = 0) -> list:
        """Split audio into chunks with optional overlap"""
        if len(audio) <= chunk_size:
            return [audio]

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)

            if end >= len(audio):
                break

            start = end - overlap

        return chunks

    @staticmethod
    def calculate_audio_features(audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Calculate basic audio features"""
        features = {
            "duration": len(audio) / sample_rate,
            "sample_rate": sample_rate,
            "samples": len(audio),
            "rms_energy": np.sqrt(np.mean(audio ** 2)),
            "zero_crossing_rate": np.mean(np.abs(np.diff(np.sign(audio)))),
            "max_amplitude": np.max(np.abs(audio))
        }

        return features


def validate_audio_data(audio_bytes: bytes, max_size: int = 50 * 1024 * 1024) -> bool:
    """Validate audio data"""
    if not audio_bytes:
        return False

    if len(audio_bytes) > max_size:
        logger.warning(f"Audio data too large: {len(audio_bytes)} bytes")
        return False

    # Basic WAV header validation
    if len(audio_bytes) < 44:  # Minimum WAV header size
        return False

    # Check for WAV signature
    if audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
        return False

    return True
