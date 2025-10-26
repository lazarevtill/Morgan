"""
Audio processing utilities for Morgan AI Assistant
"""
import numpy as np
import wave
import io
import asyncio
import logging
from typing import Tuple, Optional, Union, AsyncGenerator, List, Dict, Any
import base64

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


class AudioCapture:
    """Audio capture utilities for device input"""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_recording = False

    @staticmethod
    def validate_audio_format(audio_bytes: bytes) -> Dict[str, Any]:
        """Validate audio format and return metadata"""
        # Try WAV first
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                return {
                    "valid": True,
                    "format": "wav",
                    "channels": wav_file.getnchannels(),
                    "sample_rate": wav_file.getframerate(),
                    "sample_width": wav_file.getsampwidth(),
                    "frames": wav_file.getnframes(),
                    "duration": wav_file.getnframes() / wav_file.getframerate()
                }
        except Exception as e:
            # Check if it's WebM format
            try:
                # WebM files start with specific bytes
                if len(audio_bytes) > 4 and audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
                    return {
                        "valid": True,
                        "format": "webm",
                        "channels": 1,  # Assume mono for WebM
                        "sample_rate": 16000,  # Assume 16kHz
                        "sample_width": 2,  # Assume 16-bit
                        "frames": len(audio_bytes) // 2,  # Rough estimate
                        "duration": (len(audio_bytes) // 2) / 16000  # Rough estimate
                    }
                else:
                    return {
                        "valid": False,
                        "error": f"Unknown audio format. Expected WAV or WebM. Error: {str(e)}"
                    }
            except Exception as webm_error:
                return {
                    "valid": False,
                    "error": f"Invalid audio format: {str(e)}"
                }

    def encode_audio_chunk(self, audio_data: bytes) -> str:
        """Encode audio bytes to base64 string"""
        return base64.b64encode(audio_data).decode('utf-8')

    def decode_audio_chunk(self, encoded_data: str) -> bytes:
        """Decode base64 string to audio bytes"""
        return base64.b64decode(encoded_data)

    def create_wav_header(self, data_size: int, sample_rate: int = 16000,
                         channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """Create WAV file header"""
        header = bytearray()

        # RIFF chunk
        header.extend(b'RIFF')
        header.extend((data_size + 36).to_bytes(4, 'little'))  # File size - 8
        header.extend(b'WAVE')

        # Format chunk
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))  # Chunk size
        header.extend((1).to_bytes(2, 'little'))   # PCM format
        header.extend((channels).to_bytes(2, 'little'))
        header.extend((sample_rate).to_bytes(4, 'little'))
        header.extend((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
        header.extend((channels * bits_per_sample // 8).to_bytes(2, 'little'))
        header.extend((bits_per_sample).to_bytes(2, 'little'))

        # Data chunk
        header.extend(b'data')
        header.extend((data_size).to_bytes(4, 'little'))

        return bytes(header)

    def audio_bytes_to_chunks(self, audio_bytes: bytes, chunk_duration_ms: int = 100) -> List[bytes]:
        """Split audio bytes into chunks for streaming"""
        # Calculate chunk size in bytes (16-bit mono)
        chunk_size_bytes = int(self.sample_rate * (chunk_duration_ms / 1000) * 2)

        chunks = []
        for i in range(0, len(audio_bytes), chunk_size_bytes):
            chunk = audio_bytes[i:i + chunk_size_bytes]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

        return chunks

    async def stream_audio_chunks(self, audio_generator: AsyncGenerator[bytes, None],
                                 chunk_duration_ms: int = 100) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream audio chunks with metadata"""
        async for audio_bytes in audio_generator:
            chunks = self.audio_bytes_to_chunks(audio_bytes, chunk_duration_ms)

            for i, chunk in enumerate(chunks):
                yield {
                    "audio_data": self.encode_audio_chunk(chunk),
                    "chunk_index": i,
                    "timestamp": asyncio.get_event_loop().time(),
                    "sample_rate": self.sample_rate,
                    "format": "wav"
                }

    @staticmethod
    def estimate_audio_duration(audio_bytes: bytes) -> float:
        """Estimate duration of audio in seconds"""
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / rate
        except Exception:
            # Fallback estimation assuming 16kHz 16-bit mono
            return len(audio_bytes) / (16000 * 2)


class DeviceAudioCapture:
    """Device-specific audio capture utilities"""

    @staticmethod
    def get_supported_audio_formats() -> List[str]:
        """Get list of supported audio formats"""
        return [
            "wav", "mp3", "ogg", "flac", "aac",
            "webm", "m4a", "wma", "aiff"
        ]

    @staticmethod
    def get_recommended_settings(device_type: str = "microphone") -> Dict[str, Any]:
        """Get recommended audio settings for different device types"""
        settings = {
            "microphone": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "format": "wav",
                "echo_cancellation": True,
                "noise_suppression": True
            },
            "line_in": {
                "sample_rate": 44100,
                "channels": 2,
                "chunk_size": 2048,
                "format": "wav"
            },
            "bluetooth": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "format": "wav",
                "codec": "SBC"
            },
            "webcam": {
                "sample_rate": 48000,
                "channels": 2,
                "chunk_size": 2048,
                "format": "wav"
            }
        }

        return settings.get(device_type, settings["microphone"])

    @staticmethod
    def generate_device_config(device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration for a specific audio device"""
        device_type = device_info.get("type", "microphone")
        base_settings = DeviceAudioCapture.get_recommended_settings(device_type)

        return {
            **base_settings,
            "device_id": device_info.get("id"),
            "device_name": device_info.get("name"),
            "device_type": device_type,
            "is_default": device_info.get("is_default", False)
        }

    @staticmethod
    def format_device_list(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format device list for API response"""
        formatted_devices = []

        for device in devices:
            config = DeviceAudioCapture.generate_device_config(device)
            formatted_devices.append({
                "id": device.get("id"),
                "name": device.get("name"),
                "type": device.get("type", "microphone"),
                "is_default": device.get("is_default", False),
                "is_available": device.get("is_available", True),
                "config": config
            })

        return formatted_devices
