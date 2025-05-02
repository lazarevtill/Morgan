"""
TTS service interface for Morgan Core
"""
import aiohttp
import base64
from typing import Dict, Any, Optional


class TTSService:
    """Interface to the TTS service (Dia)"""

    def __init__(self, service_url: str, default_voice: str):
        self.service_url = service_url
        self.default_voice = default_voice
        self.session = None

    async def connect(self):
        """Establish connection to the TTS service"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close connection to the TTS service"""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_speech(
            self,
            text: str,
            voice_id: Optional[str] = None,
            speed: float = 1.0,
            emotion: Optional[str] = None
    ) -> bytes:
        """
        Generate speech from text

        Args:
            text: Text to convert to speech
            voice_id: Voice to use (defaults to service default)
            speed: Speech speed multiplier
            emotion: Optional emotion to apply

        Returns:
            Audio data as bytes
        """
        if self.session is None:
            await self.connect()

        voice_id = voice_id or self.default_voice

        # Prepare the payload
        payload = {
            "text": text,
            "voice_id": voice_id,
            "speed": speed
        }

        # Add emotion if provided
        if emotion:
            payload["emotion"] = emotion

        # Send request to TTS service
        async with self.session.post(
                f"{self.service_url}/synthesize",
                json=payload,
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"TTS service error: {response.status} - {error_text}")

            result = await response.json()

            # Decode base64 audio data
            audio_data = base64.b64decode(result.get("audio_data", ""))
            return audio_data

    async def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices"""
        if self.session is None:
            await self.connect()

        # Send request to TTS service
        async with self.session.get(
                f"{self.service_url}/voices",
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"TTS service error: {response.status} - {error_text}")

            result = await response.json()
            return result.get("voices", {})