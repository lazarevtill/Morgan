"""
STT service interface for Morgan Core
"""
import aiohttp
import base64
from typing import Dict, Any, Optional


class STTService:
    """Interface to the STT service (Whisper)"""

    def __init__(self, service_url: str, model_name: str):
        self.service_url = service_url
        self.model_name = model_name
        self.session = None

    async def connect(self):
        """Establish connection to the STT service"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close connection to the STT service"""
        if self.session:
            await self.session.close()
            self.session = None

    async def transcribe(
            self,
            audio_data: bytes,
            language: Optional[str] = None,
            prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe speech to text

        Args:
            audio_data: Audio data as bytes
            language: Optional language code
            prompt: Optional prompt to guide transcription

        Returns:
            Transcribed text
        """
        if self.session is None:
            await self.connect()

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "audio_data": base64.b64encode(audio_data).decode('utf-8')
        }

        # Add optional parameters
        if language:
            payload["language"] = language

        if prompt:
            payload["prompt"] = prompt

        # Send request to STT service
        async with self.session.post(
                f"{self.service_url}/transcribe",
                json=payload,
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"STT service error: {response.status} - {error_text}")

            result = await response.json()
            return result.get("text", "")