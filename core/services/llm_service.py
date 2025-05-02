"""
LLM service interface for Morgan Core
"""
import aiohttp
import json
from typing import Dict, Any, List


class LLMService:
    """Interface to the LLM service"""

    def __init__(self, service_url: str, model_name: str):
        self.service_url = service_url
        self.model_name = model_name
        self.session = None

    async def connect(self):
        """Establish connection to the LLM service"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close connection to the LLM service"""
        if self.session:
            await self.session.close()
            self.session = None

    async def process_input(self, text: str, history: List[Dict[str, Any]] = None) -> str:
        """
        Process user input and generate a response

        Args:
            text: The user input text
            history: Optional conversation history

        Returns:
            The generated text response
        """
        if self.session is None:
            await self.connect()

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": text,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        # Add history if provided
        if history:
            payload["history"] = history

        # Send request to LLM service
        async with self.session.post(
                f"{self.service_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LLM service error: {response.status} - {error_text}")

            result = await response.json()
            return result.get("generated_text", "")

    async def get_intent(self, text: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract intent and parameters from user input

        Args:
            text: The user input text
            history: Optional conversation history

        Returns:
            Dictionary containing intent and parameters
        """
        if self.session is None:
            await self.connect()

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": text,
            "max_tokens": 500,
            "temperature": 0.3,
            "task": "intent_extraction"
        }

        # Add history if provided
        if history:
            payload["history"] = history

        # Send request to LLM service
        async with self.session.post(
                f"{self.service_url}/extract_intent",
                json=payload,
                headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LLM service error: {response.status} - {error_text}")

            result = await response.json()
            return result