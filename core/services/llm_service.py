"""
Enhanced LLM service interface for Morgan Core
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class LLMService:
    """Interface to the LLM service"""

    def __init__(self, service_url: str, model_name: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        self.service_url = service_url
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are Morgan, a helpful and friendly home assistant AI."
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = None
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests in seconds

    async def connect(self):
        """Establish connection to the LLM service"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            logger.info(f"Connected to LLM service at {self.service_url}")

    async def disconnect(self):
        """Close connection to the LLM service"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from LLM service")

    async def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            # Sleep to maintain the minimum interval
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    async def process_input(self, text: str, history: List[Dict[str, Any]] = None,
                            temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Process user input and generate a response

        Args:
            text: The user input text
            history: Optional conversation history
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            The generated text response
        """
        if self.session is None:
            await self.connect()

        await self._rate_limit()

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": text,
            "system_prompt": self.system_prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature
        }

        # Add history if provided
        if history:
            payload["history"] = history

        # Send request to LLM service
        try:
            async with self.session.post(
                    f"{self.service_url}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # Add a timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM service error: {response.status} - {error_text}")
                    raise Exception(f"LLM service error: {response.status} - {error_text}")

                result = await response.json()
                return result.get("generated_text", "")
        except aiohttp.ClientError as e:
            logger.error(f"Network error when connecting to LLM service: {e}")
            raise Exception(f"Network error when connecting to LLM service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in process_input: {e}")
            raise

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

        await self._rate_limit()

        # Create a specialized system prompt for intent extraction
        intent_system_prompt = """
        You are an intent classification system for a smart home assistant called Morgan.
        Your task is to analyze user input and extract the intent and parameters.
        Respond with a JSON object containing 'intent' and 'parameters' fields.
        
        Available intents:
        - home_assistant.light - For controlling lights
        - home_assistant.switch - For controlling switches
        - home_assistant.climate - For controlling climate devices
        - home_assistant.media_player - For controlling media players
        - home_assistant.query - For querying device status
        - information.weather - For weather information
        - information.knowledge - For knowledge queries
        - system.status - For system status
        - system.restart - For restarting the system
        - system.update - For updating the system
        - general.conversation - For general conversation
        """

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": text,
            "system_prompt": intent_system_prompt,
            "max_tokens": 500,
            "temperature": 0.3,
            "task": "intent_extraction",
            "response_format": "json"
        }

        # Add history if provided
        if history:
            payload["history"] = history

        # Send request to LLM service
        try:
            async with self.session.post(
                    f"{self.service_url}/extract_intent",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10  # Shorter timeout for intent extraction
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM service error (intent extraction): {response.status} - {error_text}")
                    raise Exception(f"LLM service error: {response.status} - {error_text}")

                result = await response.json()

                # Validate the result
                if not isinstance(result, dict) or "intent" not in result:
                    # Try to parse the generated text as JSON if the result doesn't have the expected structure
                    try:
                        text_result = result.get("generated_text", "{}")
                        parsed_result = json.loads(text_result)
                        if isinstance(parsed_result, dict) and "intent" in parsed_result:
                            return parsed_result
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    # Return a default intent if we can't parse the result
                    logger.warning("Invalid intent extraction result, using default")
                    return {"intent": "general.conversation", "parameters": {"query": text}}

                return result

        except aiohttp.ClientError as e:
            logger.error(f"Network error when connecting to LLM service for intent extraction: {e}")
            raise Exception(f"Network error when connecting to LLM service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_intent: {e}")
            raise

    async def classify_text(self, text: str, categories: List[str]) -> str:
        """
        Classify text into one of several categories

        Args:
            text: The text to classify
            categories: List of possible categories

        Returns:
            The most likely category
        """
        if self.session is None:
            await self.connect()

        await self._rate_limit()

        # Create a prompt for the classification
        categories_str = "\n".join([f"- {category}" for category in categories])
        prompt = f"""
        Classify the following text into one of these categories:
        {categories_str}
        
        Text: {text}
        
        Category:
        """

        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.3
        }

        # Send request to LLM service
        try:
            async with self.session.post(
                    f"{self.service_url}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM service error (classification): {response.status} - {error_text}")
                    raise Exception(f"LLM service error: {response.status} - {error_text}")

                result = await response.json()
                classification = result.get("generated_text", "").strip()

                # Find the best matching category
                best_match = None
                best_similarity = 0

                for category in categories:
                    # Simple string similarity check (can be improved with better algorithms)
                    similarity = self._string_similarity(classification.lower(), category.lower())
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = category

                return best_match or categories[0]  # Return the best match or the first category as fallback

        except Exception as e:
            logger.error(f"Error in classify_text: {e}")
            # Return the first category as a fallback
            return categories[0]

    def _string_similarity(self, a: str, b: str) -> float:
        """Calculate simple string similarity"""
        # Count the number of matching characters
        matches = sum(1 for char_a, char_b in zip(a, b) if char_a == char_b)
        max_length = max(len(a), len(b))
        return matches / max_length if max_length > 0 else 0


import asyncio  # Add missing import
