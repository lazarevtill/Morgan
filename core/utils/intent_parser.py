"""
Intent parsing utilities for Morgan Core
"""
import json
import re
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class IntentParser:
    """Parser for extracting intents and parameters from LLM responses"""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def extract_intent(self, text: str, history: List[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Extract intent and parameters from user input using the LLM

        Args:
            text: User input text
            history: Optional conversation history

        Returns:
            Tuple of (intent_name, parameters)
        """
        try:
            # Use LLM service to get structured intent information
            intent_data = await self.llm_service.get_intent(text, history)

            # Extract intent and parameters from LLM response
            intent_name = intent_data.get("intent", "general.conversation")
            parameters = intent_data.get("parameters", {})

            # Validate and clean parameters
            cleaned_params = self._clean_parameters(parameters)

            logger.debug(f"Extracted intent: {intent_name} with parameters: {cleaned_params}")
            return intent_name, cleaned_params

        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            # Fallback to rule-based parsing if LLM intent extraction fails
            return self._rule_based_parsing(text)

    def _clean_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate parameters"""
        cleaned = {}

        for key, value in parameters.items():
            # Skip None or empty values
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Convert numeric strings to numbers where appropriate
            if isinstance(value, str) and value.strip().replace('.', '', 1).isdigit():
                try:
                    if '.' in value:
                        cleaned[key] = float(value)
                    else:
                        cleaned[key] = int(value)
                    continue
                except ValueError:
                    pass

            # Keep other values as is
            cleaned[key] = value

        return cleaned

    def _rule_based_parsing(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Simple rule-based parsing as fallback

        Args:
            text: User input text

        Returns:
            Tuple of (intent_name, parameters)
        """
        text_lower = text.lower()

        # Home Assistant light controls
        if re.search(r'turn (on|off) .*(light|lamp|lights)', text_lower):
            action = "turn_on" if "turn on" in text_lower else "turn_off"
            # Try to extract the light name
            light_match = re.search(r'(turn on|turn off) the (.+?)(?:\s|$|\.)', text_lower)
            entity = light_match.group(2) if light_match else "light"
            return "home_assistant.light", {"type": "control", "action": action, "entity": entity}

        # Home Assistant queries
        if re.search(r'(what\'s|what is|how is|status of) .*(temperature|light|switch|thermostat)', text_lower):
            entity_match = re.search(r'(temperature|light|switch|thermostat)', text_lower)
            entity = entity_match.group(1) if entity_match else ""
            return "home_assistant.query", {"type": "query", "entity": entity}

        # Weather queries
        if re.search(r'(weather|temperature|forecast)', text_lower):
            location_match = re.search(r'weather (?:in|at|for) (.+?)(?:\s|$|\.)', text_lower)
            location = location_match.group(1) if location_match else "here"
            return "information.weather", {"type": "weather", "query": location}

        # System commands
        if re.search(r'(system status|how are you running|resource usage)', text_lower):
            return "system.status", {"command": "status"}

        # Default to general conversation
        return "general.conversation", {"type": "chat", "query": text}