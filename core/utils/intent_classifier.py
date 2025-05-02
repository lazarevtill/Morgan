"""
Intent classification for Morgan
"""
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classification system

    This class provides a way to define known intents and their
    expected parameters, and helps validate and normalize intents
    extracted from user input.
    """

    def __init__(self):
        self.intent_definitions = {
            # Home Assistant intents
            "home_assistant.light": {
                "parameters": ["type", "action", "entity", "brightness"],
                "required": ["action", "entity"]
            },
            "home_assistant.switch": {
                "parameters": ["type", "action", "entity"],
                "required": ["action", "entity"]
            },
            "home_assistant.climate": {
                "parameters": ["type", "action", "entity", "temperature", "mode"],
                "required": ["action", "entity"]
            },
            "home_assistant.media_player": {
                "parameters": ["type", "action", "entity", "media", "volume"],
                "required": ["action", "entity"]
            },
            "home_assistant.query": {
                "parameters": ["type", "entity"],
                "required": ["entity"]
            },

            # Information intents
            "information.weather": {
                "parameters": ["type", "query", "location", "days"],
                "required": ["type"]
            },
            "information.knowledge": {
                "parameters": ["type", "query"],
                "required": ["query"]
            },

            # System intents
            "system.status": {
                "parameters": ["command"],
                "required": ["command"]
            },
            "system.restart": {
                "parameters": ["command"],
                "required": ["command"]
            },
            "system.update": {
                "parameters": ["command"],
                "required": ["command"]
            },

            # General conversation
            "general.conversation": {
                "parameters": ["type", "query"],
                "required": []
            }
        }

    def validate_intent(self, intent: str, params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate an intent and its parameters

        Args:
            intent: Intent identifier
            params: Intent parameters

        Returns:
            Tuple of (is_valid, modified_intent, normalized_params)
        """
        # Check if intent exists in definitions
        if intent in self.intent_definitions:
            definition = self.intent_definitions[intent]

            # Check required parameters
            missing_params = [p for p in definition["required"] if p not in params]
            if missing_params:
                # If missing required parameters, return false
                return False, intent, params

            # Filter out unexpected parameters
            normalized_params = {k: v for k, v in params.items()
                                 if k in definition["parameters"]}

            return True, intent, normalized_params

        # If intent doesn't exist, try to find a parent intent
        parts = intent.split('.')
        for i in range(len(parts) - 1, 0, -1):
            parent_intent = '.'.join(parts[:i])
            if parent_intent in self.intent_definitions:
                # If parent intent exists, incorporate the extra info into params
                modified_params = params.copy()
                if "entity" not in modified_params and i < len(parts):
                    modified_params["entity"] = parts[i]

                definition = self.intent_definitions[parent_intent]
                # Check required parameters
                missing_params = [p for p in definition["required"] if p not in modified_params]
                if not missing_params:
                    # Filter out unexpected parameters
                    normalized_params = {k: v for k, v in modified_params.items()
                                         if k in definition["parameters"]}
                    return True, parent_intent, normalized_params

        # If no valid intent found, fall back to general conversation
        return True, "general.conversation", {"query": params.get("query", "")}

    def suggest_completion(self, partial_intent: str) -> List[str]:
        """
        Suggest completions for a partial intent

        Args:
            partial_intent: Partial intent string

        Returns:
            List of suggested completions
        """
        return [intent for intent in self.intent_definitions.keys()
                if intent.startswith(partial_intent)]

    def get_intent_parameters(self, intent: str) -> List[str]:
        """
        Get the parameters for an intent

        Args:
            intent: Intent identifier

        Returns:
            List of parameter names
        """
        if intent in self.intent_definitions:
            return self.intent_definitions[intent]["parameters"]
        return []