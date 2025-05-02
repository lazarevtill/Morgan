"""
Intent classification for Morgan
"""
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import json
import re

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classification system

    This class provides a way to define known intents and their
    expected parameters, and helps validate and normalize intents
    extracted from user input.
    """

    def __init__(self):
        self.intent_definitions = self._load_intent_definitions()

        # Map of intent aliases to canonical intent names
        self.intent_aliases = {
            # Home Assistant aliases
            "lights": "home_assistant.light",
            "light": "home_assistant.light",
            "switch": "home_assistant.switch",
            "climate": "home_assistant.climate",
            "thermostat": "home_assistant.climate",
            "media": "home_assistant.media_player",
            "media_player": "home_assistant.media_player",

            # Information aliases
            "weather": "information.weather",
            "forecast": "information.weather",
            "knowledge": "information.knowledge",
            "question": "information.knowledge",
            "time": "information.time",
            "date": "information.date",

            # System aliases
            "status": "system.status",
            "restart": "system.restart",
            "reboot": "system.restart",
            "update": "system.update",
            "upgrade": "system.update",
            "help": "system.help",
            "debug": "system.debug"
        }

        # Build a dictionary of parameter normalizers for common parameter types
        self.parameter_normalizers = {
            "action": self._normalize_action,
            "brightness": self._normalize_brightness,
            "temperature": self._normalize_temperature,
            "volume": self._normalize_volume
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
        # Resolve intent aliases
        resolved_intent = self._resolve_intent_alias(intent)

        # Normalize intent name (convert to lowercase, replace spaces with underscores)
        normalized_intent = self._normalize_intent_name(resolved_intent)

        # Check if intent exists in definitions
        if normalized_intent in self.intent_definitions:
            definition = self.intent_definitions[normalized_intent]

            # Normalize parameters based on parameter type
            normalized_params = self._normalize_parameters(normalized_intent, params)

            # Check required parameters
            missing_params = [p for p in definition["required"] if p not in normalized_params]
            if missing_params:
                # If missing required parameters, return false
                logger.warning(f"Intent {normalized_intent} missing required parameters: {missing_params}")
                return False, normalized_intent, normalized_params

            # Filter out unexpected parameters
            filtered_params = {k: v for k, v in normalized_params.items()
                               if k in definition["parameters"]}

            return True, normalized_intent, filtered_params

        # If intent doesn't exist, try to find a parent intent
        parts = normalized_intent.split('.')
        for i in range(len(parts) - 1, 0, -1):
            parent_intent = '.'.join(parts[:i])
            if parent_intent in self.intent_definitions:
                # If parent intent exists, incorporate the extra info into params
                modified_params = params.copy()
                if "entity" not in modified_params and i < len(parts):
                    modified_params["entity"] = parts[i]

                definition = self.intent_definitions[parent_intent]

                # Normalize parameters
                normalized_params = self._normalize_parameters(parent_intent, modified_params)

                # Check required parameters
                missing_params = [p for p in definition["required"] if p not in normalized_params]
                if not missing_params:
                    # Filter out unexpected parameters
                    filtered_params = {k: v for k, v in normalized_params.items()
                                       if k in definition["parameters"]}
                    return True, parent_intent, filtered_params

        # If no valid intent found, fall back to general conversation
        query = params.get("query", "")
        if not query and "text" in params:
            query = params["text"]
        return True, "general.conversation", {"query": query}

    def suggest_completion(self, partial_intent: str) -> List[str]:
        """
        Suggest completions for a partial intent

        Args:
            partial_intent: Partial intent string

        Returns:
            List of suggested completions
        """
        normalized_partial = self._normalize_intent_name(partial_intent)
        return [intent for intent in self.intent_definitions.keys()
                if intent.startswith(normalized_partial)]

    def get_intent_parameters(self, intent: str) -> List[str]:
        """
        Get the parameters for an intent

        Args:
            intent: Intent identifier

        Returns:
            List of parameter names
        """
        normalized_intent = self._normalize_intent_name(intent)
        if normalized_intent in self.intent_definitions:
            return self.intent_definitions[normalized_intent]["parameters"]
        return []

    def get_required_parameters(self, intent: str) -> List[str]:
        """
        Get the required parameters for an intent

        Args:
            intent: Intent identifier

        Returns:
            List of required parameter names
        """
        normalized_intent = self._normalize_intent_name(intent)
        if normalized_intent in self.intent_definitions:
            return self.intent_definitions[normalized_intent]["required"]
        return []

    def categorize_intent(self, intent: str) -> str:
        """
        Get the category of an intent

        Args:
            intent: Intent identifier

        Returns:
            Category name
        """
        normalized_intent = self._normalize_intent_name(intent)
        parts = normalized_intent.split('.')
        if len(parts) > 0:
            return parts[0]
        return "unknown"

    def _load_intent_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load intent definitions"""
        # In a real implementation, this could load from a JSON or YAML file
        # For now, we'll define them inline
        return {
            # Home Assistant intents
            "home_assistant.light": {
                "parameters": ["type", "action", "entity", "brightness", "color"],
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
            "home_assistant.group": {
                "parameters": ["type", "action", "group", "brightness"],
                "required": ["action", "group"]
            },
            "home_assistant.scene": {
                "parameters": ["type", "scene"],
                "required": ["scene"]
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
            "information.time": {
                "parameters": ["type", "timezone"],
                "required": ["type"]
            },
            "information.date": {
                "parameters": ["type", "format"],
                "required": ["type"]
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
            "system.debug": {
                "parameters": ["command"],
                "required": ["command"]
            },
            "system.help": {
                "parameters": ["command"],
                "required": ["command"]
            },
            "system.config": {
                "parameters": ["command", "action", "section", "key", "value"],
                "required": ["command", "action"]
            },
            "system.voice": {
                "parameters": ["command", "action", "voice_id"],
                "required": ["command", "action"]
            },

            # General conversation
            "general.conversation": {
                "parameters": ["type", "query"],
                "required": []
            }
        }

    def _resolve_intent_alias(self, intent: str) -> str:
        """
        Resolve an intent alias to its canonical intent name

        Args:
            intent: Intent name or alias

        Returns:
            Canonical intent name
        """
        # Check for exact match in aliases
        if intent in self.intent_aliases:
            return self.intent_aliases[intent]

        # Check if it's already a known intent
        if intent in self.intent_definitions:
            return intent

        # Try to match parts of compound intents
        parts = intent.split('.')
        if len(parts) > 1:
            first_part = parts[0]
            if first_part in self.intent_aliases:
                # Replace the first part with its alias
                parts[0] = self.intent_aliases[first_part]
                return '.'.join(parts)

        # No match found, return the original
        return intent

    def _normalize_intent_name(self, intent: str) -> str:
        """
        Normalize an intent name

        Args:
            intent: Intent name

        Returns:
            Normalized intent name
        """
        # Convert to lowercase
        normalized = intent.lower()

        # Replace spaces with underscores
        normalized = normalized.replace(' ', '_')

        # Remove any special characters
        normalized = re.sub(r'[^a-z0-9_.]', '', normalized)

        return normalized

    def _normalize_parameters(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters based on intent and parameter types

        Args:
            intent: Intent name
            params: Parameters to normalize

        Returns:
            Normalized parameters
        """
        normalized_params = {}

        # Copy parameters that don't need normalization
        for key, value in params.items():
            if key in self.parameter_normalizers:
                # Use specific normalizer for this parameter type
                normalized_params[key] = self.parameter_normalizers[key](value, intent)
            else:
                # No specific normalizer, just use the value as-is
                normalized_params[key] = value

        # Add any missing default parameters
        if intent.startswith("home_assistant") and "type" not in normalized_params:
            # For Home Assistant intents, add a default type if missing
            if "action" in normalized_params:
                normalized_params["type"] = "control"
            else:
                normalized_params["type"] = "query"

        if intent.startswith("system") and "command" not in normalized_params:
            # For system intents, add a default command based on the intent
            command = intent.split('.')[1] if len(intent.split('.')) > 1 else "status"
            normalized_params["command"] = command

        return normalized_params

    def _normalize_action(self, value: Any, intent: str) -> str:
        """Normalize action parameter"""
        if not value:
            return ""

        # Convert to string and lowercase
        str_value = str(value).lower()

        # Define common action aliases
        action_aliases = {
            "on": "turn_on",
            "off": "turn_off",
            "up": "turn_on",
            "down": "turn_off",
            "start": "turn_on",
            "stop": "turn_off",
            "toggle": "toggle"
        }

        # Return the canonical action name or the original if not found
        return action_aliases.get(str_value, str_value)

    def _normalize_brightness(self, value: Any, intent: str) -> Any:
        """Normalize brightness parameter"""
        # If it's already a string with % sign, return as is
        if isinstance(value, str) and "%" in value:
            return value

        try:
            # Convert to float
            brightness = float(value)

            # If it's a fraction (0-1), convert to percentage
            if 0 <= brightness <= 1:
                return f"{int(brightness * 100)}%"

            # If it's a percentage (0-100), add % sign
            if 0 <= brightness <= 100:
                return f"{int(brightness)}%"

            # If it's in the range 0-255, convert to percentage
            if 0 <= brightness <= 255:
                return f"{int(brightness / 255 * 100)}%"

            # Invalid range, return as is
            return value

        except (ValueError, TypeError):
            # Not a numeric value, return as is
            return value

    def _normalize_temperature(self, value: Any, intent: str) -> Any:
        """Normalize temperature parameter"""
        try:
            # Convert to float
            temp = float(value)

            # If it's a reasonable temperature, return as is
            if -50 <= temp <= 100:
                return temp

            # If it's a large number, maybe it's in Fahrenheit?
            if 100 < temp <= 212:
                # Convert to Celsius
                celsius = (temp - 32) * 5 / 9
                return round(celsius, 1)

            # Invalid range, return as is
            return value

        except (ValueError, TypeError):
            # Not a numeric value, return as is
            return value

    def _normalize_volume(self, value: Any, intent: str) -> Any:
        """Normalize volume parameter"""
        # Similar logic to brightness
        if isinstance(value, str) and "%" in value:
            return value

        try:
            # Convert to float
            volume = float(value)

            # If it's a fraction (0-1), convert to percentage
            if 0 <= volume <= 1:
                return f"{int(volume * 100)}%"

            # If it's a percentage (0-100), add % sign
            if 0 <= volume <= 100:
                return f"{int(volume)}%"

            # Invalid range, return as is
            return value

        except (ValueError, TypeError):
            # Not a numeric value, return as is
            return value