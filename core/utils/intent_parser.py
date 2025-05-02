"""
Enhanced intent parsing utilities for Morgan Core
"""
import logging
import re
import traceback
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class IntentParser:
    """Parser for extracting intents and parameters from LLM responses"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.common_intent_patterns = self._compile_intent_patterns()

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
            # First, try rule-based parsing for common patterns for efficiency
            intent_name, parameters = self._quick_pattern_match(text)
            if intent_name:
                logger.debug(f"Quickly matched intent: {intent_name} with parameters: {parameters}")
                return intent_name, parameters

            # Use LLM service to get structured intent information for more complex requests
            prompt = self._create_intent_extraction_prompt(text, history)
            intent_data = await self.llm_service.get_intent(text, history)

            # Extract intent and parameters from LLM response
            intent_name = intent_data.get("intent", "general.conversation")
            parameters = intent_data.get("parameters", {})

            # Add the original query for reference
            if intent_name == "general.conversation" or intent_name.startswith("information"):
                if "query" not in parameters:
                    parameters["query"] = text

            # Validate and clean parameters
            cleaned_params = self._clean_parameters(parameters)

            logger.debug(f"Extracted intent: {intent_name} with parameters: {cleaned_params}")
            return intent_name, cleaned_params

        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            logger.debug(traceback.format_exc())
            # Fallback to rule-based parsing if LLM intent extraction fails
            return self._rule_based_parsing(text)

    def _create_intent_extraction_prompt(self, text: str, history: List[Dict[str, Any]] = None) -> str:
        """Create a prompt for intent extraction"""
        # Build context from history if available
        context = ""
        if history and len(history) > 0:
            context = "Conversation history:\n"
            for msg in history[-5:]:  # Include up to 5 most recent messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context += f"{role}: {content}\n"

        # Define the intent extraction task
        prompt = f"""
{context}
Task: Extract the user's intent and parameters from the following input.
User input: {text}

Identify the most likely intent from these categories:
1. home_assistant.light - Controlling lights (parameters: type, action, entity, brightness)
2. home_assistant.switch - Controlling switches (parameters: type, action, entity)
3. home_assistant.climate - Controlling climate devices (parameters: type, action, entity, temperature, mode)
4. home_assistant.media_player - Controlling media players (parameters: type, action, entity, media, volume)
5. home_assistant.query - Querying device status (parameters: type, entity)
6. information.weather - Weather information (parameters: type, query, location, days)
7. information.knowledge - General knowledge queries (parameters: type, query)
8. system.status - System status information (parameters: command)
9. system.restart - Restart the system (parameters: command)
10. system.update - Update the system (parameters: command)
11. general.conversation - General conversation (parameters: type, query)

Return a JSON object with "intent" and "parameters" fields.
"""
        return prompt

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

            # Handle boolean values
            if isinstance(value, str) and value.lower() in ('true', 'yes', 'on'):
                cleaned[key] = True
                continue
            elif isinstance(value, str) and value.lower() in ('false', 'no', 'off'):
                cleaned[key] = False
                continue

            # Handle percentages
            if isinstance(value, str) and value.endswith('%'):
                try:
                    cleaned[key] = float(value.rstrip('%'))
                    continue
                except ValueError:
                    pass

            # Keep other values as is
            cleaned[key] = value

        return cleaned

    def _compile_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Compile regex patterns for common intents"""
        patterns = {
            # Light control patterns
            "turn_on_light": {
                "pattern": re.compile(r'turn (?:on|up) (?:the )?(.*?)(?: light| lamp)?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.light",
                "action": "turn_on"
            },
            "turn_off_light": {
                "pattern": re.compile(r'turn (?:off|down) (?:the )?(.*?)(?: light| lamp)?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.light",
                "action": "turn_off"
            },
            "dim_light": {
                "pattern": re.compile(r'(?:dim|set) (?:the )?(.*?)(?: light| lamp)?(?: to)? (\d+)%?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.light",
                "action": "turn_on",
                "has_brightness": True
            },

            # Switch control patterns
            "turn_on_switch": {
                "pattern": re.compile(r'turn on (?:the )?(.*?)(?: switch)?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.switch",
                "action": "turn_on"
            },
            "turn_off_switch": {
                "pattern": re.compile(r'turn off (?:the )?(.*?)(?: switch)?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.switch",
                "action": "turn_off"
            },

            # Temperature control patterns
            "set_temperature": {
                "pattern": re.compile(r'set (?:the )?(.*?) (?:temperature |thermostat )?to (\d+)(?:\s|°C|°F|degrees)?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.climate",
                "action": "set_temperature"
            },

            # Device query patterns
            "device_status": {
                "pattern": re.compile(r'(?:what\'s|what is|how is|status of) (?:the )?(.*?)(?:\s|$|\.)', re.IGNORECASE),
                "intent": "home_assistant.query"
            },

            # Weather patterns
            "weather_query": {
                "pattern": re.compile(r'(?:what\'s|what is) the (?:weather|temperature|forecast)(?: (?:like|in|at|for) (.*?))?(?:\s|$|\.)', re.IGNORECASE),
                "intent": "information.weather"
            },

            # System status patterns
            "system_status": {
                "pattern": re.compile(r'(?:system|morgan) (?:status|health|diagnostics)', re.IGNORECASE),
                "intent": "system.status",
                "command": "status"
            },
            "system_restart": {
                "pattern": re.compile(r'(?:restart|reboot) (?:system|morgan|yourself)', re.IGNORECASE),
                "intent": "system.restart",
                "command": "restart"
            },
            "system_update": {
                "pattern": re.compile(r'(?:update|upgrade) (?:system|morgan|yourself)', re.IGNORECASE),
                "intent": "system.update",
                "command": "update"
            }
        }
        return patterns

    def _quick_pattern_match(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Quickly match text against common patterns for faster processing

        Returns:
            Tuple of (intent_name, parameters) or (None, {}) if no match
        """
        for pattern_name, pattern_info in self.common_intent_patterns.items():
            match = pattern_info["pattern"].search(text)
            if match:
                # Extract intent and build parameters
                intent = pattern_info["intent"]
                params = {}

                # Add action if present in pattern info
                if "action" in pattern_info:
                    params["action"] = pattern_info["action"]

                # Add command if present (for system commands)
                if "command" in pattern_info:
                    params["command"] = pattern_info["command"]

                # Handle device entity from regex match
                if len(match.groups()) >= 1 and match.group(1):
                    if intent.startswith("home_assistant"):
                        params["entity"] = match.group(1).strip()
                        params["type"] = "control" if "action" in params else "query"
                    elif intent == "information.weather":
                        location = match.group(1).strip() if match.group(1) else "here"
                        params["type"] = "weather"
                        params["location"] = location
                        params["query"] = location

                # Handle brightness/temperature value from regex match
                if "has_brightness" in pattern_info and len(match.groups()) >= 2:
                    try:
                        brightness_val = int(match.group(2))
                        params["brightness"] = f"{brightness_val}%"
                    except (ValueError, IndexError):
                        pass

                # Handle temperature from regex match
                if intent == "home_assistant.climate" and "action" in params and params["action"] == "set_temperature":
                    if len(match.groups()) >= 2:
                        try:
                            params["temperature"] = float(match.group(2))
                        except (ValueError, IndexError):
                            pass

                return intent, params

        # No match found
        return None, {}

    def _rule_based_parsing(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Rule-based parsing as fallback
        """
        text_lower = text.lower()

        # Home Assistant light controls
        if re.search(r'(turn|switch) (on|off) .*(light|lamp|lights)', text_lower):
            action = "turn_on" if "on" in text_lower else "turn_off"
            # Try to extract the light name
            light_match = re.search(r'(turn|switch) (on|off) (?:the )?([\w\s]+?)(?:\s|$|\.)', text_lower)
            entity = light_match.group(3) if light_match else "light"
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
            return "information.weather", {"type": "weather", "query": text, "location": location}

        # Knowledge queries
        if re.search(r'(who|what|when|where|why|how|tell me about)', text_lower):
            return "information.knowledge", {"type": "knowledge", "query": text}

        # System commands
        if re.search(r'(system status|how are you running|resource usage)', text_lower):
            return "system.status", {"command": "status"}

        # Default to general conversation
        return "general.conversation", {"type": "chat", "query": text}