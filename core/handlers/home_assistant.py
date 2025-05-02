"""
Home Assistant command handler for Morgan
"""
from typing import Dict, Any, List, Optional
import logging

from .base_handler import BaseHandler
from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class HomeAssistantHandler(BaseHandler):
    """Handler for Home Assistant commands"""

    async def handle(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Handle a Home Assistant command

        Args:
            params: Command parameters
            context: Conversation context

        Returns:
            Response dictionary
        """
        command_type = params.get("type", "")
        entity_reference = params.get("entity", "")
        action = params.get("action", "")

        # Process based on command type
        if command_type == "query":
            return await self._handle_query(entity_reference, context)
        elif command_type == "control":
            return await self._handle_control(entity_reference, action, params, context)
        else:
            return await self.format_response(
                "I'm not sure what you want me to do with your smart home devices.",
                True
            )

    async def _handle_query(self, entity_reference: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle a query command"""
        # Resolve entity reference to entity ID
        entity_id = await self.core.home_assistant.resolve_entity(entity_reference)

        if not entity_id:
            return await self.format_response(
                f"I couldn't find a device matching '{entity_reference}'.",
                True
            )

        # Get entity state
        state_info = await self.core.home_assistant.get_state(entity_id)

        if not state_info:
            return await self.format_response(
                f"I couldn't get the state of {entity_reference}.",
                True
            )

        # Format response based on entity type
        domain = entity_id.split('.')[0]
        state = state_info.get("state", "unknown")

        if domain == "light":
            if state == "on":
                brightness = state_info.get("attributes", {}).get("brightness", 0)
                brightness_pct = round((brightness / 255) * 100)
                return await self.format_response(
                    f"The {entity_reference} is on at {brightness_pct}% brightness.",
                    True
                )
            else:
                return await self.format_response(
                    f"The {entity_reference} is off.",
                    True
                )
        elif domain == "switch":
            return await self.format_response(
                f"The {entity_reference} is {state}.",
                True
            )
        elif domain == "climate":
            current_temp = state_info.get("attributes", {}).get("current_temperature")
            target_temp = state_info.get("attributes", {}).get("temperature")
            hvac_mode = state_info.get("attributes", {}).get("hvac_mode")

            if current_temp and target_temp:
                return await self.format_response(
                    f"The {entity_reference} is {state}. The current temperature is {current_temp}°C and it's set to {target_temp}°C in {hvac_mode} mode.",
                    True
                )
            else:
                return await self.format_response(
                    f"The {entity_reference} is {state}.",
                    True
                )
        else:
            return await self.format_response(
                f"The {entity_reference} is {state}.",
                True
            )

    async def _handle_control(
            self,
            entity_reference: str,
            action: str,
            params: Dict[str, Any],
            context: ConversationContext
    ) -> Dict[str, Any]:
        """Handle a control command"""
        # Resolve entity reference to entity ID
        entity_id = await self.core.home_assistant.resolve_entity(entity_reference)

        if not entity_id:
            return await self.format_response(
                f"I couldn't find a device matching '{entity_reference}'.",
                True
            )

        # Determine domain and service
        domain = entity_id.split('.')[0]

        # Process based on action and domain
        if action == "turn_on":
            service = "turn_on"
            service_data = {}

            # Add brightness if specified (for lights)
            if domain == "light" and "brightness" in params:
                brightness = params["brightness"]
                if isinstance(brightness, str) and brightness.endswith("%"):
                    # Convert percentage to 0-255
                    try:
                        brightness_pct = int(brightness.rstrip("%"))
                        brightness = int((brightness_pct / 100) * 255)
                    except ValueError:
                        brightness = 255

                service_data["brightness"] = brightness

            # Call the service
            success = await self.core.home_assistant.call_service(
                domain, service, entity_id, service_data=service_data
            )

            if success:
                if domain == "light" and "brightness" in service_data:
                    brightness_pct = round((service_data["brightness"] / 255) * 100)
                    return await self.format_response(
                        f"I've turned on the {entity_reference} at {brightness_pct}% brightness.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I've turned on the {entity_reference}.",
                        True
                    )
            else:
                return await self.format_response(
                    f"I couldn't turn on the {entity_reference}.",
                    True
                )

        elif action == "turn_off":
            service = "turn_off"

            # Call the service
            success = await self.core.home_assistant.call_service(domain, service, entity_id)

            if success:
                return await self.format_response(
                    f"I've turned off the {entity_reference}.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't turn off the {entity_reference}.",
                    True
                )

        elif action == "set_temperature" and domain == "climate":
            service = "set_temperature"
            temperature = params.get("temperature")

            if not temperature:
                return await self.format_response(
                    "I need to know what temperature to set.",
                    True
                )

            # Call the service
            success = await self.core.home_assistant.call_service(
                domain, service, entity_id, service_data={"temperature": temperature}
            )

            if success:
                return await self.format_response(
                    f"I've set the {entity_reference} to {temperature}°C.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't set the temperature for {entity_reference}.",
                    True
                )

        else:
            return await self.format_response(
                f"I don't know how to {action} the {entity_reference}.",
                True
            )