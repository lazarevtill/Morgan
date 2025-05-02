"""
Enhanced Home Assistant command handler for Morgan
"""
from typing import Dict, Any, List, Optional
import logging
import re

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

        # Handle special case for empty entity
        if not entity_reference and context.get_variable("last_entity"):
            entity_reference = context.get_variable("last_entity")
            logger.debug(f"Using last entity from context: {entity_reference}")

        # Process based on command type
        if command_type == "query":
            return await self._handle_query(entity_reference, context)
        elif command_type == "control":
            return await self._handle_control(entity_reference, action, params, context)
        elif command_type == "group":
            return await self._handle_group_command(params, context)
        elif command_type == "scene":
            return await self._handle_scene(params.get("scene", ""), context)
        else:
            # Try to infer what the user wants based on the parameters
            if entity_reference and action:
                return await self._handle_control(entity_reference, action, params, context)
            elif entity_reference:
                return await self._handle_query(entity_reference, context)
            else:
                return await self.format_response(
                    "I'm not sure what you want me to do with your smart home devices. Could you specify a device?",
                    True
                )

    async def _handle_query(self, entity_reference: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle a query command"""
        # Check if we're querying a group
        if entity_reference.lower() in ["everything", "all devices", "all"]:
            return await self._handle_all_devices_query(context)

        # Resolve entity reference to entity ID
        entity_id = await self.core.home_assistant.resolve_entity(entity_reference)

        if not entity_id:
            # Check if we're looking for a room or area
            area_entities = await self.core.home_assistant.get_entities_for_area(entity_reference)
            if area_entities:
                return await self._handle_area_query(entity_reference, area_entities, context)

            return await self.format_response(
                f"I couldn't find a device matching '{entity_reference}'.",
                True
            )

        # Save this entity as the last one used
        context.set_variable("last_entity", entity_reference)

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
        friendly_name = state_info.get("attributes", {}).get("friendly_name", entity_reference)

        if domain == "light":
            if state == "on":
                brightness = state_info.get("attributes", {}).get("brightness", 0)
                brightness_pct = round((brightness / 255) * 100)

                # Get color information if available
                color_temp = state_info.get("attributes", {}).get("color_temp")
                rgb_color = state_info.get("attributes", {}).get("rgb_color")
                color_info = ""

                if rgb_color:
                    color_info = f" with RGB color {rgb_color}"
                elif color_temp:
                    # Convert color temperature to description
                    if color_temp < 300:
                        temp_desc = "warm"
                    elif color_temp < 370:
                        temp_desc = "neutral"
                    else:
                        temp_desc = "cool"
                    color_info = f" with {temp_desc} white color temperature"

                return await self.format_response(
                    f"The {friendly_name} is on at {brightness_pct}% brightness{color_info}.",
                    True
                )
            else:
                return await self.format_response(
                    f"The {friendly_name} is off.",
                    True
                )

        elif domain == "switch":
            return await self.format_response(
                f"The {friendly_name} is {state}.",
                True
            )

        elif domain == "climate":
            current_temp = state_info.get("attributes", {}).get("current_temperature")
            target_temp = state_info.get("attributes", {}).get("temperature")
            hvac_mode = state_info.get("attributes", {}).get("hvac_mode", "")
            hvac_action = state_info.get("attributes", {}).get("hvac_action", "")

            # Create a more human-friendly description of the HVAC mode and action
            mode_descriptions = {
                "heat": "heating",
                "cool": "cooling",
                "auto": "automatic",
                "off": "turned off",
                "fan_only": "fan only",
                "dry": "dehumidifying"
            }

            mode_desc = mode_descriptions.get(hvac_mode, hvac_mode)

            if current_temp and target_temp:
                if hvac_action and hvac_action != "idle":
                    action_desc = f"currently {mode_descriptions.get(hvac_action, hvac_action)}"
                    return await self.format_response(
                        f"The {friendly_name} is {action_desc}. The current temperature is {current_temp}°C and it's set to {target_temp}°C in {mode_desc} mode.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"The {friendly_name} is {mode_desc}. The current temperature is {current_temp}°C and it's set to {target_temp}°C.",
                        True
                    )
            else:
                return await self.format_response(
                    f"The {friendly_name} is {state}.",
                    True
                )

        elif domain == "media_player":
            if state == "playing":
                title = state_info.get("attributes", {}).get("media_title", "something")
                artist = state_info.get("attributes", {}).get("media_artist")
                volume = state_info.get("attributes", {}).get("volume_level", 0) * 100

                artist_info = f" by {artist}" if artist else ""
                volume_info = f" at {int(volume)}% volume" if volume is not None else ""

                return await self.format_response(
                    f"The {friendly_name} is playing \"{title}\"{artist_info}{volume_info}.",
                    True
                )
            elif state == "paused":
                title = state_info.get("attributes", {}).get("media_title", "something")
                return await self.format_response(
                    f"The {friendly_name} is paused while playing \"{title}\".",
                    True
                )
            else:
                return await self.format_response(
                    f"The {friendly_name} is {state}.",
                    True
                )

        elif domain == "sensor":
            unit = state_info.get("attributes", {}).get("unit_of_measurement", "")
            return await self.format_response(
                f"The {friendly_name} is reading {state}{' ' + unit if unit else ''}.",
                True
            )

        else:
            return await self.format_response(
                f"The {friendly_name} is {state}.",
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
            # Check if we're controlling a room or area
            area_entities = await self.core.home_assistant.get_entities_for_area(entity_reference)
            if area_entities:
                return await self._handle_area_control(entity_reference, action, params, area_entities, context)

            return await self.format_response(
                f"I couldn't find a device matching '{entity_reference}'.",
                True
            )

        # Save this entity as the last one used
        context.set_variable("last_entity", entity_reference)

        # Determine domain and service
        domain = entity_id.split('.')[0]

        # Get the entity state for friendly name
        state_info = await self.core.home_assistant.get_state(entity_id)
        friendly_name = state_info.get("attributes", {}).get("friendly_name", entity_reference) if state_info else entity_reference

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
                elif isinstance(brightness, (int, float)) and 0 <= brightness <= 100:
                    # Assume it's a percentage
                    brightness = int((brightness / 100) * 255)

                service_data["brightness"] = brightness

            # Add color if specified (for lights)
            if domain == "light" and "color" in params:
                color = params["color"]
                # Handle color temperature or RGB
                if isinstance(color, str):
                    color_lower = color.lower()
                    if color_lower in ["warm", "warm white"]:
                        service_data["color_temp"] = 300
                    elif color_lower in ["neutral", "neutral white"]:
                        service_data["color_temp"] = 370
                    elif color_lower in ["cool", "cool white"]:
                        service_data["color_temp"] = 450
                    else:
                        # Try to map color names to RGB
                        color_map = {
                            "red": [255, 0, 0],
                            "green": [0, 255, 0],
                            "blue": [0, 0, 255],
                            "yellow": [255, 255, 0],
                            "purple": [128, 0, 128],
                            "orange": [255, 165, 0],
                            "pink": [255, 192, 203],
                            "white": [255, 255, 255]
                        }
                        if color_lower in color_map:
                            service_data["rgb_color"] = color_map[color_lower]

            # Call the service
            success = await self.core.home_assistant.call_service(
                domain, service, entity_id, service_data=service_data
            )

            if success:
                if domain == "light" and "brightness" in service_data:
                    brightness_pct = round((service_data["brightness"] / 255) * 100)
                    color_msg = ""
                    if "color_temp" in service_data:
                        if service_data["color_temp"] < 300:
                            color_msg = " with warm white color"
                        elif service_data["color_temp"] < 370:
                            color_msg = " with neutral white color"
                        else:
                            color_msg = " with cool white color"
                    elif "rgb_color" in service_data:
                        color_msg = " with the color you specified"

                    return await self.format_response(
                        f"I've turned on the {friendly_name} at {brightness_pct}% brightness{color_msg}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I've turned on the {friendly_name}.",
                        True
                    )
            else:
                return await self.format_response(
                    f"I couldn't turn on the {friendly_name}.",
                    True
                )

        elif action == "turn_off":
            service = "turn_off"

            # Call the service
            success = await self.core.home_assistant.call_service(domain, service, entity_id)

            if success:
                return await self.format_response(
                    f"I've turned off the {friendly_name}.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't turn off the {friendly_name}.",
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
                    f"I've set the {friendly_name} to {temperature}°C.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't set the temperature for {friendly_name}.",
                    True
                )

        elif action == "set_mode" and domain == "climate":
            service = "set_hvac_mode"
            mode = params.get("mode", "").lower()

            # Map common terms to HVAC modes
            mode_mapping = {
                "heat": "heat",
                "heating": "heat",
                "warm": "heat",
                "cool": "cool",
                "cooling": "cool",
                "cold": "cool",
                "auto": "auto",
                "automatic": "auto",
                "fan": "fan_only",
                "fan only": "fan_only",
                "dry": "dry",
                "dehumidify": "dry",
                "off": "off",
                "turn off": "off"
            }

            hvac_mode = mode_mapping.get(mode)
            if not hvac_mode:
                return await self.format_response(
                    f"I don't understand the mode '{mode}'. Available modes are heat, cool, auto, fan, dry, or off.",
                    True
                )

            # Call the service
            success = await self.core.home_assistant.call_service(
                domain, service, entity_id, service_data={"hvac_mode": hvac_mode}
            )

            if success:
                mode_descriptions = {
                    "heat": "heating",
                    "cool": "cooling",
                    "auto": "automatic",
                    "fan_only": "fan only",
                    "dry": "dehumidifying",
                    "off": "off"
                }
                mode_desc = mode_descriptions.get(hvac_mode, hvac_mode)

                return await self.format_response(
                    f"I've set the {friendly_name} to {mode_desc} mode.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't change the mode for {friendly_name}.",
                    True
                )

        elif domain == "media_player":
            if action == "play":
                service = "media_play"
                success = await self.core.home_assistant.call_service(domain, service, entity_id)

                if success:
                    return await self.format_response(
                        f"I've started playing media on the {friendly_name}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't start playing on the {friendly_name}.",
                        True
                    )

            elif action == "pause":
                service = "media_pause"
                success = await self.core.home_assistant.call_service(domain, service, entity_id)

                if success:
                    return await self.format_response(
                        f"I've paused the {friendly_name}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't pause the {friendly_name}.",
                        True
                    )

            elif action == "stop":
                service = "media_stop"
                success = await self.core.home_assistant.call_service(domain, service, entity_id)

                if success:
                    return await self.format_response(
                        f"I've stopped the {friendly_name}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't stop the {friendly_name}.",
                        True
                    )

            elif action == "next":
                service = "media_next_track"
                success = await self.core.home_assistant.call_service(domain, service, entity_id)

                if success:
                    return await self.format_response(
                        f"I've skipped to the next track on the {friendly_name}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't skip to the next track on the {friendly_name}.",
                        True
                    )

            elif action == "previous":
                service = "media_previous_track"
                success = await self.core.home_assistant.call_service(domain, service, entity_id)

                if success:
                    return await self.format_response(
                        f"I've gone back to the previous track on the {friendly_name}.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't go back to the previous track on the {friendly_name}.",
                        True
                    )

            elif action == "set_volume":
                service = "volume_set"
                volume = params.get("volume")

                if not volume:
                    return await self.format_response(
                        "I need to know what volume to set.",
                        True
                    )

                # Convert to 0-1 scale if needed
                if isinstance(volume, str) and volume.endswith("%"):
                    try:
                        volume_pct = int(volume.rstrip("%"))
                        volume = volume_pct / 100
                    except ValueError:
                        volume = 0.5
                elif isinstance(volume, (int, float)) and volume > 1:
                    # Assume it's a percentage
                    volume = volume / 100

                success = await self.core.home_assistant.call_service(
                    domain, service, entity_id, service_data={"volume_level": volume}
                )

                if success:
                    return await self.format_response(
                        f"I've set the volume on the {friendly_name} to {int(volume * 100)}%.",
                        True
                    )
                else:
                    return await self.format_response(
                        f"I couldn't set the volume on the {friendly_name}.",
                        True
                    )

        else:
            return await self.format_response(
                f"I don't know how to {action} the {friendly_name}.",
                True
            )

    async def _handle_area_query(self, area_name: str, entities: List[str], context: ConversationContext) -> Dict[str, Any]:
        """Handle a query for all devices in an area"""
        # Get state for all entities
        states = []
        for entity_id in entities:
            state_info = await self.core.home_assistant.get_state(entity_id)
            if state_info:
                states.append((entity_id, state_info))

        if not states:
            return await self.format_response(
                f"I couldn't find any devices in {area_name}.",
                True
            )

        # Group by domain
        domain_states = {}
        for entity_id, state_info in states:
            domain = entity_id.split('.')[0]
            if domain not in domain_states:
                domain_states[domain] = []
            domain_states[domain].append((entity_id, state_info))

        # Build a summary
        response_parts = [f"Here's the status of devices in {area_name}:"]

        # Handle lights
        if "light" in domain_states:
            lights_on = [s[1].get("attributes", {}).get("friendly_name", s[0]) for s in domain_states["light"] if s[1].get("state") == "on"]
            lights_off = [s[1].get("attributes", {}).get("friendly_name", s[0]) for s in domain_states["light"] if s[1].get("state") == "off"]

            if lights_on:
                response_parts.append(f"Lights on: {', '.join(lights_on)}")
            if lights_off:
                response_parts.append(f"Lights off: {', '.join(lights_off)}")

        # Handle climate
        if "climate" in domain_states:
            for entity_id, state_info in domain_states["climate"]:
                friendly_name = state_info.get("attributes", {}).get("friendly_name", entity_id)
                current_temp = state_info.get("attributes", {}).get("current_temperature")
                target_temp = state_info.get("attributes", {}).get("temperature")
                hvac_mode = state_info.get("attributes", {}).get("hvac_mode", "")

                if current_temp and target_temp:
                    response_parts.append(f"The {friendly_name} is currently at {current_temp}°C and set to {target_temp}°C in {hvac_mode} mode.")

        # Add other domains as needed

        # Return the response
        return await self.format_response(
            "\n".join(response_parts),
            True
        )

    async def _handle_area_control(
            self,
            area_name: str,
            action: str,
            params: Dict[str, Any],
            entities: List[str],
            context: ConversationContext
    ) -> Dict[str, Any]:
        """Handle a control command for all devices in an area"""
        # Filter entities by type based on the action
        target_entities = []

        for entity_id in entities:
            domain = entity_id.split('.')[0]

            if action in ["turn_on", "turn_off"]:
                # These actions apply to lights, switches, media_players, etc.
                if domain in ["light", "switch", "media_player", "climate"]:
                    target_entities.append(entity_id)
            elif action == "set_temperature":
                # Only applies to climate devices
                if domain == "climate":
                    target_entities.append(entity_id)

        if not target_entities:
            return await self.format_response(
                f"I couldn't find any devices in {area_name} that I can {action}.",
                True
            )

        # Execute the action on all targets
        success_count = 0

        for entity_id in target_entities:
            domain = entity_id.split('.')[0]

            if action == "turn_on":
                service_data = {}
                # Add brightness for lights if specified
                if domain == "light" and "brightness" in params:
                    brightness = params["brightness"]
                    if isinstance(brightness, str) and brightness.endswith("%"):
                        try:
                            brightness_pct = int(brightness.rstrip("%"))
                            brightness = int((brightness_pct / 100) * 255)
                        except ValueError:
                            brightness = 255
                    elif isinstance(brightness, (int, float)) and 0 <= brightness <= 100:
                        brightness = int((brightness / 100) * 255)
                    service_data["brightness"] = brightness

                success = await self.core.home_assistant.call_service(
                    domain, "turn_on", entity_id, service_data=service_data
                )
                if success:
                    success_count += 1

            elif action == "turn_off":
                success = await self.core.home_assistant.call_service(
                    domain, "turn_off", entity_id
                )
                if success:
                    success_count += 1

            elif action == "set_temperature" and domain == "climate":
                temperature = params.get("temperature")
                if temperature:
                    success = await self.core.home_assistant.call_service(
                        domain, "set_temperature", entity_id, service_data={"temperature": temperature}
                    )
                    if success:
                        success_count += 1

        # Return response based on successes
        if success_count == len(target_entities):
            if action == "turn_on":
                brightness_info = ""
                if "brightness" in params:
                    try:
                        brightness_pct = int(params["brightness"].rstrip("%")) if isinstance(params["brightness"], str) else params["brightness"]
                        brightness_info = f" at {brightness_pct}% brightness"
                    except (ValueError, AttributeError):
                        pass

                return await self.format_response(
                    f"I've turned on all devices in {area_name}{brightness_info}.",
                    True
                )
            elif action == "turn_off":
                return await self.format_response(
                    f"I've turned off all devices in {area_name}.",
                    True
                )
            elif action == "set_temperature":
                return await self.format_response(
                    f"I've set all climate devices in {area_name} to {params.get('temperature')}°C.",
                    True
                )
        elif success_count > 0:
            return await self.format_response(
                f"I was able to {action} {success_count} out of {len(target_entities)} devices in {area_name}.",
                True
            )
        else:
            return await self.format_response(
                f"I couldn't {action} any devices in {area_name}.",
                True
            )

    async def _handle_all_devices_query(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a query for all devices"""
        # Get all states
        all_states = await self.core.home_assistant.get_states()

        if not all_states:
            return await self.format_response(
                "I couldn't get the state of any devices.",
                True
            )

        # Group by area and domain
        area_devices = {}

        for state_info in all_states:
            entity_id = state_info.get("entity_id", "")
            if not entity_id:
                continue

            # Skip non-relevant entities
            domain = entity_id.split('.')[0]
            if domain not in ["light", "switch", "climate", "media_player"]:
                continue

            # Get area
            area = state_info.get("attributes", {}).get("area", "Other")
            if area not in area_devices:
                area_devices[area] = {}

            if domain not in area_devices[area]:
                area_devices[area][domain] = []

            area_devices[area][domain].append(state_info)

        # Build a summary
        response_parts = ["Here's a summary of your smart home:"]

        for area, domains in area_devices.items():
            area_summary = []

            # Lights summary
            if "light" in domains:
                lights_on = len([s for s in domains["light"] if s.get("state") == "on"])
                total_lights = len(domains["light"])
                if lights_on > 0:
                    area_summary.append(f"{lights_on} out of {total_lights} lights are on")

            # Climate summary
            if "climate" in domains:
                for climate in domains["climate"]:
                    friendly_name = climate.get("attributes", {}).get("friendly_name", climate.get("entity_id", ""))
                    current_temp = climate.get("attributes", {}).get("current_temperature")
                    if current_temp:
                        area_summary.append(f"{friendly_name} is at {current_temp}°C")

            # Add the area summary if we have info
            if area_summary:
                response_parts.append(f"\n{area}: {'; '.join(area_summary)}")

        # If nothing meaningful was found
        if len(response_parts) == 1:
            response_parts.append("I couldn't find any interesting device states to report.")

        # Return the response
        return await self.format_response(
            "\n".join(response_parts),
            True
        )

    async def _handle_group_command(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Handle a command for a device group"""
        group_name = params.get("group", "")
        action = params.get("action", "")

        if not group_name or not action:
            return await self.format_response(
                "I need to know what group to control and what action to take.",
                True
            )

        # Get the group entities
        group_entities = await self.core.home_assistant.get_group_entities(group_name)

        if not group_entities:
            return await self.format_response(
                f"I couldn't find a device group named '{group_name}'.",
                True
            )

        # Handle the action
        if action in ["turn_on", "turn_off"]:
            success_count = 0

            for entity_id in group_entities:
                domain = entity_id.split('.')[0]
                service = action

                # Skip entities that don't support this action
                if domain not in ["light", "switch", "media_player", "climate"]:
                    continue

                service_data = {}
                # Add brightness for lights if turning on and specified
                if action == "turn_on" and domain == "light" and "brightness" in params:
                    brightness = params["brightness"]
                    if isinstance(brightness, str) and brightness.endswith("%"):
                        try:
                            brightness_pct = int(brightness.rstrip("%"))
                            brightness = int((brightness_pct / 100) * 255)
                        except ValueError:
                            brightness = 255
                    elif isinstance(brightness, (int, float)) and 0 <= brightness <= 100:
                        brightness = int((brightness / 100) * 255)
                    service_data["brightness"] = brightness

                success = await self.core.home_assistant.call_service(
                    domain, service, entity_id, service_data=service_data
                )
                if success:
                    success_count += 1

            # Return response based on successes
            if success_count > 0:
                brightness_info = ""
                if action == "turn_on" and "brightness" in params:
                    try:
                        brightness_pct = int(params["brightness"].rstrip("%")) if isinstance(params["brightness"], str) else params["brightness"]
                        brightness_info = f" at {brightness_pct}% brightness"
                    except (ValueError, AttributeError):
                        pass

                return await self.format_response(
                    f"I've turned {action.replace('_', ' ')} the {group_name} group{brightness_info}.",
                    True
                )
            else:
                return await self.format_response(
                    f"I couldn't {action.replace('_', ' ')} any devices in the {group_name} group.",
                    True
                )
        else:
            return await self.format_response(
                f"I don't know how to {action} a device group.",
                True
            )

    async def _handle_scene(self, scene_name: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle activating a scene"""
        if not scene_name:
            return await self.format_response(
                "I need to know what scene to activate.",
                True
            )

        # Find the scene entity
        scene_entity = await self.core.home_assistant.find_scene(scene_name)

        if not scene_entity:
            return await self.format_response(
                f"I couldn't find a scene named '{scene_name}'.",
                True
            )

        # Activate the scene
        success = await self.core.home_assistant.call_service(
            "scene", "turn_on", scene_entity
        )

        if success:
            return await self.format_response(
                f"I've activated the {scene_name} scene.",
                True
            )
        else:
            return await self.format_response(
                f"I couldn't activate the {scene_name} scene.",
                True
            )