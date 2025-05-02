"""
Enhanced Home Assistant integration for Morgan Core
"""
import aiohttp
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class HomeAssistantIntegration:
    """Integration with Home Assistant for smart home control"""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.session = None
        self.ws_client = None
        self.device_registry = {}
        self.entity_registry = {}
        self.area_registry = {}
        self.device_groups = {}
        self.device_aliases = {}
        self.scenes = {}
        self.reconnect_interval = 10
        self.connected = False
        self.state_listeners = []

    async def connect(self):
        """Connect to Home Assistant and initialize data"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

        try:
            # Test API connection
            await self._test_connection()

            # Load device and entity information
            await self._load_registries()

            # Load device groups and aliases from config
            await self._load_device_mappings()

            # Start WebSocket connection for state updates
            asyncio.create_task(self._start_ws_client())

            logger.info("Connected to Home Assistant successfully")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from Home Assistant"""
        self.connected = False

        if self.ws_client:
            await self.ws_client.close()
            self.ws_client = None

        if self.session:
            await self.session.close()
            self.session = None

    async def call_service(
            self,
            domain: str,
            service: str,
            entity_id: Optional[str] = None,
            target: Optional[Dict[str, Any]] = None,
            service_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Call a Home Assistant service

        Args:
            domain: Service domain (e.g., 'light', 'switch')
            service: Service name (e.g., 'turn_on', 'turn_off')
            entity_id: Optional entity ID to target
            target: Optional target specification
            service_data: Optional service data

        Returns:
            Success status
        """
        if not self.connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Failed to reconnect to Home Assistant: {e}")
                return False

        if self.session is None:
            logger.error("No active session for Home Assistant")
            return False

        # Prepare the service call data
        data = service_data or {}

        # Add entity_id to data if provided
        if entity_id:
            if target:
                target["entity_id"] = entity_id
            else:
                target = {"entity_id": entity_id}

        # Prepare the final payload
        payload = {
            "type": "call_service",
            "domain": domain,
            "service": service
        }

        if target:
            payload["target"] = target

        if data:
            payload["service_data"] = data

        # Send request to Home Assistant
        url = f"{self.url}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=10) as response:
                if response.status < 200 or response.status >= 300:
                    error_text = await response.text()
                    logger.error(f"Home Assistant service call error: {response.status} - {error_text}")
                    return False

                return True
        except aiohttp.ClientError as e:
            logger.error(f"Network error when calling Home Assistant service: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error when calling Home Assistant service: {e}")
            return False

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the current state of an entity

        Args:
            entity_id: Entity ID to query

        Returns:
            Entity state information
        """
        if not self.connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Failed to reconnect to Home Assistant: {e}")
                return {}

        if self.session is None:
            logger.error("No active session for Home Assistant")
            return {}

        url = f"{self.url}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Home Assistant state query error: {response.status} - {error_text}")
                    return {}

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Network error when querying Home Assistant state: {e}")
            self.connected = False
            return {}
        except Exception as e:
            logger.error(f"Unexpected error when querying Home Assistant state: {e}")
            return {}

    async def get_states(self) -> List[Dict[str, Any]]:
        """
        Get states of all entities

        Returns:
            List of all entity states
        """
        if not self.connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Failed to reconnect to Home Assistant: {e}")
                return []

        if self.session is None:
            logger.error("No active session for Home Assistant")
            return []

        url = f"{self.url}/api/states"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Home Assistant states query error: {response.status} - {error_text}")
                    return []

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Network error when querying Home Assistant states: {e}")
            self.connected = False
            return []
        except Exception as e:
            logger.error(f"Unexpected error when querying Home Assistant states: {e}")
            return []

    async def resolve_entity(self, entity_reference: str) -> Optional[str]:
        """
        Resolve an entity reference to an entity ID

        Args:
            entity_reference: Entity reference, which could be an ID, alias, or group

        Returns:
            Resolved entity ID or None if not found
        """
        if not entity_reference:
            return None

        # Normalize reference for comparison
        normalized_ref = entity_reference.lower().strip()

        # Check if it's already a valid entity ID
        if normalized_ref in [entity_id.lower() for entity_id in self.entity_registry]:
            for entity_id in self.entity_registry:
                if entity_id.lower() == normalized_ref:
                    return entity_id

        # Check if it's an alias
        for alias, entity_id in self.device_aliases.items():
            if alias.lower() == normalized_ref:
                return entity_id

        # Check if it's a device group
        for group_name, entities in self.device_groups.items():
            if group_name.lower() == normalized_ref and entities:
                # Return the first entity in the group
                return entities[0]

        # Try to find a partial match by friendly name
        all_states = await self.get_states()
        for state in all_states:
            entity_id = state.get("entity_id")
            friendly_name = state.get("attributes", {}).get("friendly_name", "")

            if friendly_name and friendly_name.lower() == normalized_ref:
                return entity_id

            # Check for partial match in friendly name
            if friendly_name and normalized_ref in friendly_name.lower():
                return entity_id

        # Try to find a partial match in entity ID
        for entity_id in self.entity_registry:
            if normalized_ref in entity_id.lower():
                return entity_id

        # No match found
        return None

    async def get_entities_for_area(self, area_name: str) -> List[str]:
        """
        Get all entities for a specific area/room

        Args:
            area_name: Name of the area/room

        Returns:
            List of entity IDs in the area
        """
        # Normalize area name
        normalized_area = area_name.lower().strip()

        # Find area ID by name
        area_id = None
        for id, area in self.area_registry.items():
            if area.get("name", "").lower() == normalized_area:
                area_id = id
                break

        if not area_id:
            # Try partial match
            for id, area in self.area_registry.items():
                if normalized_area in area.get("name", "").lower():
                    area_id = id
                    break

        if not area_id:
            return []

        # Find all entities in this area
        entities = []

        # Check device-entity relationships
        for entity_id, entity in self.entity_registry.items():
            device_id = entity.get("device_id")
            if device_id and device_id in self.device_registry:
                device = self.device_registry[device_id]
                if device.get("area_id") == area_id:
                    entities.append(entity_id)

        # Check entity area IDs directly
        for entity_id, entity in self.entity_registry.items():
            if entity.get("area_id") == area_id and entity_id not in entities:
                entities.append(entity_id)

        return entities

    async def get_group_entities(self, group_name: str) -> List[str]:
        """
        Get all entities in a device group

        Args:
            group_name: Name of the device group

        Returns:
            List of entity IDs in the group
        """
        # Normalize group name
        normalized_group = group_name.lower().strip()

        # Check exact match
        for name, entities in self.device_groups.items():
            if name.lower() == normalized_group:
                return entities

        # Check partial match
        for name, entities in self.device_groups.items():
            if normalized_group in name.lower():
                return entities

        # Check for Home Assistant groups
        group_entity_id = f"group.{normalized_group.replace(' ', '_')}"
        group_state = await self.get_state(group_entity_id)

        if group_state:
            entities = group_state.get("attributes", {}).get("entity_id", [])
            if isinstance(entities, list):
                return entities

        return []

    async def find_scene(self, scene_name: str) -> Optional[str]:
        """
        Find a scene by name

        Args:
            scene_name: Name of the scene

        Returns:
            Scene entity ID or None if not found
        """
        # Normalize scene name
        normalized_scene = scene_name.lower().strip()

        # Check in cached scenes
        for scene_id, scene_info in self.scenes.items():
            if scene_info.get("name", "").lower() == normalized_scene:
                return scene_id

        # Fetch all scenes from Home Assistant
        all_states = await self.get_states()
        scene_entities = {}

        for state in all_states:
            entity_id = state.get("entity_id", "")
            if entity_id.startswith("scene."):
                friendly_name = state.get("attributes", {}).get("friendly_name", "")
                scene_entities[entity_id] = friendly_name

        # Update cache
        self.scenes = {entity_id: {"name": name} for entity_id, name in scene_entities.items()}

        # Look for exact match
        for entity_id, name in scene_entities.items():
            if name.lower() == normalized_scene:
                return entity_id

        # Look for partial match
        for entity_id, name in scene_entities.items():
            if normalized_scene in name.lower():
                return entity_id

        # Try entity ID match
        scene_entity_id = f"scene.{normalized_scene.replace(' ', '_')}"
        if scene_entity_id in scene_entities:
            return scene_entity_id

        return None

    async def register_state_listener(self, callback):
        """
        Register a callback for state change events

        Args:
            callback: Async function to call with state change information
        """
        if callback not in self.state_listeners:
            self.state_listeners.append(callback)

    async def unregister_state_listener(self, callback):
        """
        Unregister a state change callback

        Args:
            callback: Previously registered callback
        """
        if callback in self.state_listeners:
            self.state_listeners.remove(callback)

    async def _test_connection(self):
        """Test the connection to Home Assistant"""
        url = f"{self.url}/api/"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Home Assistant connection test failed: {response.status} - {error_text}")

                # Get API version
                result = await response.json()
                version = result.get("version", "unknown")
                logger.info(f"Connected to Home Assistant version {version}")
        except Exception as e:
            logger.error(f"Home Assistant connection test failed: {e}")
            raise

    async def _load_registries(self):
        """Load device and entity registries from Home Assistant"""
        # Get entity registry
        url = f"{self.url}/api/config/entity_registry"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    entities = await response.json()
                    self.entity_registry = {entity["entity_id"]: entity for entity in entities}
                    logger.info(f"Loaded {len(self.entity_registry)} entities from registry")
                else:
                    logger.error(f"Failed to load entity registry: {response.status}")
        except Exception as e:
            logger.error(f"Error loading entity registry: {e}")

        # Get device registry
        url = f"{self.url}/api/config/device_registry"

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    devices = await response.json()
                    self.device_registry = {device["id"]: device for device in devices}
                    logger.info(f"Loaded {len(self.device_registry)} devices from registry")
                else:
                    logger.error(f"Failed to load device registry: {response.status}")
        except Exception as e:
            logger.error(f"Error loading device registry: {e}")

        # Get area registry
        url = f"{self.url}/api/config/area_registry"

        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    areas = await response.json()
                    self.area_registry = {area["id"]: area for area in areas}
                    logger.info(f"Loaded {len(self.area_registry)} areas from registry")
                else:
                    logger.error(f"Failed to load area registry: {response.status}")
        except Exception as e:
            logger.error(f"Error loading area registry: {e}")

        # Load scenes
        all_states = await self.get_states()
        scene_entities = {}

        for state in all_states:
            entity_id = state.get("entity_id", "")
            if entity_id.startswith("scene."):
                friendly_name = state.get("attributes", {}).get("friendly_name", "")
                scene_entities[entity_id] = friendly_name

        self.scenes = {entity_id: {"name": name} for entity_id, name in scene_entities.items()}
        logger.info(f"Loaded {len(self.scenes)} scenes")

    async def _load_device_mappings(self):
        """Load device groups and aliases from configuration"""
        # In a full implementation, this would load from a configuration file
        # For now, we'll use some example mappings

        try:
            # Load device groups from Home Assistant
            all_states = await self.get_states()
            group_entities = {}

            for state in all_states:
                entity_id = state.get("entity_id", "")
                if entity_id.startswith("group."):
                    members = state.get("attributes", {}).get("entity_id", [])
                    if isinstance(members, list) and members:
                        friendly_name = state.get("attributes", {}).get("friendly_name", entity_id.split(".")[1])
                        group_entities[friendly_name] = members

            # Merge with predefined groups
            self.device_groups = {
                "living_room": [
                    "light.living_room",
                    "media_player.living_room_tv",
                    "climate.living_room"
                ],
                "kitchen": [
                    "light.kitchen",
                    "switch.coffee_maker"
                ]
            }

            # Add groups from Home Assistant
            self.device_groups.update(group_entities)

            # Device aliases mapping
            self.device_aliases = {
                "tv": "media_player.living_room_tv",
                "main_lights": "light.living_room",
                "coffee": "switch.coffee_maker"
            }

            # Add aliases based on friendly names
            for state in all_states:
                entity_id = state.get("entity_id", "")
                friendly_name = state.get("attributes", {}).get("friendly_name")

                if friendly_name and entity_id:
                    domain = entity_id.split('.')[0]
                    if domain in ["light", "switch", "climate", "media_player", "sensor"]:
                        # Add simple alias if it doesn't already exist
                        simple_alias = friendly_name.lower().replace(" ", "_")
                        if simple_alias not in self.device_aliases:
                            self.device_aliases[simple_alias] = entity_id

            logger.info(f"Loaded {len(self.device_groups)} device groups and {len(self.device_aliases)} device aliases")

        except Exception as e:
            logger.error(f"Error loading device mappings: {e}")

    async def _start_ws_client(self):
        """Start WebSocket client for real-time updates"""
        ws_url = f"{self.url}/api/websocket"

        while self.session and not self.session.closed:
            try:
                async with self.session.ws_connect(ws_url, timeout=30) as ws:
                    self.ws_client = ws
                    self.connected = True
                    logger.info("WebSocket connection to Home Assistant established")

                    # Authentication
                    auth_msg = await ws.receive_json()

                    if auth_msg["type"] == "auth_required":
                        await ws.send_json({"type": "auth", "access_token": self.token})
                        auth_result = await ws.receive_json()

                        if auth_result["type"] != "auth_ok":
                            logger.error(f"WebSocket authentication failed: {auth_result}")
                            self.connected = False
                            break

                    # Subscribe to state changes
                    await ws.send_json({
                        "id": 1,
                        "type": "subscribe_events",
                        "event_type": "state_changed"
                    })

                    # Message handling loop
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)

                            # Process state change events
                            if data.get("type") == "event" and data.get("event", {}).get("event_type") == "state_changed":
                                event_data = data.get("event", {}).get("data", {})
                                entity_id = event_data.get("entity_id")
                                new_state = event_data.get("new_state")
                                old_state = event_data.get("old_state")

                                # Notify listeners
                                for listener in self.state_listeners:
                                    try:
                                        await listener(entity_id, new_state, old_state)
                                    except Exception as e:
                                        logger.error(f"Error in state listener: {e}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            self.connected = False
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket connection closed")
                            self.connected = False
                            break
            except aiohttp.ClientConnectorError as e:
                logger.error(f"WebSocket connection error: {e}")
                self.connected = False
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False

            # Reconnect after a delay
            self.ws_client = None
            logger.info(f"Reconnecting to Home Assistant in {self.reconnect_interval} seconds")
            await asyncio.sleep(self.reconnect_interval)

        logger.warning("WebSocket client stopped")