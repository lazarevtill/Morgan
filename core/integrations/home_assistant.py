"""
Home Assistant integration for Morgan Core
"""
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

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
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Home Assistant"""
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
        if self.session is None:
            await self.connect()

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

        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                logger.error(f"Home Assistant service call error: {response.status} - {error_text}")
                return False

            return True

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the current state of an entity

        Args:
            entity_id: Entity ID to query

        Returns:
            Entity state information
        """
        if self.session is None:
            await self.connect()

        url = f"{self.url}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with self.session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Home Assistant state query error: {response.status} - {error_text}")
                return {}

            return await response.json()

    async def get_states(self) -> List[Dict[str, Any]]:
        """
        Get states of all entities

        Returns:
            List of all entity states
        """
        if self.session is None:
            await self.connect()

        url = f"{self.url}/api/states"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with self.session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Home Assistant states query error: {response.status} - {error_text}")
                return []

            return await response.json()

    async def resolve_entity(self, entity_reference: str) -> str:
        """
        Resolve an entity reference to an entity ID

        Args:
            entity_reference: Entity reference, which could be an ID, alias, or group

        Returns:
            Resolved entity ID
        """
        # Check if it's already a valid entity ID
        if entity_reference in self.entity_registry:
            return entity_reference

        # Check if it's an alias
        if entity_reference in self.device_aliases:
            return self.device_aliases[entity_reference]

        # Check if it's a device group
        if entity_reference in self.device_groups:
            # Return the first entity in the group
            # A more sophisticated implementation might handle this differently
            return self.device_groups[entity_reference][0]

        # Try to find a partial match
        for entity_id in self.entity_registry:
            if entity_reference.lower() in entity_id.lower():
                return entity_id

        # No match found
        return None

    async def _test_connection(self):
        """Test the connection to Home Assistant"""
        url = f"{self.url}/api/"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with self.session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Home Assistant connection test failed: {response.status} - {error_text}")

    async def _load_registries(self):
        """Load device and entity registries from Home Assistant"""
        # Get entity registry
        url = f"{self.url}/api/config/entity_registry"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                entities = await response.json()
                for entity in entities:
                    self.entity_registry[entity["entity_id"]] = entity
            else:
                logger.error(f"Failed to load entity registry: {response.status}")

        # Get device registry
        url = f"{self.url}/api/config/device_registry"

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                devices = await response.json()
                for device in devices:
                    self.device_registry[device["id"]] = device
            else:
                logger.error(f"Failed to load device registry: {response.status}")

        # Get area registry
        url = f"{self.url}/api/config/area_registry"

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                areas = await response.json()
                for area in areas:
                    self.area_registry[area["id"]] = area
            else:
                logger.error(f"Failed to load area registry: {response.status}")

    async def _load_device_mappings(self):
        """Load device groups and aliases from configuration"""
        # In a real implementation, this would load from a configuration file
        # For now, we'll use some example mappings

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

        self.device_aliases = {
            "tv": "media_player.living_room_tv",
            "main_lights": "light.living_room",
            "coffee": "switch.coffee_maker"
        }

    async def _start_ws_client(self):
        """Start WebSocket client for real-time updates"""
        ws_url = f"{self.url}/api/websocket"

        while True:
            try:
                async with self.session.ws_connect(ws_url) as ws:
                    self.ws_client = ws

                    # Authentication
                    auth_msg = await ws.receive_json()

                    if auth_msg["type"] == "auth_required":
                        await ws.send_json({"type": "auth", "access_token": self.token})
                        auth_result = await ws.receive_json()

                        if auth_result["type"] != "auth_ok":
                            logger.error(f"WebSocket authentication failed: {auth_result}")
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
                            if data.get("type") == "event" and data.get("event", {}).get(
                                    "event_type") == "state_changed":
                                # Handle state change
                                # In a real implementation, you might want to notify subscribers
                                pass
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")

            # Reconnect after a delay
            self.ws_client = None
            await asyncio.sleep(10)