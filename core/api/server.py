"""
Enhanced API Server for Morgan Core
"""
import asyncio
import json
import logging
import aiohttp
from aiohttp import web
import base64
from typing import Dict, Any, Optional, List
import time
import os
from pathlib import Path
import uuid
import functools

logger = logging.getLogger(__name__)


class APIServer:
    """HTTP API Server for Morgan Core"""

    def __init__(self, core, host="0.0.0.0", port=8000):
        self.core = core
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[self._error_middleware])
        self.active_tasks = {}  # For tracking long-running background tasks
        self.setup_routes()
        self.auth_enabled = self.core.config.get('api', {}).get('auth_enabled', False)
        self.api_tokens = self.core.config.get('api', {}).get('tokens', {})
        self.cors_origins = self.core.config.get('api', {}).get('cors_origins', ["*"])

    def setup_routes(self):
        """Configure API routes"""
        # Setup CORS
        import aiohttp_cors
        cors = aiohttp_cors.setup(self.app, defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            ) for origin in self.cors_origins
        })

        # Define routes
        routes = [
            # Core interaction endpoints
            web.post('/api/text', self.handle_text_input),
            web.post('/api/audio', self.handle_audio_input),
            web.post('/api/conversation/reset', self.reset_conversation),

            # Status and management endpoints
            web.get('/api/health', self.health_check),
            web.get('/api/status', self.get_status),
            web.post('/api/restart', self.restart_system),

            # Configuration endpoints
            web.get('/api/config', self.get_config),
            web.post('/api/config', self.update_config),

            # Voice endpoints
            web.get('/api/voices', self.get_voices),
            web.post('/api/voices', self.set_voice),

            # Task management
            web.get('/api/tasks', self.get_tasks),
            web.delete('/api/tasks/{task_id}', self.cancel_task),

            # Home Assistant integration
            web.get('/api/home-assistant/status', self.get_ha_status),
            web.get('/api/home-assistant/devices', self.get_ha_devices),
            web.get('/api/home-assistant/states', self.get_ha_states),

            # Static web UI
            web.static('/ui/', '/opt/morgan/web-ui', show_index=True)
        ]

        # Apply CORS to all routes
        for route in routes:
            self.app.router.add_route(route.method, route.path, route.handler)
            if route.method != 'OPTIONS':  # Skip OPTIONS as it's handled by CORS
                cors.add(self.app.router.add_resource(route.path))

    async def start(self):
        """Start the API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.site = web.TCPSite(runner, self.host, self.port)
        await self.site.start()
        logger.info(f"API server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the API server"""
        # Cancel any active tasks
        for task_id, task_info in list(self.active_tasks.items()):
            if not task_info['task'].done():
                task_info['task'].cancel()
                logger.info(f"Canceled task {task_id}")

        # Stop the server
        await self.site.stop()
        logger.info("API server stopped")

    @web.middleware
    async def _error_middleware(self, request, handler):
        """Middleware to handle errors"""
        try:
            return await handler(request)
        except web.HTTPException as ex:
            # Pass through HTTP exceptions (like 404, 403, etc.)
            raise
        except Exception as e:
            # Log the error and return a 500 response
            logger.exception(f"Unhandled exception in API request: {e}")
            return web.json_response({
                "error": f"Internal server error: {str(e)}"
            }, status=500)

    async def _check_auth(self, request):
        """Check API authentication"""
        if not self.auth_enabled:
            return True

        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise web.HTTPUnauthorized(reason="Missing Authorization header")

        try:
            scheme, token = auth_header.strip().split(' ', 1)
            if scheme.lower() != 'bearer':
                raise web.HTTPUnauthorized(reason="Invalid authentication scheme")

            if token not in self.api_tokens.values():
                raise web.HTTPUnauthorized(reason="Invalid token")

            return True
        except ValueError:
            raise web.HTTPUnauthorized(reason="Invalid Authorization header format")

    async def handle_text_input(self, request):
        """Handle text input API requests"""
        await self._check_auth(request)

        try:
            data = await request.json()
            text = data.get('text')
            user_id = data.get('user_id', 'default')

            if not text:
                return web.json_response({
                    "error": "Missing required field: text"
                }, status=400)

            # Process the text input
            response = await self.core.process_text_input(text, user_id)

            # If there's audio in the response, encode it as base64
            if 'audio' in response and response['audio']:
                response['audio'] = base64.b64encode(response['audio']).decode('utf-8')

            return web.json_response(response)

        except Exception as e:
            logger.exception(f"Error handling text input: {e}")
            return web.json_response({
                "error": f"Error processing request: {str(e)}"
            }, status=500)

    async def handle_audio_input(self, request):
        """Handle audio input API requests"""
        await self._check_auth(request)

        try:
            reader = await request.multipart()

            # Get user ID
            user_id = 'default'
            field = await reader.next()
            if field and field.name == 'user_id':
                user_id = await field.text()
                field = await reader.next()

            # Get audio data
            if not field or field.name != 'audio':
                return web.json_response({
                    "error": "Missing required field: audio"
                }, status=400)

            # Read audio data
            audio_data = await field.read()
            if not audio_data:
                return web.json_response({
                    "error": "Empty audio data"
                }, status=400)

            # Process the audio input
            response = await self.core.process_audio_input(audio_data, user_id)

            # If there's audio in the response, encode it as base64
            if 'audio' in response and response['audio']:
                response['audio'] = base64.b64encode(response['audio']).decode('utf-8')

            return web.json_response(response)

        except Exception as e:
            logger.exception(f"Error handling audio input: {e}")
            return web.json_response({
                "error": f"Error processing request: {str(e)}"
            }, status=500)

    async def reset_conversation(self, request):
        """Reset a conversation context"""
        await self._check_auth(request)

        try:
            data = await request.json()
            user_id = data.get('user_id', 'default')

            success = self.core.state_manager.reset_context(user_id)
            return web.json_response({
                "success": success
            })

        except Exception as e:
            logger.exception(f"Error resetting conversation: {e}")
            return web.json_response({
                "error": f"Error resetting conversation: {str(e)}"
            }, status=500)

    async def health_check(self, request):
        """API health check endpoint"""
        # This endpoint is public (no auth check)
        return web.json_response({
            "status": "ok",
            "timestamp": time.time()
        })

    async def get_status(self, request):
        """Get detailed system status"""
        await self._check_auth(request)

        try:
            system_info = await self.core.get_system_info()
            return web.json_response(system_info)
        except Exception as e:
            logger.exception(f"Error getting system status: {e}")
            return web.json_response({
                "error": f"Error getting system status: {str(e)}"
            }, status=500)

    async def restart_system(self, request):
        """Restart the system"""
        await self._check_auth(request)

        try:
            # Check if system handler allows restart
            handlers_config = self.core.config_manager.load_handlers_config()
            system_config = handlers_config.get('handlers', {}).get('system', {})
            allow_restart = system_config.get('allow_restart', True)

            if not allow_restart:
                return web.json_response({
                    "error": "System restart is not allowed by configuration"
                }, status=403)

            # Create a background task for restart
            restart_task = asyncio.create_task(self._restart_system_task())
            task_id = str(uuid.uuid4())

            self.active_tasks[task_id] = {
                "task": restart_task,
                "type": "restart",
                "created_at": time.time(),
                "status": "running"
            }

            return web.json_response({
                "success": True,
                "message": "System restart initiated",
                "task_id": task_id
            })

        except Exception as e:
            logger.exception(f"Error restarting system: {e}")
            return web.json_response({
                "error": f"Error restarting system: {str(e)}"
            }, status=500)

    async def _restart_system_task(self):
        """Background task for system restart"""
        try:
            # Wait a moment for response to be sent
            await asyncio.sleep(2)

            # Stop the core
            await self.core.stop()

            # Exit the process - systemd or Docker should restart us
            os._exit(0)
        except Exception as e:
            logger.exception(f"Error in restart task: {e}")

    async def get_config(self, request):
        """Get system configuration"""
        await self._check_auth(request)

        try:
            # Get section from query parameters
            section = request.query.get('section')

            config = self.core.config_manager.load_core_config()

            # Filter sensitive information
            if 'home_assistant' in config and 'token' in config['home_assistant']:
                # Mask the token
                token = config['home_assistant']['token']
                if token and len(token) > 8:
                    config['home_assistant']['token'] = token[:4] + '*' * (len(token) - 8) + token[-4:]

            if 'api' in config and 'tokens' in config['api']:
                # Remove tokens completely
                config['api']['tokens'] = {"redacted": True}

            if section:
                if section in config:
                    return web.json_response({section: config[section]})
                else:
                    return web.json_response({
                        "error": f"Configuration section '{section}' not found"
                    }, status=404)
            else:
                return web.json_response(config)

        except Exception as e:
            logger.exception(f"Error getting configuration: {e}")
            return web.json_response({
                "error": f"Error getting configuration: {str(e)}"
            }, status=500)

    async def update_config(self, request):
        """Update system configuration"""
        await self._check_auth(request)

        try:
            data = await request.json()
            section = data.get('section')
            key = data.get('key')
            value = data.get('value')

            if not section or not key:
                return web.json_response({
                    "error": "Missing required fields: section and key"
                }, status=400)

            config = self.core.config_manager.load_core_config()

            if section not in config:
                return web.json_response({
                    "error": f"Configuration section '{section}' not found"
                }, status=404)

            # Handle nested keys (e.g., services.llm.model)
            keys = key.split('.')
            current = config[section]

            # Navigate to the nested dictionary
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the value
            current[keys[-1]] = value

            # Save the configuration
            success = self.core.config_manager.save_config("core", config)

            if success:
                return web.json_response({
                    "success": True,
                    "message": f"Configuration updated: {section}.{key} = {value}"
                })
            else:
                return web.json_response({
                    "error": "Failed to save configuration"
                }, status=500)

        except Exception as e:
            logger.exception(f"Error updating configuration: {e}")
            return web.json_response({
                "error": f"Error updating configuration: {str(e)}"
            }, status=500)

    async def get_voices(self, request):
        """Get available TTS voices"""
        await self._check_auth(request)

        try:
            voices = await self.core.tts_service.get_available_voices()
            return web.json_response({"voices": voices})
        except Exception as e:
            logger.exception(f"Error getting voices: {e}")
            return web.json_response({
                "error": f"Error getting voices: {str(e)}"
            }, status=500)

    async def set_voice(self, request):
        """Set voice preference for a user"""
        await self._check_auth(request)

        try:
            data = await request.json()
            user_id = data.get('user_id', 'default')
            voice_id = data.get('voice_id')

            if not voice_id:
                return web.json_response({
                    "error": "Missing required field: voice_id"
                }, status=400)

            # Get available voices to validate
            voices = await self.core.tts_service.get_available_voices()

            if voice_id not in voices:
                return web.json_response({
                    "error": f"Voice '{voice_id}' not found"
                }, status=404)

            # Get the user's context
            context = self.core.state_manager.get_context(user_id)

            # Set the voice preference
            context.set_variable("voice_preference", voice_id)

            # Get the voice description
            voice_info = voices.get(voice_id, {})
            description = voice_info.get("description", "No description")

            return web.json_response({
                "success": True,
                "message": f"Voice preference set to {voice_id}: {description}",
                "voice_id": voice_id,
                "description": description
            })

        except Exception as e:
            logger.exception(f"Error setting voice preference: {e}")
            return web.json_response({
                "error": f"Error setting voice preference: {str(e)}"
            }, status=500)

    async def get_tasks(self, request):
        """Get active tasks"""
        await self._check_auth(request)

        try:
            # Update task status
            for task_id, task_info in list(self.active_tasks.items()):
                task = task_info['task']
                if task.done():
                    if task.cancelled():
                        task_info['status'] = "cancelled"
                    elif task.exception():
                        task_info['status'] = "failed"
                        task_info['error'] = str(task.exception())
                    else:
                        task_info['status'] = "completed"
                        if task.result():
                            task_info['result'] = task.result()

                    # Remove completed tasks older than 1 hour
                    if time.time() - task_info['created_at'] > 3600:
                        del self.active_tasks[task_id]

            # Format response
            tasks = []
            for task_id, task_info in self.active_tasks.items():
                tasks.append({
                    "id": task_id,
                    "type": task_info['type'],
                    "status": task_info['status'],
                    "created_at": task_info['created_at'],
                    "error": task_info.get('error'),
                    "result": task_info.get('result')
                })

            return web.json_response({"tasks": tasks})

        except Exception as e:
            logger.exception(f"Error getting tasks: {e}")
            return web.json_response({
                "error": f"Error getting tasks: {str(e)}"
            }, status=500)

    async def cancel_task(self, request):
        """Cancel a running task"""
        await self._check_auth(request)

        try:
            task_id = request.match_info.get('task_id')

            if task_id not in self.active_tasks:
                return web.json_response({
                    "error": f"Task '{task_id}' not found"
                }, status=404)

            task_info = self.active_tasks[task_id]
            task = task_info['task']

            if task.done():
                return web.json_response({
                    "error": f"Task '{task_id}' is already completed"
                }, status=400)

            # Cancel the task
            task.cancel()
            task_info['status'] = "cancelled"

            return web.json_response({
                "success": True,
                "message": f"Task '{task_id}' cancelled"
            })

        except Exception as e:
            logger.exception(f"Error cancelling task: {e}")
            return web.json_response({
                "error": f"Error cancelling task: {str(e)}"
            }, status=500)

    async def get_ha_status(self, request):
        """Get Home Assistant connection status"""
        await self._check_auth(request)

        try:
            if not self.core.home_assistant:
                return web.json_response({
                    "connected": False,
                    "message": "Home Assistant integration is not configured"
                })

            return web.json_response({
                "connected": self.core.home_assistant.connected,
                "url": self.core.home_assistant.url,
                "device_count": len(self.core.home_assistant.device_registry),
                "entity_count": len(self.core.home_assistant.entity_registry),
                "area_count": len(self.core.home_assistant.area_registry)
            })

        except Exception as e:
            logger.exception(f"Error getting Home Assistant status: {e}")
            return web.json_response({
                "error": f"Error getting Home Assistant status: {str(e)}"
            }, status=500)

    async def get_ha_devices(self, request):
        """Get Home Assistant devices and entities"""
        await self._check_auth(request)

        try:
            if not self.core.home_assistant:
                return web.json_response({
                    "error": "Home Assistant integration is not configured"
                }, status=404)

            if not self.core.home_assistant.connected:
                return web.json_response({
                    "error": "Home Assistant is not connected"
                }, status=503)

            # Get areas
            areas = {}
            for area_id, area in self.core.home_assistant.area_registry.items():
                areas[area_id] = area

            # Get devices
            devices = {}
            for device_id, device in self.core.home_assistant.device_registry.items():
                area_id = device.get("area_id")
                area_name = areas.get(area_id, {}).get("name", "No Area") if area_id else "No Area"

                devices[device_id] = {
                    "id": device_id,
                    "name": device.get("name", "Unknown"),
                    "area_id": area_id,
                    "area_name": area_name,
                    "model": device.get("model"),
                    "manufacturer": device.get("manufacturer"),
                    "entities": []
                }

            # Get entities and map to devices
            entities = {}
            for entity_id, entity in self.core.home_assistant.entity_registry.items():
                device_id = entity.get("device_id")

                entity_info = {
                    "id": entity_id,
                    "name": entity.get("name", entity_id),
                    "device_id": device_id,
                    "domain": entity_id.split(".")[0] if "." in entity_id else "",
                    "disabled": entity.get("disabled", False),
                    "area_id": entity.get("area_id")
                }

                entities[entity_id] = entity_info

                # Add to device if available
                if device_id in devices:
                    devices[device_id]["entities"].append(entity_info)

            # Format response grouped by areas
            area_data = {}
            for area_id, area in areas.items():
                area_data[area_id] = {
                    "id": area_id,
                    "name": area.get("name", "Unknown"),
                    "devices": [],
                    "entities": []
                }

            # Add devices to areas
            for device_id, device in devices.items():
                area_id = device.get("area_id")
                if area_id and area_id in area_data:
                    area_data[area_id]["devices"].append(device)

            # Add entities directly associated with areas
            for entity_id, entity in entities.items():
                area_id = entity.get("area_id")
                if area_id and area_id in area_data:
                    # Only add if not already included via a device
                    if not entity.get("device_id"):
                        area_data[area_id]["entities"].append(entity)

            return web.json_response({
                "areas": list(area_data.values()),
                "devices": list(devices.values()),
                "entities": list(entities.values())
            })

        except Exception as e:
            logger.exception(f"Error getting Home Assistant devices: {e}")
            return web.json_response({
                "error": f"Error getting Home Assistant devices: {str(e)}"
            }, status=500)

    async def get_ha_states(self, request):
        """Get Home Assistant entity states"""
        await self._check_auth(request)

        try:
            if not self.core.home_assistant:
                return web.json_response({
                    "error": "Home Assistant integration is not configured"
                }, status=404)

            if not self.core.home_assistant.connected:
                return web.json_response({
                    "error": "Home Assistant is not connected"
                }, status=503)

            # Get entity ID from query parameters
            entity_id = request.query.get('entity_id')
            domain = request.query.get('domain')

            if entity_id:
                # Get state for a specific entity
                state = await self.core.home_assistant.get_state(entity_id)
                if not state:
                    return web.json_response({
                        "error": f"Entity '{entity_id}' not found"
                    }, status=404)

                return web.json_response(state)
            else:
                # Get all states or filter by domain
                states = await self.core.home_assistant.get_states()

                if domain:
                    states = [state for state in states if state.get("entity_id", "").startswith(f"{domain}.")]

                return web.json_response(states)

        except Exception as e:
            logger.exception(f"Error getting Home Assistant states: {e}")
            return web.json_response({
                "error": f"Error getting Home Assistant states: {str(e)}"
            }, status=500)