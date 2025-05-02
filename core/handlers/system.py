"""
Enhanced system command handler for Morgan
"""
import os
import platform
import psutil
import subprocess
import asyncio
import shutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_handler import BaseHandler
from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class SystemHandler(BaseHandler):
    """Handler for system commands"""

    def __init__(self, core_instance):
        super().__init__(core_instance)
        # Load system handler configuration
        try:
            handlers_config = self.core.config_manager.load_handlers_config()
            system_config = handlers_config.get('handlers', {}).get('system', {})
            self.allow_restart = system_config.get('allow_restart', True)
            self.allow_update = system_config.get('allow_update', True)
        except Exception as e:
            logger.error(f"Error loading system handler config: {e}")
            self.allow_restart = True
            self.allow_update = True

    async def handle(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Handle a system command

        Args:
            params: Command parameters
            context: Conversation context

        Returns:
            Response dictionary
        """
        command = params.get("command", "")

        # Process based on command
        if command == "status":
            return await self._handle_status(context)
        elif command == "restart":
            return await self._handle_restart(context)
        elif command == "update":
            return await self._handle_update(context)
        elif command == "debug":
            return await self._handle_debug(context)
        elif command == "help":
            return await self._handle_help(context)
        elif command == "config":
            return await self._handle_config(params, context)
        elif command == "voice":
            return await self._handle_voice_config(params, context)
        else:
            return await self.format_response(
                "I'm not sure what system command you want me to execute. Available commands are: status, restart, update, debug, help, config, and voice.",
                True
            )

    async def _handle_status(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a status command"""
        # Get system information
        system_info = await self.core.get_system_info()

        # Format response
        response = (
            f"System Status:\n"
            f"Version: {system_info['version']}\n"
            f"Uptime: {system_info['uptime']}\n"
            f"CPU Usage: {system_info['cpu_percent']}%\n"
            f"Memory Usage: {system_info['memory_percent']}%\n"
            f"Disk Usage: {system_info['disk_percent']}%\n\n"
            f"Platform: {system_info['platform']}\n"
            f"Python: {system_info['python_version']}\n\n"
            f"Services:\n"
            f"LLM Service: {'Connected' if system_info['service_status']['llm'] else 'Disconnected'}\n"
            f"TTS Service: {'Connected' if system_info['service_status']['tts'] else 'Disconnected'}\n"
            f"STT Service: {'Connected' if system_info['service_status']['stt'] else 'Disconnected'}\n"
            f"Home Assistant: {'Connected' if system_info['service_status']['home_assistant'] else 'Disconnected or Not Configured'}\n\n"
            f"Morgan is running normally."
        )

        return await self.format_response(response, True)

    async def _handle_restart(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a restart command"""
        if not self.allow_restart:
            return await self.format_response(
                "I'm sorry, but I'm not allowed to restart the system. Please contact the administrator if you need to restart Morgan.",
                True
            )

        # Schedule the restart
        asyncio.create_task(self._restart_system())

        return await self.format_response(
            "I'll restart now. This will take a few moments.",
            True,
            [{"type": "restart", "delay": 5}]
        )

    async def _handle_update(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle an update command"""
        if not self.allow_update:
            return await self.format_response(
                "I'm sorry, but I'm not allowed to update the system. Please contact the administrator if you need to update Morgan.",
                True
            )

        # Schedule the update
        update_task = asyncio.create_task(self._update_system())

        return await self.format_response(
            "I'll check for updates now. This might take a few minutes and may require a restart when complete.",
            True,
            [{"type": "update", "background": True, "task_id": id(update_task)}]
        )

    async def _handle_debug(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a debug command"""
        # Get detailed system information
        import json

        # System info
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "hostname": platform.node()
        }

        # Process info
        process = psutil.Process(os.getpid())
        process_info = {
            "pid": process.pid,
            "create_time": datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
            "status": process.status(),
            "cpu_percent": process.cpu_percent(),
            "memory_info": {
                "rss": process.memory_info().rss / (1024 * 1024),  # MB
                "vms": process.memory_info().vms / (1024 * 1024)   # MB
            },
            "threads": len(process.threads()),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }

        # Service status
        service_info = {
            "llm_service": {
                "connected": self.core.llm_service.session is not None,
                "model": self.core.llm_service.model_name,
                "url": self.core.llm_service.service_url
            },
            "tts_service": {
                "connected": self.core.tts_service.session is not None,
                "default_voice": self.core.tts_service.default_voice,
                "url": self.core.tts_service.service_url
            },
            "stt_service": {
                "connected": self.core.stt_service.session is not None,
                "model": self.core.stt_service.model_name,
                "url": self.core.stt_service.service_url
            }
        }

        if self.core.home_assistant:
            service_info["home_assistant"] = {
                "connected": self.core.home_assistant.connected,
                "url": self.core.home_assistant.url,
                "device_count": len(self.core.home_assistant.device_registry),
                "entity_count": len(self.core.home_assistant.entity_registry),
                "area_count": len(self.core.home_assistant.area_registry)
            }

        # Conversation data
        context_count = len(self.core.state_manager.contexts)
        active_contexts = [
            {
                "user_id": user_id,
                "message_count": len(context.history),
                "created_at": datetime.fromtimestamp(context.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.fromtimestamp(context.last_updated_at).strftime("%Y-%m-%d %H:%M:%S")
            }
            for user_id, context in self.core.state_manager.contexts.items()
        ]

        debug_info = {
            "system": system_info,
            "process": process_info,
            "services": service_info,
            "conversations": {
                "count": context_count,
                "active_contexts": active_contexts
            }
        }

        # Format the debug information
        formatted_debug = json.dumps(debug_info, indent=2)

        return await self.format_response(
            f"Debug Information:\n```json\n{formatted_debug}\n```\n\nThis debug information may be useful for troubleshooting.",
            True,
            [{"type": "debug_info", "data": debug_info}]
        )

    async def _handle_help(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a help command"""
        help_text = """
        Morgan System Commands:
        
        - status: Show system status including CPU, memory, disk usage, and service status.
        - restart: Restart Morgan services (if enabled).
        - update: Check for and apply updates (if enabled).
        - debug: Show detailed debug information for troubleshooting.
        - help: Show this help message.
        - config: View or modify system configuration.
        - voice: Configure voice settings.
        
        You can also ask me to control your smart home devices, provide information, 
        or just have a general conversation.
        """

        return await self.format_response(help_text, True)

    async def _handle_config(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Handle a configuration command"""
        action = params.get("action", "show")
        section = params.get("section", "")
        key = params.get("key", "")
        value = params.get("value", "")

        if action == "show":
            if not section:
                # Show available configuration sections
                return await self.format_response(
                    "Available configuration sections: system, services, home_assistant, api\n" +
                    "Use 'config show [section]' to view a specific section.",
                    True
                )
            else:
                # Show configuration for a specific section
                try:
                    config = self.core.config_manager.load_core_config()
                    if section in config:
                        import json
                        section_config = json.dumps(config[section], indent=2)
                        return await self.format_response(
                            f"{section} configuration:\n```json\n{section_config}\n```",
                            True
                        )
                    else:
                        return await self.format_response(
                            f"Configuration section '{section}' not found.",
                            True
                        )
                except Exception as e:
                    logger.error(f"Error loading configuration: {e}")
                    return await self.format_response(
                        f"Error loading configuration: {str(e)}",
                        True
                    )
        elif action == "set":
            # This is simplified - in a real implementation, you would want more validation
            # and proper handling of different value types
            if not section or not key:
                return await self.format_response(
                    "Please specify both a section and a key to set a configuration value.",
                    True
                )

            try:
                config = self.core.config_manager.load_core_config()

                if section not in config:
                    return await self.format_response(
                        f"Configuration section '{section}' not found.",
                        True
                    )

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
                    return await self.format_response(
                        f"Configuration updated: {section}.{key} = {value}",
                        True
                    )
                else:
                    return await self.format_response(
                        "Failed to save configuration.",
                        True
                    )
            except Exception as e:
                logger.error(f"Error updating configuration: {e}")
                return await self.format_response(
                    f"Error updating configuration: {str(e)}",
                    True
                )
        else:
            return await self.format_response(
                f"Unknown configuration action: {action}. Available actions: show, set",
                True
            )

    async def _handle_voice_config(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Handle voice configuration command"""
        action = params.get("action", "show")
        voice_id = params.get("voice_id", "")

        if action == "show":
            try:
                # Get list of available voices
                voices = await self.core.tts_service.get_available_voices()

                # Format the response
                if not voices:
                    return await self.format_response(
                        "No voices available.",
                        True
                    )

                # Get current voice preference for this user
                current_voice = context.get_variable("voice_preference", self.core.tts_service.default_voice)

                response_parts = ["Available voices:"]
                for voice_id, voice_info in voices.items():
                    current_marker = " (current)" if voice_id == current_voice else ""
                    description = voice_info.get("description", "No description")
                    response_parts.append(f"- {voice_id}: {description}{current_marker}")

                return await self.format_response(
                    "\n".join(response_parts),
                    True
                )
            except Exception as e:
                logger.error(f"Error getting available voices: {e}")
                return await self.format_response(
                    f"Error getting available voices: {str(e)}",
                    True
                )
        elif action == "set":
            if not voice_id:
                return await self.format_response(
                    "Please specify a voice_id to set as your preference.",
                    True
                )

            try:
                # Get list of available voices to validate
                voices = await self.core.tts_service.get_available_voices()

                if voice_id not in voices:
                    return await self.format_response(
                        f"Voice '{voice_id}' not found. Use 'voice show' to see available voices.",
                        True
                    )

                # Set the voice preference for this user
                context.set_variable("voice_preference", voice_id)

                # Get the voice description
                voice_info = voices.get(voice_id, {})
                description = voice_info.get("description", "No description")

                # Generate a test message with the new voice
                test_message = f"I've set my voice to {voice_id}. This is how I sound now."
                audio_data = await self.core.tts_service.generate_speech(test_message, voice_id)

                return await self.format_response(
                    f"Voice preference set to {voice_id}: {description}. I'll use this voice for future responses.",
                    True,
                    [{"type": "voice_set", "voice_id": voice_id}],
                    audio_data
                )
            except Exception as e:
                logger.error(f"Error setting voice preference: {e}")
                return await self.format_response(
                    f"Error setting voice preference: {str(e)}",
                    True
                )
        elif action == "reset":
            # Reset to default voice
            context.set_variable("voice_preference", None)

            return await self.format_response(
                f"Voice preference reset to system default.",
                True
            )
        else:
            return await self.format_response(
                f"Unknown voice configuration action: {action}. Available actions: show, set, reset",
                True
            )

    async def _restart_system(self):
        """Restart the system"""
        logger.info("Restarting system...")

        # Wait a moment for the response to be sent
        await asyncio.sleep(2)

        # Signal the core to stop
        await self.core.stop()

        # In a real implementation, you might want to use systemctl or another method
        # to properly restart the service. For now, we'll just exit and rely on
        # the service manager to restart us.
        os._exit(0)

    async def _update_system(self) -> Dict[str, Any]:
        """Update the system"""
        logger.info("Checking for updates...")

        try:
            # Check if we're running from a Git repository
            if os.path.exists(os.path.join(os.path.dirname(__file__), '../../../.git')):
                # Git pull to update
                update_result = await self._run_command("git pull")

                if "Already up to date" in update_result:
                    logger.info("No updates available.")
                    return {
                        "success": True,
                        "updated": False,
                        "message": "No updates available. System is already up to date."
                    }
                else:
                    logger.info(f"Update successful: {update_result}")
                    # Check if we need to update dependencies
                    if os.path.exists(os.path.join(os.path.dirname(__file__), '../../../requirements.txt')):
                        pip_result = await self._run_command("pip install -r ../../../requirements.txt --upgrade")
                        logger.info(f"Dependencies updated: {pip_result}")

                    return {
                        "success": True,
                        "updated": True,
                        "message": "Update successful! I'll need to restart to apply the changes.",
                        "restart_required": True
                    }
            else:
                # Check for updated container images if running in Docker
                if shutil.which('docker'):
                    # Check if we're running in Docker
                    docker_result = await self._run_command("docker-compose pull")
                    logger.info(f"Docker images updated: {docker_result}")

                    return {
                        "success": True,
                        "updated": "up to date" not in docker_result.lower(),
                        "message": "Docker images have been updated. Run 'docker-compose up -d' to apply the changes.",
                        "restart_required": True
                    }
                else:
                    logger.warning("System update not implemented for this deployment method.")
                    return {
                        "success": False,
                        "updated": False,
                        "message": "Automatic updates are not configured for this deployment method. Please update manually."
                    }
        except Exception as e:
            logger.error(f"Error during update: {e}")
            return {
                "success": False,
                "updated": False,
                "message": f"Error during update: {str(e)}"
            }

    async def _run_command(self, command: str) -> str:
        """Run a shell command asynchronously"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Command '{command}' failed with exit code {process.returncode}: {stderr.decode().strip()}")
            raise Exception(f"Command failed with exit code {process.returncode}: {stderr.decode().strip()}")

        return stdout.decode().strip()

    async def format_response(
            self,
            text: str,
            voice: bool = True,
            actions: Optional[List[Dict[str, Any]]] = None,
            audio_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Format a standard response, potentially with audio data

        Args:
            text: Response text
            voice: Whether to generate voice for this response
            actions: Optional actions to include
            audio_data: Optional pre-generated audio data

        Returns:
            Formatted response dictionary
        """
        response = {
            "text": text,
            "voice": voice,
            "actions": actions or []
        }

        if audio_data:
            response["audio"] = audio_data

        return response