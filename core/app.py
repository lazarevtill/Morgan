#!/usr/bin/env python3
"""
Morgan: A Self-Hosted Home Lab AI Assistant
Enhanced main application entry point
"""
import asyncio
import signal
import sys
import logging
import os
from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path

from config.config_manager import ConfigManager
from conversation.state_manager import ConversationStateManager
from handlers import get_handler_registry
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.stt_service import STTService
from integrations.home_assistant import HomeAssistantIntegration
from utils.logging import setup_logging
from utils.intent_parser import IntentParser
from utils.command_resolver import CommandResolver
from utils.error_handler import ErrorHandler
from api.server import APIServer


class MorganCore:
    """Main application class for Morgan Core Service"""

    def __init__(self, config_dir: Optional[str] = None):
        self.running = False
        self.start_time = time.time()

        # Set up configuration
        self.config_dir = config_dir or "/opt/morgan/config"
        self.config_manager = ConfigManager(self.config_dir)

        try:
            self.config = self.config_manager.load_core_config()
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            print("Creating default configuration...")
            self._create_default_config()
            self.config = self.config_manager.load_core_config()

        # Set up data directory
        self.data_dir = self.config['system'].get('data_dir', '/opt/morgan/data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Set up logging
        log_level_str = self.config['system'].get('log_level', 'info').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = os.path.join(self.data_dir, "logs/core.log")
        self.logger = setup_logging("morgan_core", log_level, log_file)

        # Initialize components
        self.logger.info("Initializing Morgan Core Service...")

        # Set up error handler
        self.error_handler = ErrorHandler()

        # Initialize services
        llm_config = self.config['services']['llm']
        self.llm_service = LLMService(
            service_url=llm_config['url'],
            model_name=llm_config['model'],
            system_prompt=llm_config.get('system_prompt'),
            max_tokens=llm_config.get('max_tokens', 1000),
            temperature=llm_config.get('temperature', 0.7)
        )

        tts_config = self.config['services']['tts']
        self.tts_service = TTSService(
            service_url=tts_config['url'],
            default_voice=tts_config.get('default_voice', 'morgan_default')
        )

        stt_config = self.config['services']['stt']
        self.stt_service = STTService(
            service_url=stt_config['url'],
            model_name=stt_config['model']
        )

        # Initialize integrations
        ha_config = self.config.get('home_assistant', {})
        if ha_config and 'url' in ha_config and 'token' in ha_config:
            self.home_assistant = HomeAssistantIntegration(
                url=ha_config['url'],
                token=ha_config['token']
            )
            self.home_assistant.reconnect_interval = ha_config.get('reconnect_interval', 10)
        else:
            self.logger.warning("Home Assistant configuration is missing or incomplete")
            self.home_assistant = None

        # Initialize state management
        self.state_manager = ConversationStateManager(
            data_dir=self.data_dir,
            max_history=self.config['system'].get('max_history', 20),
            context_timeout=self.config['system'].get('context_timeout', 1800),
            save_interval=self.config['system'].get('save_interval', 60)
        )

        # Initialize command handlers
        self.handlers = get_handler_registry(self)

        # Initialize intent parser and command resolver
        self.intent_parser = IntentParser(self.llm_service)
        self.command_resolver = CommandResolver(self.handlers)

        # Initialize API server
        api_config = self.config.get('api', {})
        self.api_server = APIServer(
            self,
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 8000)
        )

        self.logger.info("Morgan Core Service initialized successfully")

    async def process_text_input(self, text: str, user_id: str = "default") -> Dict[str, Any]:
        """Process text input from user and return response"""
        self.logger.info(f"Processing text input: {text[:50]}...")

        # Update conversation context
        context = self.state_manager.get_context(user_id)
        context.add_user_message(text)

        try:
            # Extract intent and parameters using the intent parser
            intent, params = await self.intent_parser.extract_intent(text, context.get_history())
            self.logger.debug(f"Extracted intent: {intent} with parameters: {params}")

            # Push the intent to the context
            context.push_active_intent(intent)

            # Resolve and execute the command
            response = await self.command_resolver.resolve_and_execute(intent, params, context)

            # Pop the intent from the context when done
            context.pop_active_intent()

        except Exception as e:
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            response = self.error_handler.create_error_response(
                "I'm sorry, I'm having trouble understanding that request. Could you try again?"
            )

        # Update conversation context with response
        context.add_assistant_message(response["text"])

        # Generate voice response if requested
        if response.get("voice", False) and not response.get("audio"):
            try:
                voice_id = context.get_variable("voice_preference", None)
                response["audio"] = await self.tts_service.generate_speech(
                    text=response["text"],
                    voice_id=voice_id
                )
            except Exception as e:
                self.logger.error(f"Error generating speech: {e}", exc_info=True)
                # Don't add the audio field if speech generation fails

        self.logger.info(f"Completed processing input, response: {response['text'][:50]}...")
        return response

    async def process_audio_input(self, audio_data: bytes, user_id: str = "default") -> Dict[str, Any]:
        """Process audio input from user and return response"""
        self.logger.info("Processing audio input...")

        try:
            # Get conversation context
            context = self.state_manager.get_context(user_id)

            # Get prompts from recent conversation to help with transcription
            recent_messages = context.get_last_n_messages(3)
            prompt = None
            if recent_messages:
                # Extract last assistant message as context for STT
                for msg in reversed(recent_messages):
                    if msg.get("role") == "assistant":
                        prompt = msg.get("content", "")
                        break

            # Convert speech to text
            text = await self.stt_service.transcribe(audio_data, prompt=prompt)
            self.logger.info(f"Transcribed text: {text}")

            # Process the resulting text
            return await self.process_text_input(text, user_id)

        except Exception as e:
            self.logger.error(f"Error processing audio input: {e}", exc_info=True)
            return self.error_handler.create_error_response(
                "I'm sorry, I couldn't understand the audio. Could you try again?"
            )

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil

        # Get system info
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Get uptime
        uptime_seconds = time.time() - self.start_time
        uptime_str = self._format_uptime(uptime_seconds)

        # Get version from pyproject.toml if available
        version = "0.1.0"  # Default version
        try:
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    for line in f:
                        if line.startswith("version = "):
                            version = line.split("=")[1].strip().strip('"\'')
                            break
        except Exception as e:
            self.logger.error(f"Error reading version from pyproject.toml: {e}")

        return {
            "version": version,
            "uptime": uptime_str,
            "uptime_seconds": uptime_seconds,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "service_status": {
                "llm": self.llm_service.session is not None,
                "tts": self.tts_service.session is not None,
                "stt": self.stt_service.session is not None,
                "home_assistant": self.home_assistant.connected if self.home_assistant else False
            }
        }

    async def start(self):
        """Start the Morgan Core Service"""
        self.running = True
        self.logger.info("Morgan Core Service starting...")

        # Initialize connections to services
        await self.llm_service.connect()
        await self.tts_service.connect()
        await self.stt_service.connect()

        # Connect to Home Assistant if configured
        if self.home_assistant:
            try:
                await self.home_assistant.connect()
                self.logger.info("Connected to Home Assistant successfully")
            except Exception as e:
                self.logger.error(f"Failed to connect to Home Assistant: {e}")

        # Start conversation context save task
        self.state_manager.start_save_task()

        # Start API server
        try:
            await self.api_server.start()
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            self.running = False
            return

        self.logger.info("Morgan Core Service started")

        # Periodic tasks
        while self.running:
            try:
                # Clean up expired conversation contexts
                self.state_manager.clear_expired_contexts()

                # Check Home Assistant connection
                if self.home_assistant and not self.home_assistant.connected:
                    try:
                        await self.home_assistant.connect()
                    except Exception as e:
                        self.logger.error(f"Failed to reconnect to Home Assistant: {e}")

                await asyncio.sleep(60)  # Run cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic tasks: {e}")
                await asyncio.sleep(60)  # Continue with the loop even if there's an error

    async def stop(self):
        """Stop the Morgan Core Service"""
        self.logger.info("Morgan Core Service stopping...")
        self.running = False

        # Stop conversation context save task
        self.state_manager.stop_save_task()

        # Stop API server
        try:
            await self.api_server.stop()
        except Exception as e:
            self.logger.error(f"Error stopping API server: {e}")

        # Close connections to services
        await self.llm_service.disconnect()
        await self.tts_service.disconnect()
        await self.stt_service.disconnect()

        if self.home_assistant:
            await self.home_assistant.disconnect()

        self.logger.info("Morgan Core Service stopped")

    def _create_default_config(self):
        """Create default configuration files"""
        try:
            # Ensure config directory exists
            os.makedirs(self.config_dir, exist_ok=True)

            # Create core.yaml if it doesn't exist
            core_config_path = os.path.join(self.config_dir, "core.yaml")
            if not os.path.exists(core_config_path):
                default_core_config = {
                    "system": {
                        "name": "Morgan",
                        "log_level": "info",
                        "data_dir": "/opt/morgan/data",
                        "max_history": 20,
                        "context_timeout": 1800,
                        "save_interval": 60
                    },
                    "services": {
                        "llm": {
                            "url": "http://llm-service:8001",
                            "model": "mistral",
                            "system_prompt": "You are Morgan, a helpful and friendly home assistant AI. You assist with smart home controls, answer questions, and perform various tasks.",
                            "max_tokens": 1000,
                            "temperature": 0.7
                        },
                        "tts": {
                            "url": "http://tts-service:8002",
                            "default_voice": "morgan_default"
                        },
                        "stt": {
                            "url": "http://stt-service:8003",
                            "model": "whisper-large-v3"
                        }
                    },
                    "home_assistant": {
                        "url": "http://homeassistant:8123",
                        "token": "your_long_lived_access_token",
                        "reconnect_interval": 10
                    },
                    "api": {
                        "host": "0.0.0.0",
                        "port": 8000,
                        "cors_origins": ["*"],
                        "auth_enabled": False
                    }
                }

                with open(core_config_path, 'w') as f:
                    import yaml
                    yaml.dump(default_core_config, f, default_flow_style=False)

            # Create handlers.yaml if it doesn't exist
            handlers_config_path = os.path.join(self.config_dir, "handlers.yaml")
            if not os.path.exists(handlers_config_path):
                default_handlers_config = {
                    "handlers": {
                        "home_assistant": {
                            "enabled": True,
                            "domains": ["light", "switch", "climate", "media_player"]
                        },
                        "information": {
                            "enabled": True,
                            "weather_api_key": ""
                        },
                        "system": {
                            "enabled": True,
                            "allow_restart": True,
                            "allow_update": True
                        }
                    }
                }

                with open(handlers_config_path, 'w') as f:
                    import yaml
                    yaml.dump(default_handlers_config, f, default_flow_style=False)

            # Create devices.yaml if it doesn't exist
            devices_config_path = os.path.join(self.config_dir, "devices.yaml")
            if not os.path.exists(devices_config_path):
                default_devices_config = {
                    "device_groups": {
                        "living_room": [
                            "light.living_room",
                            "media_player.living_room_tv",
                            "climate.living_room"
                        ],
                        "kitchen": [
                            "light.kitchen",
                            "switch.coffee_maker"
                        ]
                    },
                    "device_aliases": {
                        "tv": "media_player.living_room_tv",
                        "main_lights": "light.living_room",
                        "coffee": "switch.coffee_maker"
                    }
                }

                with open(devices_config_path, 'w') as f:
                    import yaml
                    yaml.dump(default_devices_config, f, default_flow_style=False)

            # Create voices.yaml if it doesn't exist
            voices_config_path = os.path.join(self.config_dir, "voices.yaml")
            if not os.path.exists(voices_config_path):
                default_voices_config = {
                    "voices": {
                        "morgan_default": {
                            "description": "Default Morgan voice",
                            "type": "preset",
                            "preset_id": 12
                        }
                    }
                }

                with open(voices_config_path, 'w') as f:
                    import yaml
                    yaml.dump(default_voices_config, f, default_flow_style=False)

            print("Default configuration files created successfully")
        except Exception as e:
            print(f"Error creating default configuration: {e}")
            raise

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in a human-readable string"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        return ", ".join(parts)


async def main():
    """Main entry point"""
    # Get config directory from environment or use default
    config_dir = os.environ.get("MORGAN_CONFIG_DIR", "/opt/morgan/config")

    morgan = MorganCore(config_dir)

    # Setup signal handling for graceful shutdown
    def signal_handler():
        print("Shutting down...")
        asyncio.create_task(morgan.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

    # Start the core service
    try:
        await morgan.start()
    except KeyboardInterrupt:
        await morgan.stop()
    except Exception as e:
        print(f"Error: {e}")
        await morgan.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())