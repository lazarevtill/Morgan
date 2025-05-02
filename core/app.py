#!/usr/bin/env python3
"""
Morgan: A Self-Hosted Home Lab AI Assistant
Main application entry point
"""
import asyncio
import signal
import sys
import logging
from typing import Dict, Any

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
from api.server import APIServer


class MorganCore:
    """Main application class for Morgan Core Service"""

    def __init__(self):
        self.running = False
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_core_config()

        # Setup logging
        log_level = getattr(logging, self.config['system']['log_level'].upper())
        self.logger = setup_logging("morgan_core", log_level, self.config['system']['data_dir'] + "/logs/core.log")

        # Initialize components
        self.logger.info("Initializing Morgan Core Service...")

        # Initialize services
        self.llm_service = LLMService(
            self.config['services']['llm']['url'],
            self.config['services']['llm']['model']
        )

        self.tts_service = TTSService(
            self.config['services']['tts']['url'],
            self.config['services']['tts'].get('default_voice', 'morgan_default')
        )

        self.stt_service = STTService(
            self.config['services']['stt']['url'],
            self.config['services']['stt']['model']
        )

        # Initialize integrations
        self.home_assistant = HomeAssistantIntegration(
            self.config['home_assistant']['url'],
            self.config['home_assistant']['token']
        )

        # Initialize state management
        self.state_manager = ConversationStateManager()

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

            # Push the intent to the context
            context.push_active_intent(intent)

            # Resolve and execute the command
            response = await self.command_resolver.resolve_and_execute(intent, params, context)

            # Pop the intent from the context when done
            context.pop_active_intent()

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            response = {
                "text": "I'm sorry, I'm having trouble understanding that request. Could you try again?",
                "voice": True,
                "actions": []
            }

        # Update conversation context with response
        context.add_assistant_message(response["text"])

        # Generate voice response if requested
        if response.get("voice", False) and not response.get("audio"):
            try:
                response["audio"] = await self.tts_service.generate_speech(response["text"])
            except Exception as e:
                self.logger.error(f"Error generating speech: {e}")
                # Don't add the audio field if speech generation fails

        self.logger.info(f"Completed processing input, response: {response['text'][:50]}...")
        return response

    async def process_audio_input(self, audio_data: bytes, user_id: str = "default") -> Dict[str, Any]:
        """Process audio input from user and return response"""
        self.logger.info("Processing audio input...")

        try:
            # Convert speech to text
            text = await self.stt_service.transcribe(audio_data)
            self.logger.info(f"Transcribed text: {text}")

            # Process the resulting text
            return await self.process_text_input(text, user_id)

        except Exception as e:
            self.logger.error(f"Error processing audio input: {e}")
            return {
                "text": "I'm sorry, I couldn't understand the audio. Could you try again?",
                "voice": True,
                "actions": []
            }

    async def start(self):
        """Start the Morgan Core Service"""
        self.running = True
        self.logger.info("Morgan Core Service starting...")

        # Initialize connections to services
        await self.llm_service.connect()
        await self.tts_service.connect()
        await self.stt_service.connect()

        # Connect to Home Assistant
        try:
            await self.home_assistant.connect()
            self.logger.info("Connected to Home Assistant successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Home Assistant: {e}")

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
            # Clean up expired conversation contexts
            self.state_manager.clear_expired_contexts()
            await asyncio.sleep(60)  # Run cleanup every minute

    async def stop(self):
        """Stop the Morgan Core Service"""
        self.logger.info("Morgan Core Service stopping...")
        self.running = False

        # Stop API server
        try:
            await self.api_server.stop()
        except Exception as e:
            self.logger.error(f"Error stopping API server: {e}")

        # Close connections to services
        await self.llm_service.disconnect()
        await self.tts_service.disconnect()
        await self.stt_service.disconnect()
        await self.home_assistant.disconnect()

        self.logger.info("Morgan Core Service stopped")


async def main():
    """Main entry point"""
    morgan = MorganCore()

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