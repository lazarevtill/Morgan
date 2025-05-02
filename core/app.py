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
            self.config['services']['tts']['default_voice']
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

        self.logger.info("Morgan Core Service initialized successfully")

    async def process_text_input(self, text: str, user_id: str = "default") -> Dict[str, Any]:
        """Process text input from user and return response"""
        self.logger.info(f"Processing text input: {text[:50]}...")

        # Update conversation context
        context = self.state_manager.get_context(user_id)
        context.add_user_message(text)

        # Generate LLM response to determine intent
        llm_response = await self.llm_service.process_input(text, context.get_history())

        # Extract intent and parameters
        intent, params = self.parse_intent(llm_response)

        # Execute appropriate handler
        if intent in self.handlers:
            handler = self.handlers[intent]
            response = await handler.handle(params, context)
        else:
            # Default to direct LLM response if no specific handler
            response = {
                "text": llm_response,
                "voice": True,
                "actions": []
            }

        # Update conversation context with response
        context.add_assistant_message(response["text"])

        # Generate voice response if requested
        if response.get("voice", False):
            response["audio"] = await self.tts_service.generate_speech(response["text"])

        self.logger.info(f"Completed processing input, response: {response['text'][:50]}...")
        return response

    async def process_audio_input(self, audio_data: bytes, user_id: str = "default") -> Dict[str, Any]:
        """Process audio input from user and return response"""
        self.logger.info("Processing audio input...")

        # Convert speech to text
        text = await self.stt_service.transcribe(audio_data)

        # Process the resulting text
        return await self.process_text_input(text, user_id)

    def parse_intent(self, llm_response: str) -> tuple:
        """Parse intent and parameters from LLM response"""
        # This is a simplified implementation
        # In a real system, we'd have more sophisticated intent parsing

        # For now, assume the LLM returns a JSON or structured format
        # that can be parsed to extract intent and parameters

        # Simple example:
        if "turn on" in llm_response.lower() and "light" in llm_response.lower():
            return "home_assistant.light", {"action": "turn_on", "entity": "light"}
        elif "weather" in llm_response.lower():
            return "information.weather", {}
        else:
            return "general.conversation", {}

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

        self.logger.info("Morgan Core Service started")

        # Keep the service running
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """Stop the Morgan Core Service"""
        self.logger.info("Morgan Core Service stopping...")
        self.running = False

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