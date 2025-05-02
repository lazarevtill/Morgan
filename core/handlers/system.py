"""
System command handler for Morgan
"""
import os
import platform
import psutil
from typing import Dict, Any
import logging

from .base_handler import BaseHandler
from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class SystemHandler(BaseHandler):
    """Handler for system commands"""

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
        else:
            return await self.format_response(
                "I'm not sure what system command you want me to execute.",
                True
            )

    async def _handle_status(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a status command"""
        # Get system information
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Get model information
        llm_model = self.core.llm_service.model_name
        tts_voice = self.core.tts_service.default_voice
        stt_model = self.core.stt_service.model_name

        # Format response
        response = (
            f"System Status:\n"
            f"CPU Usage: {cpu_percent}%\n"
            f"Memory Usage: {memory_percent}%\n"
            f"Disk Usage: {disk_percent}%\n\n"
            f"Models:\n"
            f"LLM: {llm_model}\n"
            f"TTS Voice: {tts_voice}\n"
            f"STT: {stt_model}\n\n"
            f"Morgan is running normally."
        )

        return await self.format_response(response, True)

    async def _handle_restart(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle a restart command"""
        # In a real implementation, this would trigger a restart
        # For now, we'll just return a message

        return await self.format_response(
            "I'll restart now. This will take a few moments.",
            True,
            [{"type": "restart", "delay": 5}]
        )

    async def _handle_update(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle an update command"""
        # In a real implementation, this would trigger an update
        # For now, we'll just return a message

        return await self.format_response(
            "I'll check for updates now. This might take a few minutes.",
            True,
            [{"type": "update", "background": True}]
        )