"""
Information retrieval handler for Morgan
"""
from typing import Dict, Any
import logging

from .base_handler import BaseHandler
from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class InformationHandler(BaseHandler):
    """Handler for information retrieval commands"""

    async def handle(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Handle an information retrieval command

        Args:
            params: Command parameters
            context: Conversation context

        Returns:
            Response dictionary
        """
        info_type = params.get("type", "")
        query = params.get("query", "")

        # Process based on information type
        if info_type == "weather":
            return await self._handle_weather(query, context)
        elif info_type == "knowledge":
            return await self._handle_knowledge(query, context)
        else:
            # Pass to LLM for general information
            response = await self.core.llm_service.process_input(query, context.get_history())
            return await self.format_response(response, True)

    async def _handle_weather(self, location: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle a weather query"""
        # In a real implementation, this would connect to a weather API
        # For now, we'll return a placeholder response

        if not location or location.lower() == "here":
            location = "your current location"

        # Generate response using LLM
        prompt = f"Provide a weather forecast for {location}. This is a placeholder as we don't have a real weather API integration yet."
        response = await self.core.llm_service.process_input(prompt)

        return await self.format_response(
            f"I don't have access to weather data yet, but I can help you find this information online. {response}",
            True
        )

    async def _handle_knowledge(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle a knowledge query"""
        # For knowledge queries, we use the LLM directly
        response = await self.core.llm_service.process_input(query, context.get_history())

        return await self.format_response(response, True)