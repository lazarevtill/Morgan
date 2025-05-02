"""
Command resolution utilities for Morgan Core
"""
from typing import Dict, Any, Optional, Tuple
import logging

from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class CommandResolver:
    """Resolves intents to appropriate command handlers"""

    def __init__(self, handlers):
        self.handlers = handlers

    async def resolve_and_execute(self, intent: str, params: Dict[str, Any],
                                  context: ConversationContext) -> Dict[str, Any]:
        """
        Resolve intent to appropriate handler and execute command

        Args:
            intent: Intent identifier
            params: Command parameters
            context: Conversation context

        Returns:
            Command execution result
        """
        # Try exact match first
        if intent in self.handlers:
            logger.debug(f"Executing handler for intent: {intent}")
            try:
                return await self.handlers[intent].handle(params, context)
            except Exception as e:
                logger.error(f"Error executing handler for intent {intent}: {e}")
                return self._create_error_response(f"I had trouble processing that command: {str(e)}")

        # Try to find a partial match (e.g. if intent is home_assistant.light.bedroom
        # but we only have home_assistant.light handler)
        parts = intent.split('.')
        for i in range(len(parts) - 1, 0, -1):
            partial_intent = '.'.join(parts[:i])
            if partial_intent in self.handlers:
                logger.debug(f"Using partial match handler {partial_intent} for intent: {intent}")
                try:
                    # Update params to include any missing information from the intent
                    if "entity" not in params and i < len(parts):
                        params["entity"] = parts[i]
                    return await self.handlers[partial_intent].handle(params, context)
                except Exception as e:
                    logger.error(f"Error executing handler for partial intent {partial_intent}: {e}")
                    return self._create_error_response(f"I had trouble processing that command: {str(e)}")

        # Fallback to general conversation handler
        if "general.conversation" in self.handlers:
            logger.debug(f"Using fallback handler for intent: {intent}")
            try:
                return await self.handlers["general.conversation"].handle(params, context)
            except Exception as e:
                logger.error(f"Error executing fallback handler: {e}")
                return self._create_error_response("I'm not sure how to handle that request right now.")

        # Last resort error response
        return self._create_error_response("I don't know how to process that command.")

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "text": message,
            "voice": True,
            "actions": []
        }