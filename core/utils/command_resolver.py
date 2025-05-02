"""
Enhanced command resolution utilities for Morgan Core
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
import traceback
import re
import time

from conversation.context import ConversationContext
from utils.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


class CommandResolver:
    """Resolves intents to appropriate command handlers with enhanced capabilities"""

    def __init__(self, handlers):
        self.handlers = handlers
        self.intent_classifier = IntentClassifier()
        self.conversation_handler_id = "general.conversation"

        # Define fallback handlers based on intent patterns
        self.fallback_handlers = {
            r"^home_assistant\..*": "home_assistant.query",
            r"^information\..*": "information.knowledge",
            r"^system\..*": "system.status"
        }

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
        start_time = time.time()

        # Validate the intent and parameters
        valid, intent, params = self.intent_classifier.validate_intent(intent, params)

        if not valid:
            logger.warning(f"Invalid intent: {intent}")
            return self._create_error_response("I didn't understand that command. Could you try rephrasing it?")

        # Keep track of the original intent for logging and analytics
        original_intent = intent

        # Try exact match first
        if intent in self.handlers:
            logger.info(f"Executing handler for intent: {intent}")
            try:
                result = await self.handlers[intent].handle(params, context)
                self._log_execution_time(original_intent, start_time)
                return result
            except Exception as e:
                logger.error(f"Error executing handler for intent {intent}: {e}")
                logger.debug(traceback.format_exc())
                return self._create_error_response(f"I had trouble processing that command: {str(e)}")

        # Try to find a partial match (e.g. if intent is home_assistant.light.bedroom
        # but we only have home_assistant.light handler)
        matched_handler = await self._find_closest_handler(intent, params)
        if matched_handler:
            self._log_execution_time(original_intent, start_time)
            return matched_handler

        # Check for special handling of information requests
        if intent.startswith("information.") and "information.knowledge" in self.handlers:
            try:
                # Add intent type to parameters to help with processing
                params["type"] = intent.split(".")[1] if len(intent.split(".")) > 1 else "knowledge"
                # Ensure query is set
                if "query" not in params and "text" in params:
                    params["query"] = params["text"]
                elif "query" not in params:
                    # Reconstruct query from the params
                    params["query"] = " ".join([f"{k}:{v}" for k, v in params.items() if k != "type"])

                logger.info(f"Using information handler for intent: {intent}")
                result = await self.handlers["information.knowledge"].handle(params, context)
                self._log_execution_time(original_intent, start_time)
                return result
            except Exception as e:
                logger.error(f"Error executing information handler for intent {intent}: {e}")
                logger.debug(traceback.format_exc())

        # Fallback to general conversation handler
        if self.conversation_handler_id in self.handlers:
            logger.info(f"Using fallback handler for intent: {intent}")
            try:
                # Include the intent and original params in the query for better context
                if "query" not in params:
                    # Create a natural language query from the intent and parameters
                    query = f"Intent: {intent}, Parameters: {params}"
                else:
                    query = params["query"]

                conv_params = {"type": "chat", "query": query}
                result = await self.handlers[self.conversation_handler_id].handle(conv_params, context)
                self._log_execution_time(original_intent, start_time)
                return result
            except Exception as e:
                logger.error(f"Error executing fallback handler: {e}")
                logger.debug(traceback.format_exc())
                return self._create_error_response("I'm not sure how to handle that request right now.")

        # Last resort error response
        self._log_execution_time(original_intent, start_time)
        return self._create_error_response("I don't know how to process that command.")

    async def _find_closest_handler(self, intent: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the closest matching handler for an intent"""
        # Try to find a partial match
        parts = intent.split('.')
        for i in range(len(parts) - 1, 0, -1):
            partial_intent = '.'.join(parts[:i])
            if partial_intent in self.handlers:
                logger.debug(f"Using partial match handler {partial_intent} for intent: {intent}")
                try:
                    # Update params to include any missing information from the intent
                    if "entity" not in params and i < len(parts):
                        params["entity"] = parts[i]
                    return await self.handlers[partial_intent].handle(params, params)
                except Exception as e:
                    logger.error(f"Error executing handler for partial intent {partial_intent}: {e}")
                    logger.debug(traceback.format_exc())
                    # Continue trying other partial matches instead of immediately giving up

        # Try fallback handlers based on regex patterns
        for pattern, handler_id in self.fallback_handlers.items():
            if re.match(pattern, intent) and handler_id in self.handlers:
                logger.debug(f"Using pattern fallback handler {handler_id} for intent: {intent}")
                try:
                    return await self.handlers[handler_id].handle(params, params)
                except Exception as e:
                    logger.error(f"Error executing fallback handler {handler_id}: {e}")
                    logger.debug(traceback.format_exc())

        # No suitable handler found
        return None

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "text": message,
            "voice": True,
            "actions": [],
            "error": True
        }

    def _log_execution_time(self, intent: str, start_time: float):
        """Log the execution time for analytics"""
        execution_time = time.time() - start_time
        logger.debug(f"Intent {intent} executed in {execution_time:.4f} seconds")


class ContextualCommandResolver(CommandResolver):
    """Command resolver that takes conversation context into account"""

    async def resolve_and_execute(self, intent: str, params: Dict[str, Any],
                                  context: ConversationContext) -> Dict[str, Any]:
        """
        Resolve intent with context awareness and execute command

        Args:
            intent: Intent identifier
            params: Command parameters
            context: Conversation context

        Returns:
            Command execution result
        """
        # Check for continuation of previous intent
        active_intent = context.get_active_intent()
        if active_intent and not params.get("force_new_intent"):
            # Check if we should continue with the active intent
            if self._is_continuation(intent, active_intent, params):
                logger.info(f"Continuing active intent: {active_intent}")
                # Merge the previous params with the new ones
                combined_params = self._merge_parameters(context, params)
                # Use the active intent
                intent = active_intent
                params = combined_params

        # Check for missing parameters that might be in the context
        if "entity" not in params:
            # Check for recently mentioned entities
            recent_entities = context.get_recently_mentioned_entities()
            if recent_entities:
                logger.debug(f"Using recently mentioned entity: {recent_entities[0]}")
                params["entity"] = recent_entities[0]

        # Run the standard resolution and execution
        return await super().resolve_and_execute(intent, params, context)

    def _is_continuation(self, new_intent: str, active_intent: str, params: Dict[str, Any]) -> bool:
        """Check if a new intent is a continuation of an active intent"""
        # If the new intent is general conversation, it might be a continuation
        if new_intent == self.conversation_handler_id:
            return True

        # If it's the same domain, it might be a continuation
        if new_intent.split('.')[0] == active_intent.split('.')[0]:
            return True

        # If it's a different domain but we explicitly requested to continue
        if params.get("continue"):
            return True

        return False

    def _merge_parameters(self, context: ConversationContext, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parameters from context with new parameters"""
        # Get the previous parameters from context
        prev_params = context.get_variable("last_params", {})

        # Start with the previous parameters
        merged = {**prev_params}

        # Override with new parameters
        for key, value in new_params.items():
            merged[key] = value

        return merged