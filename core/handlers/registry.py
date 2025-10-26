"""
Handler registry for Morgan Core Service
"""
import logging
from typing import Dict, Any, Optional, List

from shared.models.base import Command, ConversationContext
from shared.utils.logging import setup_logging
from handlers.remember_handler import RememberHandler


class BaseHandler:
    """Base class for command handlers"""

    def __init__(self, core_service):
        self.core_service = core_service
        self.logger = logging.getLogger(f"handler_{self.__class__.__name__}")

    async def handle(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle a command - to be implemented by subclasses"""
        raise NotImplementedError

    async def validate(self, command: Command, context: ConversationContext) -> bool:
        """Validate command parameters - to be implemented by subclasses"""
        return True


class SystemHandler(BaseHandler):
    """System command handler"""

    async def handle(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle system commands"""
        action = command.action.lower()

        if action == "status":
            return await self._handle_status(command, context)
        elif action == "restart":
            return await self._handle_restart(command, context)
        elif action == "reset":
            return await self._handle_reset(command, context)
        else:
            return {
                "success": False,
                "error": f"Unknown system action: {action}",
                "response": f"I don't understand the system command '{action}'. Available commands: status, restart, reset."
            }

    async def _handle_status(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle status command"""
        try:
            status = await self.core_service.get_system_status()
            return {
                "success": True,
                "response": f"System status: {status['status']}. Uptime: {status['uptime']}. Processed {status['request_count']} requests.",
                "data": status
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Unable to get system status."
            }

    async def _handle_restart(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle restart command"""
        # In a real implementation, this would trigger a restart
        return {
            "success": True,
            "response": "Restart command acknowledged. The system will restart shortly.",
            "action": "restart"
        }

    async def _handle_reset(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle reset command"""
        user_id = context.user_id
        self.core_service.conversation_manager.reset_context(user_id)

        return {
            "success": True,
            "response": "Conversation has been reset. How can I help you now?",
            "action": "reset_conversation"
        }


class GeneralHandler(BaseHandler):
    """General conversation handler"""

    async def handle(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Handle general conversation"""
        # This is a fallback handler for general conversation
        return {
            "success": True,
            "response": "I understand you're having a general conversation. How can I assist you today?",
            "action": "general_conversation"
        }


class HandlerRegistry:
    """Registry for command handlers"""

    def __init__(self, core_service):
        self.core_service = core_service
        self.logger = setup_logging("handler_registry", "INFO", "logs/handlers.log")

        # Initialize handlers
        self.handlers = {
            "system": SystemHandler(core_service),
            "general": GeneralHandler(core_service)
        }

        # Initialize remember handler if memory manager is available
        if hasattr(core_service, 'memory_manager') and core_service.memory_manager:
            self.remember_handler = RememberHandler(core_service.memory_manager)
            self.logger.info("Remember handler initialized")
        else:
            self.remember_handler = None

        self.logger.info(f"Handler registry initialized with {len(self.handlers)} handlers")

    async def process_command(self, command: Command, context: ConversationContext) -> Dict[str, Any]:
        """Process a command through the appropriate handler"""
        try:
            # Get handler based on command action
            handler_name = command.action.split('.')[0] if '.' in command.action else 'general'
            handler = self.handlers.get(handler_name, self.handlers['general'])

            # Validate command
            if not await handler.validate(command, context):
                return {
                    "success": False,
                    "error": "Command validation failed",
                    "response": f"I'm sorry, I can't process that command. Please check the parameters and try again."
                }

            # Execute command
            result = await handler.handle(command, context)

            self.logger.info(f"Command processed: {command.action} -> {result['success']}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing command {command.action}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I'm sorry, I encountered an error while processing your command. Please try again."
            }

    def register_handler(self, name: str, handler_class):
        """Register a new handler"""
        self.handlers[name] = handler_class(self.core_service)
        self.logger.info(f"Registered handler: {name}")

    def get_available_handlers(self) -> List[str]:
        """Get list of available handlers"""
        return list(self.handlers.keys())
