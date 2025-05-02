"""
Base handler for Morgan command processing
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from conversation.context import ConversationContext


class BaseHandler(ABC):
    """Base class for all command handlers"""

    def __init__(self, core_instance):
        self.core = core_instance

    @abstractmethod
    async def handle(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Handle a command

        Args:
            params: Command parameters
            context: Conversation context

        Returns:
            Response dictionary
        """
        pass

    async def format_response(
            self,
            text: str,
            voice: bool = True,
            actions: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Format a standard response

        Args:
            text: Response text
            voice: Whether to generate voice for this response
            actions: Optional actions to include

        Returns:
            Formatted response dictionary
        """
        return {
            "text": text,
            "voice": voice,
            "actions": actions or []
        }