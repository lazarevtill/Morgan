"""
Command handler registry for Morgan
"""
from typing import Dict

from .base_handler import BaseHandler
from .home_assistant import HomeAssistantHandler
from .information import InformationHandler
from .system import SystemHandler


def get_handler_registry(core_instance) -> Dict[str, BaseHandler]:
    """
    Create and return the handler registry

    Args:
        core_instance: The MorganCore instance

    Returns:
        Dictionary mapping command types to handlers
    """
    handlers = {
        # Home Assistant handlers
        "home_assistant.light": HomeAssistantHandler(core_instance),
        "home_assistant.switch": HomeAssistantHandler(core_instance),
        "home_assistant.climate": HomeAssistantHandler(core_instance),
        "home_assistant.media_player": HomeAssistantHandler(core_instance),
        "home_assistant.query": HomeAssistantHandler(core_instance),

        # Information handlers
        "information.weather": InformationHandler(core_instance),
        "information.knowledge": InformationHandler(core_instance),
        "general.conversation": InformationHandler(core_instance),

        # System handlers
        "system.status": SystemHandler(core_instance),
        "system.restart": SystemHandler(core_instance),
        "system.update": SystemHandler(core_instance)
    }

    return handlers