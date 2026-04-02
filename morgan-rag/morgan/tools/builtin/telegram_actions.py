"""
Telegram channel action tools.

Exposes Telegram Bot API capabilities as tools the LLM can call:
- create_forum_topic: Create a new forum topic/thread in a supergroup
- edit_forum_topic: Rename or change icon of a forum topic
- pin_message: Pin a message in a chat
- delete_message: Delete a message
- send_reaction: React to a message with an emoji

These tools require an active Telegram channel with a bot instance.
The bot reference is set at runtime via set_telegram_bot().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult

logger = logging.getLogger(__name__)

# Runtime reference to the Telegram bot — set by TelegramChannel on start()
_telegram_bot = None
_telegram_bot_id: Optional[int] = None


def set_telegram_bot(bot: Any, bot_id: Optional[int] = None) -> None:
    """Register the active Telegram bot instance for action tools."""
    global _telegram_bot, _telegram_bot_id
    _telegram_bot = bot
    _telegram_bot_id = bot_id


def get_telegram_bot():
    """Get the active Telegram bot instance."""
    return _telegram_bot


# --- Forum Topic Creation ---------------------------------------------------

class CreateForumTopicTool(BaseTool):
    """Create a new forum topic (thread) in a Telegram supergroup.

    The supergroup must have forum/topics enabled. The bot must be an admin
    with the ``can_manage_topics`` permission.
    """

    name = "create_forum_topic"
    description = (
        "Create a new forum topic/thread in the current Telegram group. "
        "Use this when users ask to start a new discussion thread, "
        "organize conversations by topic, or create a space for a specific subject."
    )
    aliases = ("new_thread", "new_topic")
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID of the supergroup (from message context)",
            },
            "name": {
                "type": "string",
                "description": "Topic name (1-128 characters)",
            },
            "icon_color": {
                "type": "integer",
                "description": "Optional icon color (one of: 7322096, 16766590, 13338331, 9367192, 16749490, 16478047)",
            },
        },
        required=("chat_id", "name"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(
                output="Telegram bot is not available. Cannot create forum topic.",
                is_error=True,
                error_code="BOT_NOT_AVAILABLE",
            )

        chat_id = input_data["chat_id"]
        name = input_data["name"].strip()
        if len(name) > 128:
            name = name[:128]

        try:
            # Parse chat_id — strip topic qualifier if present (e.g. "-100123:topic:5")
            raw_chat_id = str(chat_id).split(":")[0]
            int_chat_id = int(raw_chat_id)
        except (ValueError, TypeError):
            return ToolResult(
                output=f"Invalid chat_id: {chat_id}",
                is_error=True,
                error_code="INVALID_INPUT",
            )

        kwargs: Dict[str, Any] = {}
        if "icon_color" in input_data:
            kwargs["icon_color"] = input_data["icon_color"]

        try:
            result = await bot.create_forum_topic(
                chat_id=int_chat_id,
                name=name,
                **kwargs,
            )

            # Cache the topic name for context isolation
            try:
                from morgan.channels.telegram_channel import TelegramChannel
                # Find active channel instance and update its cache
                cache_key = f"{int_chat_id}:{result.message_thread_id}"
                # Use module-level cache as fallback
                if not hasattr(TelegramChannel, "_shared_topic_cache"):
                    TelegramChannel._shared_topic_cache = {}
                TelegramChannel._shared_topic_cache[cache_key] = name
            except Exception:
                pass

            return ToolResult(
                output=f"Created forum topic '{name}' (thread_id={result.message_thread_id})",
                metadata={
                    "thread_id": result.message_thread_id,
                    "chat_id": int_chat_id,
                    "name": name,
                },
            )
        except Exception as exc:
            logger.error("Failed to create forum topic: %s", exc)
            return ToolResult(
                output=f"Failed to create forum topic: {exc}",
                is_error=True,
                error_code="TELEGRAM_API_ERROR",
            )


# --- Forum Topic Edit -------------------------------------------------------

class EditForumTopicTool(BaseTool):
    """Rename or change the icon of an existing forum topic."""

    name = "edit_forum_topic"
    description = "Rename or update an existing forum topic/thread in a Telegram group."
    aliases = ("rename_topic", "rename_thread")
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID of the supergroup",
            },
            "message_thread_id": {
                "type": "integer",
                "description": "Thread ID of the topic to edit",
            },
            "name": {
                "type": "string",
                "description": "New topic name (1-128 characters)",
            },
        },
        required=("chat_id", "message_thread_id"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(output="Telegram bot not available.", is_error=True, error_code="BOT_NOT_AVAILABLE")

        try:
            int_chat_id = int(str(input_data["chat_id"]).split(":")[0])
        except (ValueError, TypeError):
            return ToolResult(output=f"Invalid chat_id: {input_data['chat_id']}", is_error=True, error_code="INVALID_INPUT")

        thread_id = input_data["message_thread_id"]
        kwargs: Dict[str, Any] = {}
        if "name" in input_data:
            kwargs["name"] = input_data["name"].strip()[:128]

        if not kwargs:
            return ToolResult(output="Nothing to edit — provide at least 'name'.", is_error=True, error_code="INVALID_INPUT")

        try:
            await bot.edit_forum_topic(chat_id=int_chat_id, message_thread_id=thread_id, **kwargs)
            return ToolResult(output=f"Forum topic {thread_id} updated: {kwargs}")
        except Exception as exc:
            return ToolResult(output=f"Failed to edit forum topic: {exc}", is_error=True, error_code="TELEGRAM_API_ERROR")


# --- Pin Message -------------------------------------------------------------

class PinMessageTool(BaseTool):
    """Pin a message in a Telegram chat."""

    name = "pin_message"
    description = "Pin a message in the current Telegram chat or forum topic."
    aliases = ("pin",)
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {"type": "string", "description": "Telegram chat ID"},
            "message_id": {"type": "integer", "description": "ID of the message to pin"},
        },
        required=("chat_id", "message_id"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(output="Telegram bot not available.", is_error=True, error_code="BOT_NOT_AVAILABLE")

        try:
            int_chat_id = int(str(input_data["chat_id"]).split(":")[0])
            await bot.pin_chat_message(chat_id=int_chat_id, message_id=input_data["message_id"])
            return ToolResult(output=f"Message {input_data['message_id']} pinned.")
        except Exception as exc:
            return ToolResult(output=f"Failed to pin message: {exc}", is_error=True, error_code="TELEGRAM_API_ERROR")


# --- Delete Message ----------------------------------------------------------

class DeleteMessageTool(BaseTool):
    """Delete a message from a Telegram chat."""

    name = "delete_message"
    description = "Delete a message in the current Telegram chat."
    aliases = ("remove_message",)
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {"type": "string", "description": "Telegram chat ID"},
            "message_id": {"type": "integer", "description": "ID of the message to delete"},
        },
        required=("chat_id", "message_id"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(output="Telegram bot not available.", is_error=True, error_code="BOT_NOT_AVAILABLE")

        try:
            int_chat_id = int(str(input_data["chat_id"]).split(":")[0])
            await bot.delete_message(chat_id=int_chat_id, message_id=input_data["message_id"])
            return ToolResult(output=f"Message {input_data['message_id']} deleted.")
        except Exception as exc:
            return ToolResult(output=f"Failed to delete message: {exc}", is_error=True, error_code="TELEGRAM_API_ERROR")


# --- React to Message --------------------------------------------------------

class ReactToMessageTool(BaseTool):
    """React to a message with an emoji."""

    name = "react_to_message"
    description = "Add an emoji reaction to a message in the current Telegram chat."
    aliases = ("react", "add_reaction")
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {"type": "string", "description": "Telegram chat ID"},
            "message_id": {"type": "integer", "description": "ID of the message to react to"},
            "emoji": {"type": "string", "description": "Emoji to react with (e.g. '👍', '❤️', '🔥')"},
        },
        required=("chat_id", "message_id", "emoji"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(output="Telegram bot not available.", is_error=True, error_code="BOT_NOT_AVAILABLE")

        try:
            from telegram import ReactionTypeEmoji
            int_chat_id = int(str(input_data["chat_id"]).split(":")[0])
            await bot.set_message_reaction(
                chat_id=int_chat_id,
                message_id=input_data["message_id"],
                reaction=[ReactionTypeEmoji(emoji=input_data["emoji"])],
            )
            return ToolResult(output=f"Reacted with {input_data['emoji']}.")
        except ImportError:
            return ToolResult(output="python-telegram-bot not installed.", is_error=True, error_code="DEPENDENCY_MISSING")
        except Exception as exc:
            return ToolResult(output=f"Failed to react: {exc}", is_error=True, error_code="TELEGRAM_API_ERROR")


# --- Send to Forum Topic (message in a specific thread) ---------------------

class SendToTopicTool(BaseTool):
    """Send a message to a specific forum topic/thread."""

    name = "send_to_topic"
    description = (
        "Send a message to a specific forum topic/thread in a Telegram group. "
        "Use this to post in a thread that was just created or an existing one."
    )
    aliases = ("post_in_thread",)
    input_schema = ToolInputSchema(
        properties={
            "chat_id": {"type": "string", "description": "Telegram chat ID"},
            "message_thread_id": {"type": "integer", "description": "Thread/topic ID to post in"},
            "text": {"type": "string", "description": "Message text to send"},
        },
        required=("chat_id", "message_thread_id", "text"),
    )

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        bot = get_telegram_bot()
        if bot is None:
            return ToolResult(output="Telegram bot not available.", is_error=True, error_code="BOT_NOT_AVAILABLE")

        try:
            int_chat_id = int(str(input_data["chat_id"]).split(":")[0])
            msg = await bot.send_message(
                chat_id=int_chat_id,
                message_thread_id=input_data["message_thread_id"],
                text=input_data["text"],
            )
            return ToolResult(
                output=f"Sent message to topic {input_data['message_thread_id']} (msg_id={msg.message_id})",
                metadata={"message_id": msg.message_id},
            )
        except Exception as exc:
            return ToolResult(output=f"Failed to send to topic: {exc}", is_error=True, error_code="TELEGRAM_API_ERROR")


# --- Collection of all Telegram action tools --------------------------------

TELEGRAM_ACTION_TOOLS: List[BaseTool] = [
    CreateForumTopicTool(),
    EditForumTopicTool(),
    PinMessageTool(),
    DeleteMessageTool(),
    ReactToMessageTool(),
    SendToTopicTool(),
]
