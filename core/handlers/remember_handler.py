"""
Remember Command Handler
Handles "remember" commands to store information in memory
"""
import logging
import re
from typing import Dict, Any, Optional

from shared.utils.logging import setup_logging


class RememberHandler:
    """
    Handler for 'remember' commands

    Examples:
    - "Remember that I like coffee"
    - "Remember: my birthday is June 15th"
    - "Please remember my favorite color is blue"
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.logger = setup_logging("remember_handler", "INFO", "logs/handlers.log")

        # Patterns to detect "remember" commands
        self.patterns = [
            r"^remember\s+(?:that\s+)?(.+)$",  # "remember that..."
            r"^remember:\s*(.+)$",  # "remember: ..."
            r"^(?:please\s+)?remember\s+(?:that\s+)?(.+)$",  # "please remember..."
            r"^(?:can\s+you\s+)?remember\s+(?:that\s+)?(.+)$",  # "can you remember..."
            r"^(?:i\s+want\s+you\s+to\s+)?remember\s+(?:that\s+)?(.+)$",  # "I want you to remember..."
        ]

    def can_handle(self, text: str) -> bool:
        """Check if this handler can process the given text"""
        text_lower = text.lower().strip()

        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    async def handle(self, text: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a remember command

        Args:
            text: The input text
            user_id: User identifier
            metadata: Additional context

        Returns:
            Response dict with status and message
        """
        text_lower = text.lower().strip()

        # Extract the content to remember
        content_to_remember = None
        for pattern in self.patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                content_to_remember = match.group(1).strip()
                break

        if not content_to_remember:
            return {
                "success": False,
                "response": "I couldn't understand what you want me to remember.",
                "metadata": {}
            }

        # Analyze the content to determine category and importance
        category, importance = self._analyze_content(content_to_remember)

        # Store in memory
        try:
            memory = await self.memory_manager.create_memory(
                user_id=user_id,
                content=content_to_remember,
                memory_type="fact",
                category=category,
                importance=importance,
                metadata=metadata or {}
            )

            self.logger.info(f"Stored memory for user {user_id}: {content_to_remember[:50]}...")

            return {
                "success": True,
                "response": f"I've remembered that {content_to_remember}",
                "metadata": {
                    "memory_id": memory.id,
                    "category": category,
                    "importance": importance
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "response": "I had trouble remembering that. Please try again.",
                "metadata": {"error": str(e)}
            }

    def _analyze_content(self, content: str) -> tuple[str, int]:
        """
        Analyze content to determine category and importance

        Returns:
            (category, importance)
        """
        content_lower = content.lower()

        # Determine category
        category = "general"

        if any(word in content_lower for word in ["birthday", "anniversary", "date"]):
            category = "dates"
            importance = 8
        elif any(word in content_lower for word in ["like", "love", "favorite", "prefer", "enjoy"]):
            category = "preferences"
            importance = 7
        elif any(word in content_lower for word in ["name", "called", "email", "phone", "address"]):
            category = "personal_info"
            importance = 9
        elif any(word in content_lower for word in ["work", "job", "career", "company", "office"]):
            category = "professional"
            importance = 7
        elif any(word in content_lower for word in ["family", "friend", "spouse", "child", "parent"]):
            category = "relationships"
            importance = 8
        elif any(word in content_lower for word in ["hobby", "interest", "passion"]):
            category = "hobbies"
            importance = 6
        elif any(word in content_lower for word in ["allergy", "allergic", "medical", "health", "medication"]):
            category = "health"
            importance = 9
        elif any(word in content_lower for word in ["important", "critical", "must", "never forget"]):
            importance = 10
        else:
            importance = 5

        return category, importance
