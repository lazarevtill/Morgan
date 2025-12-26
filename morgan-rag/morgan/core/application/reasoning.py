"""
Reasoning Engine for Morgan Core.
Handles query contextualization and intent analysis for multi-turn conversations.
"""

from typing import Any, Dict, List, Optional
from morgan.services.llm import LLMService
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine:
    """
    Engine for multi-turn reasoning and query contextualization.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def contextualize_query(
        self, question: str, history: List[Dict[str, str]]
    ) -> str:
        """
        Convert a potentially context-dependent question into a standalone query.

        Args:
            question: The current user question.
            history: List of recent conversation turns (question, answer).

        Returns:
            A standalone query for searching knowledge.
        """
        if not history:
            return question

        # Build history string
        history_str = ""
        for turn in history[-3:]:  # Only look at last 3 turns
            history_str += f"User: {turn['question']}\nMorgan: {turn['answer']}\n"

        prompt = f"""
Given the following conversation history and a new user question, 
determine if the new question depends on the context of the history.
If it does, rewrite it as a standalone, complete question that can be understood without the history.
If it is already standalone, return it as is.

Conversation History:
{history_str}

New Question: {question}

Standalone Question:"""

        try:
            # We want a very short, direct response
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a query rewriting assistant. Only output the rewritten question and nothing else.",
                max_tokens=50,
            )
            contextualized = response.content.strip()
            logger.debug(f"Contextualized query: '{question}' -> '{contextualized}'")
            return contextualized
        except Exception as e:
            logger.error(f"Failed to contextualize query: {e}")
            return question

    def build_system_prompt(
        self, emotional_state: Any, style: Any, profile: Any
    ) -> str:
        """
        Build a comprehensive system prompt based on context.
        """
        prompt = "You are Morgan, a helpful, knowledgeable, and emotionally aware AI assistant.\n"

        if emotional_state:
            prompt += f"The user is currently feeling {emotional_state.primary_emotion.value}.\n"
            prompt += "Respond with appropriate empathy and support.\n"

        if profile:
            prompt += (
                f"You are talking to {profile.get('preferred_name', 'the user')}.\n"
            )

        return prompt
