"""
Morgan Assistant - Human-First AI Interface with Emotional Intelligence

The main interface for human interaction with Morgan RAG.
Designed to be intuitive, helpful, conversational, and emotionally aware.
Now includes companion features for building meaningful relationships.

KISS Principle: Clean interface that orchestrates specialized modules.
"""

import time
from typing import Any, Dict, Iterator, List, Optional

from ..companion.relationship_manager import CompanionRelationshipManager
from ..config import get_settings

# Legacy imports removed
from morgan.intelligence.core.intelligence_engine import (
    get_emotional_intelligence_engine,
)
from morgan.intelligence.core.models import ConversationContext, EmotionalState
from ..services.llm_service import LLMService
from ..utils.logger import get_logger
from .conversation_manager import ConversationManager
from .emotional_processor import EmotionalProcessor
from .milestone_tracker import MilestoneTracker

# Import our new modular components
from .response_handler import Response, ResponseHandler

# New imports for refactoring
from .knowledge import KnowledgeService
from .memory import MemoryService
from .application.orchestrators import ConversationOrchestrator

logger = get_logger(__name__)


class MorganAssistant:
    """
    Morgan - Your Human-First AI Assistant with Emotional Intelligence.
    A thin facade delegating to specialized services and orchestrators.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Morgan assistant with emotional intelligence.

        Args:
            config_path: Optional configuration file path
        """
        self.settings = get_settings(config_path)

        # Core services
        self.knowledge = KnowledgeService()
        self.memory = MemoryService()
        self.llm = LLMService()

        # Enhanced companion components
        self.emotional_engine = get_emotional_intelligence_engine()
        self.relationship_manager = CompanionRelationshipManager()
        self.memory_processor = get_memory_processor()

        # Specialized processors
        self.response_handler = ResponseHandler()
        self.conversation_manager = ConversationManager()
        self.emotional_processor = EmotionalProcessor(
            self.emotional_engine, self.relationship_manager, self.memory_processor
        )
        self.milestone_tracker = MilestoneTracker()

        # Orchestrator
        self.orchestrator = ConversationOrchestrator(
            self.knowledge, self.memory, self.llm, self.emotional_processor
        )

        # Human-friendly state
        self.name = "Morgan"
        self.personality = (
            "helpful, knowledgeable, conversational, and emotionally aware"
        )

        logger.info(f"{self.name} assistant initialized with clean DDD architecture!")

    async def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        max_context: Optional[int] = None,
    ) -> Response:
        """
        Ask Morgan a question with emotional intelligence and companion awareness.
        """
        return await self.orchestrator.answer_question(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
            include_sources=include_sources,
            max_context=max_context,
        )

    async def ask_stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Ask Morgan a question with streaming response.
        """
        # For now, delegating to LLM directly but using Knowledge/Memory
        search_results = self.knowledge.search_knowledge(question)
        memory_context = ""
        if conversation_id:
            memory_context = self.memory.get_conversation_context(conversation_id)

        context = f"Knowledge: {search_results}\nMemory: {memory_context}"

        full_response = ""
        for chunk in self.llm.stream_generate(
            prompt=f"Question: {question}", system_prompt=context
        ):
            chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += chunk_text
            yield chunk_text

        # Update memory at the end
        if conversation_id:
            self.memory.add_turn(
                conversation_id=conversation_id,
                question=question,
                answer=full_response,
                sources=[res.get("source") for res in search_results],
            )

    def learn_from_documents(
        self, source_path: str, document_type: str = "auto", show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Teach Morgan new knowledge from documents.
        """
        try:
            result = self.knowledge.ingest_documents(
                source_path=source_path,
                document_type=document_type,
                show_progress=show_progress,
            )
            return result
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return {"success": False, "error": str(e)}

    def start_conversation(
        self, topic: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """
        Start a new conversation with Morgan with companion awareness.

        Args:
            topic: Optional conversation topic
            user_id: Optional user ID for personalization

        Returns:
            Conversation ID for continuing the chat
        """
        conversation_id = self.memory.create_conversation(topic=topic)

        # Generate personalized greeting if user is known
        if user_id:
            user_profile = self.emotional_processor.get_or_create_user_profile(user_id)

            # Check if we should generate a personalized greeting
            if user_profile.interaction_count > 0:
                greeting_obj = self.emotional_processor.generate_personalized_greeting(
                    user_profile
                )
                greeting = (
                    greeting_obj.greeting_text
                    if hasattr(greeting_obj, "greeting_text")
                    else f"Welcome back, {user_profile.preferred_name}!"
                )
            else:
                greeting = f"Hello! I'm {self.name}, your emotionally intelligent AI companion."
        else:
            greeting = f"Hello! I'm {self.name}, your AI assistant with emotional intelligence."

        if topic:
            greeting += f" I understand you'd like to discuss {topic}."

        greeting += " How can I help you today?"

        logger.info(
            f"Started new conversation: {conversation_id} (user: {user_id or 'anonymous'})"
        )
        return conversation_id

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of conversation turns with questions and answers
        """
        return self.memory.get_conversation_history(conversation_id)

    def provide_feedback(
        self,
        conversation_id: str,
        rating: int,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Provide feedback to help Morgan learn and improve with emotional awareness.

        Args:
            conversation_id: ID of the conversation
            rating: Rating from 1-5 (5 being excellent)
            comment: Optional feedback comment
            user_id: Optional user ID for relationship tracking

        Returns:
            True if feedback was recorded successfully
        """
        try:
            success = self.memory.add_feedback(
                conversation_id=conversation_id, rating=rating, comment=comment
            )

            # Process feedback for relationship building
            if success and user_id and rating >= 4:
                # High rating might indicate a milestone
                milestone_message = self.emotional_processor.celebrate_milestone(
                    user_id,
                    "breakthrough_moment" if rating == 5 else "positive_feedback",
                )
                if milestone_message:
                    logger.info(f"Feedback triggered celebration: {milestone_message}")

            if success:
                logger.info(
                    f"Received feedback: {rating}/5 - {comment or 'No comment'}"
                )
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Morgan's knowledge base.

        Returns:
            Human-readable statistics about what Morgan knows
        """
        try:
            stats = self.knowledge.get_statistics()

            return {
                "total_documents": stats.get("document_count", 0),
                "knowledge_chunks": stats.get("chunk_count", 0),
                "knowledge_areas": stats.get("topics", []),
                "last_updated": stats.get("last_updated", "Unknown"),
                "storage_size": stats.get("storage_size_mb", 0),
                "message": f"I have knowledge from {stats.get('document_count', 0)} documents "
                f"covering {len(stats.get('topics', []))} different areas.",
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": "Unable to retrieve knowledge statistics"}

    def celebrate_milestone(
        self, user_id: str, milestone_type: str, custom_message: Optional[str] = None
    ) -> str:
        """
        Celebrate a relationship milestone with the user.

        Args:
            user_id: User identifier
            milestone_type: Type of milestone to celebrate
            custom_message: Optional custom celebration message

        Returns:
            Celebration message
        """
        return self.emotional_processor.celebrate_milestone(
            user_id, milestone_type, custom_message
        )

    def get_relationship_insights(self, user_id: str):
        """
        Get insights about the relationship with a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with relationship insights
        """
        return self.emotional_processor.get_relationship_insights(user_id)

    def suggest_conversation_topics(self, user_id: str) -> List[str]:
        """
        Suggest conversation topics based on user interests and history.

        Args:
            user_id: User identifier

        Returns:
            List of suggested topics
        """
        return self.emotional_processor.suggest_conversation_topics(user_id)

    def get_milestone_statistics(self, user_id: str):
        """Get milestone statistics for a user."""
        user_profile = self.emotional_processor.get_or_create_user_profile(user_id)
        return self.milestone_tracker.get_milestone_statistics(user_profile)

    def __str__(self) -> str:
        """Human-friendly string representation."""
        return f"{self.name} - Your Emotionally Intelligent AI Companion (Ready to help and understand!)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"MorganAssistant(name='{self.name}', personality='{self.personality}', emotional_intelligence=True)"


# Human-friendly helper functions
def quick_ask(question: str, user_id: Optional[str] = None) -> str:
    """
    Quick way to ask Morgan a question with emotional intelligence.

    Args:
        question: Your question
        user_id: Optional user ID for personalization

    Returns:
        Morgan's answer as a string
    """
    morgan = MorganAssistant()
    response = morgan.ask(question, user_id=user_id)
    return response.answer


def chat_with_morgan(user_id: Optional[str] = None):
    """
    Start an interactive chat session with Morgan with emotional intelligence.

    Perfect for command-line usage where humans want to have
    a natural, emotionally aware conversation with Morgan.

    Args:
        user_id: Optional user ID for personalized experience
    """
    morgan = MorganAssistant()
    conversation_id = morgan.start_conversation(user_id=user_id)

    # Generate personalized greeting
    if user_id:
        user_profile = morgan.emotional_processor.get_or_create_user_profile(user_id)
        if user_profile.interaction_count > 0:
            greeting_obj = morgan.emotional_processor.generate_personalized_greeting(
                user_profile
            )
            greeting = (
                greeting_obj.greeting_text
                if hasattr(greeting_obj, "greeting_text")
                else f"Hello, {user_profile.preferred_name}!"
            )
        else:
            greeting = f"Hello! I'm {morgan.name}, your emotionally intelligent AI companion. Type 'quit' to exit."
    else:
        greeting = f"Hello! I'm {morgan.name}, your AI assistant with emotional intelligence. Type 'quit' to exit."

    print(f"\n🤖 {morgan.name}: {greeting}\n")

    # Show conversation topics if user is known
    if user_id:
        topics = morgan.suggest_conversation_topics(user_id)
        if topics and len(topics) > 1:
            print("💡 Some things we could talk about:")
            for i, topic in enumerate(topics[:3], 1):
                print(f"   {i}. {topic}")
            print()

    while True:
        try:
            question = input("👤 You: ").strip()

            if question.lower() in ["quit", "exit", "bye"]:
                # Generate personalized goodbye
                if user_id:
                    user_profile = morgan.relationship_manager.profiles.get(user_id)
                    if user_profile and user_profile.preferred_name != user_id:
                        goodbye = f"Goodbye, {user_profile.preferred_name}! I've enjoyed our conversation."
                    else:
                        goodbye = "Goodbye! I've enjoyed getting to know you better."
                else:
                    goodbye = "Goodbye! Feel free to ask me anything anytime."

                print(f"\n🤖 {morgan.name}: {goodbye}")
                break

            if not question:
                continue

            print(f"\n🤖 {morgan.name}: ", end="", flush=True)

            # Get response with emotional intelligence
            response = morgan.ask(question, conversation_id, user_id)

            # Show emotional context if significant
            if response.empathy_level > 0.7:
                print(f"[Speaking with {response.emotional_tone}]")

            print(response.answer)

            # Show milestone celebration if any
            if response.milestone_celebration:
                celebration_msg = morgan.milestone_tracker.generate_celebration_message(
                    response.milestone_celebration
                )
                print(f"\n🎉 Milestone: {celebration_msg}")

            # Ask for feedback occasionally
            import random

            if random.random() < 0.1:  # 10% chance
                feedback_prompt = input(
                    "\n💭 How was that response? (1-5 or press Enter to skip): "
                ).strip()
                if feedback_prompt.isdigit() and 1 <= int(feedback_prompt) <= 5:
                    morgan.provide_feedback(
                        conversation_id, int(feedback_prompt), user_id=user_id
                    )
                    print("Thank you for the feedback! 😊")

            print()

        except KeyboardInterrupt:
            print(f"\n\n🤖 {morgan.name}: Goodbye!")
            break
        except Exception as e:
            print(f"\n🤖 {morgan.name}: I encountered an error: {e}")


if __name__ == "__main__":
    # Demo Morgan's enhanced capabilities with emotional intelligence
    print("🤖 Morgan RAG Assistant with Emotional Intelligence Demo")
    print("=" * 60)

    # Quick test with emotional awareness
    morgan = MorganAssistant()
    print(f"Assistant: {morgan}")

    # Test question with user ID for personalization
    test_user_id = "demo_user"
    response = morgan.ask("What is Docker?", user_id=test_user_id)
    print("\nQuestion: What is Docker?")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Emotional Tone: {response.emotional_tone}")
    print(f"Empathy Level: {response.empathy_level:.2f}")

    # Show relationship insights
    insights = morgan.get_relationship_insights(test_user_id)
    print(f"\nRelationship Insights: {insights}")

    # Show suggested topics
    topics = morgan.suggest_conversation_topics(test_user_id)
    print(f"\nSuggested Topics: {topics}")

    # Interactive chat with emotional intelligence
    print("\nStarting emotionally intelligent interactive chat...")
    chat_with_morgan(test_user_id)
