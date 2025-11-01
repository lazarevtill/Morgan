"""
Morgan Assistant - Human-First AI Interface with Emotional Intelligence

The main interface for human interaction with Morgan RAG.
Designed to be intuitive, helpful, conversational, and emotionally aware.
Now includes companion features for building meaningful relationships.

KISS Principle: Clean interface that orchestrates specialized modules.
"""

import time
from typing import Optional, List, Iterator, Dict, Any

from ..config import get_settings
from ..utils.logger import get_logger
from ..core.knowledge import KnowledgeBase
from ..core.memory import ConversationMemory
from ..core.search import SmartSearch
from ..services.llm_service import LLMService
from ..emotional.intelligence_engine import get_emotional_intelligence_engine
from ..companion.relationship_manager import CompanionRelationshipManager
from ..memory.memory_processor import get_memory_processor

# Import our new modular components
from .response_handler import Response, ResponseHandler
from .conversation_manager import ConversationManager
from .emotional_processor import EmotionalProcessor
from .milestone_tracker import MilestoneTracker

logger = get_logger(__name__)


class MorganAssistant:
    """
    Morgan - Your Human-First AI Assistant with Emotional Intelligence
    
    Designed to be:
    - Conversational and helpful
    - Emotionally aware and empathetic
    - Transparent about sources and reasoning
    - Easy to interact with
    - Continuously learning from conversations
    - Building meaningful relationships over time
    
    KISS: Clean orchestrator that coordinates specialized modules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Morgan assistant with emotional intelligence.
        
        Args:
            config_path: Optional configuration file path
        """
        self.settings = get_settings(config_path)
        
        # Core components - each with single responsibility
        self.knowledge = KnowledgeBase()
        self.memory = ConversationMemory()
        self.search = SmartSearch()
        self.llm = LLMService()
        
        # Enhanced companion components
        self.emotional_engine = get_emotional_intelligence_engine()
        self.relationship_manager = CompanionRelationshipManager()
        self.memory_processor = get_memory_processor()
        
        # Specialized processors (new modular approach)
        self.response_handler = ResponseHandler()
        self.conversation_manager = ConversationManager()
        self.emotional_processor = EmotionalProcessor(
            self.emotional_engine,
            self.relationship_manager,
            self.memory_processor
        )
        self.milestone_tracker = MilestoneTracker()
        
        # Human-friendly state
        self.name = "Morgan"
        self.personality = "helpful, knowledgeable, conversational, and emotionally aware"
        
        logger.info(f"{self.name} assistant initialized with emotional intelligence and ready to help!")
    
    def ask(
        self, 
        question: str, 
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        max_context: Optional[int] = None
    ) -> Response:
        """
        Ask Morgan a question with emotional intelligence and companion awareness.
        
        Args:
            question: Your question in natural language
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID for personalization
            include_sources: Whether to include source references
            max_context: Maximum context length to use
            
        Returns:
            Response with answer, sources, suggestions, and emotional context
        """
        start_time = time.time()
        
        try:
            # Step 1: Create conversation context
            conversation_context = self.conversation_manager.create_conversation_context(
                question, conversation_id, user_id
            )
            
            # Step 2: Analyze emotional state
            emotional_state = self.emotional_engine.analyze_emotion(
                question, conversation_context
            )
            
            # Step 3: Handle user profile and personalization
            user_profile = None
            if user_id:
                user_profile = self.emotional_processor.get_or_create_user_profile(user_id)
                
                # Generate personalized greeting if needed
                if self.emotional_processor.should_generate_greeting(user_profile):
                    self.emotional_processor.generate_personalized_greeting(user_profile)
                    logger.info("Generated personalized greeting for %s", user_id)
            
            # Step 4: Adapt conversation style
            conversation_style = None
            if user_profile:
                conversation_style = self.relationship_manager.adapt_conversation_style(
                    user_profile, emotional_state
                )
            
            # Step 5: Search for relevant knowledge
            search_results = self.search.find_relevant_info(
                query=question,
                max_results=self.settings.morgan_max_search_results
            )
            
            # Step 6: Get conversation memory context
            memory_context = ""
            if conversation_id:
                memory_context = self.memory.get_conversation_context(
                    conversation_id, max_turns=5
                )
            
            # Step 7: Build context for LLM
            context = self.conversation_manager.build_emotional_context(
                question=question,
                search_results=search_results,
                memory_context=memory_context,
                emotional_state=emotional_state,
                conversation_style=conversation_style,
                user_profile=user_profile,
                max_context=max_context or self.settings.morgan_max_context
            )
            
            # Step 8: Generate empathetic response
            empathetic_response = self.emotional_engine.generate_empathetic_response(
                emotional_state, context
            )
            
            # Step 9: Generate main response
            llm_response = self.llm.generate(
                prompt=f"Question: {question}",
                system_prompt=context
            )
            
            # Step 10: Process response
            enhanced_response = self.response_handler.enhance_response_with_emotion(
                llm_response.content, emotional_state, empathetic_response.empathy_level
            )
            
            # Step 11: Check for milestones
            milestone = None
            if user_profile:
                milestone = self.milestone_tracker.check_milestones(
                    user_profile, conversation_context, emotional_state
                )
                if milestone:
                    user_profile.relationship_milestones.append(milestone)
            
            # Step 12: Generate suggestions
            topics_discussed = self.conversation_manager.extract_topics_from_question(question)
            user_interests = user_profile.communication_preferences.topics_of_interest if user_profile else []
            suggestions = self.response_handler.generate_suggestions(
                question, context, user_interests
            )
            
            # Step 13: Create final response
            response = self.response_handler.create_response(
                answer=enhanced_response.get("answer", llm_response.content),
                sources=self.response_handler.extract_sources(search_results) if include_sources else [],
                confidence=0.8,
                thinking="Generated with emotional awareness",
                suggestions=suggestions,
                conversation_id=conversation_id,
                emotional_tone=enhanced_response.get("emotional_tone"),
                empathy_level=enhanced_response.get("empathy_level", 0.0),
                personalization_elements=enhanced_response.get("personalization_elements", []),
                relationship_context=empathetic_response.relationship_context if hasattr(empathetic_response, 'relationship_context') else None,
                milestone_celebration=milestone
            )
            
            # Step 14: Process memories and update profile
            if conversation_id:
                self.emotional_processor.process_conversation_memory(
                    conversation_context, emotional_state, response.answer, response.sources
                )
            
            if user_profile:
                self.emotional_processor.update_user_profile(
                    user_profile, conversation_context, emotional_state, 
                    response.confidence, topics_discussed
                )
            
            # Step 15: Log completion
            elapsed = time.time() - start_time
            logger.info(
                f"Morgan answered in {elapsed:.2f}s "
                f"(confidence: {response.confidence:.2f}, "
                f"emotion: {emotional_state.primary_emotion.value})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Morgan encountered an error: {e}")
            return self.response_handler.format_error_response(e)
    
    def ask_stream(
        self, 
        question: str, 
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Iterator[str]:
        """
        Ask Morgan a question with streaming response.
        
        Perfect for real-time chat interfaces where humans want to see
        Morgan "thinking" and responding in real-time.
        
        Args:
            question: Your question
            conversation_id: Optional conversation ID
            user_id: Optional user ID for personalization
            
        Yields:
            Chunks of the response as Morgan generates it
        """
        try:
            # Prepare context using our modular approach
            search_results = self.search.find_relevant_info(question)
            memory_context = ""
            if conversation_id:
                memory_context = self.memory.get_conversation_context(conversation_id)
            
            # Build context (use basic context for streaming to keep it fast)
            context = self.conversation_manager.build_basic_context(
                question, search_results, memory_context
            )
            
            # Stream the response
            full_response = ""
            for chunk in self.llm.stream_generate(
                prompt=f"Question: {question}",
                system_prompt=context
            ):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += chunk_text
                yield chunk_text
            
            # Remember the complete conversation
            if conversation_id:
                sources = self.response_handler.extract_sources(search_results)
                self.memory.add_turn(
                    conversation_id=conversation_id,
                    question=question,
                    answer=full_response,
                    sources=sources
                )
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\nI apologize, but I encountered an error: {str(e)}"
    
    def learn_from_documents(
        self, 
        source_path: str, 
        document_type: str = "auto",
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Teach Morgan new knowledge from documents.
        
        Human-friendly learning interface that makes it easy to add knowledge.
        
        Args:
            source_path: Path to documents or URL
            document_type: Type of documents (auto-detect by default)
            show_progress: Show progress to human
            
        Returns:
            Learning summary with human-readable statistics
        """
        logger.info(f"Morgan is learning from: {source_path}")
        
        try:
            result = self.knowledge.ingest_documents(
                source_path=source_path,
                document_type=document_type,
                show_progress=show_progress
            )
            
            # Human-friendly summary
            summary = {
                "success": True,
                "documents_processed": result.get("documents_processed", 0),
                "chunks_created": result.get("chunks_created", 0),
                "knowledge_areas": result.get("knowledge_areas", []),
                "learning_time": result.get("processing_time", 0),
                "message": f"Great! I've learned from {result.get('documents_processed', 0)} documents. "
                          f"I'm now more knowledgeable about: {', '.join(result.get('knowledge_areas', [])[:3])}"
            }
            
            logger.info(f"Morgan learned successfully: {summary['message']}")
            return summary
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "I had trouble learning from those documents. Could you check the path and try again?"
            }
    
    def start_conversation(
        self, 
        topic: Optional[str] = None, 
        user_id: Optional[str] = None
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
                greeting_obj = self.emotional_processor.generate_personalized_greeting(user_profile)
                greeting = greeting_obj.greeting_text if hasattr(greeting_obj, 'greeting_text') else f"Welcome back, {user_profile.preferred_name}!"
            else:
                greeting = f"Hello! I'm {self.name}, your emotionally intelligent AI companion."
        else:
            greeting = f"Hello! I'm {self.name}, your AI assistant with emotional intelligence."
        
        if topic:
            greeting += f" I understand you'd like to discuss {topic}."
        
        greeting += " How can I help you today?"
        
        logger.info(f"Started new conversation: {conversation_id} (user: {user_id or 'anonymous'})")
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
        user_id: Optional[str] = None
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
                conversation_id=conversation_id,
                rating=rating,
                comment=comment
            )
            
            # Process feedback for relationship building
            if success and user_id and rating >= 4:
                # High rating might indicate a milestone
                milestone_message = self.emotional_processor.celebrate_milestone(
                    user_id, "breakthrough_moment" if rating == 5 else "positive_feedback"
                )
                if milestone_message:
                    logger.info(f"Feedback triggered celebration: {milestone_message}")
            
            if success:
                logger.info(f"Received feedback: {rating}/5 - {comment or 'No comment'}")
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
                          f"covering {len(stats.get('topics', []))} different areas."
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": "Unable to retrieve knowledge statistics"}
    
    def celebrate_milestone(
        self,
        user_id: str,
        milestone_type: str,
        custom_message: Optional[str] = None
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
            greeting_obj = morgan.emotional_processor.generate_personalized_greeting(user_profile)
            greeting = greeting_obj.greeting_text if hasattr(greeting_obj, 'greeting_text') else f"Hello, {user_profile.preferred_name}!"
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
            
            if question.lower() in ['quit', 'exit', 'bye']:
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
                feedback_prompt = input("\n💭 How was that response? (1-5 or press Enter to skip): ").strip()
                if feedback_prompt.isdigit() and 1 <= int(feedback_prompt) <= 5:
                    morgan.provide_feedback(conversation_id, int(feedback_prompt), user_id=user_id)
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
    print(f"\nQuestion: What is Docker?")
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