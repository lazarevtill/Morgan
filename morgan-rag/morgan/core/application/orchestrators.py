"""
Application layer orchestrators for Morgan Core.
Centralizes the end-to-end flows for conversation and learning.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.core.domain.entities import ConversationTurn
from morgan.core.knowledge import KnowledgeService
from morgan.core.memory import MemoryService
from morgan.services.llm_service import LLMService
from morgan.utils.logger import get_logger
from morgan.intelligence.core.models import ConversationContext
from .reasoning import ReasoningEngine

logger = get_logger(__name__)


class ConversationOrchestrator:
    """
    Orchestrates the end-to-end conversation flow.
    Coordinates Knowledge, Memory, LLM, and Emotional services.
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        memory_service: MemoryService,
        llm_service: LLMService,
        emotional_processor: Any, # Avoid circular imports
    ):
        self.knowledge = knowledge_service
        self.memory = memory_service
        self.llm = llm_service
        self.emotional_processor = emotional_processor
        self.reasoning = ReasoningEngine(llm_service)
        self.settings = get_settings()

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        max_context: Optional[int] = None,
    ) -> Any:
        """
        Ask Morgan a question with emotional intelligence and companion awareness.
        """
        start_time = time.time()
        
        # Ensure IDs
        conv_id = conversation_id or str(uuid.uuid4())
        uid = user_id or "anonymous"

        try:
            # Step 1: Get conversation memory context
            memory_history = []
            memory_context = ""
            if conversation_id:
                history_turns = self.memory.get_conversation_history(conversation_id)
                memory_history = [
                    {"question": t.get("question", ""), "answer": t.get("answer", "")}
                    for t in history_turns[-3:]
                ]
                memory_context = self.memory.get_conversation_context(
                    conversation_id, max_turns=5
                )

            # Step 2: Contextualize query for multi-turn reasoning
            search_query = await self.reasoning.contextualize_query(question, memory_history)

            # Create conversation context
            conv_context = ConversationContext(
                user_id=uid,
                conversation_id=conv_id,
                message_text=question,
                timestamp=datetime.utcnow(),
                previous_messages=[m.get("question", "") for m in memory_history]
            )

            # Step 3: Analyze emotional state
            emotional_state = self.emotional_processor.emotional_engine.analyze_emotion(
                question, conv_context
            )

            # Step 4: Handle user profile and personalization
            user_profile = None
            if user_id:
                user_profile = self.emotional_processor.get_or_create_user_profile(user_id)

            # Step 5: Adapt conversation style
            conversation_style = None
            if user_profile:
                conversation_style = self.emotional_processor.relationship_manager.adapt_conversation_style(
                    user_profile, emotional_state
                )

            # Step 6: Search for relevant knowledge using contextualized query
            search_results = self.knowledge.search_knowledge(
                query=search_query, max_results=self.settings.morgan_max_search_results
            )

            # Step 7: Build context for LLM
            # (Simplified for now, should ideally use a context builder)
            context = f"Emotional State: {emotional_state}\nStyle: {conversation_style}\nKnowledge: {search_results}\nMemory: {memory_context}"

            # Step 8 & 9: Generate responses
            empathetic_response = self.emotional_processor.emotional_engine.generate_empathetic_response(
                emotional_state, context
            )
            
            llm_response = self.llm.generate(
                prompt=f"Question: {question}", system_prompt=context
            )

            # Step 11: Check for milestones
            milestone = None
            if user_profile:
                milestone = self.emotional_processor.check_for_milestones(
                    user_profile, conv_context, emotional_state
                )

            # Step 13: Create final response
            # (This logic should probably be in a ResponseHandler service)
            from .response_handler import Response # Local import for now
            
            response = Response(
                answer=llm_response.content,
                sources=[res.get("source") for res in search_results] if include_sources else [],
                confidence=0.8,
                conversation_id=conv_id,
                emotional_tone=emotional_state.primary_emotion.value,
                milestone_celebration=milestone.description if milestone else None,
            )

            # Step 14: Process memories and update profile
            if conversation_id:
                self.emotional_processor.process_conversation_memory(
                    conv_context, emotional_state, response.answer, response.sources
                )

            if user_profile:
                self.emotional_processor.update_user_profile(
                    user_profile, conv_context, emotional_state, response.confidence, []
                )

            logger.info(f"Answered in {time.time() - start_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise
