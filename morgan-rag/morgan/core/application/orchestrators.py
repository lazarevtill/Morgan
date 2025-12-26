"""
Application layer orchestrators for Morgan Core.
Centralizes the end-to-end flows for conversation and learning.

Features:
- Multi-source knowledge retrieval (local + external)
- Emotional intelligence integration
- MCP server integration for web search and Context7 docs
"""

import time
import uuid
from datetime import datetime
from typing import Any, List, Optional

from morgan.config import get_settings
from morgan.core.knowledge import KnowledgeService
from morgan.core.memory import MemoryService
from morgan.core.response_handler import Response
from morgan.services.llm_service import LLMService
from morgan.services.external_knowledge import (
    ExternalKnowledgeService,
    ExternalKnowledgeResult,
    KnowledgeSource,
    get_external_knowledge_service,
)
from morgan.utils.logger import get_logger
from morgan.intelligence.core.models import ConversationContext
from .reasoning import ReasoningEngine

logger = get_logger(__name__)


class ConversationOrchestrator:
    """
    Orchestrates the end-to-end conversation flow.

    Coordinates Knowledge, Memory, LLM, External Knowledge,
    and Emotional services.

    Features:
    - Local knowledge base search
    - External knowledge via MCP (web search, Context7)
    - Emotional intelligence integration
    - Multi-turn conversation context
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        memory_service: MemoryService,
        llm_service: LLMService,
        emotional_processor: Any,  # Avoid circular imports
        external_knowledge_service: Optional[ExternalKnowledgeService] = None,
    ):
        self.knowledge = knowledge_service
        self.memory = memory_service
        self.llm = llm_service
        self.emotional_processor = emotional_processor
        self.reasoning = ReasoningEngine(llm_service)
        self.settings = get_settings()

        # External knowledge service (web search, Context7)
        self.external_knowledge = (
            external_knowledge_service or get_external_knowledge_service()
        )

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        max_context: Optional[int] = None,
        use_external_knowledge: bool = True,
    ) -> Any:
        """
        Ask Morgan a question with emotional intelligence and external.

        Args:
            question: The question to answer
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID for personalization
            include_sources: Whether to include sources in response
            max_context: Maximum context tokens
            use_external_knowledge: Whether to use web search/Context7
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
            search_query = await self.reasoning.contextualize_query(
                question, memory_history
            )

            # Create conversation context
            conv_context = ConversationContext(
                user_id=uid,
                conversation_id=conv_id,
                message_text=question,
                timestamp=datetime.utcnow(),
                previous_messages=[m.get("question", "") for m in memory_history],
            )

            # Step 3: Analyze emotional state
            emotional_state = self.emotional_processor.emotional_engine.analyze_emotion(
                question, conv_context
            )

            # Step 4: Handle user profile and personalization
            user_profile = None
            if user_id:
                processor = self.emotional_processor
                user_profile = processor.get_or_create_user_profile(user_id)

            # Step 5: Adapt conversation style
            conversation_style = None
            if user_profile:
                conversation_style = self.emotional_processor.relationship_manager.adapt_conversation_style(
                    user_profile, emotional_state
                )

            # Step 6: Search for relevant knowledge (local)
            search_results = self.knowledge.search_knowledge(
                query=search_query, max_results=self.settings.morgan_max_search_results
            )

            # Step 6b: Get external knowledge (web search, Context7)
            external_results: List[ExternalKnowledgeResult] = []
            if use_external_knowledge:
                external_results = await self._fetch_external_knowledge(search_query)

            # Step 7: Build context for LLM
            knowledge_context = self._build_knowledge_context(
                search_results, external_results
            )
            context = (
                f"Emotional State: {emotional_state}\n"
                f"Style: {conversation_style}\n"
                f"Knowledge:\n{knowledge_context}\n"
                f"Memory: {memory_context}"
            )

            # Step 8 & 9: Generate responses
            empathetic_response = (
                self.emotional_processor.emotional_engine.generate_empathetic_response(
                    emotional_state, context
                )
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

            # Step 13: Create final response with empathetic context
            all_sources = self._collect_sources(
                search_results, external_results, include_sources
            )
            response = Response(
                answer=llm_response.content,
                sources=all_sources,
                confidence=0.8,
                conversation_id=conv_id,
                emotional_tone=empathetic_response.emotional_tone,
                empathy_level=empathetic_response.empathy_level,
                personalization_elements=(empathetic_response.personalization_elements),
                milestone_celebration=(milestone.description if milestone else None),
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

    async def _fetch_external_knowledge(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[ExternalKnowledgeResult]:
        """
        Fetch external knowledge using MCP services.

        Uses intelligent routing to determine best sources
        (web search for current info, Context7 for docs).
        """
        try:
            return await self.external_knowledge.intelligent_search(
                query=query,
                max_results=max_results,
            )
        except Exception as e:
            logger.warning(f"External knowledge fetch failed: {e}")
            return []

    def _build_knowledge_context(
        self,
        local_results: List[Any],
        external_results: List[ExternalKnowledgeResult],
    ) -> str:
        """Build combined knowledge context from all sources."""
        context_parts = []

        # Local knowledge
        if local_results:
            context_parts.append("=== Local Knowledge Base ===")
            for result in local_results[:5]:
                content = result.get("content", str(result))[:500]
                source = result.get("source", "local")
                context_parts.append(f"[{source}]: {content}")

        # External knowledge
        if external_results:
            context_parts.append("\n=== External Sources ===")
            for result in external_results[:5]:
                source_type = result.source.value
                title = result.title or "External"
                content = result.content[:500]
                context_parts.append(f"[{source_type}:{title}]: {content}")

                # Include code examples if available
                if result.code_examples:
                    num_examples = len(result.code_examples)
                    context_parts.append(f"  Code examples: {num_examples} available")

        if context_parts:
            return "\n".join(context_parts)
        return "No relevant knowledge found"

    def _collect_sources(
        self,
        local_results: List[Any],
        external_results: List[ExternalKnowledgeResult],
        include_sources: bool,
    ) -> List[str]:
        """Collect all sources from local and external results."""
        if not include_sources:
            return []

        sources = []

        # Local sources
        for result in local_results:
            source = result.get("source")
            if source:
                sources.append(source)

        # External sources
        for result in external_results:
            if result.source_url:
                sources.append(result.source_url)
            elif result.title:
                sources.append(f"{result.source.value}:{result.title}")

        return sources

    async def search_external(
        self,
        query: str,
        source: KnowledgeSource = KnowledgeSource.ALL,
        max_results: int = 10,
    ) -> List[ExternalKnowledgeResult]:
        """
        Search external knowledge sources directly.

        Useful for explicit external lookups without full conversation flow.

        Args:
            query: Search query
            source: Knowledge source (WEB, CONTEXT7, ALL)
            max_results: Maximum results

        Returns:
            List of external knowledge results
        """
        return await self.external_knowledge.search(
            query=query,
            sources=source,
            max_results=max_results,
        )

    async def get_library_docs(
        self,
        library_name: str,
        topic: Optional[str] = None,
    ) -> Optional[ExternalKnowledgeResult]:
        """
        Get documentation for a specific library via Context7.

        Args:
            library_name: Name of the library (e.g., "fastapi", "react")
            topic: Optional specific topic

        Returns:
            Documentation result or None if not found
        """
        return await self.external_knowledge.get_library_info(
            library_name=library_name,
            topic=topic,
            include_examples=True,
        )

    async def get_best_practices(
        self,
        technology: str,
        topic: Optional[str] = None,
    ) -> List[ExternalKnowledgeResult]:
        """
        Get best practices for a technology.

        Args:
            technology: Technology name
            topic: Optional specific topic

        Returns:
            List of best practice results from docs and web
        """
        return await self.external_knowledge.get_best_practices(
            technology=technology,
            topic=topic,
        )
