"""
Application layer orchestrators for Morgan Core.
Centralizes the end-to-end flows for conversation and learning.

Features:
- Multi-source knowledge retrieval (local + external)
- Emotional intelligence integration
- MCP server integration for web search and Context7 docs
- Multi-step reasoning for complex queries
- Proactive assistance (anticipation + suggestions)
"""

import time
import uuid
from datetime import datetime
from typing import Any, List, Optional

from morgan.config import get_settings
from morgan.core.knowledge import KnowledgeService
from morgan.core.memory import MemoryService
from morgan.core.response_handler import Response
from morgan.services.llm import LLMService
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
    Emotional, Reasoning, and Proactive services.

    Features:
    - Local knowledge base search
    - External knowledge via MCP (web search, Context7)
    - Emotional intelligence integration
    - Multi-turn conversation context
    - Multi-step reasoning for complex queries
    - Proactive suggestions and anticipation
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        memory_service: MemoryService,
        llm_service: LLMService,
        emotional_processor: Any,  # Avoid circular imports
        external_knowledge_service: Optional[ExternalKnowledgeService] = None,
        enable_reasoning: bool = True,
        enable_proactive: bool = True,
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

        # Feature flags
        self.enable_reasoning = enable_reasoning
        self.enable_proactive = enable_proactive

        # Proactive services (lazy loaded)
        self._full_reasoning_engine = None
        self._task_planner = None
        self._context_monitor = None
        self._task_anticipator = None
        self._suggestion_engine = None

        # Initialize proactive services if enabled
        if enable_proactive:
            self._init_proactive_services()

    def _init_proactive_services(self):
        """Initialize proactive assistance services."""
        try:
            from morgan.proactive.monitor import ContextMonitor, get_context_monitor
            from morgan.proactive.anticipator import (
                TaskAnticipator,
                get_task_anticipator,
            )
            from morgan.proactive.suggestions import (
                SuggestionEngine,
                get_suggestion_engine,
            )

            self._context_monitor = get_context_monitor()
            self._task_anticipator = get_task_anticipator()
            self._suggestion_engine = get_suggestion_engine()

            logger.info("Proactive services initialized")

        except ImportError as e:
            logger.warning(f"Proactive services not available: {e}")
            self.enable_proactive = False

    def _get_full_reasoning_engine(self):
        """Get full multi-step reasoning engine (lazy loaded)."""
        if self._full_reasoning_engine is None:
            try:
                from morgan.reasoning.engine import (
                    ReasoningEngine as FullReasoningEngine,
                )

                self._full_reasoning_engine = FullReasoningEngine(
                    llm_service=self.llm,
                    max_steps=10,
                    min_confidence=0.7,
                    enable_reflection=True,
                )
            except ImportError:
                logger.warning("Full reasoning engine not available")
                return None
        return self._full_reasoning_engine

    def _get_task_planner(self):
        """Get task planner for complex multi-step workflows (lazy loaded)."""
        if self._task_planner is None:
            try:
                from morgan.reasoning.planner import TaskPlanner

                self._task_planner = TaskPlanner(
                    llm_service=self.llm,
                    max_concurrent_tasks=3,
                )
            except ImportError:
                logger.warning("Task planner not available")
                return None
        return self._task_planner

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        max_context: Optional[int] = None,
        use_external_knowledge: bool = True,
        use_deep_reasoning: bool = False,
    ) -> Any:
        """
        Ask Morgan a question with emotional intelligence and external knowledge.

        Args:
            question: The question to answer
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID for personalization
            include_sources: Whether to include sources in response
            max_context: Maximum context tokens
            use_external_knowledge: Whether to use web search/Context7
            use_deep_reasoning: Use multi-step reasoning for complex queries
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

            # Step 6: Check for anticipated task (proactive)
            anticipated_response = None
            if self.enable_proactive and self._task_anticipator:
                match = await self._task_anticipator.check_match(uid, question)
                if match and match.prepared_response:
                    anticipated_response = match.prepared_response
                    logger.info(f"Using anticipated response for query")

            # Step 7: Search for relevant knowledge (local)
            search_results = self.knowledge.search_knowledge(
                query=search_query, max_results=self.settings.morgan_max_search_results
            )

            # Step 7b: Get external knowledge (web search, Context7)
            external_results: List[ExternalKnowledgeResult] = []
            if use_external_knowledge:
                external_results = await self._fetch_external_knowledge(search_query)

            # Step 8: Build context for LLM
            knowledge_context = self._build_knowledge_context(
                search_results, external_results
            )
            context = (
                f"Emotional State: {emotional_state}\n"
                f"Style: {conversation_style}\n"
                f"Knowledge:\n{knowledge_context}\n"
                f"Memory: {memory_context}"
            )

            # Step 9: Generate response (with optional deep reasoning)
            if use_deep_reasoning and self.enable_reasoning:
                llm_response_content = await self._generate_with_reasoning(
                    question, context
                )
            elif anticipated_response:
                llm_response_content = anticipated_response
            else:
                # Standard generation
                empathetic_response = self.emotional_processor.emotional_engine.generate_empathetic_response(
                    emotional_state, context
                )

                llm_response = self.llm.generate(
                    prompt=f"Question: {question}", system_prompt=context
                )
                llm_response_content = llm_response.content

            # Step 10: Check for milestones
            milestone = None
            if user_profile:
                milestone = self.emotional_processor.check_for_milestones(
                    user_profile, conv_context, emotional_state
                )

            # Step 11: Create final response
            all_sources = self._collect_sources(
                search_results, external_results, include_sources
            )

            # Get empathetic response for metadata
            empathetic_response = (
                self.emotional_processor.emotional_engine.generate_empathetic_response(
                    emotional_state, context
                )
            )

            response = Response(
                answer=llm_response_content,
                sources=all_sources,
                confidence=0.8,
                conversation_id=conv_id,
                emotional_tone=empathetic_response.emotional_tone,
                empathy_level=empathetic_response.empathy_level,
                personalization_elements=(empathetic_response.personalization_elements),
                milestone_celebration=(milestone.description if milestone else None),
            )

            # Step 12: Process memories and update profile
            if conversation_id:
                self.emotional_processor.process_conversation_memory(
                    conv_context, emotional_state, response.answer, response.sources
                )

            if user_profile:
                self.emotional_processor.update_user_profile(
                    user_profile, conv_context, emotional_state, response.confidence, []
                )

            # Step 13: Update proactive context and anticipate next tasks
            if self.enable_proactive:
                await self._update_proactive_context(
                    uid, question, response.answer, emotional_state
                )

            logger.info(f"Answered in {time.time() - start_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise

    async def _generate_with_reasoning(self, question: str, context: str) -> str:
        """Generate response using multi-step reasoning."""
        reasoning_engine = self._get_full_reasoning_engine()

        if reasoning_engine is None:
            # Fall back to standard generation
            response = self.llm.generate(
                prompt=f"Question: {question}",
                system_prompt=context,
            )
            return response.content

        try:
            result = await reasoning_engine.reason(
                query=question,
                context=context,
            )

            logger.info(
                f"Reasoning completed: {len(result.steps)} steps, "
                f"confidence: {result.overall_confidence:.2f}"
            )

            return result.final_answer

        except Exception as e:
            logger.warning(f"Reasoning failed, falling back to standard: {e}")
            response = self.llm.generate(
                prompt=f"Question: {question}",
                system_prompt=context,
            )
            return response.content

    async def _update_proactive_context(
        self,
        user_id: str,
        query: str,
        response: str,
        emotional_state: Any,
    ):
        """Update proactive context and anticipate next tasks."""
        try:
            # Update context monitor
            if self._context_monitor:
                await self._context_monitor.update_context(
                    user_id=user_id,
                    query=query,
                    emotional_state=str(emotional_state) if emotional_state else None,
                )

            # Anticipate next tasks
            if self._task_anticipator:
                await self._task_anticipator.anticipate(
                    user_id=user_id,
                    current_query=query,
                    response=response,
                )

        except Exception as e:
            logger.warning(f"Failed to update proactive context: {e}")

    async def get_suggestions(
        self,
        user_id: str,
        count: int = 3,
    ) -> List[Any]:
        """
        Get proactive suggestions for a user.

        Args:
            user_id: User identifier
            count: Number of suggestions

        Returns:
            List of suggestions
        """
        if not self.enable_proactive or not self._suggestion_engine:
            return []

        try:
            return await self._suggestion_engine.generate_suggestions(
                user_id=user_id,
                count=count,
            )
        except Exception as e:
            logger.warning(f"Failed to generate suggestions: {e}")
            return []

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

    async def reason_about(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> Any:
        """
        Perform deep multi-step reasoning about a question.

        Use this for complex analytical questions that benefit from
        step-by-step reasoning, problem decomposition, and reflection.

        Args:
            question: The question to reason about
            context: Optional additional context

        Returns:
            ReasoningResult with steps, conclusion, and confidence
        """
        reasoning_engine = self._get_full_reasoning_engine()

        if reasoning_engine is None:
            raise RuntimeError("Reasoning engine not available")

        return await reasoning_engine.reason(
            query=question,
            context=context,
        )

    async def create_task_plan(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> Any:
        """
        Create a task plan to achieve a complex goal.

        Decomposes the goal into discrete tasks with dependencies,
        priorities, and execution order.

        Args:
            goal: The goal to achieve
            context: Optional additional context

        Returns:
            TaskPlan with decomposed tasks

        Example:
            >>> plan = await orchestrator.create_task_plan(
            ...     "Research and summarize recent AI safety developments"
            ... )
            >>> print(f"Plan has {len(plan.tasks)} tasks")
            >>> for task in plan.tasks:
            ...     print(f"  - {task.name}: {task.description}")
        """
        planner = self._get_task_planner()

        if planner is None:
            raise RuntimeError("Task planner not available")

        return await planner.create_plan(goal=goal, context=context)

    async def execute_task_plan(
        self,
        plan: Any,
        on_progress: Optional[Any] = None,
    ) -> Any:
        """
        Execute a task plan created by create_task_plan().

        Executes tasks in dependency order with progress tracking.

        Args:
            plan: TaskPlan to execute
            on_progress: Optional callback for progress updates

        Returns:
            Updated TaskPlan with results

        Example:
            >>> plan = await orchestrator.create_task_plan("Research AI safety")
            >>> result = await orchestrator.execute_task_plan(
            ...     plan,
            ...     on_progress=lambda p: print(f"Progress: {p.progress:.0%}")
            ... )
            >>> if result.is_complete:
            ...     for task in result.tasks:
            ...         print(f"{task.name}: {task.result}")
        """
        planner = self._get_task_planner()

        if planner is None:
            raise RuntimeError("Task planner not available")

        return await planner.execute_plan(plan=plan, on_progress=on_progress)

    async def accomplish_goal(
        self,
        goal: str,
        context: Optional[str] = None,
        on_progress: Optional[Any] = None,
    ) -> Any:
        """
        Create and execute a complete task plan for a goal.

        This is a convenience method that combines create_task_plan()
        and execute_task_plan() for simple use cases.

        Args:
            goal: The goal to achieve
            context: Optional additional context
            on_progress: Optional callback for progress updates

        Returns:
            Completed TaskPlan with all results

        Example:
            >>> result = await orchestrator.accomplish_goal(
            ...     "Create a comprehensive comparison of Python web frameworks",
            ...     on_progress=lambda p: print(f"Progress: {p.progress:.0%}")
            ... )
            >>> print(f"Goal achieved: {result.is_complete}")
        """
        plan = await self.create_task_plan(goal=goal, context=context)
        return await self.execute_task_plan(plan=plan, on_progress=on_progress)

    async def start_proactive_monitoring(self):
        """Start proactive monitoring services."""
        if self._context_monitor:
            await self._context_monitor.start()

        if self._task_anticipator:
            await self._task_anticipator.start_preparation_worker()

        logger.info("Proactive monitoring started")

    async def stop_proactive_monitoring(self):
        """Stop proactive monitoring services."""
        if self._context_monitor:
            await self._context_monitor.stop()

        if self._task_anticipator:
            await self._task_anticipator.stop_preparation_worker()

        logger.info("Proactive monitoring stopped")
