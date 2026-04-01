"""
Application layer orchestrators for Morgan Core.
Centralizes the end-to-end flows for conversation and learning.

Features:
- Multi-source knowledge retrieval (local + external)
- Emotional intelligence integration
- MCP server integration for web search and Context7 docs
- Multi-step reasoning for complex queries
- Proactive assistance (anticipation + suggestions)
- Conversation intelligence (flow, quality, topics, interruptions)
- Communication adaptation (style, nonverbal, cultural, preferences)
- Learning engine integration
- Habit detection and adaptation
"""

import json
import time
import uuid
from datetime import datetime, timezone
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
from morgan.compaction import AutoCompactTracker, Compactor, should_auto_compact
from morgan.hook_system import HookManager, HookType
from morgan.memory_consolidation import DailyLogManager
from morgan.security import MemoryGate
from morgan.tools import ToolExecutor
from morgan.utils.logger import get_logger
from morgan.intelligence.core.models import ConversationContext
from morgan.workspace import WorkspaceManager
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
    - Conversation intelligence (flow management, quality assessment)
    - Communication adaptation (style, nonverbal, cultural awareness)
    - Learning engine for behavioral adaptation
    - Habit detection and wellness tracking
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
        tool_executor: Optional[ToolExecutor] = None,
        hook_manager: Optional[HookManager] = None,
        workspace_manager: Optional[WorkspaceManager] = None,
        auto_compact_tracker: Optional[AutoCompactTracker] = None,
        compactor: Optional[Compactor] = None,
        daily_log_manager: Optional[DailyLogManager] = None,
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
        self._tool_executor = tool_executor
        self._hook_manager = hook_manager
        self._workspace_manager = workspace_manager
        self._auto_compact_tracker = auto_compact_tracker
        self._compactor = compactor
        self._daily_log_manager = daily_log_manager
        self._memory_gate = MemoryGate()
        self._session_histories: dict[str, list[dict[str, Any]]] = {}

        # Proactive services (lazy loaded)
        self._full_reasoning_engine = None
        self._task_planner = None
        self._context_monitor = None
        self._task_anticipator = None
        self._suggestion_engine = None

        # Conversation intelligence services (lazy loaded)
        self._flow_manager = None
        self._quality_assessor = None
        self._topic_learner = None
        self._interruption_handler = None

        # Communication services (lazy loaded)
        self._communication_adapter = None
        self._nonverbal_detector = None
        self._cultural_awareness = None
        self._comm_preference_learner = None

        # Learning services (lazy loaded)
        self._learning_engine = None

        # Habit services (lazy loaded)
        self._habit_detector = None
        self._habit_adaptation = None
        self._wellness_tracker = None

        # Initialize proactive services if enabled
        if enable_proactive:
            self._init_proactive_services()

        # Initialize feature module services
        self._init_conversation_services()
        self._init_communication_services()
        self._init_learning_services()
        self._init_habit_services()

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

    def _init_conversation_services(self):
        """Initialize conversation intelligence services."""
        try:
            from morgan.conversation.flow import get_conversation_flow_manager
            from morgan.conversation.quality import get_conversation_quality_assessor
            from morgan.conversation.topics import get_topic_preference_learner
            from morgan.conversation.interruption import get_interruption_handler

            settings = get_settings()
            if settings.morgan_enable_conversation_flow:
                self._flow_manager = get_conversation_flow_manager()
            if settings.morgan_enable_quality_assessment:
                self._quality_assessor = get_conversation_quality_assessor()
            self._topic_learner = get_topic_preference_learner()
            self._interruption_handler = get_interruption_handler()
            logger.info("Conversation intelligence services initialized")
        except ImportError as e:
            logger.warning(f"Conversation intelligence services not available: {e}")

    def _init_communication_services(self):
        """Initialize communication services."""
        try:
            from morgan.communication.style import get_communication_style_adapter
            from morgan.communication.nonverbal import get_nonverbal_cue_detector
            from morgan.communication.cultural import get_cultural_emotional_awareness
            from morgan.communication.preferences import get_user_preference_learner

            settings = get_settings()
            if settings.morgan_enable_communication_adapter:
                self._communication_adapter = get_communication_style_adapter()
            if settings.morgan_enable_nonverbal_detection:
                self._nonverbal_detector = get_nonverbal_cue_detector()
            if settings.morgan_enable_cultural_awareness:
                self._cultural_awareness = get_cultural_emotional_awareness()
            self._comm_preference_learner = get_user_preference_learner()
            logger.info("Communication services initialized")
        except ImportError as e:
            logger.warning(f"Communication services not available: {e}")

    def _init_learning_services(self):
        """Initialize learning engine services."""
        try:
            from morgan.learning.engine import get_learning_engine

            settings = get_settings()
            if settings.morgan_enable_learning:
                self._learning_engine = get_learning_engine()
            logger.info("Learning services initialized")
        except ImportError as e:
            logger.warning(f"Learning services not available: {e}")

    def _init_habit_services(self):
        """Initialize habit detection and adaptation services."""
        try:
            from morgan.habits.detector import HabitDetector
            from morgan.habits.adaptation import HabitBasedAdaptation
            from morgan.habits.wellness import WellnessHabitTracker

            settings = get_settings()
            if settings.morgan_enable_habits:
                self._habit_detector = HabitDetector()
                self._habit_adaptation = HabitBasedAdaptation()
                self._wellness_tracker = WellnessHabitTracker()
            logger.info("Habit services initialized")
        except ImportError as e:
            logger.warning(f"Habit services not available: {e}")

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
        session_type: str = "main",
    ) -> Any:
        """
        Ask Morgan a question with emotional intelligence and external knowledge.

        Extended pipeline with conversation intelligence, communication adaptation,
        learning, and habit-based personalization.
        """
        start_time = time.time()

        # Ensure IDs
        conv_id = conversation_id or str(uuid.uuid4())
        uid = user_id or "anonymous"

        await self._trigger_hook(
            HookType.MESSAGE_INBOUND,
            {
                "conversation_id": conv_id,
                "user_id": uid,
                "message": question,
                "session_type": session_type,
            },
        )

        # Module results (populated throughout pipeline)
        flow_result = None
        nonverbal_analysis = None
        cultural_context = None
        quality_assessment = None
        habit_adaptations = []
        learning_adaptations = None

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

            # Step 1b: Check for interruption (conversation intelligence)
            interruption_result = None
            if self._interruption_handler and memory_history:
                try:
                    previous_content = memory_history[-1].get("answer", "") if memory_history else ""
                    interruption_result = self._interruption_handler.detect_interruption(
                        current_message=question,
                        previous_context=previous_content,
                        conversation_id=conv_id,
                        user_id=uid,
                    )
                except Exception as e:
                    logger.debug(f"Interruption detection skipped: {e}")

            # Step 2: Contextualize query for multi-turn reasoning
            search_query = await self.reasoning.contextualize_query(
                question, memory_history
            )

            # Create conversation context
            conv_context = ConversationContext(
                user_id=uid,
                conversation_id=conv_id,
                message_text=question,
                timestamp=datetime.now(timezone.utc),
                previous_messages=[m.get("question", "") for m in memory_history],
            )

            # Step 2b: Detect non-verbal cues (communication module)
            if self._nonverbal_detector:
                try:
                    nonverbal_analysis = self._nonverbal_detector.analyze_text(
                        text=question,
                        context=conv_context,
                    )
                except Exception as e:
                    logger.debug(f"Non-verbal detection skipped: {e}")

            # Step 3: Analyze emotional state
            emotional_state = self.emotional_processor.emotional_engine.analyze_emotion(
                question, conv_context
            )

            # Step 3b: Manage conversation flow (conversation intelligence)
            if self._flow_manager:
                try:
                    flow_result = self._flow_manager.process_turn(
                        conversation_id=conv_id,
                        user_id=uid,
                        message=question,
                        emotional_state=emotional_state,
                    )
                except Exception as e:
                    logger.debug(f"Flow management skipped: {e}")

            # Step 4: Handle user profile and personalization
            user_profile = None
            if user_id:
                processor = self.emotional_processor
                user_profile = processor.get_or_create_user_profile(user_id)

            # Step 4b: Detect cultural context (communication module)
            if self._cultural_awareness:
                try:
                    cultural_context = self._cultural_awareness.detect_cultural_context(
                        user_id=uid,
                        message=question,
                        conversation_context=conv_context,
                    )
                except Exception as e:
                    logger.debug(f"Cultural context detection skipped: {e}")

            # Step 5: Adapt conversation style
            conversation_style = None
            if user_profile:
                conversation_style = self.emotional_processor.relationship_manager.adapt_conversation_style(
                    user_profile, emotional_state
                )

            # Step 5b: Enhanced style adaptation (communication module)
            comm_style_result = None
            if self._communication_adapter and user_profile:
                try:
                    comm_style_result = self._communication_adapter.adapt_style(
                        user_id=uid,
                        emotional_state=emotional_state,
                        conversation_context=conv_context,
                        cultural_context=cultural_context,
                    )
                except Exception as e:
                    logger.debug(f"Communication style adaptation skipped: {e}")

            # Step 5c: Apply habit-based adaptations
            if self._habit_adaptation:
                try:
                    habit_result = self._habit_adaptation.get_adaptations(
                        user_id=uid,
                        context=conv_context,
                    )
                    if habit_result:
                        habit_adaptations = [
                            a.description if hasattr(a, 'description') else str(a)
                            for a in (habit_result if isinstance(habit_result, list) else [habit_result])
                        ]
                except Exception as e:
                    logger.debug(f"Habit adaptation skipped: {e}")

            # Step 5d: Get behavioral adaptations from learning engine
            if self._learning_engine:
                try:
                    learning_adaptations = self._learning_engine.adapt_behavior(
                        user_id=uid,
                        context=conv_context,
                    )
                except Exception as e:
                    logger.debug(f"Learning adaptation skipped: {e}")

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

            # Step 8: Build context for LLM (enhanced with module data)
            knowledge_context = self._build_knowledge_context(
                search_results, external_results
            )

            # Build enhanced style context
            style_context = str(conversation_style) if conversation_style else ""
            if comm_style_result:
                style_context += f"\nCommunication: {comm_style_result}"
            if cultural_context:
                style_context += f"\nCultural: {cultural_context}"
            if flow_result:
                style_context += f"\nFlow: {flow_result}"
            if habit_adaptations:
                style_context += f"\nHabit adaptations: {', '.join(habit_adaptations)}"
            if learning_adaptations:
                style_context += f"\nLearning: {learning_adaptations}"

            context = (
                f"Emotional State: {emotional_state}\n"
                f"Style: {style_context}\n"
                f"Knowledge:\n{knowledge_context}\n"
                f"Memory: {memory_context}"
            )
            context = self._prepend_workspace_context(
                context=context,
                session_type=session_type,
            )

            # Step 9: Generate response (with optional deep reasoning)
            if use_deep_reasoning and self.enable_reasoning:
                llm_response_content = await self._generate_with_reasoning(
                    question, context
                )
                history = self._session_histories.get(conv_id, [])
                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": llm_response_content},
                    ]
                )
                self._session_histories[conv_id] = history[-40:]
            elif anticipated_response:
                llm_response_content = anticipated_response
                history = self._session_histories.get(conv_id, [])
                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": llm_response_content},
                    ]
                )
                self._session_histories[conv_id] = history[-40:]
            else:
                # Standard generation
                llm_response_content = await self._generate_response_with_tool_loop(
                    conv_id=conv_id,
                    user_id=uid,
                    question=question,
                    system_prompt=context,
                )

            # Step 10: Check for milestones
            milestone = None
            if user_profile:
                milestone = self.emotional_processor.check_for_milestones(
                    user_profile, conv_context, emotional_state
                )

            # Step 11: Create final response (enhanced with module data)
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
                # New fields from integrated modules
                flow_state=(flow_result.state.value if flow_result and hasattr(flow_result, 'state') else None),
                quality_score=None,  # Populated after quality assessment
                nonverbal_cues_detected=(
                    [c.description for c in nonverbal_analysis.detected_cues]
                    if nonverbal_analysis and hasattr(nonverbal_analysis, 'detected_cues')
                    else None
                ),
                cultural_adaptations=(
                    [str(cultural_context)]
                    if cultural_context
                    else None
                ),
                habit_adaptations=habit_adaptations or None,
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

            # Step 12b: Quality assessment (non-blocking)
            if self._quality_assessor:
                try:
                    quality_assessment = self._quality_assessor.assess_turn(
                        conversation_id=conv_id,
                        user_id=uid,
                        user_message=question,
                        assistant_response=llm_response_content,
                        emotional_state=emotional_state,
                    )
                    if quality_assessment and hasattr(quality_assessment, 'overall_score'):
                        response.quality_score = quality_assessment.overall_score
                except Exception as e:
                    logger.debug(f"Quality assessment skipped: {e}")

            # Step 12c: Update topic preferences (conversation intelligence)
            if self._topic_learner:
                try:
                    self._topic_learner.learn_from_conversation(
                        user_id=uid,
                        message=question,
                        response=llm_response_content,
                        emotional_state=emotional_state,
                    )
                except Exception as e:
                    logger.debug(f"Topic learning skipped: {e}")

            # Step 12d: Learn from interaction (learning engine)
            if self._learning_engine:
                try:
                    self._learning_engine.process_interaction(
                        user_id=uid,
                        context=conv_context,
                        response=llm_response_content,
                        emotional_state=emotional_state,
                    )
                except Exception as e:
                    logger.debug(f"Learning engine update skipped: {e}")

            # Step 12e: Record interaction for habit detection
            if self._habit_detector:
                try:
                    self._habit_detector.record_interaction(
                        user_id=uid,
                        message=question,
                        timestamp=datetime.now(timezone.utc),
                    )
                except Exception as e:
                    logger.debug(f"Habit recording skipped: {e}")

            # Step 12f: Learn communication preferences
            if self._comm_preference_learner:
                try:
                    self._comm_preference_learner.learn_from_interaction(
                        user_id=uid,
                        message=question,
                        response=llm_response_content,
                        emotional_state=emotional_state,
                    )
                except Exception as e:
                    logger.debug(f"Communication preference learning skipped: {e}")

            # Step 13: Update proactive context and anticipate next tasks
            if self.enable_proactive:
                await self._update_proactive_context(
                    uid, question, response.answer, emotional_state
                )

            await self._append_daily_log(
                question=question,
                answer=response.answer,
                user_id=uid,
                conversation_id=conv_id,
            )
            await self._maybe_auto_compact(conv_id)
            await self._trigger_hook(
                HookType.MESSAGE_REPLY,
                {
                    "conversation_id": conv_id,
                    "user_id": uid,
                    "answer": response.answer,
                    "session_type": session_type,
                },
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

    def _prepend_workspace_context(self, context: str, session_type: str) -> str:
        """Prefix the LLM system prompt with workspace context when enabled."""
        workspace_context = self._render_workspace_context(session_type=session_type)
        if not workspace_context:
            return context
        return f"{workspace_context}\n\n{context}"

    def _render_workspace_context(self, session_type: str) -> str:
        """Render workspace files into a compact system-prompt section."""
        if self._workspace_manager is None:
            return ""

        try:
            loaded = self._workspace_manager.load_session_context(session_type=session_type)
        except Exception as exc:
            logger.debug("Workspace context unavailable: %s", exc)
            return ""

        sections: list[str] = []
        section_map = [
            ("soul", "SOUL.md"),
            ("user", "USER.md"),
            ("tools", "TOOLS.md"),
            ("daily_log_today", "Daily log (today)"),
            ("daily_log_yesterday", "Daily log (yesterday)"),
        ]
        for key, title in section_map:
            value = loaded.get(key)
            if isinstance(value, str) and value.strip():
                sections.append(f"## {title}\n{value.strip()}")

        if self._memory_gate.should_load_memory(session_type):
            memory_text = loaded.get("memory")
            if isinstance(memory_text, str) and memory_text.strip():
                sections.append(f"## MEMORY.md\n{memory_text.strip()}")

        if not sections:
            return ""
        return "Workspace Context:\n" + "\n\n".join(sections)

    def _format_tool_schema_instructions(self) -> str:
        """Provide tool schemas and response contract for tool calls."""
        if self._tool_executor is None:
            return ""

        try:
            registered_tools = self._tool_executor.list_tools()
        except Exception as exc:
            logger.debug("Tool schema listing failed: %s", exc)
            return ""

        if not registered_tools:
            return ""

        tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "aliases": list(tool.aliases),
                "input_schema": tool.input_schema.to_dict(),
            }
            for tool in registered_tools
        ]
        schemas_text = json.dumps(tool_schemas, ensure_ascii=True)
        return (
            "Tool usage is enabled.\n"
            "If you need a tool, reply with ONLY valid JSON in this exact shape:\n"
            '{"tool_use":{"name":"<tool_name>","input":{...}}}\n'
            "Do not add prose around tool JSON.\n\n"
            f"Available tool schemas:\n{schemas_text}"
        )

    @staticmethod
    def _extract_tool_call(content: str) -> Optional[dict[str, Any]]:
        """Parse a tool_use JSON object from model output."""
        text = (content or "").strip()
        if not text:
            return None

        candidates = [text]
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    candidates.append(candidate)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and isinstance(payload.get("tool_use"), dict):
                tool_use = payload["tool_use"]
                name = tool_use.get("name")
                tool_input = tool_use.get("input", {})
                if isinstance(name, str) and isinstance(tool_input, dict):
                    return {"name": name, "input": tool_input}

            if (
                isinstance(payload, dict)
                and payload.get("type") == "tool_use"
                and isinstance(payload.get("name"), str)
            ):
                tool_input = payload.get("input", {})
                if isinstance(tool_input, dict):
                    return {"name": payload["name"], "input": tool_input}

        return None

    async def _generate_response_with_tool_loop(
        self,
        conv_id: str,
        user_id: str,
        question: str,
        system_prompt: str,
        max_tool_rounds: int = 5,
    ) -> str:
        """Run LLM -> tool -> LLM loop until final text response."""
        session_history = list(self._session_histories.get(conv_id, []))
        tool_instructions = self._format_tool_schema_instructions()
        system_content = (
            f"{system_prompt}\n\n{tool_instructions}".strip()
            if tool_instructions
            else system_prompt
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(session_history)
        messages.append({"role": "user", "content": question})

        final_answer: Optional[str] = None

        for _ in range(max_tool_rounds):
            llm_response = await self.llm.achat(messages=messages)
            assistant_content = (llm_response.content or "").strip()

            tool_call = (
                self._extract_tool_call(assistant_content)
                if self._tool_executor is not None
                else None
            )
            if tool_call is None:
                final_answer = assistant_content
                messages.append({"role": "assistant", "content": assistant_content})
                break

            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            await self._trigger_hook(
                HookType.PRE_TOOL_USE,
                {
                    "conversation_id": conv_id,
                    "user_id": user_id,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                },
            )

            tool_result = await self._tool_executor.execute(
                name=tool_name,
                input_data=tool_input,
                conversation_id=conv_id,
                user_id=user_id,
            )

            await self._trigger_hook(
                HookType.POST_TOOL_USE,
                {
                    "conversation_id": conv_id,
                    "user_id": user_id,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": tool_result.output,
                    "tool_error": tool_result.is_error,
                },
            )

            tool_feedback = {
                "tool_result": {
                    "name": tool_name,
                    "is_error": tool_result.is_error,
                    "error_code": tool_result.error_code,
                    "output": tool_result.output,
                    "metadata": tool_result.metadata,
                }
            }
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool execution result. Continue responding to the user:\n"
                        + json.dumps(tool_feedback, ensure_ascii=True)
                    ),
                }
            )

        if final_answer is None:
            final_answer = (
                "I reached the tool-usage iteration limit while working on your request."
            )
            messages.append({"role": "assistant", "content": final_answer})

        self._session_histories[conv_id] = [
            message for message in messages if message.get("role") != "system"
        ][-40:]

        return final_answer

    async def _append_daily_log(
        self,
        question: str,
        answer: str,
        user_id: str,
        conversation_id: str,
    ) -> None:
        """Append the current turn to memory daily logs."""
        if self._daily_log_manager is None:
            return
        try:
            entry = (
                f"user={user_id} conv={conversation_id} "
                f"Q: {question.strip()[:300]} | A: {answer.strip()[:500]}"
            )
            self._daily_log_manager.append(entry)
        except Exception as exc:
            logger.debug("Daily log append failed: %s", exc)

    async def _maybe_auto_compact(self, conversation_id: str) -> None:
        """Run auto-compaction when token thresholds are exceeded."""
        if self._auto_compact_tracker is None or self._compactor is None:
            return

        history = self._session_histories.get(conversation_id, [])
        if not history:
            return

        if not should_auto_compact(self._auto_compact_tracker, history):
            return

        try:
            compacted = await self._compactor.compact(messages=history)
            if compacted.get("was_compacted"):
                compacted_messages = compacted.get("compacted_messages", history)
                if isinstance(compacted_messages, list):
                    self._session_histories[conversation_id] = compacted_messages
                self._auto_compact_tracker.record_success()
            else:
                self._auto_compact_tracker.record_failure()
        except Exception as exc:
            logger.warning("Auto-compaction failed: %s", exc)
            self._auto_compact_tracker.record_failure()

    async def _trigger_hook(self, hook_type: HookType, context: Any) -> None:
        """Trigger hook handlers if the hook manager is available."""
        if self._hook_manager is None:
            return
        try:
            await self._hook_manager.trigger(hook_type, context)
        except Exception as exc:
            logger.debug("Hook %s failed: %s", hook_type.value, exc)

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

    async def get_wellness_insights(self, user_id: str) -> dict:
        """Get wellness insights for a user."""
        if not self._wellness_tracker:
            return {"message": "Wellness tracking not enabled"}
        try:
            return self._wellness_tracker.get_wellness_summary(user_id)
        except Exception as e:
            logger.warning(f"Failed to get wellness insights: {e}")
            return {"error": str(e)}

    async def get_habit_patterns(self, user_id: str) -> dict:
        """Get detected habit patterns for a user."""
        if not self._habit_detector:
            return {"message": "Habit detection not enabled"}
        try:
            analysis = self._habit_detector.get_user_habits(user_id)
            if analysis:
                return {
                    "habits": [
                        {
                            "name": h.name,
                            "type": h.habit_type.value,
                            "frequency": h.frequency.value,
                            "confidence": h.confidence.value,
                            "consistency": h.consistency_score,
                        }
                        for h in analysis.habits
                    ] if hasattr(analysis, 'habits') else [],
                    "total_interactions": analysis.total_interactions if hasattr(analysis, 'total_interactions') else 0,
                }
            return {"habits": [], "total_interactions": 0}
        except Exception as e:
            logger.warning(f"Failed to get habit patterns: {e}")
            return {"error": str(e)}

    async def get_conversation_quality(self, conversation_id: str) -> dict:
        """Get conversation quality assessment."""
        if not self._quality_assessor:
            return {"message": "Quality assessment not enabled"}
        try:
            summary = self._quality_assessor.get_quality_summary(conversation_id)
            return summary if summary else {"message": "No quality data available"}
        except Exception as e:
            logger.warning(f"Failed to get conversation quality: {e}")
            return {"error": str(e)}

    async def start_proactive_monitoring(self):
        """Start proactive monitoring services."""
        if self._context_monitor:
            await self._context_monitor.start()

        if self._task_anticipator:
            await self._task_anticipator.start_preparation_worker()

        logger.info("Proactive monitoring started")

        # Start habit monitoring services
        if self._habit_detector:
            try:
                # Habit detector doesn't have start/stop but we log readiness
                logger.info("Habit detection active")
            except Exception as e:
                logger.warning(f"Failed to start habit monitoring: {e}")

    async def stop_proactive_monitoring(self):
        """Stop proactive monitoring services."""
        if self._context_monitor:
            await self._context_monitor.stop()

        if self._task_anticipator:
            await self._task_anticipator.stop_preparation_worker()

        logger.info("Proactive monitoring stopped")

        # Stop habit monitoring services
        if self._habit_detector:
            logger.info("Habit detection stopped")
