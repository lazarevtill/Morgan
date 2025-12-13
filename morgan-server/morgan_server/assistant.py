"""
Morgan Assistant - Core orchestration of all engines.

This module integrates the Empathic Engine, Knowledge Engine, and Personalization Layer
to provide a complete personal assistant experience.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from morgan_server.empathic import (
    EmotionalIntelligence,
    PersonalitySystem,
    RoleplaySystem,
    EmotionalDetection,
    EmotionalAdjustment,
    PersonalityApplication,
    RoleplayResponse,
)
from morgan_server.knowledge.rag import RAGSystem
from morgan_server.knowledge.search import SearchSystem
from morgan_server.personalization.profile import UserProfile, ProfileManager
from morgan_server.personalization.preferences import PreferenceManager
from morgan_server.personalization.memory import MemoryManager, MessageRole
from morgan_server.llm import LLMClient, LLMMessage


@dataclass
class AssistantContext:
    """Context for an assistant interaction."""
    user_id: str
    conversation_id: Optional[str] = None
    message: str = ""
    user_profile: Optional[UserProfile] = None
    emotional_detection: Optional[EmotionalDetection] = None
    emotional_adjustment: Optional[EmotionalAdjustment] = None
    personality_application: Optional[PersonalityApplication] = None
    roleplay_response: Optional[RoleplayResponse] = None
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_context: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    """Response from the assistant."""
    answer: str
    conversation_id: str
    emotional_tone: Optional[str] = None
    empathy_level: Optional[float] = None
    personalization_elements: List[str] = field(default_factory=list)
    milestone_celebration: Optional[str] = None
    confidence: float = 1.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MorganAssistant:
    """
    Morgan Assistant - Orchestrates all engines for complete assistant experience.
    
    Integrates:
    - Empathic Engine (emotional intelligence, personality, roleplay)
    - Knowledge Engine (RAG, semantic search)
    - Personalization Layer (profiles, preferences, memory)
    - LLM Client (for response generation)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        personality_system: Optional[PersonalitySystem] = None,
        roleplay_system: Optional[RoleplaySystem] = None,
        rag_system: Optional[RAGSystem] = None,
        search_system: Optional[SearchSystem] = None,
        profile_manager: Optional[ProfileManager] = None,
        preference_manager: Optional[PreferenceManager] = None,
        memory_system: Optional[MemoryManager] = None,
    ):
        """
        Initialize the Morgan Assistant.

        Args:
            llm_client: LLM client for response generation
            emotional_intelligence: Emotional intelligence engine
            personality_system: Personality system
            roleplay_system: Roleplay system
            rag_system: RAG system for knowledge retrieval
            search_system: Search system for semantic search
            profile_manager: User profile manager
            preference_manager: User preference manager
            memory_system: Conversation memory system
        """
        self.llm_client = llm_client
        
        # Empathic Engine components
        self.emotional_intelligence = emotional_intelligence or EmotionalIntelligence()
        self.personality_system = personality_system or PersonalitySystem()
        self.roleplay_system = roleplay_system or RoleplaySystem()
        
        # Knowledge Engine components
        self.rag_system = rag_system
        self.search_system = search_system
        
        # Personalization Layer components
        self.profile_manager = profile_manager or ProfileManager()
        self.preference_manager = preference_manager or PreferenceManager()
        self.memory_system = memory_system or MemoryManager()

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        use_knowledge: bool = True,
        use_memory: bool = True,
    ) -> AssistantResponse:
        """
        Process a chat message and generate a response.

        This is the main entry point for the assistant. It orchestrates all engines:
        1. Detect emotional tone
        2. Retrieve user profile and preferences
        3. Retrieve relevant memories
        4. Retrieve relevant knowledge (if enabled)
        5. Apply personality and roleplay configuration
        6. Generate response with LLM
        7. Store interaction in memory

        Args:
            message: User's message
            user_id: User identifier
            conversation_id: Optional conversation identifier
            use_knowledge: Whether to use knowledge retrieval
            use_memory: Whether to use memory retrieval

        Returns:
            AssistantResponse with generated answer and metadata
        """
        # Build context
        context = await self._build_context(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            use_knowledge=use_knowledge,
            use_memory=use_memory,
        )
        
        # Generate response
        response = await self._generate_response(context)
        
        # Store interaction in memory
        if use_memory and self.memory_system:
            await self._store_interaction(context, response)
        
        return response

    async def _build_context(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        use_knowledge: bool = True,
        use_memory: bool = True,
    ) -> AssistantContext:
        """
        Build context for response generation.

        Args:
            message: User's message
            user_id: User identifier
            conversation_id: Optional conversation identifier
            use_knowledge: Whether to use knowledge retrieval
            use_memory: Whether to use memory retrieval

        Returns:
            AssistantContext with all relevant information
        """
        context = AssistantContext(
            user_id=user_id,
            conversation_id=conversation_id or f"conv_{user_id}_{datetime.now().timestamp()}",
            message=message,
        )
        
        # 1. Detect emotional tone
        context.emotional_detection = self.emotional_intelligence.detect_tone(
            message, user_id=user_id
        )
        
        # 2. Get emotional adjustment
        context.emotional_adjustment = self.emotional_intelligence.adjust_response_tone(
            context.emotional_detection, user_id=user_id
        )
        
        # 3. Retrieve user profile
        context.user_profile = self.profile_manager.get_profile(user_id)
        
        # 4. Retrieve relevant memories
        if use_memory and self.memory_system:
            # Search for relevant conversations
            search_results = self.memory_system.search_conversations(
                user_id=user_id,
                query=message,
                limit=5,
            )
            context.relevant_memories = [
                {
                    "content": f"{msg.role.value}: {msg.content}",
                    "timestamp": msg.timestamp.isoformat(),
                    "conversation_id": conv.conversation_id,
                    "relevance_score": score,
                }
                for conv, msg, score in search_results
            ]
        
        # 5. Retrieve relevant knowledge
        if use_knowledge and self.rag_system:
            knowledge_results = await self.rag_system.retrieve(
                query=message,
                limit=3,
            )
            # Extract sources from RAG results
            if isinstance(knowledge_results, dict):
                context.knowledge_context = knowledge_results.get("sources", [])
            else:
                context.knowledge_context = knowledge_results
        
        # 6. Apply personality
        context.personality_application = self.personality_system.apply_personality(
            response=message,
            user_id=user_id,
            relationship_depth=context.user_profile.trust_level if context.user_profile else 0.0,
        )
        
        # 7. Apply roleplay configuration
        from morgan_server.empathic.roleplay import RoleplayContext
        roleplay_context = RoleplayContext(
            user_id=user_id,
            relationship_depth=context.user_profile.trust_level if context.user_profile else 0.0,
            emotional_state=context.emotional_detection,
        )
        context.roleplay_response = self.roleplay_system.generate_response(
            base_response=message,
            context=roleplay_context,
        )
        
        return context

    async def _generate_response(self, context: AssistantContext) -> AssistantResponse:
        """
        Generate response using LLM with all context.

        Args:
            context: AssistantContext with all relevant information

        Returns:
            AssistantResponse with generated answer
        """
        # Build system prompt with personality, roleplay, and emotional guidance
        system_prompt = self._build_system_prompt(context)
        
        # Build user prompt with context
        user_prompt = self._build_user_prompt(context)
        
        # Generate response
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        llm_response = await self.llm_client.generate(messages, temperature=0.7)
        
        # Build assistant response
        response = AssistantResponse(
            answer=llm_response.content,
            conversation_id=context.conversation_id,
            emotional_tone=context.emotional_detection.primary_tone.value if context.emotional_detection else None,
            empathy_level=context.emotional_adjustment.intensity if context.emotional_adjustment else None,
            personalization_elements=self._extract_personalization_elements(context),
            milestone_celebration=context.emotional_adjustment.celebration if context.emotional_adjustment else None,
            confidence=1.0,
            sources=context.knowledge_context,
            metadata={
                "emotional_detection": context.emotional_detection.__dict__ if context.emotional_detection else None,
                "personality_traits": context.personality_application.traits_applied if context.personality_application else [],
                "roleplay_tone": context.roleplay_response.tone_applied.value if context.roleplay_response else None,
            },
        )
        
        return response

    def _build_system_prompt(self, context: AssistantContext) -> str:
        """Build system prompt with personality and roleplay guidance."""
        prompt_parts = []
        
        # Base personality
        if context.personality_application:
            traits_str = ', '.join([f"{k.value}={v:.2f}" for k, v in context.personality_application.traits_applied.items()])
            prompt_parts.append(f"Personality Traits: {traits_str}")
            if context.personality_application.style_notes:
                prompt_parts.append(f"Style: {', '.join(context.personality_application.style_notes)}")
        
        # Roleplay configuration
        if context.roleplay_response:
            prompt_parts.append(f"Tone: {context.roleplay_response.tone_applied.value}")
            prompt_parts.append(f"Style: {context.roleplay_response.style_applied.value}")
            if context.roleplay_response.relationship_notes:
                prompt_parts.append(f"Relationship: {', '.join(context.roleplay_response.relationship_notes)}")
            if context.roleplay_response.personality_notes:
                prompt_parts.append(f"Personality Notes: {', '.join(context.roleplay_response.personality_notes)}")
        
        # Emotional adjustment
        if context.emotional_adjustment:
            prompt_parts.append(f"Emotional Response: {context.emotional_adjustment.target_tone.value}")
            if context.emotional_adjustment.suggestions:
                prompt_parts.append(f"Suggestions: {', '.join(context.emotional_adjustment.suggestions)}")
            if context.emotional_adjustment.support:
                prompt_parts.append(f"Support Message: {context.emotional_adjustment.support}")
            if context.emotional_adjustment.celebration:
                prompt_parts.append(f"Celebration: {context.emotional_adjustment.celebration}")
        
        # User preferences
        if context.user_profile:
            if context.user_profile.preferred_name:
                prompt_parts.append(f"User's preferred name: {context.user_profile.preferred_name}")
            prompt_parts.append(f"Communication style: {context.user_profile.communication_style}")
            prompt_parts.append(f"Response length: {context.user_profile.response_length}")
        
        return "\n".join(prompt_parts)

    def _build_user_prompt(self, context: AssistantContext) -> str:
        """Build user prompt with message and context."""
        prompt_parts = []
        
        # Add relevant memories
        if context.relevant_memories:
            prompt_parts.append("Relevant conversation history:")
            for memory in context.relevant_memories[:3]:  # Limit to top 3
                prompt_parts.append(f"- {memory.get('content', '')}")
            prompt_parts.append("")
        
        # Add knowledge context
        if context.knowledge_context:
            prompt_parts.append("Relevant knowledge:")
            for knowledge in context.knowledge_context[:3]:  # Limit to top 3
                prompt_parts.append(f"- {knowledge.get('content', '')}")
            prompt_parts.append("")
        
        # Add user message
        prompt_parts.append(f"User message: {context.message}")
        
        return "\n".join(prompt_parts)

    def _extract_personalization_elements(self, context: AssistantContext) -> List[str]:
        """Extract personalization elements applied to the response."""
        elements = []
        
        if context.emotional_adjustment:
            elements.append(f"emotional_tone:{context.emotional_adjustment.target_tone.value}")
        
        if context.personality_application and context.personality_application.traits_applied:
            elements.extend([f"trait:{trait}" for trait in context.personality_application.traits_applied])
        
        if context.roleplay_response:
            elements.append(f"roleplay_tone:{context.roleplay_response.tone_applied.value}")
            elements.append(f"roleplay_style:{context.roleplay_response.style_applied.value}")
        
        if context.user_profile:
            elements.append(f"communication_style:{context.user_profile.communication_style}")
        
        return elements

    async def _store_interaction(
        self,
        context: AssistantContext,
        response: AssistantResponse,
    ) -> None:
        """Store interaction in memory system."""
        if not self.memory_system:
            return
        
        # Get or create conversation
        conversation = self.memory_system.get_or_create_conversation(
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            metadata={
                "emotional_tone": response.emotional_tone,
                "personalization_elements": response.personalization_elements,
            },
        )
        
        # Add user message
        self.memory_system.add_message(
            conversation_id=context.conversation_id,
            role=MessageRole.USER,
            content=context.message,
            metadata={"timestamp": datetime.now().isoformat()},
        )
        
        # Add assistant response
        self.memory_system.add_message(
            conversation_id=context.conversation_id,
            role=MessageRole.ASSISTANT,
            content=response.answer,
            metadata={
                "emotional_tone": response.emotional_tone,
                "personalization_elements": response.personalization_elements,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages
        """
        if not self.memory_system:
            return []
        
        if conversation_id:
            # Get specific conversation
            messages = self.memory_system.get_conversation_context(
                conversation_id=conversation_id,
                limit=limit,
            )
            return [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in messages
            ]
        else:
            # Get all user conversations
            conversations = self.memory_system.get_user_conversations(user_id)
            all_messages = []
            for conv in conversations:
                for msg in conv.messages[-limit:]:
                    all_messages.append({
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "conversation_id": conv.conversation_id,
                        "metadata": msg.metadata,
                    })
            return sorted(all_messages, key=lambda x: x["timestamp"], reverse=True)[:limit]

    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if not self.search_system:
            return []
        
        return await self.search_system.search(query=query, limit=limit)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile.

        Args:
            user_id: User identifier

        Returns:
            UserProfile or None if not found
        """
        return self.profile_manager.get_profile(user_id)

    def update_user_profile(
        self,
        user_id: str,
        **updates
    ) -> UserProfile:
        """
        Update user profile.

        Args:
            user_id: User identifier
            **updates: Profile fields to update

        Returns:
            Updated UserProfile
        """
        return self.profile_manager.update_profile(user_id, **updates)

    async def analyze_emotional_state(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze user's emotional state over time.

        Args:
            user_id: User identifier

        Returns:
            Emotional analysis including trends and patterns
        """
        return self.emotional_intelligence.analyze_emotional_trend(user_id)
