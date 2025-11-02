"""
Companion-Aware Memory Search Integration

Integrates memory search with companion features for personalized, emotionally-aware
conversation history retrieval and memory-based response personalization.

This module implements the requirements for task 6.2:
- Add conversation history search with emotional context
- Implement memory-based personalization for responses
- Create relationship-aware memory retrieval and ranking
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from morgan.utils.logger import get_logger
from morgan.search.multi_stage_search import get_multi_stage_search_engine, SearchResult
from morgan.memory.memory_processor import get_memory_processor
from morgan.companion.relationship_manager import CompanionRelationshipManager
from morgan.emotional.intelligence_engine import get_emotional_intelligence_engine
from morgan.emotional.models import EmotionalState, ConversationContext

logger = get_logger(__name__)


@dataclass
class CompanionMemorySearchResult:
    """Enhanced search result with companion and emotional context."""
    content: str
    source: str
    score: float
    result_type: str
    timestamp: datetime
    emotional_context: Dict[str, Any]
    relationship_significance: float
    personalization_factors: List[str]
    memory_type: str  # "conversation_turn", "enhanced_memory", "emotional_moment"
    user_engagement_score: float
    
    def get_summary(self, max_length: int = 150) -> str:
        """Get a summary with emotional and relationship context."""
        summary = self.content[:max_length-3] + "..." if len(self.content) > max_length else self.content
        
        # Add context indicators
        context_indicators = []
        if self.emotional_context.get("has_emotional_content"):
            context_indicators.append("emotional")
        if self.relationship_significance > 0.5:
            context_indicators.append("significant")
        if self.user_engagement_score > 0.7:
            context_indicators.append("engaging")
        
        if context_indicators:
            summary += f" [{', '.join(context_indicators)}]"
        
        return summary


class CompanionMemorySearchEngine:
    """
    Companion-aware memory search engine that integrates emotional intelligence
    and relationship context for personalized conversation history retrieval.
    
    Features:
    - Emotional context-aware search
    - Relationship significance weighting
    - Memory-based personalization
    - User engagement scoring
    - Temporal relevance adjustment
    """
    
    def __init__(self):
        """Initialize companion memory search engine."""
        self.search_engine = get_multi_stage_search_engine()
        self.memory_processor = get_memory_processor()
        self.companion_manager = CompanionRelationshipManager()
        self.emotional_engine = get_emotional_intelligence_engine()
        
        # Search configuration
        self.emotional_boost_factor = 0.3
        self.relationship_boost_factor = 0.4
        self.recency_boost_factor = 0.2
        self.engagement_boost_factor = 0.25
        
        logger.info("Companion memory search engine initialized")
    
    def search_with_emotional_context(
        self,
        query: str,
        user_id: str = "default_user",
        max_results: int = 10,
        include_emotional_moments: bool = True,
        min_relationship_significance: float = 0.0
    ) -> List[CompanionMemorySearchResult]:
        """
        Search conversation history with emotional context awareness.
        
        Enhanced implementation for task 8.2 requirements:
        - Add conversation history search with emotional context
        - Implement memory-based personalization for responses
        - Create relationship-aware memory retrieval and ranking
        
        Args:
            query: Search query
            user_id: User identifier for personalization
            max_results: Maximum number of results
            include_emotional_moments: Whether to include emotionally significant memories
            min_relationship_significance: Minimum relationship significance threshold
            
        Returns:
            List of companion-aware memory search results
        """
        try:
            logger.debug(f"Searching memories with emotional context for user {user_id}: '{query}'")
            
            # Analyze query emotion for context
            query_emotion = self._analyze_query_emotion(query, user_id)
            
            # Get user profile for personalization
            user_profile = self._get_user_profile(user_id)
            
            # Execute enhanced memory search with multiple strategies
            search_results = self._execute_enhanced_memory_search(
                query, user_id, max_results * 3, query_emotion, user_profile
            )
            
            # Filter to memory results only
            memory_results = [r for r in search_results if r.result_type in ["memory", "enhanced_memory", "companion_memory"]]
            
            # Enhance results with companion context and personalization
            enhanced_results = []
            for result in memory_results:
                enhanced_result = self._enhance_memory_result_with_personalization(
                    result, query_emotion, user_id, user_profile, min_relationship_significance
                )
                if enhanced_result:
                    enhanced_results.append(enhanced_result)
            
            # Apply advanced companion-aware ranking with relationship context
            ranked_results = self._apply_advanced_companion_ranking(
                enhanced_results, query_emotion, user_profile
            )
            
            # Apply memory-based personalization filtering
            personalized_results = self._apply_memory_based_personalization(
                ranked_results, query, user_profile
            )
            
            # Filter by emotional moments if requested
            if include_emotional_moments:
                final_results = personalized_results
            else:
                final_results = [r for r in personalized_results if r.memory_type != "emotional_moment"]
            
            logger.debug(f"Found {len(final_results)} enhanced companion-aware memory results")
            return final_results[:max_results]
            
        except Exception as e:
            logger.error(f"Enhanced companion memory search failed: {e}")
            return []
    
    def get_personalized_memories(
        self,
        user_id: str = "default_user",
        memory_types: Optional[List[str]] = None,
        max_results: int = 20,
        days_back: int = 30
    ) -> List[CompanionMemorySearchResult]:
        """
        Get personalized memories based on user relationship and preferences.
        
        Args:
            user_id: User identifier
            memory_types: Types of memories to include
            max_results: Maximum number of results
            days_back: Number of days to look back
            
        Returns:
            List of personalized memory results
        """
        try:
            logger.debug(f"Getting personalized memories for user {user_id}")
            
            # Get user profile for personalization
            user_profile = self._get_user_profile(user_id)
            
            # Build personalized query based on user interests
            personalized_query = self._build_personalized_query(user_profile)
            
            # Search with personalized context
            results = self.search_with_emotional_context(
                query=personalized_query,
                user_id=user_id,
                max_results=max_results,
                include_emotional_moments=True,
                min_relationship_significance=0.3
            )
            
            # Filter by memory types if specified
            if memory_types:
                results = [r for r in results if r.memory_type in memory_types]
            
            # Filter by time range
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            results = [r for r in results if r.timestamp >= cutoff_date]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get personalized memories: {e}")
            return []
    
    def get_relationship_memories(
        self,
        user_id: str = "default_user",
        milestone_types: Optional[List[str]] = None,
        max_results: int = 15
    ) -> List[CompanionMemorySearchResult]:
        """
        Get memories related to relationship milestones and significant moments.
        
        Args:
            user_id: User identifier
            milestone_types: Types of milestones to include
            max_results: Maximum number of results
            
        Returns:
            List of relationship-focused memory results
        """
        try:
            logger.debug(f"Getting relationship memories for user {user_id}")
            
            # Search for relationship-significant memories
            relationship_query = "trust relationship milestone breakthrough support understanding"
            
            results = self.search_with_emotional_context(
                query=relationship_query,
                user_id=user_id,
                max_results=max_results,
                include_emotional_moments=True,
                min_relationship_significance=0.5  # Higher threshold for relationship memories
            )
            
            # Filter for high relationship significance
            relationship_results = [
                r for r in results 
                if r.relationship_significance > 0.5 or "trust" in r.personalization_factors
            ]
            
            return relationship_results
            
        except Exception as e:
            logger.error(f"Failed to get relationship memories: {e}")
            return []
    
    def search_similar_conversations(
        self,
        current_query: str,
        user_id: str = "default_user",
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[CompanionMemorySearchResult]:
        """
        Find similar conversations from the past for context and continuity.
        
        Args:
            current_query: Current user query
            user_id: User identifier
            max_results: Maximum number of similar conversations
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar conversation results
        """
        try:
            logger.debug(f"Searching for similar conversations to: '{current_query}'")
            
            # Search for similar conversations
            results = self.search_with_emotional_context(
                query=current_query,
                user_id=user_id,
                max_results=max_results * 2,
                include_emotional_moments=False,
                min_relationship_significance=0.0
            )
            
            # Filter by similarity threshold
            similar_results = [r for r in results if r.score >= similarity_threshold]
            
            # Prefer conversation turns over enhanced memories for similarity
            conversation_results = [r for r in similar_results if r.memory_type == "conversation_turn"]
            
            return conversation_results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search similar conversations: {e}")
            return []
    
    def _analyze_query_emotion(self, query: str, user_id: str) -> EmotionalState:
        """Analyze emotional context of the query."""
        try:
            from morgan.emotional.models import ConversationContext
            context = ConversationContext(
                user_id=user_id,
                conversation_id="search_context",
                message_text=query,
                timestamp=datetime.utcnow()
            )
            
            return self.emotional_engine.analyze_emotion(query, context)
            
        except Exception as e:
            logger.warning(f"Failed to analyze query emotion: {e}")
            # Return neutral emotional state as fallback
            from morgan.emotional.models import EmotionalState, EmotionType
            return EmotionalState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                confidence=0.0,
                secondary_emotions=[],
                emotional_indicators=[]
            )
    
    def _get_recent_context(self, user_id: str) -> Optional[str]:
        """Get recent conversation context for the user."""
        try:
            # This is a simplified implementation
            # In practice, you'd get actual recent conversation context
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get recent context: {e}")
            return None
    
    def _enhance_memory_result(
        self,
        result: SearchResult,
        query_emotion: EmotionalState,
        user_id: str,
        min_relationship_significance: float
    ) -> Optional[CompanionMemorySearchResult]:
        """Enhance a memory search result with companion context."""
        try:
            # Extract metadata
            metadata = result.metadata
            
            # Get emotional context
            emotional_context = metadata.get("emotional_context", {})
            if not emotional_context and result.result_type == "memory":
                # Extract emotional context from conversation turn
                emotional_context = self._extract_emotional_context_from_content(result.content)
            
            # Calculate relationship significance
            relationship_significance = metadata.get("relationship_significance", 0.0)
            if relationship_significance == 0.0:
                relationship_significance = self._calculate_relationship_significance(result, user_id)
            
            # Skip if below minimum relationship significance
            if relationship_significance < min_relationship_significance:
                return None
            
            # Determine memory type
            memory_type = self._determine_memory_type(result, emotional_context, relationship_significance)
            
            # Calculate user engagement score
            engagement_score = self._calculate_engagement_score(result, emotional_context)
            
            # Extract personalization factors
            personalization_factors = self._extract_personalization_factors(
                result, emotional_context, relationship_significance
            )
            
            # Parse timestamp
            timestamp_str = metadata.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                timestamp = datetime.utcnow()
            
            # Calculate enhanced score with companion factors
            enhanced_score = self._calculate_enhanced_score(
                result.score, emotional_context, relationship_significance, 
                engagement_score, query_emotion
            )
            
            return CompanionMemorySearchResult(
                content=result.content,
                source=result.source,
                score=enhanced_score,
                result_type=result.result_type,
                timestamp=timestamp,
                emotional_context=emotional_context,
                relationship_significance=relationship_significance,
                personalization_factors=personalization_factors,
                memory_type=memory_type,
                user_engagement_score=engagement_score
            )
            
        except Exception as e:
            logger.warning(f"Failed to enhance memory result: {e}")
            return None
    
    def _extract_emotional_context_from_content(self, content: str) -> Dict[str, Any]:
        """Extract emotional context from conversation content."""
        try:
            # Simple emotional indicator detection
            emotional_words = {
                "happy": "joy", "excited": "joy", "grateful": "joy",
                "sad": "sadness", "disappointed": "sadness", "upset": "sadness",
                "angry": "anger", "frustrated": "anger", "annoyed": "anger",
                "worried": "fear", "anxious": "fear", "concerned": "fear",
                "surprised": "surprise", "amazed": "surprise", "shocked": "surprise"
            }
            
            content_lower = content.lower()
            detected_emotions = []
            emotional_indicators = []
            
            for word, emotion in emotional_words.items():
                if word in content_lower:
                    detected_emotions.append(emotion)
                    emotional_indicators.append(word)
            
            # Calculate intensity based on emotional punctuation and words
            intensity = 0.0
            if "!" in content:
                intensity += 0.3
            if "?" in content and len(content.split("?")) > 2:
                intensity += 0.2
            if emotional_indicators:
                intensity += min(0.5, len(emotional_indicators) * 0.1)
            
            return {
                "detected_emotions": detected_emotions,
                "emotional_indicators": emotional_indicators,
                "intensity": intensity,
                "has_emotional_content": len(emotional_indicators) > 0,
                "confidence": 0.6 if emotional_indicators else 0.1
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract emotional context: {e}")
            return {}
    
    def _calculate_relationship_significance(self, result: SearchResult, user_id: str) -> float:
        """Calculate relationship significance for a memory result."""
        try:
            significance = 0.0
            
            # Check feedback rating
            feedback_rating = result.metadata.get("feedback_rating")
            if feedback_rating:
                if feedback_rating >= 4:
                    significance += 0.4
                elif feedback_rating >= 3:
                    significance += 0.2
            
            # Check content length (longer conversations often more significant)
            content_length = len(result.content)
            if content_length > 500:
                significance += 0.3
            elif content_length > 200:
                significance += 0.1
            
            # Check for relationship keywords
            relationship_keywords = [
                "trust", "understand", "help", "support", "appreciate", 
                "grateful", "thank", "personal", "share", "feel"
            ]
            
            content_lower = result.content.lower()
            keyword_matches = sum(1 for keyword in relationship_keywords if keyword in content_lower)
            significance += min(0.3, keyword_matches * 0.05)
            
            return min(1.0, significance)
            
        except Exception as e:
            logger.warning(f"Failed to calculate relationship significance: {e}")
            return 0.0
    
    def _determine_memory_type(
        self, 
        result: SearchResult, 
        emotional_context: Dict[str, Any], 
        relationship_significance: float
    ) -> str:
        """Determine the type of memory based on context."""
        try:
            # Check if it's an enhanced memory
            if result.result_type == "enhanced_memory":
                if emotional_context.get("intensity", 0.0) > 0.6:
                    return "emotional_moment"
                elif relationship_significance > 0.7:
                    return "relationship_milestone"
                else:
                    return "enhanced_memory"
            
            # For conversation turns
            if emotional_context.get("has_emotional_content", False):
                return "emotional_conversation"
            elif relationship_significance > 0.5:
                return "significant_conversation"
            else:
                return "conversation_turn"
                
        except Exception:
            return "conversation_turn"
    
    def _calculate_engagement_score(
        self, 
        result: SearchResult, 
        emotional_context: Dict[str, Any]
    ) -> float:
        """Calculate user engagement score for a memory."""
        try:
            engagement = 0.0
            
            # Base engagement from content length
            content_length = len(result.content)
            if content_length > 300:
                engagement += 0.4
            elif content_length > 100:
                engagement += 0.2
            
            # Engagement from emotional content
            if emotional_context.get("has_emotional_content", False):
                engagement += 0.3
            
            # Engagement from feedback
            feedback_rating = result.metadata.get("feedback_rating")
            if feedback_rating:
                engagement += (feedback_rating - 1) * 0.1  # 0.0 to 0.4 range
            
            # Engagement from question complexity
            if "Q:" in result.content:
                question_part = result.content.split("Q:")[1].split("A:")[0] if "A:" in result.content else result.content
                if len(question_part) > 50:
                    engagement += 0.1
            
            return min(1.0, engagement)
            
        except Exception as e:
            logger.warning(f"Failed to calculate engagement score: {e}")
            return 0.5
    
    def _extract_personalization_factors(
        self,
        result: SearchResult,
        emotional_context: Dict[str, Any],
        relationship_significance: float
    ) -> List[str]:
        """Extract factors that make this memory personally relevant."""
        factors = []
        
        try:
            # Emotional factors
            if emotional_context.get("has_emotional_content", False):
                factors.append("emotional")
            
            # Relationship factors
            if relationship_significance > 0.7:
                factors.append("highly_significant")
            elif relationship_significance > 0.4:
                factors.append("significant")
            
            # Feedback factors
            feedback_rating = result.metadata.get("feedback_rating")
            if feedback_rating and feedback_rating >= 4:
                factors.append("highly_rated")
            elif feedback_rating and feedback_rating >= 3:
                factors.append("positively_rated")
            
            # Content factors
            content_lower = result.content.lower()
            if any(word in content_lower for word in ["trust", "personal", "share"]):
                factors.append("trust")
            if any(word in content_lower for word in ["learn", "understand", "explain"]):
                factors.append("learning")
            if any(word in content_lower for word in ["help", "support", "assist"]):
                factors.append("supportive")
            
            return factors
            
        except Exception as e:
            logger.warning(f"Failed to extract personalization factors: {e}")
            return []
    
    def _calculate_enhanced_score(
        self,
        base_score: float,
        emotional_context: Dict[str, Any],
        relationship_significance: float,
        engagement_score: float,
        query_emotion: EmotionalState
    ) -> float:
        """Calculate enhanced score with companion factors."""
        try:
            enhanced_score = base_score
            
            # Emotional boost
            emotional_intensity = emotional_context.get("intensity", 0.0)
            enhanced_score += emotional_intensity * self.emotional_boost_factor
            
            # Relationship boost
            enhanced_score += relationship_significance * self.relationship_boost_factor
            
            # Engagement boost
            enhanced_score += engagement_score * self.engagement_boost_factor
            
            # Query emotion matching boost
            if query_emotion.intensity > 0.5:
                query_emotions = [query_emotion.primary_emotion.value] + query_emotion.secondary_emotions
                memory_emotions = emotional_context.get("detected_emotions", [])
                
                if any(emotion in memory_emotions for emotion in query_emotions):
                    enhanced_score += 0.2  # Boost for emotional resonance
            
            return min(1.0, enhanced_score)
            
        except Exception as e:
            logger.warning(f"Failed to calculate enhanced score: {e}")
            return base_score
    
    def _apply_companion_ranking(
        self,
        results: List[CompanionMemorySearchResult],
        query_emotion: EmotionalState
    ) -> List[CompanionMemorySearchResult]:
        """Apply companion-aware ranking to results."""
        try:
            # Sort by enhanced score first
            results.sort(key=lambda r: r.score, reverse=True)
            
            # Apply temporal relevance adjustment
            current_time = datetime.utcnow()
            for result in results:
                days_ago = (current_time - result.timestamp).days
                
                # Boost recent memories
                if days_ago <= 7:
                    result.score += 0.1
                elif days_ago <= 30:
                    result.score += 0.05
                
                # Slight penalty for very old memories
                elif days_ago > 365:
                    result.score *= 0.9
            
            # Re-sort after temporal adjustment
            results.sort(key=lambda r: r.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to apply companion ranking: {e}")
            return results
    
    def _execute_enhanced_memory_search(
        self,
        query: str,
        user_id: str,
        max_results: int,
        query_emotion: EmotionalState,
        user_profile: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Execute enhanced memory search with multiple strategies and personalization.
        
        Implements requirement 10.1: Search both document knowledge and conversation memories
        """
        try:
            # Strategy 1: Direct memory search using basic search to avoid circular dependency
            direct_results = self.search_engine._basic_memory_search(
                query=query,
                max_results=max_results // 2,
                min_score=0.5,
                request_id="companion_search"
            )
            
            # Strategy 2: Personalized search based on user interests
            personalized_query = self._build_personalized_query_enhanced(user_profile, query)
            personalized_results = self.search_engine._basic_memory_search(
                query=personalized_query,
                max_results=max_results // 3,
                min_score=0.4,
                request_id="companion_personalized"
            )
            
            # Strategy 3: Emotional context search
            emotional_results = self._search_by_emotional_context(
                query_emotion, max_results // 4, user_id
            )
            
            # Combine all results (direct_results and personalized_results are tuples from _basic_memory_search)
            all_results = []
            if direct_results:
                all_results.extend(direct_results[0])  # First element is the results list
            if personalized_results:
                all_results.extend(personalized_results[0])  # First element is the results list
            all_results.extend(emotional_results)
            
            # Remove duplicates
            unique_results = self._deduplicate_search_results(all_results)
            
            return unique_results
            
        except Exception as e:
            logger.warning(f"Enhanced memory search failed: {e}")
            # Fallback to basic search
            basic_results = self.search_engine._basic_memory_search(
                query=query,
                max_results=max_results,
                min_score=0.6,
                request_id="companion_fallback"
            )
            return basic_results[0] if basic_results else []
    
    def _enhance_memory_result_with_personalization(
        self,
        result: SearchResult,
        query_emotion: EmotionalState,
        user_id: str,
        user_profile: Dict[str, Any],
        min_relationship_significance: float
    ) -> Optional[CompanionMemorySearchResult]:
        """
        Enhanced memory result enhancement with personalization.
        
        Implements requirement 10.2: Surface previous answers and context for similar questions
        """
        try:
            # Get base enhancement
            base_result = self._enhance_memory_result(
                result, query_emotion, user_id, min_relationship_significance
            )
            
            if not base_result:
                return None
            
            # Apply personalization enhancements
            personalization_boost = self._calculate_personalization_boost(
                result, user_profile, query_emotion
            )
            
            # Apply user preference matching
            preference_factors = self._extract_user_preference_factors(
                result, user_profile
            )
            
            # Update personalization factors
            base_result.personalization_factors.extend(preference_factors)
            
            # Apply personalization score boost
            base_result.score = min(1.0, base_result.score + personalization_boost)
            
            # Add personalization metadata
            base_result.emotional_context.update({
                "personalization_boost": personalization_boost,
                "user_preference_match": len(preference_factors) > 0,
                "personalized_enhanced": True
            })
            
            return base_result
            
        except Exception as e:
            logger.warning(f"Failed to enhance memory result with personalization: {e}")
            return self._enhance_memory_result(result, query_emotion, user_id, min_relationship_significance)
    
    def _apply_advanced_companion_ranking(
        self,
        results: List[CompanionMemorySearchResult],
        query_emotion: EmotionalState,
        user_profile: Dict[str, Any]
    ) -> List[CompanionMemorySearchResult]:
        """
        Apply advanced companion-aware ranking with relationship context.
        
        Implements requirement 10.3: Weight recent and relevant conversations higher in results
        """
        try:
            if not results:
                return results
            
            # Apply multi-factor ranking
            for result in results:
                ranking_boost = 0.0
                
                # Temporal relevance boost (requirement 10.3)
                temporal_boost = self._calculate_temporal_relevance_boost(result.timestamp)
                ranking_boost += temporal_boost
                
                # Emotional resonance boost
                emotional_boost = self._calculate_emotional_resonance_boost(
                    query_emotion, result.emotional_context
                )
                ranking_boost += emotional_boost
                
                # Relationship significance boost
                relationship_boost = result.relationship_significance * 0.25
                ranking_boost += relationship_boost
                
                # User engagement boost
                engagement_boost = result.user_engagement_score * 0.2
                ranking_boost += engagement_boost
                
                # Personalization boost
                personalization_boost = self._calculate_user_personalization_boost(
                    result, user_profile
                )
                ranking_boost += personalization_boost
                
                # Apply total boost
                result.score = min(1.0, result.score + ranking_boost)
                
                # Update metadata with ranking details
                result.emotional_context.update({
                    "ranking_boost": ranking_boost,
                    "temporal_boost": temporal_boost,
                    "emotional_boost": emotional_boost,
                    "relationship_boost": relationship_boost,
                    "engagement_boost": engagement_boost,
                    "personalization_boost": personalization_boost
                })
            
            # Sort by enhanced score
            results.sort(key=lambda r: r.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to apply advanced companion ranking: {e}")
            return self._apply_companion_ranking(results, query_emotion)
    
    def _apply_memory_based_personalization(
        self,
        results: List[CompanionMemorySearchResult],
        query: str,
        user_profile: Dict[str, Any]
    ) -> List[CompanionMemorySearchResult]:
        """
        Apply memory-based personalization for responses.
        
        Implements requirement 10.4: Provide conversation timestamps and context
        """
        try:
            personalized_results = []
            
            for result in results:
                # Enhance with conversation context (requirement 10.4)
                enhanced_result = self._add_conversation_context(result)
                
                # Apply user preference filtering
                if self._matches_user_preferences(enhanced_result, user_profile):
                    # Add personalization indicators
                    enhanced_result.personalization_factors.append("user_preference_match")
                    
                    # Boost score for preference matches
                    enhanced_result.score = min(1.0, enhanced_result.score + 0.1)
                
                personalized_results.append(enhanced_result)
            
            # Sort by personalized score
            personalized_results.sort(key=lambda r: r.score, reverse=True)
            
            return personalized_results
            
        except Exception as e:
            logger.warning(f"Failed to apply memory-based personalization: {e}")
            return results
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get enhanced user profile for personalization."""
        try:
            # Enhanced user profile with more detailed preferences
            # In practice, this would come from the companion relationship manager
            return {
                "interests": ["technology", "programming", "learning", "problem-solving"],
                "communication_style": "technical",
                "preferred_topics": ["python", "docker", "api", "development", "debugging"],
                "learning_goals": ["advanced programming", "system architecture", "best practices"],
                "interaction_patterns": {
                    "prefers_detailed_explanations": True,
                    "likes_code_examples": True,
                    "values_step_by_step_guidance": True
                },
                "emotional_preferences": {
                    "supportive_tone": True,
                    "encouraging_feedback": True,
                    "patient_explanations": True
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to get enhanced user profile: {e}")
            return {}
    
    def _build_personalized_query_enhanced(
        self, 
        user_profile: Dict[str, Any], 
        original_query: str
    ) -> str:
        """Build an enhanced personalized query based on user profile and original query."""
        try:
            interests = user_profile.get("interests", [])
            topics = user_profile.get("preferred_topics", [])
            learning_goals = user_profile.get("learning_goals", [])
            
            # Combine original query with personalization terms
            query_parts = [original_query]
            
            # Add relevant interests and topics
            relevant_terms = []
            original_lower = original_query.lower()
            
            for term in interests + topics + learning_goals:
                if any(word in original_lower for word in term.lower().split()):
                    relevant_terms.append(term)
            
            # Limit personalization terms to avoid query drift
            query_parts.extend(relevant_terms[:3])
            
            return " ".join(query_parts)
            
        except Exception as e:
            logger.warning(f"Failed to build enhanced personalized query: {e}")
            return original_query
    
    def _search_by_emotional_context(
        self,
        query_emotion: EmotionalState,
        max_results: int,
        user_id: str
    ) -> List[SearchResult]:
        """Search for memories with similar emotional context."""
        try:
            # Build emotional query
            primary_emotion = query_emotion.primary_emotion.value if hasattr(query_emotion.primary_emotion, 'value') else str(query_emotion.primary_emotion)
            emotion_terms = [primary_emotion]
            emotion_terms.extend(query_emotion.secondary_emotions)
            emotion_terms.extend(query_emotion.emotional_indicators)
            
            emotional_query = " ".join(emotion_terms[:5])
            
            # Search for emotionally similar memories using basic search to avoid circular dependency
            emotional_results = self.search_engine._basic_memory_search(
                query=emotional_query,
                max_results=max_results,
                min_score=0.3,  # Lower threshold for emotional matching
                request_id="emotional_search"
            )
            
            return emotional_results[0] if emotional_results else []
            
        except Exception as e:
            logger.warning(f"Failed to search by emotional context: {e}")
            return []
    
    def _deduplicate_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results."""
        try:
            seen_content = set()
            unique_results = []
            
            for result in results:
                content_key = result.content[:100].lower().strip()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            logger.warning(f"Failed to deduplicate search results: {e}")
            return results
    
    def _calculate_personalization_boost(
        self,
        result: SearchResult,
        user_profile: Dict[str, Any],
        query_emotion: EmotionalState
    ) -> float:
        """Calculate personalization boost based on user profile and emotional context."""
        try:
            boost = 0.0
            content = result.content.lower()
            
            # Interest matching boost
            interests = user_profile.get("interests", [])
            for interest in interests:
                if interest.lower() in content:
                    boost += 0.05
            
            # Topic matching boost
            topics = user_profile.get("preferred_topics", [])
            for topic in topics:
                if topic.lower() in content:
                    boost += 0.08
            
            # Communication style boost
            comm_style = user_profile.get("communication_style", "")
            if comm_style == "technical" and any(word in content for word in ["code", "function", "api", "implementation"]):
                boost += 0.06
            
            # Emotional preference boost
            emotional_prefs = user_profile.get("emotional_preferences", {})
            if emotional_prefs.get("supportive_tone") and any(word in content for word in ["help", "support", "guide"]):
                boost += 0.04
            
            return min(0.25, boost)  # Cap at 25% boost
            
        except Exception as e:
            logger.warning(f"Failed to calculate personalization boost: {e}")
            return 0.0
    
    def _extract_user_preference_factors(
        self,
        result: SearchResult,
        user_profile: Dict[str, Any]
    ) -> List[str]:
        """Extract user preference factors from the result."""
        try:
            factors = []
            content = result.content.lower()
            
            # Check for interest matches
            interests = user_profile.get("interests", [])
            for interest in interests:
                if interest.lower() in content:
                    factors.append(f"interest:{interest}")
            
            # Check for topic matches
            topics = user_profile.get("preferred_topics", [])
            for topic in topics:
                if topic.lower() in content:
                    factors.append(f"topic:{topic}")
            
            # Check for learning goal matches
            learning_goals = user_profile.get("learning_goals", [])
            for goal in learning_goals:
                if any(word in content for word in goal.lower().split()):
                    factors.append(f"learning_goal:{goal}")
            
            return factors
            
        except Exception as e:
            logger.warning(f"Failed to extract user preference factors: {e}")
            return []
    
    def _calculate_temporal_relevance_boost(self, timestamp: datetime) -> float:
        """Calculate temporal relevance boost for recent conversations."""
        try:
            current_time = datetime.utcnow()
            time_diff = current_time - timestamp
            days_ago = time_diff.days
            
            # Apply temporal weighting
            if days_ago <= 1:
                return 0.2  # Very recent
            elif days_ago <= 7:
                return 0.15  # Recent
            elif days_ago <= 30:
                return 0.1  # Somewhat recent
            elif days_ago <= 90:
                return 0.05  # Moderately old
            else:
                return 0.0  # Old conversations
                
        except Exception as e:
            logger.warning(f"Failed to calculate temporal relevance boost: {e}")
            return 0.0
    
    def _calculate_emotional_resonance_boost(
        self,
        query_emotion: EmotionalState,
        memory_emotional_context: Dict[str, Any]
    ) -> float:
        """Calculate emotional resonance boost between query and memory."""
        try:
            boost = 0.0
            
            # Get query emotion details
            query_primary = query_emotion.primary_emotion.value if hasattr(query_emotion.primary_emotion, 'value') else str(query_emotion.primary_emotion)
            query_intensity = query_emotion.intensity
            
            # Get memory emotion details
            memory_emotions = memory_emotional_context.get("detected_emotions", [])
            memory_intensity = memory_emotional_context.get("intensity", 0.0)
            
            # Check for emotional alignment
            if query_primary in memory_emotions:
                boost += 0.12 * min(query_intensity, memory_intensity)
            elif any(emotion in memory_emotions for emotion in query_emotion.secondary_emotions):
                boost += 0.08 * min(query_intensity, memory_intensity)
            
            # Boost for high emotional intensity
            if memory_intensity > 0.7:
                boost += 0.05
            
            return min(0.15, boost)
            
        except Exception as e:
            logger.warning(f"Failed to calculate emotional resonance boost: {e}")
            return 0.0
    
    def _calculate_user_personalization_boost(
        self,
        result: CompanionMemorySearchResult,
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate user-specific personalization boost."""
        try:
            boost = 0.0
            
            # Boost for personalization factors
            personalization_factors = result.personalization_factors
            boost += len(personalization_factors) * 0.02
            
            # Boost for user engagement patterns
            interaction_patterns = user_profile.get("interaction_patterns", {})
            if interaction_patterns.get("prefers_detailed_explanations") and len(result.content) > 300:
                boost += 0.05
            
            if interaction_patterns.get("likes_code_examples") and "```" in result.content:
                boost += 0.06
            
            return min(0.12, boost)
            
        except Exception as e:
            logger.warning(f"Failed to calculate user personalization boost: {e}")
            return 0.0
    
    def _add_conversation_context(
        self,
        result: CompanionMemorySearchResult
    ) -> CompanionMemorySearchResult:
        """Add conversation context and timestamps (requirement 10.4)."""
        try:
            # Enhance source with timestamp context
            timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M")
            result.source = f"{result.source} [{timestamp_str}]"
            
            # Add context indicators to personalization factors
            if result.timestamp:
                days_ago = (datetime.utcnow() - result.timestamp).days
                if days_ago <= 1:
                    result.personalization_factors.append("very_recent")
                elif days_ago <= 7:
                    result.personalization_factors.append("recent")
                elif days_ago <= 30:
                    result.personalization_factors.append("this_month")
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to add conversation context: {e}")
            return result
    
    def _matches_user_preferences(
        self,
        result: CompanionMemorySearchResult,
        user_profile: Dict[str, Any]
    ) -> bool:
        """Check if result matches user preferences."""
        try:
            content = result.content.lower()
            
            # Check interest matches
            interests = user_profile.get("interests", [])
            if any(interest.lower() in content for interest in interests):
                return True
            
            # Check topic matches
            topics = user_profile.get("preferred_topics", [])
            if any(topic.lower() in content for topic in topics):
                return True
            
            # Check communication style match
            comm_style = user_profile.get("communication_style", "")
            if comm_style == "technical" and any(word in content for word in ["code", "function", "api"]):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check user preference match: {e}")
            return True  # Default to include if check fails
    
    def _build_personalized_query(self, user_profile: Dict[str, Any]) -> str:
        """Build a personalized query based on user profile."""
        try:
            interests = user_profile.get("interests", [])
            topics = user_profile.get("preferred_topics", [])
            
            # Combine interests and topics into a search query
            query_parts = interests + topics
            return " ".join(query_parts[:5])  # Limit to top 5 terms
            
        except Exception as e:
            logger.warning(f"Failed to build personalized query: {e}")
            return "conversation memory"


# Singleton instance for global access
_companion_memory_search_engine = None


def get_companion_memory_search_engine() -> CompanionMemorySearchEngine:
    """
    Get singleton companion memory search engine instance.
    
    Returns:
        Shared CompanionMemorySearchEngine instance
    """
    global _companion_memory_search_engine
    
    if _companion_memory_search_engine is None:
        _companion_memory_search_engine = CompanionMemorySearchEngine()
    
    return _companion_memory_search_engine