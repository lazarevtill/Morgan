"""
Habit-Based Adaptation for Morgan RAG.

Adapts Morgan's behavior, responses, and interactions based on detected user habits
and patterns to provide more personalized and contextually appropriate assistance.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .detector import HabitPattern, HabitType, HabitAnalysis, HabitConfidence
from ..intelligence.core.models import InteractionData, EmotionalState
from ..learning.patterns import InteractionPatterns
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdaptationStrategy(Enum):
    """Types of adaptation strategies."""
    
    COMMUNICATION_STYLE = "communication_style"  # Adapt communication style
    RESPONSE_TIMING = "response_timing"  # Adapt response timing
    CONTENT_PERSONALIZATION = "content_personalization"  # Personalize content
    PROACTIVE_ASSISTANCE = "proactive_assistance"  # Provide proactive help
    CONTEXT_AWARENESS = "context_awareness"  # Adapt based on context
    LEARNING_ADAPTATION = "learning_adaptation"  # Adapt learning approach
    WELLNESS_SUPPORT = "wellness_support"  # Adapt wellness support
    PRODUCTIVITY_OPTIMIZATION = "productivity_optimization"  # Optimize for productivity


class AdaptationLevel(Enum):
    """Levels of adaptation intensity."""
    
    MINIMAL = "minimal"  # Subtle adaptations
    MODERATE = "moderate"  # Noticeable adaptations
    SIGNIFICANT = "significant"  # Major adaptations
    COMPREHENSIVE = "comprehensive"  # Complete behavior adaptation


class AdaptationContext(Enum):
    """Context for adaptation decisions."""
    
    CONVERSATION_START = "conversation_start"
    ONGOING_CONVERSATION = "ongoing_conversation"
    TASK_ASSISTANCE = "task_assistance"
    LEARNING_SESSION = "learning_session"
    WELLNESS_CHECK = "wellness_check"
    PROACTIVE_INTERACTION = "proactive_interaction"


@dataclass
class AdaptationRule:
    """Represents a habit-based adaptation rule."""
    
    rule_id: str
    user_id: str
    strategy: AdaptationStrategy
    level: AdaptationLevel
    
    # Trigger conditions
    habit_types: List[HabitType]
    required_confidence: HabitConfidence
    min_consistency_score: float
    
    # Adaptation parameters
    adaptation_params: Dict[str, Any] = field(default_factory=dict)
    context_filters: List[AdaptationContext] = field(default_factory=list)
    
    # Effectiveness tracking
    application_count: int = 0
    success_rate: float = 0.5
    user_satisfaction: float = 0.5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_applied: Optional[datetime] = None


@dataclass
class AdaptationResult:
    """Result of applying habit-based adaptation."""
    
    user_id: str
    applied_strategies: List[AdaptationStrategy]
    adaptation_level: AdaptationLevel
    
    # Specific adaptations
    communication_adaptations: Dict[str, Any] = field(default_factory=dict)
    timing_adaptations: Dict[str, Any] = field(default_factory=dict)
    content_adaptations: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    triggering_habits: List[str] = field(default_factory=list)  # Habit IDs
    adaptation_context: AdaptationContext = AdaptationContext.ONGOING_CONVERSATION
    
    # Metadata
    applied_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.5


class HabitBasedAdaptation:
    """
    Adapts Morgan's behavior based on detected user habits and patterns.
    
    Analyzes user habits to determine appropriate adaptations in communication style,
    timing, content personalization, and proactive assistance.
    """
    
    # Default adaptation rules
    DEFAULT_ADAPTATION_RULES = {
        HabitType.COMMUNICATION: {
            "frequent_questions": {
                "strategy": AdaptationStrategy.COMMUNICATION_STYLE,
                "params": {"response_detail": "comprehensive", "include_examples": True}
            },
            "brief_interactions": {
                "strategy": AdaptationStrategy.COMMUNICATION_STYLE,
                "params": {"response_length": "concise", "direct_answers": True}
            }
        },
        HabitType.WORK: {
            "morning_productivity": {
                "strategy": AdaptationStrategy.PROACTIVE_ASSISTANCE,
                "params": {"morning_briefing": True, "task_suggestions": True}
            },
            "deadline_focused": {
                "strategy": AdaptationStrategy.PRODUCTIVITY_OPTIMIZATION,
                "params": {"priority_focus": True, "time_management": True}
            }
        },
        HabitType.LEARNING: {
            "consistent_learner": {
                "strategy": AdaptationStrategy.LEARNING_ADAPTATION,
                "params": {"progressive_difficulty": True, "knowledge_building": True}
            },
            "visual_learner": {
                "strategy": AdaptationStrategy.CONTENT_PERSONALIZATION,
                "params": {"include_diagrams": True, "visual_examples": True}
            }
        },
        HabitType.WELLNESS: {
            "health_conscious": {
                "strategy": AdaptationStrategy.WELLNESS_SUPPORT,
                "params": {"wellness_reminders": True, "health_tips": True}
            },
            "stress_patterns": {
                "strategy": AdaptationStrategy.WELLNESS_SUPPORT,
                "params": {"stress_monitoring": True, "relaxation_suggestions": True}
            }
        }
    }
    
    def __init__(self):
        """Initialize habit-based adaptation system."""
        self.user_adaptation_rules: Dict[str, List[AdaptationRule]] = defaultdict(list)
        self.adaptation_history: Dict[str, List[AdaptationResult]] = defaultdict(list)
        logger.info("Habit-based adaptation system initialized")
    
    def create_adaptation_rules(
        self,
        user_id: str,
        habit_analysis: HabitAnalysis,
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[AdaptationRule]:
        """
        Create adaptation rules based on user habits.
        
        Args:
            user_id: User identifier
            habit_analysis: User's habit analysis
            preferences: User preferences for adaptation
            
        Returns:
            List[AdaptationRule]: Created adaptation rules
        """
        logger.info(f"Creating adaptation rules for user {user_id}")
        
        preferences = preferences or {}
        rules = []
        
        # Create rules for each detected habit
        for habit in habit_analysis.detected_habits:
            if habit.confidence == HabitConfidence.LOW:
                continue  # Skip low-confidence habits
            
            habit_rules = self._create_rules_for_habit(user_id, habit, preferences)
            rules.extend(habit_rules)
        
        # Store rules
        self.user_adaptation_rules[user_id] = rules
        
        logger.info(f"Created {len(rules)} adaptation rules for user {user_id}")
        return rules
    
    def apply_adaptations(
        self,
        user_id: str,
        context: AdaptationContext,
        current_interaction: Optional[InteractionData] = None,
        emotional_state: Optional[EmotionalState] = None
    ) -> AdaptationResult:
        """
        Apply habit-based adaptations for a user interaction.
        
        Args:
            user_id: User identifier
            context: Current interaction context
            current_interaction: Current interaction data
            emotional_state: User's current emotional state
            
        Returns:
            AdaptationResult: Applied adaptations
        """
        logger.debug(f"Applying adaptations for user {user_id} in context {context.value}")
        
        rules = self.user_adaptation_rules.get(user_id, [])
        if not rules:
            return self._create_default_adaptation_result(user_id, context)
        
        # Filter rules by context
        applicable_rules = [
            rule for rule in rules
            if not rule.context_filters or context in rule.context_filters
        ]
        
        if not applicable_rules:
            return self._create_default_adaptation_result(user_id, context)
        
        # Apply adaptations
        result = AdaptationResult(
            user_id=user_id,
            adaptation_context=context,
            applied_strategies=[],
            adaptation_level=AdaptationLevel.MINIMAL
        )
        
        # Group rules by strategy
        strategy_rules = defaultdict(list)
        for rule in applicable_rules:
            strategy_rules[rule.strategy].append(rule)
        
        # Apply each strategy
        for strategy, rules_for_strategy in strategy_rules.items():
            self._apply_strategy(strategy, rules_for_strategy, result, emotional_state)
        
        # Determine overall adaptation level
        result.adaptation_level = self._determine_adaptation_level(result)
        
        # Store result
        self.adaptation_history[user_id].append(result)
        
        # Update rule statistics
        for rule in applicable_rules:
            rule.application_count += 1
            rule.last_applied = datetime.utcnow()
            rule.updated_at = datetime.utcnow()
        
        logger.debug(f"Applied {len(result.applied_strategies)} adaptation strategies")
        return result
    
    def _create_rules_for_habit(
        self,
        user_id: str,
        habit: HabitPattern,
        preferences: Dict[str, Any]
    ) -> List[AdaptationRule]:
        """Create adaptation rules for a specific habit."""
        rules = []
        
        # Get default rules for habit type
        default_rules = self.DEFAULT_ADAPTATION_RULES.get(habit.habit_type, {})
        
        for rule_name, rule_config in default_rules.items():
            if self._habit_matches_rule(habit, rule_name):
                rule = AdaptationRule(
                    rule_id=f"{habit.habit_id}_{rule_name}",
                    user_id=user_id,
                    strategy=rule_config["strategy"],
                    level=AdaptationLevel.MODERATE,
                    habit_types=[habit.habit_type],
                    required_confidence=habit.confidence,
                    min_consistency_score=habit.consistency_score,
                    adaptation_params=rule_config["params"].copy()
                )
                rules.append(rule)
        
        # Create timing-based rules
        if habit.typical_times:
            timing_rule = self._create_timing_rule(user_id, habit)
            if timing_rule:
                rules.append(timing_rule)
        
        # Create context-based rules
        context_rules = self._create_context_rules(user_id, habit)
        rules.extend(context_rules)
        
        return rules
    
    def _habit_matches_rule(self, habit: HabitPattern, rule_name: str) -> bool:
        """Check if a habit matches a specific rule pattern."""
        rule_patterns = {
            "frequent_questions": lambda h: "question" in h.keywords,
            "brief_interactions": lambda h: h.duration_minutes and h.duration_minutes < 10,
            "morning_productivity": lambda h: any(t.hour < 12 for t in h.typical_times),
            "deadline_focused": lambda h: "deadline" in h.keywords or "urgent" in h.keywords,
            "consistent_learner": lambda h: h.consistency_score > 0.7,
            "visual_learner": lambda h: "visual" in h.keywords or "diagram" in h.keywords,
            "health_conscious": lambda h: "health" in h.keywords or "wellness" in h.keywords,
            "stress_patterns": lambda h: "stress" in h.keywords or "tired" in h.keywords
        }
        
        pattern_func = rule_patterns.get(rule_name)
        return pattern_func(habit) if pattern_func else False
    
    def _create_timing_rule(self, user_id: str, habit: HabitPattern) -> Optional[AdaptationRule]:
        """Create timing-based adaptation rule."""
        if not habit.typical_times:
            return None
        
        # Determine optimal response timing based on habit timing
        typical_hour = habit.typical_times[0].hour
        
        if 6 <= typical_hour < 12:  # Morning
            timing_params = {"response_urgency": "high", "proactive_suggestions": True}
        elif 12 <= typical_hour < 18:  # Afternoon
            timing_params = {"response_urgency": "medium", "detailed_responses": True}
        elif 18 <= typical_hour < 22:  # Evening
            timing_params = {"response_urgency": "low", "relaxed_tone": True}
        else:  # Night
            timing_params = {"response_urgency": "low", "brief_responses": True}
        
        return AdaptationRule(
            rule_id=f"{habit.habit_id}_timing",
            user_id=user_id,
            strategy=AdaptationStrategy.RESPONSE_TIMING,
            level=AdaptationLevel.MODERATE,
            habit_types=[habit.habit_type],
            required_confidence=habit.confidence,
            min_consistency_score=habit.consistency_score,
            adaptation_params=timing_params
        )
    
    def _create_context_rules(self, user_id: str, habit: HabitPattern) -> List[AdaptationRule]:
        """Create context-based adaptation rules."""
        rules = []
        
        # Create rules based on habit context
        for context_item in habit.context:
            if context_item == "work":
                rule = AdaptationRule(
                    rule_id=f"{habit.habit_id}_work_context",
                    user_id=user_id,
                    strategy=AdaptationStrategy.PRODUCTIVITY_OPTIMIZATION,
                    level=AdaptationLevel.MODERATE,
                    habit_types=[habit.habit_type],
                    required_confidence=habit.confidence,
                    min_consistency_score=habit.consistency_score,
                    adaptation_params={"professional_tone": True, "task_focused": True},
                    context_filters=[AdaptationContext.TASK_ASSISTANCE]
                )
                rules.append(rule)
            
            elif context_item == "learning":
                rule = AdaptationRule(
                    rule_id=f"{habit.habit_id}_learning_context",
                    user_id=user_id,
                    strategy=AdaptationStrategy.LEARNING_ADAPTATION,
                    level=AdaptationLevel.MODERATE,
                    habit_types=[habit.habit_type],
                    required_confidence=habit.confidence,
                    min_consistency_score=habit.consistency_score,
                    adaptation_params={"educational_tone": True, "step_by_step": True},
                    context_filters=[AdaptationContext.LEARNING_SESSION]
                )
                rules.append(rule)
        
        return rules
    
    def _apply_strategy(
        self,
        strategy: AdaptationStrategy,
        rules: List[AdaptationRule],
        result: AdaptationResult,
        emotional_state: Optional[EmotionalState]
    ):
        """Apply a specific adaptation strategy."""
        result.applied_strategies.append(strategy)
        
        # Combine parameters from all rules for this strategy
        combined_params = {}
        for rule in rules:
            combined_params.update(rule.adaptation_params)
        
        # Apply strategy-specific adaptations
        if strategy == AdaptationStrategy.COMMUNICATION_STYLE:
            result.communication_adaptations.update(combined_params)
            
        elif strategy == AdaptationStrategy.RESPONSE_TIMING:
            result.timing_adaptations.update(combined_params)
            
        elif strategy == AdaptationStrategy.CONTENT_PERSONALIZATION:
            result.content_adaptations.update(combined_params)
            
        elif strategy == AdaptationStrategy.PROACTIVE_ASSISTANCE:
            result.content_adaptations.update({
                "proactive_mode": True,
                **combined_params
            })
            
        elif strategy == AdaptationStrategy.LEARNING_ADAPTATION:
            result.content_adaptations.update({
                "learning_optimized": True,
                **combined_params
            })
            
        elif strategy == AdaptationStrategy.WELLNESS_SUPPORT:
            result.content_adaptations.update({
                "wellness_focused": True,
                **combined_params
            })
            
        elif strategy == AdaptationStrategy.PRODUCTIVITY_OPTIMIZATION:
            result.content_adaptations.update({
                "productivity_focused": True,
                **combined_params
            })
        
        # Adjust based on emotional state
        if emotional_state and result.communication_adaptations:
            self._adjust_for_emotional_state(result, emotional_state)
    
    def _adjust_for_emotional_state(
        self, result: AdaptationResult, emotional_state: EmotionalState
    ):
        """Adjust adaptations based on user's emotional state."""
        if emotional_state.primary_emotion in ["sadness", "stress", "anxiety"]:
            # Use gentler, more supportive tone
            result.communication_adaptations.update({
                "tone": "supportive",
                "empathy_level": "high",
                "reassurance": True
            })
            
        elif emotional_state.primary_emotion in ["joy", "excitement"]:
            # Use more energetic, enthusiastic tone
            result.communication_adaptations.update({
                "tone": "enthusiastic",
                "energy_level": "high",
                "celebration": True
            })
            
        elif emotional_state.primary_emotion == "anger":
            # Use calmer, more patient tone
            result.communication_adaptations.update({
                "tone": "calm",
                "patience_level": "high",
                "de_escalation": True
            })
    
    def _determine_adaptation_level(self, result: AdaptationResult) -> AdaptationLevel:
        """Determine overall adaptation level based on applied strategies."""
        strategy_count = len(result.applied_strategies)
        
        if strategy_count >= 4:
            return AdaptationLevel.COMPREHENSIVE
        elif strategy_count >= 3:
            return AdaptationLevel.SIGNIFICANT
        elif strategy_count >= 2:
            return AdaptationLevel.MODERATE
        else:
            return AdaptationLevel.MINIMAL
    
    def _create_default_adaptation_result(
        self, user_id: str, context: AdaptationContext
    ) -> AdaptationResult:
        """Create default adaptation result when no rules apply."""
        return AdaptationResult(
            user_id=user_id,
            adaptation_context=context,
            applied_strategies=[],
            adaptation_level=AdaptationLevel.MINIMAL
        )
    
    def update_rule_effectiveness(
        self, rule_id: str, success: bool, user_satisfaction: Optional[float] = None
    ):
        """Update the effectiveness of an adaptation rule."""
        for rules in self.user_adaptation_rules.values():
            for rule in rules:
                if rule.rule_id == rule_id:
                    # Update success rate
                    if success:
                        rule.success_rate = min(1.0, rule.success_rate + 0.1)
                    else:
                        rule.success_rate = max(0.0, rule.success_rate - 0.1)
                    
                    # Update user satisfaction
                    if user_satisfaction is not None:
                        rule.user_satisfaction = (rule.user_satisfaction + user_satisfaction) / 2
                    
                    rule.updated_at = datetime.utcnow()
                    return
    
    def get_adaptation_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get adaptation statistics for a user."""
        rules = self.user_adaptation_rules.get(user_id, [])
        history = self.adaptation_history.get(user_id, [])
        
        if not rules and not history:
            return {}
        
        stats = {
            "total_rules": len(rules),
            "active_rules": len([r for r in rules if r.application_count > 0]),
            "total_adaptations": len(history),
            "average_success_rate": sum(r.success_rate for r in rules) / len(rules) if rules else 0.0,
            "average_satisfaction": sum(r.user_satisfaction for r in rules) / len(rules) if rules else 0.0,
            "strategy_usage": {
                strategy.value: len([r for r in rules if r.strategy == strategy])
                for strategy in AdaptationStrategy
            },
            "adaptation_levels": {
                level.value: len([h for h in history if h.adaptation_level == level])
                for level in AdaptationLevel
            }
        }
        
        return stats
    
    def optimize_adaptations(self, user_id: str):
        """Optimize adaptations based on effectiveness data."""
        rules = self.user_adaptation_rules.get(user_id, [])
        
        # Remove ineffective rules
        effective_rules = [
            rule for rule in rules
            if rule.success_rate > 0.3 and rule.user_satisfaction > 0.3
        ]
        
        # Adjust adaptation levels for low-performing rules
        for rule in effective_rules:
            if rule.success_rate < 0.5:
                if rule.level == AdaptationLevel.SIGNIFICANT:
                    rule.level = AdaptationLevel.MODERATE
                elif rule.level == AdaptationLevel.MODERATE:
                    rule.level = AdaptationLevel.MINIMAL
        
        self.user_adaptation_rules[user_id] = effective_rules
        
        logger.info(f"Optimized adaptations for user {user_id}: {len(effective_rules)} rules remaining")
    
    def get_recommended_adaptations(
        self, user_id: str, habit_analysis: HabitAnalysis
    ) -> List[Tuple[AdaptationStrategy, Dict[str, Any]]]:
        """Get recommended adaptations based on habit analysis."""
        recommendations = []
        
        for habit in habit_analysis.detected_habits:
            if habit.confidence == HabitConfidence.LOW:
                continue
            
            # Recommend adaptations based on habit characteristics
            if habit.habit_type == HabitType.COMMUNICATION:
                if habit.consistency_score > 0.7:
                    recommendations.append((
                        AdaptationStrategy.RESPONSE_TIMING,
                        {"optimize_for_habit_timing": True, "habit_id": habit.habit_id}
                    ))
            
            elif habit.habit_type == HabitType.LEARNING:
                recommendations.append((
                    AdaptationStrategy.LEARNING_ADAPTATION,
                    {"personalized_learning": True, "habit_id": habit.habit_id}
                ))
            
            elif habit.habit_type == HabitType.WELLNESS:
                recommendations.append((
                    AdaptationStrategy.WELLNESS_SUPPORT,
                    {"wellness_integration": True, "habit_id": habit.habit_id}
                ))
        
        return recommendations