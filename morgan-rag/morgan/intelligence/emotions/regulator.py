"""
Emotion regulation module.

Provides emotion regulation strategies, coping mechanism recommendations,
and adaptive regulation learning for emotional well-being support.
"""

import statistics
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class RegulationStrategy:
    """Represents a single emotion regulation strategy."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        description: str,
        target_emotions: List[EmotionType],
        difficulty_level: str,
        time_required: str,
        effectiveness_range: Tuple[float, float],
        contraindications: List[str] = None,
    ):
        """Initialize regulation strategy."""
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.target_emotions = target_emotions
        self.difficulty_level = difficulty_level  # "easy", "medium", "hard"
        self.time_required = time_required  # "1-2 minutes", "5-10 minutes", etc.
        self.effectiveness_range = effectiveness_range  # (min, max) effectiveness
        self.contraindications = contraindications or []


class RegulationSession:
    """Tracks a regulation attempt and its outcomes."""

    def __init__(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        strategy: RegulationStrategy,
        timestamp: datetime = None,
    ):
        """Initialize regulation session."""
        self.user_id = user_id
        self.emotional_state = emotional_state
        self.strategy = strategy
        self.timestamp = timestamp or datetime.utcnow()
        self.completed = False
        self.effectiveness_score: Optional[float] = None
        self.user_feedback: Optional[str] = None
        self.outcome_emotion: Optional[EmotionalState] = None
        self.duration: Optional[timedelta] = None


class EmotionRegulator:
    """
    Comprehensive emotion regulation system.

    Features:
    - Multi-strategy regulation recommendations
    - Adaptive strategy selection based on user history
    - Regulation effectiveness tracking
    - Personalized coping mechanism suggestions
    - Real-time regulation need assessment
    - Evidence-based regulation techniques
    """

    # Emotion regulation strategies database
    REGULATION_STRATEGIES = {
        "cognitive_reappraisal": RegulationStrategy(
            strategy_id="cognitive_reappraisal",
            name="Cognitive Reappraisal",
            description="Reframe the situation from a different perspective to change emotional response",
            target_emotions=[EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR],
            difficulty_level="medium",
            time_required="2-5 minutes",
            effectiveness_range=(0.6, 0.9),
        ),
        "mindful_breathing": RegulationStrategy(
            strategy_id="mindful_breathing",
            name="Mindful Breathing",
            description="Focus on slow, deep breaths to calm the nervous system",
            target_emotions=[EmotionType.ANGER, EmotionType.FEAR, EmotionType.SURPRISE],
            difficulty_level="easy",
            time_required="1-3 minutes",
            effectiveness_range=(0.5, 0.8),
        ),
        "progressive_relaxation": RegulationStrategy(
            strategy_id="progressive_relaxation",
            name="Progressive Muscle Relaxation",
            description="Systematically tense and relax muscle groups to reduce physical tension",
            target_emotions=[EmotionType.FEAR, EmotionType.ANGER],
            difficulty_level="medium",
            time_required="5-10 minutes",
            effectiveness_range=(0.6, 0.85),
        ),
        "positive_distraction": RegulationStrategy(
            strategy_id="positive_distraction",
            name="Positive Distraction",
            description="Engage in a pleasant activity to shift attention away from distressing emotions",
            target_emotions=[EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR],
            difficulty_level="easy",
            time_required="10-30 minutes",
            effectiveness_range=(0.4, 0.7),
        ),
        "expressive_writing": RegulationStrategy(
            strategy_id="expressive_writing",
            name="Expressive Writing",
            description="Write about your emotions and experiences to process them",
            target_emotions=[EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR],
            difficulty_level="easy",
            time_required="5-15 minutes",
            effectiveness_range=(0.5, 0.8),
        ),
        "social_support": RegulationStrategy(
            strategy_id="social_support",
            name="Seek Social Support",
            description="Connect with supportive friends or family to share feelings",
            target_emotions=[EmotionType.SADNESS, EmotionType.FEAR],
            difficulty_level="medium",
            time_required="varies",
            effectiveness_range=(0.6, 0.9),
        ),
        "grounding_technique": RegulationStrategy(
            strategy_id="grounding_technique",
            name="5-4-3-2-1 Grounding",
            description="Use your senses to anchor yourself in the present moment",
            target_emotions=[EmotionType.FEAR, EmotionType.SURPRISE],
            difficulty_level="easy",
            time_required="2-5 minutes",
            effectiveness_range=(0.6, 0.8),
        ),
        "physical_exercise": RegulationStrategy(
            strategy_id="physical_exercise",
            name="Physical Exercise",
            description="Engage in physical activity to release tension and boost mood",
            target_emotions=[EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR],
            difficulty_level="medium",
            time_required="15-30 minutes",
            effectiveness_range=(0.7, 0.95),
        ),
        "mindfulness_meditation": RegulationStrategy(
            strategy_id="mindfulness_meditation",
            name="Mindfulness Meditation",
            description="Observe your thoughts and feelings without judgment",
            target_emotions=[EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR],
            difficulty_level="hard",
            time_required="10-20 minutes",
            effectiveness_range=(0.7, 0.95),
        ),
        "acceptance": RegulationStrategy(
            strategy_id="acceptance",
            name="Emotional Acceptance",
            description="Allow yourself to feel the emotion without fighting it",
            target_emotions=[EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANGER],
            difficulty_level="hard",
            time_required="5-10 minutes",
            effectiveness_range=(0.5, 0.8),
        ),
        "problem_solving": RegulationStrategy(
            strategy_id="problem_solving",
            name="Active Problem Solving",
            description="Identify concrete steps to address the source of distress",
            target_emotions=[EmotionType.FEAR, EmotionType.ANGER, EmotionType.SADNESS],
            difficulty_level="medium",
            time_required="10-30 minutes",
            effectiveness_range=(0.6, 0.9),
        ),
        "self_compassion": RegulationStrategy(
            strategy_id="self_compassion",
            name="Self-Compassion Practice",
            description="Treat yourself with kindness and understanding",
            target_emotions=[
                EmotionType.SADNESS,
                EmotionType.ANGER,
                EmotionType.DISGUST,
            ],
            difficulty_level="medium",
            time_required="3-10 minutes",
            effectiveness_range=(0.6, 0.85),
        ),
        "humor": RegulationStrategy(
            strategy_id="humor",
            name="Humor and Levity",
            description="Find humor or lighter perspectives in the situation",
            target_emotions=[EmotionType.ANGER, EmotionType.SADNESS],
            difficulty_level="easy",
            time_required="5-15 minutes",
            effectiveness_range=(0.5, 0.8),
        ),
        "time_out": RegulationStrategy(
            strategy_id="time_out",
            name="Take a Time-Out",
            description="Temporarily remove yourself from the triggering situation",
            target_emotions=[EmotionType.ANGER],
            difficulty_level="easy",
            time_required="5-30 minutes",
            effectiveness_range=(0.6, 0.85),
        ),
    }

    # Intensity thresholds for regulation need
    REGULATION_THRESHOLDS = {
        EmotionType.ANGER: 0.6,  # High anger needs regulation
        EmotionType.SADNESS: 0.65,  # Moderate-high sadness
        EmotionType.FEAR: 0.6,  # High fear/anxiety
        EmotionType.DISGUST: 0.7,  # Very high disgust
        EmotionType.SURPRISE: 0.8,  # Extreme surprise/shock
        EmotionType.JOY: 0.95,  # Only extreme joy (mania risk)
        EmotionType.NEUTRAL: 1.0,  # Neutral rarely needs regulation
    }

    # Contextual factors that increase regulation need
    REGULATION_AMPLIFIERS = {
        "duration": {  # How long emotion has persisted
            "short": 0.0,  # < 1 hour
            "moderate": 0.1,  # 1-4 hours
            "long": 0.2,  # 4-24 hours
            "extended": 0.3,  # > 24 hours
        },
        "frequency": {  # How often emotion occurs
            "rare": 0.0,
            "occasional": 0.1,
            "frequent": 0.2,
            "constant": 0.3,
        },
        "impact": {  # Life impact level
            "minimal": 0.0,
            "moderate": 0.15,
            "significant": 0.25,
            "severe": 0.35,
        },
    }

    def __init__(self):
        """Initialize emotion regulator."""
        self.settings = get_settings()

        # Track regulation sessions per user
        self.regulation_history: Dict[str, List[RegulationSession]] = defaultdict(list)

        # Track strategy effectiveness per user
        self.strategy_effectiveness: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # User-specific strategy preferences
        self.user_strategy_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)

        logger.info("Emotion Regulator initialized with 14 regulation strategies")

    def assess_regulation_need(
        self,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext] = None,
        recent_emotions: Optional[List[EmotionalState]] = None,
    ) -> Dict[str, Any]:
        """
        Assess whether emotion regulation is needed.

        Args:
            emotional_state: Current emotional state
            context: Optional conversation context
            recent_emotions: Optional list of recent emotional states

        Returns:
            Regulation need assessment
        """
        # Get base threshold for this emotion
        base_threshold = self.REGULATION_THRESHOLDS.get(
            emotional_state.primary_emotion, 0.7
        )

        # Calculate contextual amplifiers
        amplifier_score = self._calculate_amplifier_score(
            emotional_state, recent_emotions
        )

        # Adjusted threshold based on context
        adjusted_threshold = max(0.3, base_threshold - amplifier_score)

        # Determine if regulation is needed
        needs_regulation = emotional_state.intensity >= adjusted_threshold

        # Calculate urgency level
        urgency = self._calculate_regulation_urgency(
            emotional_state, adjusted_threshold
        )

        # Identify regulation triggers
        triggers = self._identify_regulation_triggers(
            emotional_state, recent_emotions, context
        )

        assessment = {
            "needs_regulation": needs_regulation,
            "urgency": urgency,
            "base_threshold": base_threshold,
            "adjusted_threshold": adjusted_threshold,
            "amplifier_score": amplifier_score,
            "triggers": triggers,
            "emotional_intensity": emotional_state.intensity,
            "primary_emotion": emotional_state.primary_emotion.value,
            "recommendation": self._generate_regulation_recommendation(
                needs_regulation, urgency, emotional_state
            ),
        }

        logger.debug(
            f"Regulation assessment for {emotional_state.primary_emotion.value}: "
            f"needed={needs_regulation}, urgency={urgency}"
        )

        return assessment

    def recommend_strategies(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext] = None,
        max_strategies: int = 3,
        difficulty_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommend regulation strategies for the current emotional state.

        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            context: Optional conversation context
            max_strategies: Maximum number of strategies to recommend
            difficulty_filter: Optional difficulty filter ("easy", "medium", "hard")

        Returns:
            List of recommended strategies with personalization
        """
        # Get all applicable strategies
        applicable_strategies = self._get_applicable_strategies(
            emotional_state, difficulty_filter
        )

        # Score strategies based on effectiveness and user history
        scored_strategies = []
        for strategy in applicable_strategies:
            score = self._calculate_strategy_score(user_id, strategy, emotional_state)
            scored_strategies.append((strategy, score))

        # Sort by score (highest first)
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        # Get top strategies
        top_strategies = scored_strategies[:max_strategies]

        # Generate recommendations with personalization
        recommendations = []
        for strategy, score in top_strategies:
            recommendation = self._build_strategy_recommendation(
                user_id, strategy, score, emotional_state, context
            )
            recommendations.append(recommendation)

        logger.info(
            f"Recommended {len(recommendations)} strategies for user {user_id} "
            f"(emotion: {emotional_state.primary_emotion.value})"
        )

        return recommendations

    def track_regulation_attempt(
        self, user_id: str, session: RegulationSession
    ) -> Dict[str, Any]:
        """
        Track a regulation attempt for learning and adaptation.

        Args:
            user_id: User identifier
            session: Regulation session to track

        Returns:
            Tracking result with insights
        """
        # Add to regulation history
        self.regulation_history[user_id].append(session)

        # If session is completed, update effectiveness tracking
        if session.completed and session.effectiveness_score is not None:
            self.strategy_effectiveness[user_id][session.strategy.strategy_id].append(
                session.effectiveness_score
            )

            # Update user strategy preferences
            self._update_strategy_preferences(user_id, session)

        # Keep only recent history (last 100 sessions per user)
        if len(self.regulation_history[user_id]) > 100:
            self.regulation_history[user_id] = self.regulation_history[user_id][-100:]

        # Generate insights
        insights = self._generate_regulation_insights(user_id, session)

        logger.debug(
            f"Tracked regulation attempt for user {user_id}: "
            f"strategy={session.strategy.name}, "
            f"completed={session.completed}"
        )

        return insights

    def analyze_regulation_patterns(
        self, user_id: str, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze user's regulation patterns and effectiveness over time.

        Args:
            user_id: User identifier
            timeframe_days: Days to analyze

        Returns:
            Regulation pattern analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)

        # Get recent sessions
        recent_sessions = [
            s
            for s in self.regulation_history.get(user_id, [])
            if s.timestamp >= cutoff_date
        ]

        if not recent_sessions:
            return {"pattern": "insufficient_data", "session_count": 0}

        # Analyze completion rate
        completed_sessions = [s for s in recent_sessions if s.completed]
        completion_rate = len(completed_sessions) / len(recent_sessions)

        # Analyze strategy usage
        strategy_usage = Counter(s.strategy.strategy_id for s in recent_sessions)

        # Analyze effectiveness by strategy
        strategy_effectiveness = {}
        for strategy_id, sessions_list in strategy_usage.items():
            strategy_sessions = [
                s
                for s in completed_sessions
                if s.strategy.strategy_id == strategy_id
                and s.effectiveness_score is not None
            ]

            if strategy_sessions:
                effectiveness_scores = [
                    s.effectiveness_score for s in strategy_sessions
                ]
                strategy_effectiveness[strategy_id] = {
                    "mean_effectiveness": statistics.mean(effectiveness_scores),
                    "usage_count": sessions_list,
                    "success_rate": len(strategy_sessions) / sessions_list,
                }

        # Identify most effective strategies
        if strategy_effectiveness:
            best_strategies = sorted(
                strategy_effectiveness.items(),
                key=lambda x: x[1]["mean_effectiveness"],
                reverse=True,
            )[:3]
        else:
            best_strategies = []

        # Analyze emotion-specific patterns
        emotion_patterns = self._analyze_emotion_regulation_patterns(recent_sessions)

        # Calculate overall regulation success
        overall_success = self._calculate_overall_success(completed_sessions)

        return {
            "pattern": "analyzed",
            "timeframe_days": timeframe_days,
            "total_sessions": len(recent_sessions),
            "completed_sessions": len(completed_sessions),
            "completion_rate": completion_rate,
            "overall_success": overall_success,
            "strategy_usage": dict(strategy_usage),
            "strategy_effectiveness": strategy_effectiveness,
            "best_strategies": [
                {
                    "strategy_id": s_id,
                    "effectiveness": data["mean_effectiveness"],
                    "usage_count": data["usage_count"],
                }
                for s_id, data in best_strategies
            ],
            "emotion_patterns": emotion_patterns,
        }

    def get_personalized_guidance(
        self, user_id: str, emotional_state: EmotionalState, strategy_id: str
    ) -> Dict[str, Any]:
        """
        Get personalized guidance for using a specific strategy.

        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            strategy_id: Strategy to get guidance for

        Returns:
            Personalized strategy guidance
        """
        if strategy_id not in self.REGULATION_STRATEGIES:
            return {"error": "unknown_strategy"}

        strategy = self.REGULATION_STRATEGIES[strategy_id]

        # Get user's history with this strategy
        user_effectiveness = self.strategy_effectiveness.get(user_id, {}).get(
            strategy_id, []
        )

        # Generate step-by-step guidance
        steps = self._generate_strategy_steps(strategy, emotional_state)

        # Add personalized tips based on history
        personalized_tips = self._generate_personalized_tips(
            user_id, strategy, user_effectiveness
        )

        # Estimate expected effectiveness
        expected_effectiveness = self._estimate_effectiveness(
            user_id, strategy, emotional_state
        )

        return {
            "strategy": {
                "id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "difficulty": strategy.difficulty_level,
                "time_required": strategy.time_required,
            },
            "steps": steps,
            "personalized_tips": personalized_tips,
            "expected_effectiveness": expected_effectiveness,
            "user_history": {
                "previous_attempts": len(user_effectiveness),
                "average_effectiveness": (
                    statistics.mean(user_effectiveness) if user_effectiveness else None
                ),
            },
        }

    def _calculate_amplifier_score(
        self,
        emotional_state: EmotionalState,
        recent_emotions: Optional[List[EmotionalState]],
    ) -> float:
        """Calculate contextual amplifier score."""
        amplifier_score = 0.0

        # Check emotional persistence (duration)
        if recent_emotions:
            same_emotion_count = sum(
                1
                for e in recent_emotions[-10:]
                if e.primary_emotion == emotional_state.primary_emotion
            )

            if same_emotion_count >= 7:
                amplifier_score += self.REGULATION_AMPLIFIERS["duration"]["extended"]
            elif same_emotion_count >= 4:
                amplifier_score += self.REGULATION_AMPLIFIERS["duration"]["long"]
            elif same_emotion_count >= 2:
                amplifier_score += self.REGULATION_AMPLIFIERS["duration"]["moderate"]

        # Check intensity trend (is it getting worse?)
        if recent_emotions and len(recent_emotions) >= 3:
            recent_intensities = [e.intensity for e in recent_emotions[-3:]]
            if all(
                recent_intensities[i] <= recent_intensities[i + 1]
                for i in range(len(recent_intensities) - 1)
            ):
                amplifier_score += 0.15  # Escalating intensity

        return min(0.5, amplifier_score)  # Cap at 0.5

    def _calculate_regulation_urgency(
        self, emotional_state: EmotionalState, threshold: float
    ) -> str:
        """Calculate regulation urgency level."""
        intensity = emotional_state.intensity

        if not intensity >= threshold:
            return "low"

        excess = intensity - threshold

        if excess >= 0.3:
            return "high"
        elif excess >= 0.15:
            return "medium"
        else:
            return "moderate"

    def _identify_regulation_triggers(
        self,
        emotional_state: EmotionalState,
        recent_emotions: Optional[List[EmotionalState]],
        context: Optional[ConversationContext],
    ) -> List[str]:
        """Identify triggers that suggest regulation is needed."""
        triggers = []

        # High intensity trigger
        if emotional_state.intensity >= 0.75:
            triggers.append("high_intensity")

        # Persistent emotion trigger
        if recent_emotions:
            same_emotion_count = sum(
                1
                for e in recent_emotions[-5:]
                if e.primary_emotion == emotional_state.primary_emotion
            )
            if same_emotion_count >= 3:
                triggers.append("emotional_persistence")

        # Negative emotion accumulation
        if recent_emotions:
            negative_emotions = [
                EmotionType.ANGER,
                EmotionType.SADNESS,
                EmotionType.FEAR,
                EmotionType.DISGUST,
            ]
            negative_count = sum(
                1
                for e in recent_emotions[-5:]
                if e.primary_emotion in negative_emotions
            )
            if negative_count >= 4:
                triggers.append("negative_emotion_accumulation")

        # Complex emotions (multiple conflicting emotions)
        if len(emotional_state.secondary_emotions) >= 2:
            triggers.append("emotional_complexity")

        return triggers

    def _generate_regulation_recommendation(
        self, needs_regulation: bool, urgency: str, emotional_state: EmotionalState
    ) -> str:
        """Generate regulation recommendation message."""
        if not needs_regulation:
            return "No immediate regulation needed. Continue monitoring."

        if urgency == "high":
            return f"Immediate regulation recommended for intense {emotional_state.primary_emotion.value}."
        elif urgency == "medium":
            return f"Regulation suggested to manage {emotional_state.primary_emotion.value}."
        else:
            return f"Consider regulation techniques for {emotional_state.primary_emotion.value}."

    def _get_applicable_strategies(
        self, emotional_state: EmotionalState, difficulty_filter: Optional[str]
    ) -> List[RegulationStrategy]:
        """Get strategies applicable to the emotional state."""
        applicable = []

        for strategy in self.REGULATION_STRATEGIES.values():
            # Check if strategy targets this emotion
            if emotional_state.primary_emotion in strategy.target_emotions:
                # Apply difficulty filter if specified
                if difficulty_filter and strategy.difficulty_level != difficulty_filter:
                    continue
                applicable.append(strategy)

        return applicable

    def _calculate_strategy_score(
        self,
        user_id: str,
        strategy: RegulationStrategy,
        emotional_state: EmotionalState,
    ) -> float:
        """Calculate recommendation score for a strategy."""
        # Base score from strategy effectiveness range
        base_score = sum(strategy.effectiveness_range) / 2

        # Adjust for user's history with this strategy
        user_effectiveness = self.strategy_effectiveness.get(user_id, {}).get(
            strategy.strategy_id, []
        )

        if user_effectiveness:
            # Use user's actual effectiveness
            personal_effectiveness = statistics.mean(user_effectiveness)
            # Blend with base score (70% personal, 30% base)
            base_score = (personal_effectiveness * 0.7) + (base_score * 0.3)

        # Adjust for user preferences
        preference_boost = self.user_strategy_preferences.get(user_id, {}).get(
            strategy.strategy_id, 0.0
        )
        base_score += preference_boost

        # Adjust for intensity match (easier strategies for lower intensity)
        if emotional_state.intensity < 0.5 and strategy.difficulty_level == "easy":
            base_score += 0.1
        elif emotional_state.intensity >= 0.8 and strategy.difficulty_level in [
            "medium",
            "hard",
        ]:
            base_score += 0.1

        return min(1.0, max(0.0, base_score))

    def _build_strategy_recommendation(
        self,
        user_id: str,
        strategy: RegulationStrategy,
        score: float,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext],
    ) -> Dict[str, Any]:
        """Build a complete strategy recommendation."""
        return {
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "difficulty": strategy.difficulty_level,
            "time_required": strategy.time_required,
            "recommendation_score": score,
            "expected_effectiveness": self._estimate_effectiveness(
                user_id, strategy, emotional_state
            ),
            "why_recommended": self._explain_recommendation(
                user_id, strategy, emotional_state
            ),
        }

    def _update_strategy_preferences(self, user_id: str, session: RegulationSession):
        """Update user's strategy preferences based on session outcome."""
        if session.effectiveness_score is None:
            return

        strategy_id = session.strategy.strategy_id

        # Calculate preference adjustment
        # High effectiveness increases preference, low decreases it
        adjustment = (session.effectiveness_score - 0.5) * 0.1

        # Get current preference or default to 0
        current_pref = self.user_strategy_preferences[user_id].get(strategy_id, 0.0)

        # Update preference (capped between -0.3 and +0.3)
        new_pref = max(-0.3, min(0.3, current_pref + adjustment))

        self.user_strategy_preferences[user_id][strategy_id] = new_pref

    def _generate_regulation_insights(
        self, user_id: str, session: RegulationSession
    ) -> Dict[str, Any]:
        """Generate insights from regulation session."""
        insights = {
            "session_recorded": True,
            "strategy_used": session.strategy.name,
            "completed": session.completed,
        }

        if session.completed and session.effectiveness_score is not None:
            insights["effectiveness"] = session.effectiveness_score

            # Compare to typical effectiveness
            user_avg = self.strategy_effectiveness.get(user_id, {}).get(
                session.strategy.strategy_id, []
            )

            if len(user_avg) >= 2:
                avg_effectiveness = statistics.mean(user_avg[:-1])  # Exclude current
                insights["compared_to_average"] = (
                    session.effectiveness_score - avg_effectiveness
                )

        return insights

    def _analyze_emotion_regulation_patterns(
        self, sessions: List[RegulationSession]
    ) -> Dict[str, Any]:
        """Analyze emotion-specific regulation patterns."""
        emotion_patterns = defaultdict(
            lambda: {"sessions": 0, "strategies_used": Counter(), "effectiveness": []}
        )

        for session in sessions:
            emotion = session.emotional_state.primary_emotion.value
            emotion_patterns[emotion]["sessions"] += 1
            emotion_patterns[emotion]["strategies_used"][
                session.strategy.strategy_id
            ] += 1

            if session.completed and session.effectiveness_score is not None:
                emotion_patterns[emotion]["effectiveness"].append(
                    session.effectiveness_score
                )

        # Calculate summary statistics
        summary = {}
        for emotion, data in emotion_patterns.items():
            summary[emotion] = {
                "session_count": data["sessions"],
                "most_used_strategy": (
                    data["strategies_used"].most_common(1)[0][0]
                    if data["strategies_used"]
                    else None
                ),
                "average_effectiveness": (
                    statistics.mean(data["effectiveness"])
                    if data["effectiveness"]
                    else None
                ),
            }

        return summary

    def _calculate_overall_success(
        self, completed_sessions: List[RegulationSession]
    ) -> Optional[float]:
        """Calculate overall regulation success rate."""
        if not completed_sessions:
            return None

        effectiveness_scores = [
            s.effectiveness_score
            for s in completed_sessions
            if s.effectiveness_score is not None
        ]

        if not effectiveness_scores:
            return None

        return statistics.mean(effectiveness_scores)

    def _generate_strategy_steps(
        self, strategy: RegulationStrategy, emotional_state: EmotionalState
    ) -> List[str]:
        """Generate step-by-step guidance for a strategy."""
        # Predefined steps for each strategy
        strategy_steps = {
            "cognitive_reappraisal": [
                "Identify the thought or belief causing distress",
                "Consider alternative ways to interpret the situation",
                "Ask yourself: 'Is there another way to look at this?'",
                "Choose a more balanced or helpful perspective",
                "Notice how your emotion shifts with the new perspective",
            ],
            "mindful_breathing": [
                "Find a comfortable position",
                "Close your eyes or soften your gaze",
                "Breathe in slowly through your nose for 4 counts",
                "Hold for 4 counts",
                "Breathe out slowly through your mouth for 6 counts",
                "Repeat for 5-10 cycles",
            ],
            "grounding_technique": [
                "Notice 5 things you can see around you",
                "Notice 4 things you can touch or feel",
                "Notice 3 things you can hear",
                "Notice 2 things you can smell",
                "Notice 1 thing you can taste",
            ],
            # Add more as needed
        }

        return strategy_steps.get(
            strategy.strategy_id,
            [
                "Follow the strategy guidelines",
                "Take your time",
                "Be patient with yourself",
            ],
        )

    def _generate_personalized_tips(
        self,
        user_id: str,
        strategy: RegulationStrategy,
        user_effectiveness: List[float],
    ) -> List[str]:
        """Generate personalized tips based on user history."""
        tips = []

        if not user_effectiveness:
            tips.append("This is a new strategy for you - take time to learn it")
            tips.append("Be patient with yourself as you practice")
        else:
            avg_effectiveness = statistics.mean(user_effectiveness)

            if avg_effectiveness >= 0.7:
                tips.append("This strategy has worked well for you in the past")
                tips.append("Trust in your ability to use this technique effectively")
            elif avg_effectiveness < 0.5:
                tips.append("This strategy has been challenging before - that's okay")
                tips.append("Consider trying a different approach if this doesn't help")

        # Add general tips
        tips.append(f"This should take about {strategy.time_required}")
        tips.append("Find a quiet space if possible")

        return tips

    def _estimate_effectiveness(
        self,
        user_id: str,
        strategy: RegulationStrategy,
        emotional_state: EmotionalState,
    ) -> float:
        """Estimate expected effectiveness of strategy."""
        # Start with strategy baseline
        baseline = sum(strategy.effectiveness_range) / 2

        # Adjust for user history
        user_effectiveness = self.strategy_effectiveness.get(user_id, {}).get(
            strategy.strategy_id, []
        )

        if user_effectiveness:
            # Use recent history (last 5 attempts)
            recent = user_effectiveness[-5:]
            personal_estimate = statistics.mean(recent)
            # Blend: 60% personal, 40% baseline
            baseline = (personal_estimate * 0.6) + (baseline * 0.4)

        return min(1.0, max(0.0, baseline))

    def _explain_recommendation(
        self,
        user_id: str,
        strategy: RegulationStrategy,
        emotional_state: EmotionalState,
    ) -> str:
        """Explain why this strategy was recommended."""
        reasons = []

        # Check user history
        user_effectiveness = self.strategy_effectiveness.get(user_id, {}).get(
            strategy.strategy_id, []
        )

        if user_effectiveness and statistics.mean(user_effectiveness) >= 0.7:
            reasons.append("has worked well for you before")

        # Check emotion match
        if emotional_state.primary_emotion in strategy.target_emotions:
            reasons.append(f"is effective for {emotional_state.primary_emotion.value}")

        # Check difficulty
        if strategy.difficulty_level == "easy":
            reasons.append("is easy to use")

        if reasons:
            return "Recommended because it " + " and ".join(reasons)
        else:
            return "Recommended based on general effectiveness"


# Singleton instance
_regulator_instance = None
_regulator_lock = threading.Lock()


def get_emotion_regulator() -> EmotionRegulator:
    """
    Get singleton emotion regulator instance.

    Returns:
        Shared EmotionRegulator instance
    """
    global _regulator_instance

    if _regulator_instance is None:
        with _regulator_lock:
            if _regulator_instance is None:
                _regulator_instance = EmotionRegulator()

    return _regulator_instance
