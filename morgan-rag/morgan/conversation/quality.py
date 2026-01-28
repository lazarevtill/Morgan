"""
Conversation quality assessment module.

Assesses and monitors conversation quality across multiple dimensions
to ensure engaging, helpful, and satisfying user interactions.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import ConversationContext, EmotionalState
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class QualityDimension(Enum):
    """Dimensions of conversation quality."""

    COHERENCE = "coherence"
    ENGAGEMENT = "engagement"
    HELPFULNESS = "helpfulness"
    EMPATHY = "empathy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    RESPONSIVENESS = "responsiveness"
    PERSONALIZATION = "personalization"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    FLOW = "flow"


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class QualityMetric:
    """Individual quality metric assessment."""

    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    level: QualityLevel
    confidence: float
    indicators: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class ConversationQualityAssessment:
    """Complete conversation quality assessment."""

    conversation_id: str
    user_id: str
    overall_score: float
    overall_level: QualityLevel
    dimension_scores: Dict[QualityDimension, QualityMetric]
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    turn_count: int = 0
    conversation_duration: timedelta = field(default_factory=lambda: timedelta(0))
    key_strengths: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)


@dataclass
class QualityTrend:
    """Quality trend analysis over time."""

    dimension: QualityDimension
    trend_direction: str  # improving, declining, stable
    trend_strength: float  # 0.0 to 1.0
    recent_average: float
    historical_average: float
    trend_confidence: float


@dataclass
class QualityInsight:
    """Insight about conversation quality patterns."""

    insight_type: str
    description: str
    impact_level: str  # high, medium, low
    actionable_recommendations: List[str]
    supporting_data: Dict[str, Any] = field(default_factory=dict)


class ConversationQualityAssessor:
    """
    Conversation quality assessment system.

    Features:
    - Multi-dimensional quality assessment
    - Real-time quality monitoring
    - Quality trend analysis
    - Personalized quality insights
    - Improvement recommendations
    - Quality benchmarking and comparison
    """

    def __init__(self):
        """Initialize conversation quality assessor."""
        self.settings = get_settings()

        # Quality assessment data
        self.conversation_assessments: Dict[str, List[ConversationQualityAssessment]] = defaultdict(list)
        self.user_quality_history: Dict[str, List[ConversationQualityAssessment]] = defaultdict(list)
        self.quality_benchmarks: Dict[QualityDimension, float] = self._initialize_benchmarks()

        logger.info("Conversation Quality Assessor initialized")

    def assess_conversation_quality(
        self,
        conversation_id: str,
        user_id: str,
        conversation_turns: List[ConversationContext],
        emotional_states: List[EmotionalState],
        user_feedback: Optional[float] = None,
    ) -> ConversationQualityAssessment:
        """
        Assess overall conversation quality.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            conversation_turns: List of conversation turns
            emotional_states: List of emotional states
            user_feedback: Optional user feedback score (0.0-1.0)

        Returns:
            Complete conversation quality assessment
        """
        if not conversation_turns:
            return self._create_empty_assessment(conversation_id, user_id)

        # Assess each quality dimension
        dimension_scores = {}
        for dimension in QualityDimension:
            metric = self._assess_quality_dimension(
                dimension, conversation_turns, emotional_states, user_feedback
            )
            dimension_scores[dimension] = metric

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        overall_level = self._score_to_level(overall_score)

        # Calculate conversation metadata
        conversation_duration = self._calculate_conversation_duration(conversation_turns)
        turn_count = len(conversation_turns)

        # Identify strengths and improvement areas
        key_strengths = self._identify_key_strengths(dimension_scores)
        improvement_areas = self._identify_improvement_areas(dimension_scores)

        # Create assessment
        assessment = ConversationQualityAssessment(
            conversation_id=conversation_id,
            user_id=user_id,
            overall_score=overall_score,
            overall_level=overall_level,
            dimension_scores=dimension_scores,
            turn_count=turn_count,
            conversation_duration=conversation_duration,
            key_strengths=key_strengths,
            improvement_areas=improvement_areas,
        )

        # Store assessment
        self.conversation_assessments[conversation_id].append(assessment)
        self.user_quality_history[user_id].append(assessment)

        # Keep only recent assessments (last 100 per user)
        if len(self.user_quality_history[user_id]) > 100:
            self.user_quality_history[user_id] = self.user_quality_history[user_id][-100:]

        logger.debug(
            f"Assessed conversation quality for {conversation_id}: "
            f"overall={overall_score:.2f} ({overall_level.value})"
        )

        return assessment

    def assess_turn_quality(
        self,
        user_message: str,
        assistant_response: str,
        emotional_state: EmotionalState,
        context: ConversationContext,
    ) -> Dict[QualityDimension, float]:
        """
        Assess quality of a single conversation turn.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            emotional_state: User's emotional state
            context: Conversation context

        Returns:
            Quality scores by dimension
        """
        turn_scores = {}

        # Assess relevance
        turn_scores[QualityDimension.RELEVANCE] = self._assess_turn_relevance(
            user_message, assistant_response
        )

        # Assess clarity
        turn_scores[QualityDimension.CLARITY] = self._assess_turn_clarity(
            assistant_response
        )

        # Assess empathy
        turn_scores[QualityDimension.EMPATHY] = self._assess_turn_empathy(
            assistant_response, emotional_state
        )

        # Assess helpfulness
        turn_scores[QualityDimension.HELPFULNESS] = self._assess_turn_helpfulness(
            user_message, assistant_response
        )

        # Assess responsiveness
        turn_scores[QualityDimension.RESPONSIVENESS] = self._assess_turn_responsiveness(
            context
        )

        return turn_scores

    def analyze_quality_trends(
        self, user_id: str, timeframe_days: int = 30
    ) -> List[QualityTrend]:
        """
        Analyze quality trends for a user.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            List of quality trends by dimension
        """
        user_history = self.user_quality_history.get(user_id, [])
        if len(user_history) < 3:
            return []

        # Filter by timeframe
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=timeframe_days)
        recent_assessments = [
            assessment for assessment in user_history
            if assessment.assessment_timestamp >= cutoff_date
        ]

        if len(recent_assessments) < 3:
            return []

        trends = []
        for dimension in QualityDimension:
            trend = self._analyze_dimension_trend(dimension, recent_assessments, user_history)
            if trend:
                trends.append(trend)

        return trends

    def get_quality_insights(
        self, user_id: str, timeframe_days: int = 30
    ) -> List[QualityInsight]:
        """
        Get quality insights and recommendations.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            List of quality insights
        """
        insights = []

        # Get user's quality history
        user_history = self.user_quality_history.get(user_id, [])
        if not user_history:
            return insights

        # Filter by timeframe
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=timeframe_days)
        recent_assessments = [
            assessment for assessment in user_history
            if assessment.assessment_timestamp >= cutoff_date
        ]

        if not recent_assessments:
            return insights

        # Generate insights
        insights.extend(self._generate_performance_insights(recent_assessments))
        insights.extend(self._generate_trend_insights(user_id, timeframe_days))
        insights.extend(self._generate_pattern_insights(recent_assessments))
        insights.extend(self._generate_improvement_insights(recent_assessments))

        return insights

    def compare_quality_to_benchmarks(
        self, assessment: ConversationQualityAssessment
    ) -> Dict[QualityDimension, Dict[str, Any]]:
        """
        Compare conversation quality to benchmarks.

        Args:
            assessment: Conversation quality assessment

        Returns:
            Comparison results by dimension
        """
        comparisons = {}

        for dimension, metric in assessment.dimension_scores.items():
            benchmark = self.quality_benchmarks.get(dimension, 0.7)
            
            comparison = {
                "score": metric.score,
                "benchmark": benchmark,
                "difference": metric.score - benchmark,
                "performance": "above" if metric.score > benchmark else "below" if metric.score < benchmark else "at",
                "percentile": self._calculate_percentile(dimension, metric.score),
            }
            
            comparisons[dimension] = comparison

        return comparisons

    def get_quality_summary(
        self, user_id: str, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get quality summary for user.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            Quality summary statistics
        """
        user_history = self.user_quality_history.get(user_id, [])
        if not user_history:
            return {"error": "No quality data available"}

        # Filter by timeframe
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=timeframe_days)
        recent_assessments = [
            assessment for assessment in user_history
            if assessment.assessment_timestamp >= cutoff_date
        ]

        if not recent_assessments:
            return {"error": "No recent quality data"}

        # Calculate summary statistics
        summary = {
            "total_conversations": len(recent_assessments),
            "average_overall_score": self._calculate_average_score(recent_assessments),
            "quality_distribution": self._calculate_quality_distribution(recent_assessments),
            "dimension_averages": self._calculate_dimension_averages(recent_assessments),
            "best_performing_dimensions": self._identify_best_dimensions(recent_assessments),
            "improvement_opportunities": self._identify_improvement_opportunities(recent_assessments),
            "quality_consistency": self._calculate_quality_consistency(recent_assessments),
            "recent_trend": self._calculate_recent_trend(recent_assessments),
        }

        return summary

    def _assess_quality_dimension(
        self,
        dimension: QualityDimension,
        conversation_turns: List[ConversationContext],
        emotional_states: List[EmotionalState],
        user_feedback: Optional[float],
    ) -> QualityMetric:
        """Assess a specific quality dimension."""
        if dimension == QualityDimension.COHERENCE:
            return self._assess_coherence(conversation_turns)
        elif dimension == QualityDimension.ENGAGEMENT:
            return self._assess_engagement(conversation_turns, emotional_states)
        elif dimension == QualityDimension.HELPFULNESS:
            return self._assess_helpfulness(conversation_turns, user_feedback)
        elif dimension == QualityDimension.EMPATHY:
            return self._assess_empathy(conversation_turns, emotional_states)
        elif dimension == QualityDimension.RELEVANCE:
            return self._assess_relevance(conversation_turns)
        elif dimension == QualityDimension.CLARITY:
            return self._assess_clarity(conversation_turns)
        elif dimension == QualityDimension.RESPONSIVENESS:
            return self._assess_responsiveness(conversation_turns)
        elif dimension == QualityDimension.PERSONALIZATION:
            return self._assess_personalization(conversation_turns)
        elif dimension == QualityDimension.EMOTIONAL_INTELLIGENCE:
            return self._assess_emotional_intelligence(conversation_turns, emotional_states)
        elif dimension == QualityDimension.FLOW:
            return self._assess_flow(conversation_turns)
        else:
            return self._create_default_metric(dimension)

    def _assess_coherence(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation coherence."""
        if len(conversation_turns) < 2:
            return self._create_default_metric(QualityDimension.COHERENCE)

        coherence_factors = []
        indicators = []

        # Topic consistency
        topics = []
        for turn in conversation_turns:
            topic = self._extract_simple_topic(turn.message_text)
            if topic:
                topics.append(topic)

        if topics:
            unique_topics = len(set(topics))
            topic_coherence = 1.0 - (unique_topics - 1) / max(1, len(topics))
            coherence_factors.append(topic_coherence)
            
            if topic_coherence > 0.8:
                indicators.append("consistent_topic_focus")
            elif topic_coherence < 0.5:
                indicators.append("topic_drift_detected")

        # Response relevance
        relevance_scores = []
        for i in range(1, len(conversation_turns)):
            relevance = self._calculate_response_relevance(
                conversation_turns[i-1].message_text,
                conversation_turns[i].message_text
            )
            relevance_scores.append(relevance)

        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            coherence_factors.append(avg_relevance)
            
            if avg_relevance > 0.8:
                indicators.append("highly_relevant_responses")

        # Calculate overall coherence
        coherence_score = sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5
        level = self._score_to_level(coherence_score)

        # Generate improvement suggestions
        suggestions = []
        if coherence_score < 0.6:
            suggestions.append("Maintain better topic focus throughout conversation")
            suggestions.append("Ensure responses directly address user messages")

        return QualityMetric(
            dimension=QualityDimension.COHERENCE,
            score=coherence_score,
            level=level,
            confidence=0.8,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_engagement(
        self, conversation_turns: List[ConversationContext], emotional_states: List[EmotionalState]
    ) -> QualityMetric:
        """Assess conversation engagement."""
        engagement_factors = []
        indicators = []

        # Message length analysis
        message_lengths = [len(turn.message_text) for turn in conversation_turns]
        if message_lengths:
            avg_length = sum(message_lengths) / len(message_lengths)
            length_engagement = min(1.0, avg_length / 150.0)  # Normalize to 150 chars
            engagement_factors.append(length_engagement)
            
            if avg_length > 100:
                indicators.append("detailed_user_messages")

        # Emotional intensity
        if emotional_states:
            avg_intensity = sum(state.intensity for state in emotional_states) / len(emotional_states)
            engagement_factors.append(avg_intensity)
            
            if avg_intensity > 0.6:
                indicators.append("high_emotional_engagement")

        # Question frequency (indicates engagement)
        question_count = sum(1 for turn in conversation_turns if "?" in turn.message_text)
        question_ratio = question_count / len(conversation_turns) if conversation_turns else 0
        engagement_factors.append(min(1.0, question_ratio * 3))  # Scale up question impact

        if question_ratio > 0.3:
            indicators.append("active_questioning")

        # Calculate engagement score
        engagement_score = sum(engagement_factors) / len(engagement_factors) if engagement_factors else 0.5
        level = self._score_to_level(engagement_score)

        # Generate suggestions
        suggestions = []
        if engagement_score < 0.6:
            suggestions.append("Ask more engaging questions to encourage participation")
            suggestions.append("Show more interest in user's responses")

        return QualityMetric(
            dimension=QualityDimension.ENGAGEMENT,
            score=engagement_score,
            level=level,
            confidence=0.7,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_helpfulness(
        self, conversation_turns: List[ConversationContext], user_feedback: Optional[float]
    ) -> QualityMetric:
        """Assess conversation helpfulness."""
        helpfulness_factors = []
        indicators = []

        # User feedback (if available)
        if user_feedback is not None:
            helpfulness_factors.append(user_feedback)
            if user_feedback > 0.8:
                indicators.append("positive_user_feedback")

        # Solution-oriented language
        solution_keywords = ["help", "solution", "answer", "resolve", "fix", "suggest", "recommend"]
        solution_count = 0
        for turn in conversation_turns:
            text_lower = turn.message_text.lower()
            solution_count += sum(1 for keyword in solution_keywords if keyword in text_lower)

        if conversation_turns:
            solution_ratio = solution_count / len(conversation_turns)
            helpfulness_factors.append(min(1.0, solution_ratio * 2))
            
            if solution_ratio > 0.3:
                indicators.append("solution_oriented_responses")

        # Information density
        info_keywords = ["because", "therefore", "specifically", "example", "detail", "explain"]
        info_count = 0
        for turn in conversation_turns:
            text_lower = turn.message_text.lower()
            info_count += sum(1 for keyword in info_keywords if keyword in text_lower)

        if conversation_turns:
            info_ratio = info_count / len(conversation_turns)
            helpfulness_factors.append(min(1.0, info_ratio * 1.5))

        # Calculate helpfulness score
        helpfulness_score = sum(helpfulness_factors) / len(helpfulness_factors) if helpfulness_factors else 0.5
        level = self._score_to_level(helpfulness_score)

        # Generate suggestions
        suggestions = []
        if helpfulness_score < 0.6:
            suggestions.append("Provide more specific and actionable advice")
            suggestions.append("Include examples and detailed explanations")

        return QualityMetric(
            dimension=QualityDimension.HELPFULNESS,
            score=helpfulness_score,
            level=level,
            confidence=0.8 if user_feedback is not None else 0.6,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_empathy(
        self, conversation_turns: List[ConversationContext], emotional_states: List[EmotionalState]
    ) -> QualityMetric:
        """Assess empathy in conversation."""
        empathy_factors = []
        indicators = []

        # Emotional acknowledgment
        empathy_keywords = [
            "understand", "feel", "sorry", "difficult", "challenging",
            "appreciate", "recognize", "acknowledge", "support"
        ]
        
        empathy_count = 0
        for turn in conversation_turns:
            text_lower = turn.message_text.lower()
            empathy_count += sum(1 for keyword in empathy_keywords if keyword in text_lower)

        if conversation_turns:
            empathy_ratio = empathy_count / len(conversation_turns)
            empathy_factors.append(min(1.0, empathy_ratio * 2))
            
            if empathy_ratio > 0.2:
                indicators.append("empathetic_language_used")

        # Emotional responsiveness
        if emotional_states:
            high_emotion_states = [state for state in emotional_states if state.intensity > 0.6]
            if high_emotion_states:
                # Check if responses show awareness of emotional state
                emotional_response_score = 0.8  # Assume good emotional response for now
                empathy_factors.append(emotional_response_score)
                indicators.append("emotionally_responsive")

        # Calculate empathy score
        empathy_score = sum(empathy_factors) / len(empathy_factors) if empathy_factors else 0.5
        level = self._score_to_level(empathy_score)

        # Generate suggestions
        suggestions = []
        if empathy_score < 0.6:
            suggestions.append("Show more understanding of user's emotional state")
            suggestions.append("Use more empathetic language and acknowledgments")

        return QualityMetric(
            dimension=QualityDimension.EMPATHY,
            score=empathy_score,
            level=level,
            confidence=0.7,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_relevance(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation relevance."""
        if len(conversation_turns) < 2:
            return self._create_default_metric(QualityDimension.RELEVANCE)

        relevance_scores = []
        indicators = []

        # Calculate relevance between consecutive turns
        for i in range(1, len(conversation_turns)):
            relevance = self._calculate_response_relevance(
                conversation_turns[i-1].message_text,
                conversation_turns[i].message_text
            )
            relevance_scores.append(relevance)

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        level = self._score_to_level(avg_relevance)

        if avg_relevance > 0.8:
            indicators.append("highly_relevant_responses")
        elif avg_relevance < 0.5:
            indicators.append("relevance_issues_detected")

        # Generate suggestions
        suggestions = []
        if avg_relevance < 0.6:
            suggestions.append("Ensure responses directly address user questions")
            suggestions.append("Stay focused on the user's specific needs")

        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=avg_relevance,
            level=level,
            confidence=0.8,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_clarity(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation clarity."""
        clarity_factors = []
        indicators = []

        # Sentence length analysis
        total_sentences = 0
        total_words = 0
        
        for turn in conversation_turns:
            sentences = turn.message_text.split('.')
            words = turn.message_text.split()
            total_sentences += len(sentences)
            total_words += len(words)

        if total_sentences > 0:
            avg_words_per_sentence = total_words / total_sentences
            # Optimal range is 10-20 words per sentence
            if 10 <= avg_words_per_sentence <= 20:
                clarity_factors.append(0.9)
                indicators.append("optimal_sentence_length")
            elif avg_words_per_sentence > 25:
                clarity_factors.append(0.5)
                indicators.append("sentences_too_long")
            else:
                clarity_factors.append(0.7)

        # Jargon and complexity analysis
        complex_words = ["utilize", "facilitate", "implement", "subsequently", "furthermore"]
        simple_alternatives = ["use", "help", "do", "then", "also"]
        
        complex_word_count = 0
        total_word_count = 0
        
        for turn in conversation_turns:
            words = turn.message_text.lower().split()
            total_word_count += len(words)
            complex_word_count += sum(1 for word in words if word in complex_words)

        if total_word_count > 0:
            complexity_ratio = complex_word_count / total_word_count
            clarity_factors.append(1.0 - complexity_ratio)
            
            if complexity_ratio < 0.1:
                indicators.append("clear_simple_language")

        # Calculate clarity score
        clarity_score = sum(clarity_factors) / len(clarity_factors) if clarity_factors else 0.7
        level = self._score_to_level(clarity_score)

        # Generate suggestions
        suggestions = []
        if clarity_score < 0.6:
            suggestions.append("Use shorter, clearer sentences")
            suggestions.append("Avoid jargon and complex terminology")

        return QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=clarity_score,
            level=level,
            confidence=0.7,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_responsiveness(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation responsiveness."""
        responsiveness_factors = []
        indicators = []

        # Response time analysis (if available)
        if len(conversation_turns) > 1:
            response_times = []
            for i in range(1, len(conversation_turns)):
                time_diff = (conversation_turns[i].timestamp - conversation_turns[i-1].timestamp).total_seconds()
                response_times.append(time_diff)

            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                # Good response time is under 60 seconds
                if avg_response_time < 30:
                    responsiveness_factors.append(1.0)
                    indicators.append("very_responsive")
                elif avg_response_time < 60:
                    responsiveness_factors.append(0.8)
                    indicators.append("good_response_time")
                elif avg_response_time < 120:
                    responsiveness_factors.append(0.6)
                else:
                    responsiveness_factors.append(0.4)
                    indicators.append("slow_response_time")

        # Question answering completeness
        questions = [turn for turn in conversation_turns if "?" in turn.message_text]
        if questions:
            # Assume good question answering for now
            responsiveness_factors.append(0.8)
            indicators.append("addresses_user_questions")

        # Calculate responsiveness score
        responsiveness_score = sum(responsiveness_factors) / len(responsiveness_factors) if responsiveness_factors else 0.7
        level = self._score_to_level(responsiveness_score)

        # Generate suggestions
        suggestions = []
        if responsiveness_score < 0.6:
            suggestions.append("Respond more quickly to user messages")
            suggestions.append("Ensure all user questions are addressed")

        return QualityMetric(
            dimension=QualityDimension.RESPONSIVENESS,
            score=responsiveness_score,
            level=level,
            confidence=0.6,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_personalization(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation personalization."""
        personalization_factors = []
        indicators = []

        # Personal reference usage
        personal_references = ["you", "your", "yourself", "personally"]
        personal_count = 0
        total_words = 0

        for turn in conversation_turns:
            words = turn.message_text.lower().split()
            total_words += len(words)
            personal_count += sum(1 for word in words if word in personal_references)

        if total_words > 0:
            personal_ratio = personal_count / total_words
            personalization_factors.append(min(1.0, personal_ratio * 10))  # Scale up
            
            if personal_ratio > 0.05:
                indicators.append("uses_personal_references")

        # Context awareness (simplified)
        if len(conversation_turns) > 3:
            # Assume some level of personalization in longer conversations
            personalization_factors.append(0.7)
            indicators.append("maintains_conversation_context")

        # Calculate personalization score
        personalization_score = sum(personalization_factors) / len(personalization_factors) if personalization_factors else 0.5
        level = self._score_to_level(personalization_score)

        # Generate suggestions
        suggestions = []
        if personalization_score < 0.6:
            suggestions.append("Reference user's specific situation more often")
            suggestions.append("Remember and build on previous conversation points")

        return QualityMetric(
            dimension=QualityDimension.PERSONALIZATION,
            score=personalization_score,
            level=level,
            confidence=0.6,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_emotional_intelligence(
        self, conversation_turns: List[ConversationContext], emotional_states: List[EmotionalState]
    ) -> QualityMetric:
        """Assess emotional intelligence in conversation."""
        ei_factors = []
        indicators = []

        # Emotional awareness
        if emotional_states:
            high_emotion_count = sum(1 for state in emotional_states if state.intensity > 0.6)
            if high_emotion_count > 0:
                # Assume good emotional awareness for now
                ei_factors.append(0.8)
                indicators.append("recognizes_emotional_states")

        # Emotional vocabulary usage
        emotion_words = [
            "happy", "sad", "angry", "frustrated", "excited", "worried",
            "anxious", "calm", "stressed", "relieved", "disappointed", "proud"
        ]
        
        emotion_word_count = 0
        for turn in conversation_turns:
            text_lower = turn.message_text.lower()
            emotion_word_count += sum(1 for word in emotion_words if word in text_lower)

        if conversation_turns:
            emotion_ratio = emotion_word_count / len(conversation_turns)
            ei_factors.append(min(1.0, emotion_ratio * 3))
            
            if emotion_ratio > 0.2:
                indicators.append("uses_emotional_vocabulary")

        # Calculate emotional intelligence score
        ei_score = sum(ei_factors) / len(ei_factors) if ei_factors else 0.5
        level = self._score_to_level(ei_score)

        # Generate suggestions
        suggestions = []
        if ei_score < 0.6:
            suggestions.append("Show more awareness of user's emotional state")
            suggestions.append("Use more emotionally intelligent language")

        return QualityMetric(
            dimension=QualityDimension.EMOTIONAL_INTELLIGENCE,
            score=ei_score,
            level=level,
            confidence=0.7,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _assess_flow(self, conversation_turns: List[ConversationContext]) -> QualityMetric:
        """Assess conversation flow."""
        if len(conversation_turns) < 3:
            return self._create_default_metric(QualityDimension.FLOW)

        flow_factors = []
        indicators = []

        # Transition smoothness
        abrupt_transitions = 0
        for i in range(2, len(conversation_turns)):
            # Simple check for abrupt topic changes
            prev_words = set(conversation_turns[i-1].message_text.lower().split())
            curr_words = set(conversation_turns[i].message_text.lower().split())
            
            overlap = len(prev_words.intersection(curr_words))
            total_words = len(prev_words.union(curr_words))
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity < 0.1:  # Very low similarity indicates abrupt transition
                    abrupt_transitions += 1

        transition_score = 1.0 - (abrupt_transitions / max(1, len(conversation_turns) - 2))
        flow_factors.append(transition_score)

        if transition_score > 0.8:
            indicators.append("smooth_topic_transitions")
        elif transition_score < 0.5:
            indicators.append("abrupt_topic_changes")

        # Turn-taking balance
        user_turns = sum(1 for turn in conversation_turns if turn.user_id)
        assistant_turns = len(conversation_turns) - user_turns
        
        if user_turns > 0 and assistant_turns > 0:
            balance_ratio = min(user_turns, assistant_turns) / max(user_turns, assistant_turns)
            flow_factors.append(balance_ratio)
            
            if balance_ratio > 0.7:
                indicators.append("balanced_turn_taking")

        # Calculate flow score
        flow_score = sum(flow_factors) / len(flow_factors) if flow_factors else 0.7
        level = self._score_to_level(flow_score)

        # Generate suggestions
        suggestions = []
        if flow_score < 0.6:
            suggestions.append("Create smoother transitions between topics")
            suggestions.append("Maintain better conversational rhythm")

        return QualityMetric(
            dimension=QualityDimension.FLOW,
            score=flow_score,
            level=level,
            confidence=0.6,
            indicators=indicators,
            improvement_suggestions=suggestions,
        )

    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, QualityMetric]) -> float:
        """Calculate overall quality score from dimension scores."""
        if not dimension_scores:
            return 0.5

        # Weight different dimensions
        dimension_weights = {
            QualityDimension.HELPFULNESS: 0.2,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.EMPATHY: 0.15,
            QualityDimension.ENGAGEMENT: 0.1,
            QualityDimension.COHERENCE: 0.1,
            QualityDimension.RESPONSIVENESS: 0.05,
            QualityDimension.PERSONALIZATION: 0.05,
            QualityDimension.EMOTIONAL_INTELLIGENCE: 0.03,
            QualityDimension.FLOW: 0.02,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, metric in dimension_scores.items():
            weight = dimension_weights.get(dimension, 0.1)
            weighted_sum += metric.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.6:
            return QualityLevel.SATISFACTORY
        elif score >= 0.4:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.POOR

    def _create_default_metric(self, dimension: QualityDimension) -> QualityMetric:
        """Create default quality metric."""
        return QualityMetric(
            dimension=dimension,
            score=0.7,
            level=QualityLevel.SATISFACTORY,
            confidence=0.5,
            indicators=["insufficient_data"],
            improvement_suggestions=["Gather more conversation data for better assessment"],
        )

    def _create_empty_assessment(self, conversation_id: str, user_id: str) -> ConversationQualityAssessment:
        """Create empty quality assessment."""
        dimension_scores = {}
        for dimension in QualityDimension:
            dimension_scores[dimension] = self._create_default_metric(dimension)

        return ConversationQualityAssessment(
            conversation_id=conversation_id,
            user_id=user_id,
            overall_score=0.5,
            overall_level=QualityLevel.SATISFACTORY,
            dimension_scores=dimension_scores,
            key_strengths=["baseline_assessment"],
            improvement_areas=["all_dimensions"],
        )

    def _extract_simple_topic(self, text: str) -> Optional[str]:
        """Extract simple topic from text."""
        # Simple topic extraction - look for key nouns
        words = text.lower().split()
        topic_words = []
        
        for word in words:
            if len(word) > 4 and word.isalpha():
                topic_words.append(word)
        
        return topic_words[0] if topic_words else None

    def _calculate_response_relevance(self, user_message: str, response: str) -> float:
        """Calculate relevance between user message and response."""
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())
        
        if not user_words:
            return 0.5

        # Calculate word overlap
        overlap = len(user_words.intersection(response_words))
        total_unique = len(user_words.union(response_words))
        
        if total_unique == 0:
            return 0.5

        # Basic relevance score based on word overlap
        relevance = overlap / len(user_words)
        return min(1.0, relevance * 2)  # Scale up and cap at 1.0

    def _calculate_conversation_duration(self, conversation_turns: List[ConversationContext]) -> timedelta:
        """Calculate conversation duration."""
        if len(conversation_turns) < 2:
            return timedelta(0)

        start_time = min(turn.timestamp for turn in conversation_turns)
        end_time = max(turn.timestamp for turn in conversation_turns)
        
        return end_time - start_time

    def _identify_key_strengths(self, dimension_scores: Dict[QualityDimension, QualityMetric]) -> List[str]:
        """Identify key strengths from dimension scores."""
        strengths = []
        
        for dimension, metric in dimension_scores.items():
            if metric.score >= 0.8:
                strengths.append(f"Strong {dimension.value}")
        
        return strengths[:3]  # Top 3 strengths

    def _identify_improvement_areas(self, dimension_scores: Dict[QualityDimension, QualityMetric]) -> List[str]:
        """Identify improvement areas from dimension scores."""
        improvements = []
        
        for dimension, metric in dimension_scores.items():
            if metric.score < 0.6:
                improvements.append(f"Improve {dimension.value}")
        
        return improvements[:3]  # Top 3 improvement areas

    def _initialize_benchmarks(self) -> Dict[QualityDimension, float]:
        """Initialize quality benchmarks."""
        return {
            QualityDimension.COHERENCE: 0.75,
            QualityDimension.ENGAGEMENT: 0.70,
            QualityDimension.HELPFULNESS: 0.80,
            QualityDimension.EMPATHY: 0.75,
            QualityDimension.RELEVANCE: 0.80,
            QualityDimension.CLARITY: 0.75,
            QualityDimension.RESPONSIVENESS: 0.70,
            QualityDimension.PERSONALIZATION: 0.65,
            QualityDimension.EMOTIONAL_INTELLIGENCE: 0.70,
            QualityDimension.FLOW: 0.70,
        }

    # Additional helper methods for turn-level assessment
    def _assess_turn_relevance(self, user_message: str, assistant_response: str) -> float:
        """Assess relevance of a single turn."""
        return self._calculate_response_relevance(user_message, assistant_response)

    def _assess_turn_clarity(self, assistant_response: str) -> float:
        """Assess clarity of assistant response."""
        words = assistant_response.split()
        sentences = assistant_response.split('.')
        
        if not sentences:
            return 0.5

        avg_words_per_sentence = len(words) / len(sentences)
        
        # Optimal range is 10-20 words per sentence
        if 10 <= avg_words_per_sentence <= 20:
            return 0.9
        elif avg_words_per_sentence > 25:
            return 0.5
        else:
            return 0.7

    def _assess_turn_empathy(self, assistant_response: str, emotional_state: EmotionalState) -> float:
        """Assess empathy in assistant response."""
        empathy_keywords = [
            "understand", "feel", "sorry", "difficult", "challenging",
            "appreciate", "recognize", "acknowledge", "support"
        ]
        
        response_lower = assistant_response.lower()
        empathy_count = sum(1 for keyword in empathy_keywords if keyword in response_lower)
        
        base_empathy = min(1.0, empathy_count / 3.0)
        
        # Boost empathy score for high emotional intensity
        if emotional_state.intensity > 0.7:
            base_empathy = min(1.0, base_empathy * 1.2)
        
        return base_empathy

    def _assess_turn_helpfulness(self, user_message: str, assistant_response: str) -> float:
        """Assess helpfulness of assistant response."""
        # Check if user asked a question
        is_question = "?" in user_message
        
        # Check for helpful elements in response
        helpful_elements = [
            "help", "solution", "answer", "suggest", "recommend",
            "try", "consider", "might", "could", "would"
        ]
        
        response_lower = assistant_response.lower()
        helpful_count = sum(1 for element in helpful_elements if element in response_lower)
        
        base_helpfulness = min(1.0, helpful_count / 3.0)
        
        # Boost if responding to a question
        if is_question:
            base_helpfulness = min(1.0, base_helpfulness * 1.2)
        
        return base_helpfulness

    def _assess_turn_responsiveness(self, context: ConversationContext) -> float:
        """Assess responsiveness of a turn."""
        # This would use actual response time data if available
        # For now, return a default good score
        return 0.8

    # Additional methods for trend analysis and insights
    def _analyze_dimension_trend(
        self, 
        dimension: QualityDimension, 
        recent_assessments: List[ConversationQualityAssessment],
        all_assessments: List[ConversationQualityAssessment]
    ) -> Optional[QualityTrend]:
        """Analyze trend for a specific dimension."""
        if len(recent_assessments) < 3:
            return None

        # Get scores for this dimension
        recent_scores = [
            assessment.dimension_scores[dimension].score 
            for assessment in recent_assessments
            if dimension in assessment.dimension_scores
        ]
        
        all_scores = [
            assessment.dimension_scores[dimension].score 
            for assessment in all_assessments
            if dimension in assessment.dimension_scores
        ]

        if len(recent_scores) < 3 or len(all_scores) < 5:
            return None

        recent_avg = sum(recent_scores) / len(recent_scores)
        historical_avg = sum(all_scores) / len(all_scores)
        
        # Simple trend analysis
        trend_strength = abs(recent_avg - historical_avg)
        
        if recent_avg > historical_avg + 0.1:
            trend_direction = "improving"
        elif recent_avg < historical_avg - 0.1:
            trend_direction = "declining"
        else:
            trend_direction = "stable"

        return QualityTrend(
            dimension=dimension,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            recent_average=recent_avg,
            historical_average=historical_avg,
            trend_confidence=0.7,
        )

    def _generate_performance_insights(self, assessments: List[ConversationQualityAssessment]) -> List[QualityInsight]:
        """Generate performance insights."""
        insights = []
        
        if not assessments:
            return insights

        # Overall performance insight
        avg_score = sum(a.overall_score for a in assessments) / len(assessments)
        
        if avg_score > 0.8:
            insights.append(QualityInsight(
                insight_type="performance",
                description="Consistently high conversation quality",
                impact_level="high",
                actionable_recommendations=["Maintain current approach", "Share best practices"],
            ))
        elif avg_score < 0.6:
            insights.append(QualityInsight(
                insight_type="performance",
                description="Conversation quality needs improvement",
                impact_level="high",
                actionable_recommendations=[
                    "Focus on core quality dimensions",
                    "Implement quality improvement strategies"
                ],
            ))

        return insights

    def _generate_trend_insights(self, user_id: str, timeframe_days: int) -> List[QualityInsight]:
        """Generate trend-based insights."""
        insights = []
        trends = self.analyze_quality_trends(user_id, timeframe_days)
        
        improving_trends = [t for t in trends if t.trend_direction == "improving"]
        declining_trends = [t for t in trends if t.trend_direction == "declining"]
        
        if improving_trends:
            insights.append(QualityInsight(
                insight_type="trend",
                description=f"Improving trends in {len(improving_trends)} quality dimensions",
                impact_level="medium",
                actionable_recommendations=["Continue current improvement strategies"],
            ))
        
        if declining_trends:
            insights.append(QualityInsight(
                insight_type="trend",
                description=f"Declining trends in {len(declining_trends)} quality dimensions",
                impact_level="high",
                actionable_recommendations=["Address declining quality areas immediately"],
            ))

        return insights

    def _generate_pattern_insights(self, assessments: List[ConversationQualityAssessment]) -> List[QualityInsight]:
        """Generate pattern-based insights."""
        insights = []
        
        if len(assessments) < 5:
            return insights

        # Consistency analysis
        scores = [a.overall_score for a in assessments]
        score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        
        if score_variance < 0.01:
            insights.append(QualityInsight(
                insight_type="pattern",
                description="Very consistent conversation quality",
                impact_level="medium",
                actionable_recommendations=["Maintain consistency while seeking improvement"],
            ))
        elif score_variance > 0.05:
            insights.append(QualityInsight(
                insight_type="pattern",
                description="Inconsistent conversation quality",
                impact_level="high",
                actionable_recommendations=["Identify factors causing quality variation"],
            ))

        return insights

    def _generate_improvement_insights(self, assessments: List[ConversationQualityAssessment]) -> List[QualityInsight]:
        """Generate improvement-focused insights."""
        insights = []
        
        if not assessments:
            return insights

        # Find most common improvement areas
        improvement_counts = defaultdict(int)
        for assessment in assessments:
            for area in assessment.improvement_areas:
                improvement_counts[area] += 1

        if improvement_counts:
            most_common = max(improvement_counts.items(), key=lambda x: x[1])
            insights.append(QualityInsight(
                insight_type="improvement",
                description=f"Most frequent improvement area: {most_common[0]}",
                impact_level="high",
                actionable_recommendations=[f"Focus improvement efforts on {most_common[0]}"],
            ))

        return insights

    def _calculate_average_score(self, assessments: List[ConversationQualityAssessment]) -> float:
        """Calculate average overall score."""
        if not assessments:
            return 0.0
        return sum(a.overall_score for a in assessments) / len(assessments)

    def _calculate_quality_distribution(self, assessments: List[ConversationQualityAssessment]) -> Dict[str, int]:
        """Calculate distribution of quality levels."""
        distribution = defaultdict(int)
        for assessment in assessments:
            distribution[assessment.overall_level.value] += 1
        return dict(distribution)

    def _calculate_dimension_averages(self, assessments: List[ConversationQualityAssessment]) -> Dict[str, float]:
        """Calculate average scores by dimension."""
        dimension_sums = defaultdict(float)
        dimension_counts = defaultdict(int)
        
        for assessment in assessments:
            for dimension, metric in assessment.dimension_scores.items():
                dimension_sums[dimension.value] += metric.score
                dimension_counts[dimension.value] += 1

        averages = {}
        for dimension, total in dimension_sums.items():
            count = dimension_counts[dimension]
            averages[dimension] = total / count if count > 0 else 0.0

        return averages

    def _identify_best_dimensions(self, assessments: List[ConversationQualityAssessment]) -> List[str]:
        """Identify best performing dimensions."""
        dimension_averages = self._calculate_dimension_averages(assessments)
        sorted_dimensions = sorted(dimension_averages.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, _ in sorted_dimensions[:3]]

    def _identify_improvement_opportunities(self, assessments: List[ConversationQualityAssessment]) -> List[str]:
        """Identify improvement opportunities."""
        dimension_averages = self._calculate_dimension_averages(assessments)
        sorted_dimensions = sorted(dimension_averages.items(), key=lambda x: x[1])
        return [dim for dim, score in sorted_dimensions[:3] if score < 0.7]

    def _calculate_quality_consistency(self, assessments: List[ConversationQualityAssessment]) -> float:
        """Calculate quality consistency score."""
        if len(assessments) < 2:
            return 1.0

        scores = [a.overall_score for a in assessments]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score)**2 for s in scores) / len(scores)
        
        # Convert variance to consistency score (lower variance = higher consistency)
        consistency = 1.0 - min(1.0, variance * 10)  # Scale variance
        return consistency

    def _calculate_recent_trend(self, assessments: List[ConversationQualityAssessment]) -> str:
        """Calculate recent trend direction."""
        if len(assessments) < 3:
            return "insufficient_data"

        recent_scores = [a.overall_score for a in assessments[-3:]]
        older_scores = [a.overall_score for a in assessments[:-3]] if len(assessments) > 3 else [assessments[0].overall_score]

        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)

        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"

    def _calculate_percentile(self, dimension: QualityDimension, score: float) -> float:
        """Calculate percentile for a dimension score."""
        # This would use historical data in a real implementation
        # For now, return a simple percentile based on score
        if score > 0.9:
            return 95.0
        elif score > 0.8:
            return 80.0
        elif score > 0.7:
            return 65.0
        elif score > 0.6:
            return 50.0
        elif score > 0.5:
            return 35.0
        else:
            return 20.0


# Singleton instance
_conversation_quality_assessor_instance = None
_conversation_quality_assessor_lock = threading.Lock()


def get_conversation_quality_assessor() -> ConversationQualityAssessor:
    """
    Get singleton conversation quality assessor instance.

    Returns:
        Shared ConversationQualityAssessor instance
    """
    global _conversation_quality_assessor_instance

    if _conversation_quality_assessor_instance is None:
        with _conversation_quality_assessor_lock:
            if _conversation_quality_assessor_instance is None:
                _conversation_quality_assessor_instance = ConversationQualityAssessor()

    return _conversation_quality_assessor_instance