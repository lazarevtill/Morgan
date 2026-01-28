"""
Emotional trigger detection module.

Provides focused emotional trigger identification, sensitivity analysis,
and trigger management for enhanced emotional intelligence and user support.

Semantic-First Architecture:
- LLM analysis identifies CONCEPTUAL triggers, not just word triggers
- Understands context-dependent triggering (work stress only with deadlines)
- Tracks WHY things trigger, not just THAT they trigger
- Detects trigger patterns from meaning, not keyword matching
"""

import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import EmotionalState, EmotionType
from morgan.intelligence.emotions.memory import (
    EmotionalMemory,
    get_emotional_memory_storage,
)
from morgan.services.llm import get_llm_service
from morgan.utils.llm_parsing import parse_llm_json
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalTrigger:
    """
    Represents an identified emotional trigger.

    Features:
    - Trigger pattern identification
    - Emotional response tracking
    - Sensitivity scoring
    - Context awareness
    """

    def __init__(
        self,
        trigger_id: str,
        user_id: str,
        trigger_pattern: str,
        trigger_type: str,
        emotional_responses: List[EmotionType],
        sensitivity_score: float,
        confidence: float,
        context_factors: Dict[str, Any],
        detection_count: int = 1,
    ):
        """Initialize emotional trigger."""
        self.trigger_id = trigger_id
        self.user_id = user_id
        self.trigger_pattern = trigger_pattern
        self.trigger_type = trigger_type
        self.emotional_responses = emotional_responses
        self.sensitivity_score = sensitivity_score
        self.confidence = confidence
        self.context_factors = context_factors
        self.detection_count = detection_count
        self.first_detected = datetime.now(timezone.utc)
        self.last_detected = datetime.now(timezone.utc)
        self.response_intensities = []
        self.associated_memories = []

    def update_detection(
        self, emotional_response: EmotionType, intensity: float, memory_id: str
    ):
        """Update trigger with new detection."""
        self.detection_count += 1
        self.last_detected = datetime.now(timezone.utc)
        self.response_intensities.append(intensity)
        self.associated_memories.append(memory_id)

        # Update emotional responses if new
        if emotional_response not in self.emotional_responses:
            self.emotional_responses.append(emotional_response)

        # Recalculate sensitivity score
        self._recalculate_sensitivity()

    def _recalculate_sensitivity(self):
        """Recalculate sensitivity score based on detection history."""
        if not self.response_intensities:
            return

        # Base sensitivity on average intensity and frequency
        avg_intensity = sum(self.response_intensities) / len(self.response_intensities)
        frequency_factor = min(1.0, self.detection_count / 10.0)

        # Higher sensitivity for more frequent and intense responses
        self.sensitivity_score = (avg_intensity * 0.7) + (frequency_factor * 0.3)

    def get_trigger_strength(self) -> float:
        """
        Calculate overall trigger strength.

        Returns:
            Trigger strength score (0.0 to 1.0)
        """
        # Combine sensitivity, confidence, and detection frequency
        frequency_factor = min(1.0, self.detection_count / 5.0)
        return (
            (self.sensitivity_score * 0.4)
            + (self.confidence * 0.4)
            + (frequency_factor * 0.2)
        )

    def is_active(self, max_age_days: int = 60) -> bool:
        """
        Check if trigger is still active.

        Args:
            max_age_days: Maximum days since last detection

        Returns:
            True if trigger is still active
        """
        age = datetime.now(timezone.utc) - self.last_detected
        return age.days <= max_age_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary for storage."""
        return {
            "trigger_id": self.trigger_id,
            "user_id": self.user_id,
            "trigger_pattern": self.trigger_pattern,
            "trigger_type": self.trigger_type,
            "emotional_responses": [e.value for e in self.emotional_responses],
            "sensitivity_score": self.sensitivity_score,
            "confidence": self.confidence,
            "context_factors": self.context_factors,
            "detection_count": self.detection_count,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "response_intensities": self.response_intensities,
            "associated_memories": self.associated_memories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalTrigger":
        """Create trigger from dictionary."""
        trigger = cls(
            trigger_id=data["trigger_id"],
            user_id=data["user_id"],
            trigger_pattern=data["trigger_pattern"],
            trigger_type=data["trigger_type"],
            emotional_responses=[EmotionType(e) for e in data["emotional_responses"]],
            sensitivity_score=data["sensitivity_score"],
            confidence=data["confidence"],
            context_factors=data["context_factors"],
            detection_count=data["detection_count"],
        )

        # Restore metadata
        trigger.first_detected = datetime.fromisoformat(data["first_detected"])
        trigger.last_detected = datetime.fromisoformat(data["last_detected"])
        trigger.response_intensities = data["response_intensities"]
        trigger.associated_memories = data["associated_memories"]

        return trigger


class EmotionalTriggerDetector:
    """
    Detects and analyzes emotional triggers in user interactions.

    Semantic-First Architecture:
    - LLM analysis is PRIMARY for conceptual trigger detection
    - Identifies CONCEPTS that trigger, not just WORDS
    - Understands context-dependent triggering
    - Tracks WHY things trigger (reasoning), not just THAT they trigger
    - Pattern matching is SECONDARY for validation

    Features:
    - Semantic conceptual trigger detection
    - Context-aware trigger analysis
    - Trigger causality understanding
    - Sensitivity assessment with reasoning
    - Trigger management and tracking
    """

    # Trigger type definitions (expanded for semantic understanding)
    TRIGGER_TYPES = {
        "conceptual": "Abstract concepts that trigger emotions (not specific words)",
        "relational": "Relationship-related triggers (family, work relationships)",
        "achievement": "Success/failure related triggers",
        "temporal": "Time-based triggers (dates, anniversaries, time of day)",
        "contextual": "Situational triggers based on conversation context",
        "identity": "Self-concept related triggers",
        "loss": "Loss or grief related triggers",
        "control": "Autonomy and control related triggers",
    }

    # Validation patterns (for secondary verification only)
    _VALIDATION_PATTERNS = {
        "negative_self_talk": re.compile(
            r"\b(i am|i\'m)\s+(stupid|worthless|useless|terrible|awful|horrible)\b",
            re.IGNORECASE,
        ),
        "stress_indicators": re.compile(
            r"\b(stressed|overwhelmed|anxious|worried|panic|pressure)\b",
            re.IGNORECASE,
        ),
        "loss_grief": re.compile(
            r"\b(lost|death|died|gone|miss|grief|mourning)\b", re.IGNORECASE
        ),
    }

    def __init__(self):
        """Initialize emotional trigger detector with LLM service."""
        self.settings = get_settings()
        self.memory_storage = get_emotional_memory_storage()
        self.llm_service = get_llm_service()

        # Trigger storage
        self._detected_triggers = {}  # user_id -> List[EmotionalTrigger]
        self._trigger_patterns = {}  # Compiled regex patterns (for validation)

        # Initialize validation patterns
        self._initialize_trigger_patterns()

        logger.info("Emotional Trigger Detector initialized (semantic mode)")

    def detect_triggers(
        self, user_id: str, analysis_days: int = 60, min_trigger_confidence: float = 0.6
    ) -> List[EmotionalTrigger]:
        """
        Detect emotional triggers for a user using semantic-first approach.

        Semantic-First Architecture:
        1. PRIMARY: LLM semantic analysis for conceptual triggers
        2. SECONDARY: Pattern analysis for validation and additional triggers
        3. COMBINED: Merged results with calibrated confidence

        Args:
            user_id: User identifier
            analysis_days: Days of history to analyze
            min_trigger_confidence: Minimum confidence threshold

        Returns:
            List of detected emotional triggers
        """
        # Retrieve emotional memories for analysis
        memories = self.memory_storage.retrieve_memories(
            user_id=user_id, max_age_days=analysis_days, min_importance=0.3, limit=150
        )

        if len(memories) < 5:
            logger.debug(
                f"Insufficient data for trigger detection: {len(memories)} memories"
            )
            return []

        detected_triggers = []

        # PRIMARY: Semantic conceptual trigger detection
        semantic_triggers = self._analyze_triggers_semantically(user_id, memories)
        detected_triggers.extend(semantic_triggers)

        # SECONDARY: Pattern-based validation and additional triggers
        temporal_triggers = self._detect_temporal_triggers(user_id, memories)
        contextual_triggers = self._detect_contextual_triggers(user_id, memories)

        detected_triggers.extend(temporal_triggers)
        detected_triggers.extend(contextual_triggers)

        # Filter by confidence threshold
        detected_triggers = [
            trigger
            for trigger in detected_triggers
            if trigger.confidence >= min_trigger_confidence
        ]

        # Deduplicate similar triggers
        detected_triggers = self._deduplicate_triggers(detected_triggers)

        # Store triggers for user
        self._detected_triggers[user_id] = detected_triggers

        logger.info(
            f"Detected {len(detected_triggers)} emotional triggers for user {user_id}"
        )
        return detected_triggers

    def _analyze_triggers_semantically(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """
        Semantic trigger detection using LLM analysis.

        This is the PRIMARY method - it identifies:
        - CONCEPTUAL triggers (concepts, not just words)
        - WHY things trigger (reasoning and causality)
        - Context-dependent triggering patterns
        - Hidden trigger patterns not obvious from keywords

        Args:
            user_id: User identifier
            memories: Emotional memories to analyze

        Returns:
            List of semantically detected triggers
        """
        triggers = []

        # Prepare memory summaries for analysis
        memory_summaries = []
        for memory in memories[-30:]:  # Analyze recent memories
            summary = {
                "message": memory.conversation_context.message_text[:200],
                "emotion": memory.emotional_state.primary_emotion.value,
                "intensity": memory.emotional_state.intensity,
                "indicators": memory.emotional_state.emotional_indicators[:3],
            }
            memory_summaries.append(summary)

        if len(memory_summaries) < 5:
            return triggers

        try:
            prompt = f"""Analyze these emotional interactions to identify CONCEPTUAL triggers.

Interactions (message, resulting emotion, intensity):
{json.dumps(memory_summaries[:15], indent=2)}

A trigger is a CONCEPT, not a word. For example:
- "Work" might trigger stress, but only when deadlines are mentioned
- "Family" might trigger joy OR sadness depending on context
- Criticism of their ideas might trigger anger more than criticism of them directly

Identify patterns where certain CONCEPTS consistently lead to emotional responses.

For each trigger, explain:
1. The CONCEPT (not just keywords) - what abstract idea triggers them
2. WHY it likely triggers them - underlying emotional significance
3. CONTEXT dependence - when does it trigger vs not trigger
4. Emotional response pattern - what emotions and intensity

Respond with JSON ONLY:
{{
    "triggers": [
        {{
            "concept": "what the trigger concept is (e.g., 'perceived criticism', 'loss of control')",
            "trigger_type": "conceptual|relational|achievement|temporal|identity|loss|control",
            "emotional_response": "joy|sadness|anger|fear|surprise|disgust",
            "typical_intensity": 0.0-1.0,
            "why_triggers": "explanation of emotional significance",
            "context_factors": ["when it triggers", "when it doesn't"],
            "evidence": ["quotes or patterns that support this"],
            "confidence": 0.0-1.0
        }}
    ],
    "analysis_notes": "overall patterns observed"
}}

Focus on finding 3-5 most significant triggers, not every possible one."""

            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000,  # Increased for reasoning models
            )

            # Parse JSON using utility that handles reasoning blocks
            analysis = parse_llm_json(response.content)
            if analysis is None:
                logger.warning("Failed to parse semantic trigger analysis")
                return self._detect_triggers_legacy(user_id, memories)

            for trigger_data in analysis.get("triggers", []):
                trigger_id = hashlib.sha256(
                    f"{user_id}_semantic_{trigger_data['concept']}".encode()
                ).hexdigest()[:16]

                # Map emotional response to EmotionType
                emotion_response = trigger_data.get("emotional_response", "neutral")
                try:
                    emotion_type = EmotionType(emotion_response)
                except ValueError:
                    emotion_type = EmotionType.NEUTRAL

                trigger = EmotionalTrigger(
                    trigger_id=trigger_id,
                    user_id=user_id,
                    trigger_pattern=trigger_data["concept"],
                    trigger_type=trigger_data.get("trigger_type", "conceptual"),
                    emotional_responses=[emotion_type],
                    sensitivity_score=float(trigger_data.get("typical_intensity", 0.5)),
                    confidence=float(trigger_data.get("confidence", 0.6)),
                    context_factors={
                        "why_triggers": trigger_data.get("why_triggers", ""),
                        "context_factors": trigger_data.get("context_factors", []),
                        "evidence": trigger_data.get("evidence", []),
                        "semantic_analysis": True,
                    },
                )

                triggers.append(trigger)

            logger.debug(
                f"Semantic trigger analysis found {len(triggers)} conceptual triggers"
            )

        except Exception as e:
            logger.warning(f"Semantic trigger analysis failed: {e}")
            # Fall back to legacy pattern-based detection
            return self._detect_triggers_legacy(user_id, memories)

        return triggers

    def _detect_triggers_legacy(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Legacy pattern-based trigger detection (fallback)."""
        triggers = []
        triggers.extend(self._detect_keyword_triggers(user_id, memories))
        triggers.extend(self._detect_topic_triggers(user_id, memories))
        triggers.extend(self._detect_semantic_triggers(user_id, memories))
        triggers.extend(self._detect_personal_triggers(user_id, memories))
        return triggers

    def _deduplicate_triggers(
        self, triggers: List[EmotionalTrigger]
    ) -> List[EmotionalTrigger]:
        """Remove duplicate or overlapping triggers, keeping highest confidence."""
        if not triggers:
            return []

        # Group by similar concepts
        seen_concepts = {}
        for trigger in triggers:
            key = trigger.trigger_pattern.lower()
            if key not in seen_concepts or trigger.confidence > seen_concepts[key].confidence:
                seen_concepts[key] = trigger

        return list(seen_concepts.values())

    def get_user_triggers(
        self,
        user_id: str,
        trigger_types: Optional[List[str]] = None,
        active_only: bool = True,
        min_strength: float = 0.0,
    ) -> List[EmotionalTrigger]:
        """
        Get detected triggers for a user.

        Args:
            user_id: User identifier
            trigger_types: Optional filter by trigger types
            active_only: Only return active triggers
            min_strength: Minimum trigger strength

        Returns:
            List of user's emotional triggers
        """
        user_triggers = self._detected_triggers.get(user_id, [])

        # Filter by trigger types
        if trigger_types:
            user_triggers = [
                t for t in user_triggers if t.trigger_type in trigger_types
            ]

        # Filter by active status
        if active_only:
            user_triggers = [t for t in user_triggers if t.is_active()]

        # Filter by strength
        if min_strength > 0.0:
            user_triggers = [
                t for t in user_triggers if t.get_trigger_strength() >= min_strength
            ]

        return user_triggers

    def analyze_trigger_sensitivity(
        self, user_id: str, content: str, emotional_state: EmotionalState
    ) -> List[Dict[str, Any]]:
        """
        Analyze content for potential trigger activation.

        Args:
            user_id: User identifier
            content: Content to analyze
            emotional_state: Current emotional state

        Returns:
            List of potentially activated triggers
        """
        user_triggers = self.get_user_triggers(user_id, active_only=True)
        activated_triggers = []

        for trigger in user_triggers:
            activation_score = self._calculate_trigger_activation(
                trigger, content, emotional_state
            )

            if activation_score > 0.5:
                activated_triggers.append(
                    {
                        "trigger": trigger,
                        "activation_score": activation_score,
                        "predicted_response": (
                            trigger.emotional_responses[0]
                            if trigger.emotional_responses
                            else EmotionType.NEUTRAL
                        ),
                        "sensitivity_level": trigger.sensitivity_score,
                        "recommendations": self._generate_trigger_recommendations(
                            trigger
                        ),
                    }
                )

        # Sort by activation score
        activated_triggers.sort(key=lambda x: x["activation_score"], reverse=True)

        return activated_triggers

    def update_trigger_from_interaction(
        self,
        user_id: str,
        content: str,
        emotional_state: EmotionalState,
        memory_id: str,
    ):
        """
        Update triggers based on new interaction.

        Args:
            user_id: User identifier
            content: Interaction content
            emotional_state: Resulting emotional state
            memory_id: Associated memory ID
        """
        user_triggers = self.get_user_triggers(user_id, active_only=True)

        for trigger in user_triggers:
            if self._content_matches_trigger(content, trigger):
                trigger.update_detection(
                    emotional_response=emotional_state.primary_emotion,
                    intensity=emotional_state.intensity,
                    memory_id=memory_id,
                )
                logger.debug(f"Updated trigger {trigger.trigger_id} for user {user_id}")

    def _initialize_trigger_patterns(self):
        """Initialize common trigger patterns."""
        self._trigger_patterns = {
            "negative_self_talk": re.compile(
                r"\b(i am|i\'m)\s+(stupid|worthless|useless|terrible|awful|horrible)\b",
                re.IGNORECASE,
            ),
            "failure_words": re.compile(
                r"\b(fail|failed|failure|mistake|wrong|error|mess up)\b", re.IGNORECASE
            ),
            "stress_indicators": re.compile(
                r"\b(stressed|overwhelmed|anxious|worried|panic|pressure)\b",
                re.IGNORECASE,
            ),
            "loss_grief": re.compile(
                r"\b(lost|death|died|gone|miss|grief|mourning)\b", re.IGNORECASE
            ),
            "rejection_abandonment": re.compile(
                r"\b(rejected|abandoned|alone|lonely|ignored|left out)\b", re.IGNORECASE
            ),
            "criticism_judgment": re.compile(
                r"\b(criticized|judged|blamed|attacked|insulted)\b", re.IGNORECASE
            ),
            "health_concerns": re.compile(
                r"\b(sick|illness|disease|pain|hurt|hospital|doctor)\b", re.IGNORECASE
            ),
            "financial_stress": re.compile(
                r"\b(money|debt|broke|expensive|afford|financial|budget)\b",
                re.IGNORECASE,
            ),
        }

    def _detect_keyword_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect keyword-based emotional triggers."""
        triggers = []

        # Group memories by emotional indicators
        indicator_emotions = defaultdict(list)

        for memory in memories:
            for indicator in memory.emotional_state.emotional_indicators:
                indicator_emotions[indicator].append(memory)

        # Find consistent keyword triggers
        for indicator, associated_memories in indicator_emotions.items():
            if len(associated_memories) >= 3:  # Minimum occurrences
                emotions = [
                    m.emotional_state.primary_emotion for m in associated_memories
                ]
                emotion_counts = Counter(emotions)
                dominant_emotion, count = emotion_counts.most_common(1)[0]

                consistency = count / len(associated_memories)
                if consistency >= 0.7:  # High consistency threshold

                    # Calculate sensitivity
                    intensities = [
                        m.emotional_state.intensity for m in associated_memories
                    ]
                    avg_intensity = sum(intensities) / len(intensities)

                    trigger_id = hashlib.sha256(
                        f"{user_id}_keyword_{indicator}".encode()
                    ).hexdigest()[:16]

                    trigger = EmotionalTrigger(
                        trigger_id=trigger_id,
                        user_id=user_id,
                        trigger_pattern=indicator,
                        trigger_type="keyword",
                        emotional_responses=[dominant_emotion],
                        sensitivity_score=avg_intensity,
                        confidence=consistency,
                        context_factors={
                            "keyword": indicator,
                            "occurrence_count": len(associated_memories),
                            "consistency": consistency,
                        },
                        detection_count=len(associated_memories),
                    )

                    # Add associated memories
                    trigger.associated_memories = [
                        m.memory_id for m in associated_memories
                    ]
                    trigger.response_intensities = intensities

                    triggers.append(trigger)

        return triggers

    def _detect_topic_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect topic-based emotional triggers."""
        triggers = []

        # Extract topics from conversation context
        topic_emotions = defaultdict(list)

        for memory in memories:
            # Simple topic extraction from message text
            message = memory.conversation_context.message_text.lower()

            # Check for common topic patterns
            topics = []
            if any(
                word in message
                for word in ["work", "job", "career", "boss", "colleague"]
            ):
                topics.append("work")
            if any(
                word in message
                for word in ["family", "parent", "mother", "father", "sibling"]
            ):
                topics.append("family")
            if any(
                word in message
                for word in [
                    "relationship",
                    "partner",
                    "boyfriend",
                    "girlfriend",
                    "spouse",
                ]
            ):
                topics.append("relationships")
            if any(
                word in message
                for word in ["health", "medical", "doctor", "hospital", "sick"]
            ):
                topics.append("health")
            if any(
                word in message
                for word in ["money", "financial", "debt", "budget", "expensive"]
            ):
                topics.append("finances")

            for topic in topics:
                topic_emotions[topic].append(memory)

        # Analyze topic-emotion associations
        for topic, associated_memories in topic_emotions.items():
            if len(associated_memories) >= 4:  # Minimum occurrences for topics
                emotions = [
                    m.emotional_state.primary_emotion for m in associated_memories
                ]

                # Check for negative emotional bias
                negative_emotions = [
                    EmotionType.SADNESS,
                    EmotionType.ANGER,
                    EmotionType.FEAR,
                ]
                negative_count = sum(1 for e in emotions if e in negative_emotions)
                negative_ratio = negative_count / len(emotions)

                if (
                    negative_ratio >= 0.6
                ):  # Topic consistently triggers negative emotions
                    dominant_emotion = max(
                        [e for e in emotions if e in negative_emotions],
                        key=emotions.count,
                    )

                    intensities = [
                        m.emotional_state.intensity for m in associated_memories
                    ]
                    avg_intensity = sum(intensities) / len(intensities)

                    trigger_id = hashlib.sha256(
                        f"{user_id}_topic_{topic}".encode()
                    ).hexdigest()[:16]

                    trigger = EmotionalTrigger(
                        trigger_id=trigger_id,
                        user_id=user_id,
                        trigger_pattern=topic,
                        trigger_type="topic",
                        emotional_responses=[dominant_emotion],
                        sensitivity_score=avg_intensity,
                        confidence=negative_ratio,
                        context_factors={
                            "topic": topic,
                            "negative_ratio": negative_ratio,
                            "occurrence_count": len(associated_memories),
                        },
                        detection_count=len(associated_memories),
                    )

                    trigger.associated_memories = [
                        m.memory_id for m in associated_memories
                    ]
                    trigger.response_intensities = intensities

                    triggers.append(trigger)

        return triggers

    def _detect_temporal_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect temporal emotional triggers."""
        triggers = []

        # Group memories by time patterns
        time_patterns = {
            "anniversary_dates": defaultdict(list),
            "seasonal_patterns": defaultdict(list),
            "time_of_day": defaultdict(list),
        }

        for memory in memories:
            timestamp = memory.created_at

            # Check for anniversary patterns (same month/day)
            anniversary_key = f"{timestamp.month:02d}-{timestamp.day:02d}"
            time_patterns["anniversary_dates"][anniversary_key].append(memory)

            # Check for seasonal patterns
            season = self._get_season(timestamp.month)
            time_patterns["seasonal_patterns"][season].append(memory)

            # Check for time of day patterns
            hour_group = self._get_hour_group(timestamp.hour)
            time_patterns["time_of_day"][hour_group].append(memory)

        # Analyze temporal patterns
        for pattern_type, pattern_data in time_patterns.items():
            for pattern_key, associated_memories in pattern_data.items():
                if len(associated_memories) >= 3:
                    emotions = [
                        m.emotional_state.primary_emotion for m in associated_memories
                    ]

                    # Check for consistent negative emotions
                    negative_emotions = [
                        EmotionType.SADNESS,
                        EmotionType.ANGER,
                        EmotionType.FEAR,
                    ]
                    negative_count = sum(1 for e in emotions if e in negative_emotions)
                    negative_ratio = negative_count / len(emotions)

                    if negative_ratio >= 0.7:
                        dominant_emotion = max(
                            [e for e in emotions if e in negative_emotions],
                            key=emotions.count,
                        )

                        intensities = [
                            m.emotional_state.intensity for m in associated_memories
                        ]
                        avg_intensity = sum(intensities) / len(intensities)

                        trigger_id = hashlib.sha256(
                            f"{user_id}_temporal_{pattern_type}_{pattern_key}".encode()
                        ).hexdigest()[:16]

                        trigger = EmotionalTrigger(
                            trigger_id=trigger_id,
                            user_id=user_id,
                            trigger_pattern=f"{pattern_type}:{pattern_key}",
                            trigger_type="temporal",
                            emotional_responses=[dominant_emotion],
                            sensitivity_score=avg_intensity,
                            confidence=negative_ratio,
                            context_factors={
                                "pattern_type": pattern_type,
                                "pattern_key": pattern_key,
                                "negative_ratio": negative_ratio,
                            },
                            detection_count=len(associated_memories),
                        )

                        trigger.associated_memories = [
                            m.memory_id for m in associated_memories
                        ]
                        trigger.response_intensities = intensities

                        triggers.append(trigger)

        return triggers

    def _detect_contextual_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect contextual emotional triggers."""
        triggers = []

        # Group by conversation context characteristics
        context_groups = {
            "conversation_length": defaultdict(list),
            "session_type": defaultdict(list),
            "feedback_context": defaultdict(list),
        }

        for memory in memories:
            context = memory.conversation_context

            # Group by conversation length
            msg_length = len(context.message_text)
            if msg_length < 50:
                context_groups["conversation_length"]["short"].append(memory)
            elif msg_length < 200:
                context_groups["conversation_length"]["medium"].append(memory)
            else:
                context_groups["conversation_length"]["long"].append(memory)

            # Group by session type
            if context.previous_messages:
                context_groups["session_type"]["continued"].append(memory)
            else:
                context_groups["session_type"]["new"].append(memory)

            # Group by feedback context
            if context.user_feedback:
                if context.user_feedback <= 2:
                    context_groups["feedback_context"]["negative_feedback"].append(
                        memory
                    )
                elif context.user_feedback >= 4:
                    context_groups["feedback_context"]["positive_feedback"].append(
                        memory
                    )

        # Analyze contextual patterns
        for context_type, context_data in context_groups.items():
            for context_key, associated_memories in context_data.items():
                if len(associated_memories) >= 4:
                    emotions = [
                        m.emotional_state.primary_emotion for m in associated_memories
                    ]

                    # Look for consistent emotional patterns
                    emotion_counts = Counter(emotions)
                    dominant_emotion, count = emotion_counts.most_common(1)[0]
                    consistency = count / len(emotions)

                    if consistency >= 0.6:
                        intensities = [
                            m.emotional_state.intensity for m in associated_memories
                        ]
                        avg_intensity = sum(intensities) / len(intensities)

                        trigger_id = hashlib.sha256(
                            f"{user_id}_contextual_{context_type}_{context_key}".encode()
                        ).hexdigest()[:16]

                        trigger = EmotionalTrigger(
                            trigger_id=trigger_id,
                            user_id=user_id,
                            trigger_pattern=f"{context_type}:{context_key}",
                            trigger_type="contextual",
                            emotional_responses=[dominant_emotion],
                            sensitivity_score=avg_intensity,
                            confidence=consistency,
                            context_factors={
                                "context_type": context_type,
                                "context_key": context_key,
                                "consistency": consistency,
                            },
                            detection_count=len(associated_memories),
                        )

                        trigger.associated_memories = [
                            m.memory_id for m in associated_memories
                        ]
                        trigger.response_intensities = intensities

                        triggers.append(trigger)

        return triggers

    def _detect_semantic_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect semantic emotional triggers using pattern matching."""
        triggers = []

        # Check for common semantic trigger patterns
        for pattern_name, pattern_regex in self._trigger_patterns.items():
            matching_memories = []

            for memory in memories:
                content = memory.conversation_context.message_text
                if pattern_regex.search(content):
                    matching_memories.append(memory)

            if len(matching_memories) >= 3:
                emotions = [
                    m.emotional_state.primary_emotion for m in matching_memories
                ]

                # Check for negative emotional bias
                negative_emotions = [
                    EmotionType.SADNESS,
                    EmotionType.ANGER,
                    EmotionType.FEAR,
                ]
                negative_count = sum(1 for e in emotions if e in negative_emotions)
                negative_ratio = negative_count / len(emotions)

                if negative_ratio >= 0.6:
                    dominant_emotion = max(
                        [e for e in emotions if e in negative_emotions],
                        key=emotions.count,
                    )

                    intensities = [
                        m.emotional_state.intensity for m in matching_memories
                    ]
                    avg_intensity = sum(intensities) / len(intensities)

                    trigger_id = hashlib.sha256(
                        f"{user_id}_semantic_{pattern_name}".encode()
                    ).hexdigest()[:16]

                    trigger = EmotionalTrigger(
                        trigger_id=trigger_id,
                        user_id=user_id,
                        trigger_pattern=pattern_name,
                        trigger_type="semantic",
                        emotional_responses=[dominant_emotion],
                        sensitivity_score=avg_intensity,
                        confidence=negative_ratio,
                        context_factors={
                            "pattern_name": pattern_name,
                            "negative_ratio": negative_ratio,
                            "pattern_description": self.TRIGGER_TYPES.get(
                                "semantic", ""
                            ),
                        },
                        detection_count=len(matching_memories),
                    )

                    trigger.associated_memories = [
                        m.memory_id for m in matching_memories
                    ]
                    trigger.response_intensities = intensities

                    triggers.append(trigger)

        return triggers

    def _detect_personal_triggers(
        self, user_id: str, memories: List[EmotionalMemory]
    ) -> List[EmotionalTrigger]:
        """Detect personal reference triggers."""
        triggers = []

        # Look for personal references that consistently trigger emotions
        personal_patterns = {
            "self_reference": re.compile(
                r"\b(i am|i\'m|my|me|myself)\b", re.IGNORECASE
            ),
            "family_reference": re.compile(
                r"\b(family|parent|mother|father|mom|dad|sibling|brother|sister)\b",
                re.IGNORECASE,
            ),
            "past_reference": re.compile(
                r"\b(remember|past|before|used to|back then|childhood)\b", re.IGNORECASE
            ),
            "future_concern": re.compile(
                r"\b(future|tomorrow|next|will|going to|plan|hope|worry about)\b",
                re.IGNORECASE,
            ),
        }

        for pattern_name, pattern_regex in personal_patterns.items():
            matching_memories = []

            for memory in memories:
                content = memory.conversation_context.message_text
                if pattern_regex.search(content):
                    matching_memories.append(memory)

            if len(matching_memories) >= 4:
                emotions = [
                    m.emotional_state.primary_emotion for m in matching_memories
                ]
                emotion_counts = Counter(emotions)

                # Look for consistent emotional responses
                dominant_emotion, count = emotion_counts.most_common(1)[0]
                consistency = count / len(emotions)

                if consistency >= 0.5:  # Lower threshold for personal triggers
                    intensities = [
                        m.emotional_state.intensity for m in matching_memories
                    ]
                    avg_intensity = sum(intensities) / len(intensities)

                    trigger_id = hashlib.sha256(
                        f"{user_id}_personal_{pattern_name}".encode()
                    ).hexdigest()[:16]

                    trigger = EmotionalTrigger(
                        trigger_id=trigger_id,
                        user_id=user_id,
                        trigger_pattern=pattern_name,
                        trigger_type="personal",
                        emotional_responses=[dominant_emotion],
                        sensitivity_score=avg_intensity,
                        confidence=consistency,
                        context_factors={
                            "pattern_name": pattern_name,
                            "consistency": consistency,
                            "personal_reference_type": pattern_name,
                        },
                        detection_count=len(matching_memories),
                    )

                    trigger.associated_memories = [
                        m.memory_id for m in matching_memories
                    ]
                    trigger.response_intensities = intensities

                    triggers.append(trigger)

        return triggers

    def _calculate_trigger_activation(
        self, trigger: EmotionalTrigger, content: str, emotional_state: EmotionalState
    ) -> float:
        """Calculate trigger activation score for given content."""
        activation_score = 0.0

        # Check for pattern match
        if self._content_matches_trigger(content, trigger):
            activation_score += 0.6

        # Check emotional state alignment
        if emotional_state.primary_emotion in trigger.emotional_responses:
            activation_score += 0.3

        # Consider trigger sensitivity
        activation_score *= trigger.sensitivity_score

        return min(1.0, activation_score)

    def _content_matches_trigger(self, content: str, trigger: EmotionalTrigger) -> bool:
        """Check if content matches trigger pattern."""
        content_lower = content.lower()
        pattern = trigger.trigger_pattern.lower()

        if trigger.trigger_type == "keyword":
            return pattern in content_lower
        elif trigger.trigger_type == "topic":
            # Simple topic matching
            topic_words = {
                "work": ["work", "job", "career", "boss", "colleague", "office"],
                "family": ["family", "parent", "mother", "father", "sibling"],
                "relationships": ["relationship", "partner", "boyfriend", "girlfriend"],
                "health": ["health", "medical", "doctor", "hospital", "sick"],
                "finances": ["money", "financial", "debt", "budget", "expensive"],
            }

            if pattern in topic_words:
                return any(word in content_lower for word in topic_words[pattern])
        elif trigger.trigger_type == "semantic":
            if pattern in self._trigger_patterns:
                return bool(self._trigger_patterns[pattern].search(content))

        return False

    def _generate_trigger_recommendations(self, trigger: EmotionalTrigger) -> List[str]:
        """Generate recommendations for handling a trigger."""
        recommendations = []

        if trigger.sensitivity_score > 0.8:
            recommendations.append("High sensitivity - approach with extra care")

        if trigger.trigger_type == "keyword":
            recommendations.append("Avoid or reframe triggering keywords")
        elif trigger.trigger_type == "topic":
            recommendations.append(
                "Provide emotional support when discussing this topic"
            )
        elif trigger.trigger_type == "temporal":
            recommendations.append("Be aware of time-sensitive emotional patterns")
        elif trigger.trigger_type == "contextual":
            recommendations.append("Adapt interaction style based on context")

        if EmotionType.SADNESS in trigger.emotional_responses:
            recommendations.append("Offer comfort and validation")
        elif EmotionType.ANGER in trigger.emotional_responses:
            recommendations.append("Remain calm and de-escalate")
        elif EmotionType.FEAR in trigger.emotional_responses:
            recommendations.append("Provide reassurance and support")

        return recommendations

    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _get_hour_group(self, hour: int) -> str:
        """Get hour group from hour."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"


# Singleton instance
_trigger_detector_instance = None
_trigger_detector_lock = threading.Lock()


def get_emotional_trigger_detector() -> EmotionalTriggerDetector:
    """
    Get singleton emotional trigger detector instance.

    Returns:
        Shared EmotionalTriggerDetector instance
    """
    global _trigger_detector_instance

    if _trigger_detector_instance is None:
        with _trigger_detector_lock:
            if _trigger_detector_instance is None:
                _trigger_detector_instance = EmotionalTriggerDetector()

    return _trigger_detector_instance
