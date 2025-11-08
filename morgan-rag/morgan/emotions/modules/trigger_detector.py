"""
Trigger Detector Module.

Identifies specific words, phrases, or patterns that trigger emotional responses.
Useful for understanding what causes emotional reactions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionAnalysisError
from morgan.emotions.types import Emotion, EmotionTrigger, EmotionType


class TriggerDetector(EmotionModule):
    """
    Detects emotional triggers in text.

    Identifies specific textual elements that likely triggered
    the detected emotional response.
    """

    def __init__(self) -> None:
        super().__init__("TriggerDetector")
        self._trigger_keywords: Dict[EmotionType, Set[str]] = {}
        self._trigger_patterns: Dict[EmotionType, List[re.Pattern]] = {}

    async def initialize(self) -> None:
        """Initialize trigger detector."""
        self._load_trigger_keywords()
        self._compile_trigger_patterns()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._trigger_keywords.clear()
        self._trigger_patterns.clear()

    async def detect_triggers(
        self,
        text: str,
        detected_emotions: List[Emotion],
    ) -> List[EmotionTrigger]:
        """
        Detect emotional triggers in text.

        Args:
            text: Text to analyze
            detected_emotions: Emotions that were detected

        Returns:
            List of detected triggers

        Raises:
            EmotionAnalysisError: If trigger detection fails
        """
        await self.ensure_initialized()

        try:
            triggers: List[EmotionTrigger] = []
            text_lower = text.lower()

            # Only look for triggers for detected emotions
            for emotion in detected_emotions:
                if emotion.intensity < 0.3:
                    continue  # Skip weak emotions

                emotion_type = emotion.emotion_type

                # Detect keyword triggers
                keyword_triggers = self._detect_keyword_triggers(
                    text_lower, emotion_type
                )
                triggers.extend(keyword_triggers)

                # Detect pattern triggers
                pattern_triggers = self._detect_pattern_triggers(
                    text_lower, emotion_type
                )
                triggers.extend(pattern_triggers)

            # Remove duplicates and sort by confidence
            unique_triggers = self._deduplicate_triggers(triggers)

            return sorted(unique_triggers, key=lambda t: t.confidence, reverse=True)

        except Exception as e:
            raise EmotionAnalysisError(
                f"Failed to detect triggers: {str(e)}", cause=e
            )

    def _detect_keyword_triggers(
        self, text: str, emotion_type: EmotionType
    ) -> List[EmotionTrigger]:
        """Detect keyword-based triggers."""
        triggers: List[EmotionTrigger] = []

        keywords = self._trigger_keywords.get(emotion_type, set())

        for keyword in keywords:
            if keyword in text:
                position = text.find(keyword)
                triggers.append(
                    EmotionTrigger(
                        trigger_text=keyword,
                        trigger_type="keyword",
                        related_emotions=[emotion_type],
                        confidence=0.7,
                        position=position,
                    )
                )

        return triggers

    def _detect_pattern_triggers(
        self, text: str, emotion_type: EmotionType
    ) -> List[EmotionTrigger]:
        """Detect pattern-based triggers."""
        triggers: List[EmotionTrigger] = []

        patterns = self._trigger_patterns.get(emotion_type, [])

        for pattern in patterns:
            matches = list(pattern.finditer(text))
            for match in matches:
                triggers.append(
                    EmotionTrigger(
                        trigger_text=match.group(0),
                        trigger_type="pattern",
                        related_emotions=[emotion_type],
                        confidence=0.8,
                        position=match.start(),
                    )
                )

        return triggers

    def _deduplicate_triggers(
        self, triggers: List[EmotionTrigger]
    ) -> List[EmotionTrigger]:
        """Remove duplicate triggers."""
        seen: Set[tuple[str, int]] = set()
        unique: List[EmotionTrigger] = []

        for trigger in triggers:
            key = (trigger.trigger_text, trigger.position)
            if key not in seen:
                seen.add(key)
                unique.append(trigger)

        return unique

    def _load_trigger_keywords(self) -> None:
        """Load trigger keywords for each emotion."""
        self._trigger_keywords = {
            EmotionType.JOY: {
                "birthday",
                "wedding",
                "promotion",
                "success",
                "achievement",
                "victory",
                "celebration",
                "congratulations",
                "won",
                "passed",
                "graduated",
            },
            EmotionType.SADNESS: {
                "death",
                "died",
                "loss",
                "lost",
                "goodbye",
                "farewell",
                "breakup",
                "broke up",
                "divorced",
                "fired",
                "rejected",
                "failed",
                "failure",
                "depression",
            },
            EmotionType.ANGER: {
                "betrayed",
                "lied",
                "cheated",
                "unfair",
                "injustice",
                "disrespect",
                "insulted",
                "abuse",
                "abused",
                "violated",
                "scammed",
                "stolen",
            },
            EmotionType.FEAR: {
                "threat",
                "danger",
                "emergency",
                "crisis",
                "attack",
                "violence",
                "death",
                "dying",
                "illness",
                "disease",
                "accident",
                "disaster",
                "terrorism",
            },
            EmotionType.SURPRISE: {
                "unexpected",
                "suddenly",
                "shocking",
                "unbelievable",
                "never thought",
                "didn't expect",
                "out of nowhere",
            },
            EmotionType.DISGUST: {
                "corruption",
                "filth",
                "contamination",
                "infection",
                "rotten",
                "decay",
                "vomit",
                "feces",
                "bodily fluids",
            },
            EmotionType.TRUST: {
                "honest",
                "integrity",
                "loyal",
                "faithful",
                "dependable",
                "consistent",
                "transparent",
            },
            EmotionType.ANTICIPATION: {
                "tomorrow",
                "next week",
                "next month",
                "soon",
                "upcoming",
                "planning",
                "will be",
                "going to",
            },
        }

    def _compile_trigger_patterns(self) -> None:
        """Compile regex patterns for trigger detection."""
        self._trigger_patterns = {
            EmotionType.JOY: [
                re.compile(r"just (?:got|received|won|achieved)"),
                re.compile(r"so proud of"),
                re.compile(r"dreams? (?:came|come) true"),
                re.compile(r"best (?:day|news|moment)"),
            ],
            EmotionType.SADNESS: [
                re.compile(r"(?:just|recently) (?:lost|died|passed away)"),
                re.compile(r"can'?t (?:stop|help) (?:crying|tears)"),
                re.compile(r"(?:heart)?broken"),
                re.compile(r"no longer (?:here|with us)"),
            ],
            EmotionType.ANGER: [
                re.compile(r"how dare (?:you|they|he|she)"),
                re.compile(r"had enough of"),
                re.compile(r"sick (?:of|and tired)"),
                re.compile(r"(?:lied|cheated|betrayed) (?:to|on)"),
            ],
            EmotionType.FEAR: [
                re.compile(r"what if"),
                re.compile(r"scared (?:that|of|about)"),
                re.compile(r"(?:afraid|worried|anxious) (?:that|about)"),
                re.compile(r"in danger"),
            ],
            EmotionType.SURPRISE: [
                re.compile(r"never (?:thought|expected|imagined)"),
                re.compile(r"can'?t believe"),
                re.compile(r"out of (?:the blue|nowhere)"),
            ],
            EmotionType.DISGUST: [
                re.compile(r"makes? me (?:sick|gag|puke)"),
                re.compile(r"can'?t (?:stand|stomach)"),
            ],
            EmotionType.TRUST: [
                re.compile(r"(?:can|will) (?:trust|count on|rely on)"),
                re.compile(r"(?:believe|have faith) in"),
            ],
            EmotionType.ANTICIPATION: [
                re.compile(r"can'?t wait (?:to|for|until)"),
                re.compile(r"looking forward to"),
                re.compile(r"excited (?:about|for)"),
            ],
        }
