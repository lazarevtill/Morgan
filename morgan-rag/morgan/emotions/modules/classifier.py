"""
Emotion Classifier Module.

Classifies text into 8 basic emotion types using multiple detection strategies:
1. Keyword-based classification
2. Pattern-based classification
3. Contextual classification
4. Negation handling
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionClassificationError
from morgan.emotions.types import Emotion, EmotionIntensity, EmotionType


class EmotionClassifier(EmotionModule):
    """
    Classifies text into emotion categories.

    Uses a multi-strategy approach:
    - Lexicon-based keyword matching
    - Pattern recognition for emotional expressions
    - Negation detection and handling
    - Intensity boosters and diminishers
    """

    def __init__(self) -> None:
        super().__init__("EmotionClassifier")
        self._emotion_lexicon: Dict[EmotionType, Set[str]] = {}
        self._boosters: Set[str] = set()
        self._diminishers: Set[str] = set()
        self._negations: Set[str] = set()
        self._patterns: Dict[EmotionType, List[re.Pattern]] = {}

    async def initialize(self) -> None:
        """Initialize emotion lexicons and patterns."""
        self._load_emotion_lexicon()
        self._load_intensity_modifiers()
        self._load_negation_words()
        self._compile_patterns()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._emotion_lexicon.clear()
        self._patterns.clear()

    async def classify(self, text: str) -> List[Emotion]:
        """
        Classify text into emotions.

        Args:
            text: Text to classify

        Returns:
            List of detected emotions with intensities

        Raises:
            EmotionClassificationError: If classification fails
        """
        await self.ensure_initialized()

        if not text or not text.strip():
            raise EmotionClassificationError("Cannot classify empty text")

        try:
            # Normalize text
            normalized = self._normalize_text(text)

            # Detect emotions using multiple strategies
            keyword_emotions = self._classify_by_keywords(normalized)
            pattern_emotions = self._classify_by_patterns(normalized)

            # Merge results
            merged = self._merge_classifications(keyword_emotions, pattern_emotions)

            # Apply intensity modifiers
            adjusted = self._apply_intensity_modifiers(merged, normalized)

            # Handle negations
            final = self._handle_negations(adjusted, normalized)

            # Filter out low-confidence detections
            significant = [e for e in final if e.intensity >= 0.1]

            return sorted(significant, key=lambda e: e.intensity, reverse=True)

        except Exception as e:
            raise EmotionClassificationError(
                f"Failed to classify emotion: {str(e)}", cause=e
            )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Convert to lowercase
        text = text.lower()
        # Remove excessive punctuation but keep some for pattern matching
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        return text

    def _classify_by_keywords(self, text: str) -> Dict[EmotionType, float]:
        """Classify using keyword matching."""
        emotion_scores: Dict[EmotionType, float] = {
            emo: 0.0 for emo in EmotionType.all_emotions()
        }

        words = set(text.split())

        for emotion_type, keywords in self._emotion_lexicon.items():
            matches = words.intersection(keywords)
            if matches:
                # Score based on number of matches and keyword strength
                emotion_scores[emotion_type] = min(
                    len(matches) * 0.3, 1.0
                )

        return emotion_scores

    def _classify_by_patterns(self, text: str) -> Dict[EmotionType, float]:
        """Classify using regex patterns."""
        emotion_scores: Dict[EmotionType, float] = {
            emo: 0.0 for emo in EmotionType.all_emotions()
        }

        for emotion_type, patterns in self._patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Pattern matches are stronger signals
                    current_score = emotion_scores[emotion_type]
                    emotion_scores[emotion_type] = min(
                        current_score + (len(matches) * 0.4), 1.0
                    )

        return emotion_scores

    def _merge_classifications(
        self,
        keyword_results: Dict[EmotionType, float],
        pattern_results: Dict[EmotionType, float],
    ) -> Dict[EmotionType, float]:
        """Merge multiple classification results."""
        merged: Dict[EmotionType, float] = {}

        for emotion_type in EmotionType.all_emotions():
            keyword_score = keyword_results.get(emotion_type, 0.0)
            pattern_score = pattern_results.get(emotion_type, 0.0)

            # Weighted average, patterns have higher weight
            merged[emotion_type] = (keyword_score * 0.4) + (pattern_score * 0.6)

        return merged

    def _apply_intensity_modifiers(
        self, emotions: Dict[EmotionType, float], text: str
    ) -> Dict[EmotionType, float]:
        """Apply intensity boosters and diminishers."""
        words = set(text.split())

        booster_count = len(words.intersection(self._boosters))
        diminisher_count = len(words.intersection(self._diminishers))

        modifier = 1.0 + (booster_count * 0.2) - (diminisher_count * 0.2)
        modifier = max(0.5, min(modifier, 1.5))  # Clamp to reasonable range

        return {
            emo_type: min(score * modifier, 1.0)
            for emo_type, score in emotions.items()
        }

    def _handle_negations(
        self, emotions: Dict[EmotionType, float], text: str
    ) -> List[Emotion]:
        """Handle negations and create final emotion list."""
        # Simple negation detection: if negation word appears,
        # reduce positive emotions and boost opposite emotions
        words = text.split()
        has_negation = any(word in self._negations for word in words)

        results: List[Emotion] = []

        for emotion_type, intensity in emotions.items():
            if intensity < 0.1:
                continue

            if has_negation:
                # Reduce intensity and boost opposite
                adjusted_intensity = intensity * 0.6
                opposite = emotion_type.opposite()
                opposite_boost = intensity * 0.3

                # Add negated emotion
                if adjusted_intensity >= 0.1:
                    results.append(
                        Emotion(
                            emotion_type=emotion_type,
                            intensity=EmotionIntensity(adjusted_intensity),
                            confidence=0.6,
                        )
                    )

                # Add boosted opposite
                if opposite_boost >= 0.1:
                    existing_opposite = next(
                        (e for e in results if e.emotion_type == opposite), None
                    )
                    if existing_opposite:
                        # Update existing
                        results.remove(existing_opposite)
                        new_intensity = min(
                            float(existing_opposite.intensity) + opposite_boost, 1.0
                        )
                        results.append(
                            Emotion(
                                emotion_type=opposite,
                                intensity=EmotionIntensity(new_intensity),
                                confidence=0.7,
                            )
                        )
                    else:
                        results.append(
                            Emotion(
                                emotion_type=opposite,
                                intensity=EmotionIntensity(opposite_boost),
                                confidence=0.6,
                            )
                        )
            else:
                results.append(
                    Emotion(
                        emotion_type=emotion_type,
                        intensity=EmotionIntensity(intensity),
                        confidence=0.8,
                    )
                )

        return results

    def _load_emotion_lexicon(self) -> None:
        """Load emotion keyword lexicon."""
        self._emotion_lexicon = {
            EmotionType.JOY: {
                "happy",
                "joy",
                "glad",
                "pleased",
                "delighted",
                "excited",
                "cheerful",
                "joyful",
                "elated",
                "ecstatic",
                "thrilled",
                "wonderful",
                "fantastic",
                "great",
                "awesome",
                "amazing",
                "love",
                "loving",
                "loved",
            },
            EmotionType.SADNESS: {
                "sad",
                "unhappy",
                "depressed",
                "miserable",
                "sorrowful",
                "gloomy",
                "melancholy",
                "dejected",
                "downcast",
                "heartbroken",
                "grief",
                "mourning",
                "crying",
                "tears",
                "lonely",
                "alone",
                "hurt",
                "pain",
                "painful",
            },
            EmotionType.ANGER: {
                "angry",
                "mad",
                "furious",
                "enraged",
                "irritated",
                "annoyed",
                "frustrated",
                "outraged",
                "livid",
                "infuriated",
                "rage",
                "hatred",
                "hate",
                "hating",
                "resent",
                "bitter",
                "hostile",
                "aggressive",
            },
            EmotionType.FEAR: {
                "afraid",
                "scared",
                "fearful",
                "frightened",
                "terrified",
                "horrified",
                "panicked",
                "anxious",
                "worried",
                "nervous",
                "uneasy",
                "apprehensive",
                "dread",
                "terror",
                "panic",
                "alarmed",
                "threatened",
            },
            EmotionType.SURPRISE: {
                "surprised",
                "shocked",
                "astonished",
                "amazed",
                "astounded",
                "startled",
                "stunned",
                "unexpected",
                "sudden",
                "wow",
                "whoa",
                "unbelievable",
                "incredible",
            },
            EmotionType.DISGUST: {
                "disgusted",
                "repulsed",
                "revolted",
                "nauseated",
                "sickened",
                "appalled",
                "gross",
                "nasty",
                "vile",
                "foul",
                "offensive",
                "repugnant",
                "loathsome",
            },
            EmotionType.TRUST: {
                "trust",
                "faith",
                "confidence",
                "reliable",
                "dependable",
                "safe",
                "secure",
                "comfortable",
                "assured",
                "certain",
                "belief",
                "believe",
                "believing",
            },
            EmotionType.ANTICIPATION: {
                "anticipate",
                "expect",
                "await",
                "hope",
                "hopeful",
                "looking forward",
                "eager",
                "excited",
                "ready",
                "prepared",
                "waiting",
                "soon",
                "upcoming",
                "future",
            },
        }

    def _load_intensity_modifiers(self) -> None:
        """Load intensity boosters and diminishers."""
        self._boosters = {
            "very",
            "extremely",
            "incredibly",
            "absolutely",
            "completely",
            "totally",
            "utterly",
            "highly",
            "so",
            "really",
            "truly",
            "deeply",
        }

        self._diminishers = {
            "slightly",
            "somewhat",
            "fairly",
            "rather",
            "quite",
            "kind of",
            "sort of",
            "a bit",
            "a little",
            "moderately",
        }

    def _load_negation_words(self) -> None:
        """Load negation words."""
        self._negations = {
            "not",
            "no",
            "never",
            "nothing",
            "nobody",
            "nowhere",
            "neither",
            "nor",
            "hardly",
            "scarcely",
            "barely",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "shouldn't",
            "can't",
            "cannot",
            "couldn't",
        }

    def _compile_patterns(self) -> None:
        """Compile regex patterns for emotion detection."""
        self._patterns = {
            EmotionType.JOY: [
                re.compile(r"(?:so|very)\s+happy"),
                re.compile(r"feel(?:s|ing)?\s+great"),
                re.compile(r"love\s+(?:this|it|that)"),
                re.compile(r"!+\s*$"),  # Exclamation marks often indicate joy
            ],
            EmotionType.SADNESS: [
                re.compile(r"feel(?:s|ing)?\s+(?:sad|down|low)"),
                re.compile(r"(?:so|very)\s+sad"),
                re.compile(r"can't\s+stop\s+crying"),
                re.compile(r"miss\s+(?:you|him|her|them)"),
            ],
            EmotionType.ANGER: [
                re.compile(r"piss(?:ed|es)\s+(?:off|me)"),
                re.compile(r"so\s+(?:angry|mad|frustrated)"),
                re.compile(r"hate\s+(?:this|it|that|you)"),
                re.compile(r"!{2,}"),  # Multiple exclamation marks can indicate anger
            ],
            EmotionType.FEAR: [
                re.compile(r"(?:so|very)\s+(?:scared|afraid|worried)"),
                re.compile(r"terrif(?:ied|ying)"),
                re.compile(r"what\s+if"),
                re.compile(r"worried\s+(?:about|that)"),
            ],
            EmotionType.SURPRISE: [
                re.compile(r"can't\s+believe"),
                re.compile(r"didn't\s+expect"),
                re.compile(r"wow!?"),
                re.compile(r"\?!+"),  # Interrobang indicates surprise
            ],
            EmotionType.DISGUST: [
                re.compile(r"so\s+(?:gross|disgusting|nasty)"),
                re.compile(r"makes?\s+me\s+sick"),
            ],
            EmotionType.TRUST: [
                re.compile(r"trust\s+(?:you|him|her|them)"),
                re.compile(r"have\s+faith"),
                re.compile(r"believe\s+in"),
            ],
            EmotionType.ANTICIPATION: [
                re.compile(r"can't\s+wait"),
                re.compile(r"looking\s+forward"),
                re.compile(r"excited\s+(?:for|about)"),
                re.compile(r"hope\s+(?:that|to)"),
            ],
        }
