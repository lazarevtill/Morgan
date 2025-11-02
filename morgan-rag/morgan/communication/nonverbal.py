"""
Non-verbal cue detection module.

Detects and interprets non-verbal communication cues from text patterns,
timing, and behavioral indicators to enhance emotional understanding
and communication effectiveness.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotional.models import (
    EmotionalState, ConversationContext
)

logger = get_logger(__name__)


class NonVerbalCueType(Enum):
    """Types of non-verbal cues that can be detected."""
    PUNCTUATION_EMPHASIS = "punctuation_emphasis"
    CAPITALIZATION = "capitalization"
    REPETITION = "repetition"
    SPACING_PATTERNS = "spacing_patterns"
    TIMING_PATTERNS = "timing_patterns"
    MESSAGE_LENGTH = "message_length"
    EMOJI_USAGE = "emoji_usage"
    TYPING_PATTERNS = "typing_patterns"


class EmotionalIntensity(Enum):
    """Emotional intensity levels detected from non-verbal cues."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class NonVerbalCue:
    """A detected non-verbal communication cue."""
    cue_type: NonVerbalCueType
    intensity: EmotionalIntensity
    confidence: float
    description: str
    indicators: List[str]
    emotional_implication: str
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NonVerbalAnalysis:
    """Complete analysis of non-verbal cues in communication."""
    detected_cues: List[NonVerbalCue]
    overall_intensity: EmotionalIntensity
    emotional_state_indicators: Dict[str, float]
    communication_urgency: float
    engagement_level: float
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


class NonVerbalCueDetector:
    """
    Non-verbal cue detection system.
    
    Features:
    - Text pattern analysis for emotional indicators
    - Punctuation and capitalization pattern detection
    - Timing pattern analysis for urgency and engagement
    - Repetition and emphasis detection
    - Emoji and emoticon interpretation
    - Message structure and length analysis
    - Behavioral pattern recognition
    """
    
    def __init__(self):
        """Initialize non-verbal cue detector."""
        self.settings = get_settings()
        
        # Pattern storage for learning
        self.user_patterns: Dict[str, List[NonVerbalAnalysis]] = {}
        
        # Emoji emotion mapping
        self.emoji_emotions = {
            "😊": ("joy", 0.7), "😄": ("joy", 0.9), "😃": ("joy", 0.8),
            "😢": ("sadness", 0.8), "😭": ("sadness", 0.9), "😔": ("sadness", 0.6),
            "😠": ("anger", 0.8), "😡": ("anger", 0.9), "🤬": ("anger", 1.0),
            "😰": ("fear", 0.7), "😨": ("fear", 0.8), "😱": ("fear", 0.9),
            "😮": ("surprise", 0.7), "😲": ("surprise", 0.8), "🤯": ("surprise", 0.9),
            "🤢": ("disgust", 0.7), "🤮": ("disgust", 0.9), "😷": ("disgust", 0.6),
            "❤️": ("love", 0.9), "💕": ("love", 0.8), "🥰": ("love", 0.8),
            "😴": ("tired", 0.7), "😪": ("tired", 0.8), "🥱": ("tired", 0.6)
        }
        
        logger.info("Non-Verbal Cue Detector initialized")
    
    def detect_nonverbal_cues(
        self,
        user_id: str,
        message_text: str,
        context: ConversationContext,
        timing_data: Optional[Dict[str, Any]] = None
    ) -> NonVerbalAnalysis:
        """
        Detect non-verbal cues from message and context.
        
        Args:
            user_id: User identifier
            message_text: Text message to analyze
            context: Conversation context
            timing_data: Optional timing information
            
        Returns:
            Non-verbal analysis result
        """
        detected_cues = []
        
        # Detect punctuation emphasis
        punctuation_cues = self._detect_punctuation_emphasis(message_text)
        detected_cues.extend(punctuation_cues)
        
        # Detect capitalization patterns
        capitalization_cues = self._detect_capitalization_patterns(message_text)
        detected_cues.extend(capitalization_cues)
        
        # Detect repetition patterns
        repetition_cues = self._detect_repetition_patterns(message_text)
        detected_cues.extend(repetition_cues)
        
        # Detect spacing patterns
        spacing_cues = self._detect_spacing_patterns(message_text)
        detected_cues.extend(spacing_cues)
        
        # Detect emoji usage
        emoji_cues = self._detect_emoji_usage(message_text)
        detected_cues.extend(emoji_cues)
        
        # Detect message length patterns
        length_cues = self._detect_message_length_patterns(message_text, context)
        detected_cues.extend(length_cues)
        
        # Detect timing patterns if available
        if timing_data:
            timing_cues = self._detect_timing_patterns(timing_data, context)
            detected_cues.extend(timing_cues)
        
        # Calculate overall analysis
        overall_intensity = self._calculate_overall_intensity(detected_cues)
        emotional_indicators = self._extract_emotional_indicators(detected_cues)
        urgency = self._calculate_communication_urgency(detected_cues, timing_data)
        engagement = self._calculate_engagement_level(detected_cues, message_text)
        confidence = self._calculate_confidence_score(detected_cues)
        
        analysis = NonVerbalAnalysis(
            detected_cues=detected_cues,
            overall_intensity=overall_intensity,
            emotional_state_indicators=emotional_indicators,
            communication_urgency=urgency,
            engagement_level=engagement,
            confidence_score=confidence
        )
        
        # Store analysis for pattern learning
        self._store_analysis(user_id, analysis)
        
        logger.debug(
            f"Detected {len(detected_cues)} non-verbal cues for user {user_id}: "
            f"intensity={overall_intensity.value}, urgency={urgency:.2f}"
        )
        
        return analysis
    
    def _detect_punctuation_emphasis(self, text: str) -> List[NonVerbalCue]:
        """Detect emphasis through punctuation patterns."""
        cues = []
        
        # Multiple exclamation marks
        exclamation_matches = re.findall(r'!{2,}', text)
        if exclamation_matches:
            intensity = EmotionalIntensity.HIGH if len(max(exclamation_matches, key=len)) > 3 else EmotionalIntensity.MODERATE
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.PUNCTUATION_EMPHASIS,
                intensity=intensity,
                confidence=0.8,
                description=f"Multiple exclamation marks detected ({len(exclamation_matches)} instances)",
                indicators=exclamation_matches,
                emotional_implication="High excitement, emphasis, or urgency"
            ))
        
        # Multiple question marks
        question_matches = re.findall(r'\?{2,}', text)
        if question_matches:
            intensity = EmotionalIntensity.MODERATE if len(max(question_matches, key=len)) > 2 else EmotionalIntensity.LOW
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.PUNCTUATION_EMPHASIS,
                intensity=intensity,
                confidence=0.7,
                description=f"Multiple question marks detected ({len(question_matches)} instances)",
                indicators=question_matches,
                emotional_implication="Confusion, disbelief, or strong questioning"
            ))
        
        # Ellipsis usage
        ellipsis_matches = re.findall(r'\.{3,}', text)
        if ellipsis_matches:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.PUNCTUATION_EMPHASIS,
                intensity=EmotionalIntensity.MODERATE,
                confidence=0.6,
                description=f"Ellipsis usage detected ({len(ellipsis_matches)} instances)",
                indicators=ellipsis_matches,
                emotional_implication="Hesitation, trailing thought, or dramatic pause"
            ))
        
        return cues
    
    def _detect_capitalization_patterns(self, text: str) -> List[NonVerbalCue]:
        """Detect emphasis through capitalization patterns."""
        cues = []
        
        # All caps words
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if caps_words:
            intensity = EmotionalIntensity.HIGH if len(caps_words) > 2 else EmotionalIntensity.MODERATE
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.CAPITALIZATION,
                intensity=intensity,
                confidence=0.9,
                description=f"All caps words detected: {', '.join(caps_words)}",
                indicators=caps_words,
                emotional_implication="Shouting, strong emphasis, or high emotion"
            ))
        
        # Mixed case emphasis (e.g., "sO aNnOyInG")
        mixed_case_pattern = re.findall(r'\b[a-zA-Z]*[a-z][A-Z][a-zA-Z]*\b', text)
        if mixed_case_pattern:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.CAPITALIZATION,
                intensity=EmotionalIntensity.MODERATE,
                confidence=0.7,
                description=f"Mixed case emphasis detected: {', '.join(mixed_case_pattern)}",
                indicators=mixed_case_pattern,
                emotional_implication="Sarcasm, mockery, or playful emphasis"
            ))
        
        return cues
    
    def _detect_repetition_patterns(self, text: str) -> List[NonVerbalCue]:
        """Detect emphasis through repetition patterns."""
        cues = []
        
        # Letter repetition (e.g., "sooooo", "noooo")
        letter_repetition = re.findall(r'\b\w*([a-zA-Z])\1{2,}\w*\b', text)
        if letter_repetition:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.REPETITION,
                intensity=EmotionalIntensity.MODERATE,
                confidence=0.8,
                description=f"Letter repetition detected ({len(letter_repetition)} instances)",
                indicators=[f"Repeated '{letter}'" for letter in letter_repetition],
                emotional_implication="Emphasis, prolonged emotion, or dramatic effect"
            ))
        
        # Word repetition
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 2:  # Ignore short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 1]
        if repeated_words:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.REPETITION,
                intensity=EmotionalIntensity.LOW,
                confidence=0.6,
                description=f"Word repetition detected: {', '.join(repeated_words)}",
                indicators=repeated_words,
                emotional_implication="Emphasis, nervousness, or strong feeling"
            ))
        
        return cues
    
    def _detect_spacing_patterns(self, text: str) -> List[NonVerbalCue]:
        """Detect patterns in spacing and formatting."""
        cues = []
        
        # Excessive spacing
        if '  ' in text:  # Multiple spaces
            space_count = text.count('  ')
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.SPACING_PATTERNS,
                intensity=EmotionalIntensity.LOW,
                confidence=0.5,
                description=f"Excessive spacing detected ({space_count} instances)",
                indicators=[f"{space_count} double spaces"],
                emotional_implication="Dramatic pause, emphasis, or formatting for effect"
            ))
        
        # Line breaks in short messages
        line_breaks = text.count('\n')
        if line_breaks > 0 and len(text) < 100:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.SPACING_PATTERNS,
                intensity=EmotionalIntensity.MODERATE,
                confidence=0.6,
                description=f"Line breaks in short message ({line_breaks} breaks)",
                indicators=[f"{line_breaks} line breaks"],
                emotional_implication="Dramatic effect, emphasis, or structured thinking"
            ))
        
        return cues
    
    def _detect_emoji_usage(self, text: str) -> List[NonVerbalCue]:
        """Detect and interpret emoji usage."""
        cues = []
        
        # Find all emojis in text
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        emojis = emoji_pattern.findall(text)
        if emojis:
            # Analyze emotional content of emojis
            detected_emotions = []
            total_intensity = 0.0
            
            for emoji in emojis:
                if emoji in self.emoji_emotions:
                    emotion, intensity = self.emoji_emotions[emoji]
                    detected_emotions.append((emotion, intensity))
                    total_intensity += intensity
            
            if detected_emotions:
                avg_intensity = total_intensity / len(detected_emotions)
                intensity_level = self._intensity_to_enum(avg_intensity)
                
                cues.append(NonVerbalCue(
                    cue_type=NonVerbalCueType.EMOJI_USAGE,
                    intensity=intensity_level,
                    confidence=0.8,
                    description=f"Emoji usage detected: {', '.join(emojis)}",
                    indicators=emojis,
                    emotional_implication=f"Emotional expression: {', '.join([e[0] for e in detected_emotions])}"
                ))
        
        return cues
    
    def _detect_message_length_patterns(
        self,
        text: str,
        context: ConversationContext
    ) -> List[NonVerbalCue]:
        """Detect patterns in message length."""
        cues = []
        
        message_length = len(text)
        
        # Very short messages (potential urgency or brevity)
        if message_length < 10:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.MESSAGE_LENGTH,
                intensity=EmotionalIntensity.MODERATE,
                confidence=0.6,
                description=f"Very short message ({message_length} characters)",
                indicators=[f"Length: {message_length}"],
                emotional_implication="Urgency, brevity, or low engagement"
            ))
        
        # Very long messages (potential high engagement or stress)
        elif message_length > 500:
            cues.append(NonVerbalCue(
                cue_type=NonVerbalCueType.MESSAGE_LENGTH,
                intensity=EmotionalIntensity.HIGH,
                confidence=0.7,
                description=f"Very long message ({message_length} characters)",
                indicators=[f"Length: {message_length}"],
                emotional_implication="High engagement, detailed sharing, or emotional outpouring"
            ))
        
        return cues
    
    def _detect_timing_patterns(
        self,
        timing_data: Dict[str, Any],
        context: ConversationContext
    ) -> List[NonVerbalCue]:
        """Detect patterns in message timing."""
        cues = []
        
        # Rapid response (very quick reply)
        if "response_time" in timing_data:
            response_time = timing_data["response_time"]
            if response_time < 30:  # Less than 30 seconds
                cues.append(NonVerbalCue(
                    cue_type=NonVerbalCueType.TIMING_PATTERNS,
                    intensity=EmotionalIntensity.HIGH,
                    confidence=0.8,
                    description=f"Rapid response ({response_time}s)",
                    indicators=[f"Response time: {response_time}s"],
                    emotional_implication="High engagement, urgency, or immediate reaction"
                ))
        
        # Typing indicators
        if "typing_duration" in timing_data:
            typing_duration = timing_data["typing_duration"]
            message_length = len(context.message_text)
            
            # Long typing time for short message (hesitation)
            if typing_duration > 60 and message_length < 50:
                cues.append(NonVerbalCue(
                    cue_type=NonVerbalCueType.TYPING_PATTERNS,
                    intensity=EmotionalIntensity.MODERATE,
                    confidence=0.7,
                    description=f"Long typing time for short message ({typing_duration}s for {message_length} chars)",
                    indicators=[f"Typing: {typing_duration}s, Length: {message_length}"],
                    emotional_implication="Hesitation, careful consideration, or difficulty expressing"
                ))
        
        return cues
    
    def _calculate_overall_intensity(self, cues: List[NonVerbalCue]) -> EmotionalIntensity:
        """Calculate overall emotional intensity from detected cues."""
        if not cues:
            return EmotionalIntensity.VERY_LOW
        
        intensity_scores = {
            EmotionalIntensity.VERY_LOW: 0.1,
            EmotionalIntensity.LOW: 0.3,
            EmotionalIntensity.MODERATE: 0.5,
            EmotionalIntensity.HIGH: 0.8,
            EmotionalIntensity.VERY_HIGH: 1.0
        }
        
        total_score = sum(intensity_scores[cue.intensity] * cue.confidence for cue in cues)
        avg_score = total_score / len(cues)
        
        # Convert back to intensity enum
        if avg_score >= 0.9:
            return EmotionalIntensity.VERY_HIGH
        elif avg_score >= 0.7:
            return EmotionalIntensity.HIGH
        elif avg_score >= 0.4:
            return EmotionalIntensity.MODERATE
        elif avg_score >= 0.2:
            return EmotionalIntensity.LOW
        else:
            return EmotionalIntensity.VERY_LOW
    
    def _extract_emotional_indicators(self, cues: List[NonVerbalCue]) -> Dict[str, float]:
        """Extract emotional state indicators from cues."""
        indicators = {}
        
        for cue in cues:
            # Map cue types to emotional indicators
            if cue.cue_type == NonVerbalCueType.PUNCTUATION_EMPHASIS:
                if "exclamation" in cue.description:
                    indicators["excitement"] = indicators.get("excitement", 0.0) + cue.confidence * 0.8
                elif "question" in cue.description:
                    indicators["confusion"] = indicators.get("confusion", 0.0) + cue.confidence * 0.7
                elif "ellipsis" in cue.description:
                    indicators["hesitation"] = indicators.get("hesitation", 0.0) + cue.confidence * 0.6
            
            elif cue.cue_type == NonVerbalCueType.CAPITALIZATION:
                indicators["intensity"] = indicators.get("intensity", 0.0) + cue.confidence * 0.9
                if "mixed case" in cue.description:
                    indicators["sarcasm"] = indicators.get("sarcasm", 0.0) + cue.confidence * 0.7
            
            elif cue.cue_type == NonVerbalCueType.REPETITION:
                indicators["emphasis"] = indicators.get("emphasis", 0.0) + cue.confidence * 0.8
            
            elif cue.cue_type == NonVerbalCueType.EMOJI_USAGE:
                indicators["emotional_expression"] = indicators.get("emotional_expression", 0.0) + cue.confidence * 0.9
            
            elif cue.cue_type == NonVerbalCueType.TIMING_PATTERNS:
                if "rapid" in cue.description:
                    indicators["urgency"] = indicators.get("urgency", 0.0) + cue.confidence * 0.8
                elif "typing" in cue.description:
                    indicators["hesitation"] = indicators.get("hesitation", 0.0) + cue.confidence * 0.6
        
        # Normalize indicators to 0-1 range
        max_value = max(indicators.values()) if indicators else 1.0
        if max_value > 1.0:
            indicators = {k: v / max_value for k, v in indicators.items()}
        
        return indicators
    
    def _calculate_communication_urgency(
        self,
        cues: List[NonVerbalCue],
        timing_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate communication urgency score."""
        urgency_score = 0.0
        
        for cue in cues:
            if cue.cue_type == NonVerbalCueType.PUNCTUATION_EMPHASIS and "exclamation" in cue.description:
                urgency_score += 0.3 * cue.confidence
            elif cue.cue_type == NonVerbalCueType.CAPITALIZATION:
                urgency_score += 0.4 * cue.confidence
            elif cue.cue_type == NonVerbalCueType.TIMING_PATTERNS and "rapid" in cue.description:
                urgency_score += 0.5 * cue.confidence
            elif cue.cue_type == NonVerbalCueType.MESSAGE_LENGTH and "short" in cue.description:
                urgency_score += 0.2 * cue.confidence
        
        return min(1.0, urgency_score)
    
    def _calculate_engagement_level(self, cues: List[NonVerbalCue], text: str) -> float:
        """Calculate engagement level from cues and message content."""
        engagement_score = 0.0
        
        # Base engagement from message length
        message_length = len(text)
        if message_length > 100:
            engagement_score += 0.3
        elif message_length > 50:
            engagement_score += 0.2
        else:
            engagement_score += 0.1
        
        # Add engagement from cues
        for cue in cues:
            if cue.cue_type == NonVerbalCueType.EMOJI_USAGE:
                engagement_score += 0.2 * cue.confidence
            elif cue.cue_type == NonVerbalCueType.REPETITION:
                engagement_score += 0.1 * cue.confidence
            elif cue.cue_type == NonVerbalCueType.MESSAGE_LENGTH and "long" in cue.description:
                engagement_score += 0.3 * cue.confidence
        
        return min(1.0, engagement_score)
    
    def _calculate_confidence_score(self, cues: List[NonVerbalCue]) -> float:
        """Calculate overall confidence in the analysis."""
        if not cues:
            return 0.1
        
        # Average confidence of all cues
        avg_confidence = sum(cue.confidence for cue in cues) / len(cues)
        
        # Boost confidence if multiple cues detected
        cue_count_boost = min(0.2, len(cues) * 0.05)
        
        return min(1.0, avg_confidence + cue_count_boost)
    
    def _intensity_to_enum(self, intensity_value: float) -> EmotionalIntensity:
        """Convert intensity value to enum."""
        if intensity_value >= 0.9:
            return EmotionalIntensity.VERY_HIGH
        elif intensity_value >= 0.7:
            return EmotionalIntensity.HIGH
        elif intensity_value >= 0.4:
            return EmotionalIntensity.MODERATE
        elif intensity_value >= 0.2:
            return EmotionalIntensity.LOW
        else:
            return EmotionalIntensity.VERY_LOW
    
    def _store_analysis(self, user_id: str, analysis: NonVerbalAnalysis) -> None:
        """Store analysis for pattern learning."""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = []
        
        self.user_patterns[user_id].append(analysis)
        
        # Keep only recent analyses (last 50)
        if len(self.user_patterns[user_id]) > 50:
            self.user_patterns[user_id] = self.user_patterns[user_id][-50:]


# Singleton instance
_nonverbal_cue_detector_instance = None
_nonverbal_cue_detector_lock = threading.Lock()


def get_nonverbal_cue_detector() -> NonVerbalCueDetector:
    """
    Get singleton non-verbal cue detector instance.
    
    Returns:
        Shared NonVerbalCueDetector instance
    """
    global _nonverbal_cue_detector_instance
    
    if _nonverbal_cue_detector_instance is None:
        with _nonverbal_cue_detector_lock:
            if _nonverbal_cue_detector_instance is None:
                _nonverbal_cue_detector_instance = NonVerbalCueDetector()
    
    return _nonverbal_cue_detector_instance