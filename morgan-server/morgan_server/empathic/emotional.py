"""
Emotional Intelligence module for the Empathic Engine.

This module provides emotional tone detection, adjustment, and pattern tracking
to make Morgan feel emotionally aware and responsive.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re


class EmotionalTone(str, Enum):
    """Emotional tone categories."""
    JOYFUL = "joyful"
    EXCITED = "excited"
    CONTENT = "content"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    SAD = "sad"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    GRATEFUL = "grateful"
    HOPEFUL = "hopeful"
    CONFUSED = "confused"


@dataclass
class EmotionalDetection:
    """Result of emotional tone detection."""
    primary_tone: EmotionalTone
    confidence: float  # 0.0 to 1.0
    secondary_tones: List[Tuple[EmotionalTone, float]] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)  # Words/phrases that indicated the tone


@dataclass
class EmotionalPattern:
    """Tracked emotional pattern over time."""
    user_id: str
    timestamp: datetime
    tone: EmotionalTone
    confidence: float
    context: Optional[str] = None


@dataclass
class EmotionalAdjustment:
    """Suggested adjustment for response tone."""
    target_tone: EmotionalTone
    intensity: float  # 0.0 to 1.0
    suggestions: List[str] = field(default_factory=list)
    celebration: Optional[str] = None  # For positive moments
    support: Optional[str] = None  # For difficult times


class EmotionalIntelligence:
    """
    Emotional intelligence system for detecting and responding to emotional tones.
    
    This class provides:
    - Emotional tone detection from user messages
    - Response tone adjustment recommendations
    - Emotional pattern tracking over time
    - Support for celebrating positive moments and providing comfort
    """
    
    # Emotional indicators - words and phrases that suggest different tones
    TONE_INDICATORS = {
        EmotionalTone.JOYFUL: [
            "happy", "joy", "wonderful", "amazing", "fantastic", "great",
            "love", "excited", "thrilled", "delighted", "pleased", "glad",
            "yay", "awesome", "brilliant", "excellent", "perfect", "😊", "😄", "🎉"
        ],
        EmotionalTone.EXCITED: [
            "excited", "can't wait", "looking forward", "pumped", "eager",
            "thrilled", "stoked", "hyped", "omg", "wow", "incredible",
            "!!!", "amazing", "🎉", "🚀", "⚡"
        ],
        EmotionalTone.CONTENT: [
            "good", "fine", "okay", "alright", "satisfied", "comfortable",
            "peaceful", "calm", "relaxed", "content", "pleasant"
        ],
        EmotionalTone.CONCERNED: [
            "worried", "concerned", "nervous", "unsure", "uncertain",
            "uneasy", "troubled", "bothered", "wondering if"
        ],
        EmotionalTone.SAD: [
            "sad", "unhappy", "down", "depressed", "miserable", "upset",
            "disappointed", "heartbroken", "crying", "tears", "hurt",
            "lonely", "blue", "struggling", "😢", "😞", "💔"
        ],
        EmotionalTone.ANXIOUS: [
            "anxious", "stressed", "panic", "overwhelmed", "scared",
            "afraid", "terrified", "worried sick", "freaking out",
            "nervous breakdown", "can't handle", "too much"
        ],
        EmotionalTone.FRUSTRATED: [
            "frustrated", "annoyed", "irritated", "fed up", "sick of",
            "tired of", "can't stand", "driving me crazy", "ugh",
            "annoying", "infuriating", "😤", "😠"
        ],
        EmotionalTone.ANGRY: [
            "angry", "furious", "mad", "rage", "pissed", "livid",
            "outraged", "hate", "disgusted", "enraged", "fuming"
        ],
        EmotionalTone.GRATEFUL: [
            "thank", "thanks", "grateful", "appreciate", "thankful",
            "blessed", "fortunate", "lucky", "indebted", "🙏", "❤️"
        ],
        EmotionalTone.HOPEFUL: [
            "hope", "hopeful", "optimistic", "positive", "confident",
            "believe", "faith", "looking up", "better", "improve"
        ],
        EmotionalTone.CONFUSED: [
            "confused", "don't understand", "unclear", "puzzled",
            "bewildered", "lost", "huh", "??", "how come",
            "doesn't make sense", "makes no sense"
        ],
    }
    
    # Intensity modifiers
    INTENSITY_MODIFIERS = {
        "very": 1.3,
        "really": 1.3,
        "extremely": 1.5,
        "incredibly": 1.5,
        "so": 1.2,
        "quite": 1.1,
        "somewhat": 0.8,
        "a bit": 0.7,
        "slightly": 0.7,
        "kind of": 0.7,
        "sort of": 0.7,
    }
    
    def __init__(self, pattern_window_days: int = 30):
        """
        Initialize the emotional intelligence system.
        
        Args:
            pattern_window_days: Number of days to track emotional patterns
        """
        self.pattern_window_days = pattern_window_days
        self.patterns: Dict[str, List[EmotionalPattern]] = {}
    
    def detect_tone(self, message: str, user_id: Optional[str] = None) -> EmotionalDetection:
        """
        Detect emotional tone from a user message.
        
        Args:
            message: The user's message
            user_id: Optional user ID for context
            
        Returns:
            EmotionalDetection with primary tone, confidence, and indicators
        """
        message_lower = message.lower()
        
        # Score each tone based on indicators found
        tone_scores: Dict[EmotionalTone, float] = {}
        tone_indicators: Dict[EmotionalTone, List[str]] = {}
        
        for tone, indicators in self.TONE_INDICATORS.items():
            score = 0.0
            found_indicators = []
            
            for indicator in indicators:
                if indicator in message_lower:
                    # Base score for finding the indicator
                    indicator_score = 1.0
                    
                    # Check for intensity modifiers before the indicator
                    for modifier, multiplier in self.INTENSITY_MODIFIERS.items():
                        if f"{modifier} {indicator}" in message_lower:
                            indicator_score *= multiplier
                            break
                    
                    score += indicator_score
                    found_indicators.append(indicator)
            
            if score > 0:
                tone_scores[tone] = score
                tone_indicators[tone] = found_indicators
        
        # If no tones detected, default to neutral
        if not tone_scores:
            return EmotionalDetection(
                primary_tone=EmotionalTone.NEUTRAL,
                confidence=0.5,
                secondary_tones=[],
                indicators=[]
            )
        
        # Sort tones by score
        sorted_tones = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary tone is the highest scoring
        primary_tone, primary_score = sorted_tones[0]
        
        # Calculate confidence based on score and message length
        # Longer messages with more indicators = higher confidence
        message_words = len(message.split())
        confidence = min(1.0, (primary_score / max(1, message_words / 10)) * 0.8 + 0.2)
        
        # Secondary tones are others with significant scores
        secondary_tones = [
            (tone, score / primary_score)
            for tone, score in sorted_tones[1:]
            if score / primary_score > 0.3  # At least 30% of primary score
        ]
        
        detection = EmotionalDetection(
            primary_tone=primary_tone,
            confidence=confidence,
            secondary_tones=secondary_tones,
            indicators=tone_indicators[primary_tone]
        )
        
        # Track this pattern if user_id provided
        if user_id:
            self._track_pattern(user_id, detection)
        
        return detection
    
    def adjust_response_tone(
        self,
        detected_tone: EmotionalDetection,
        user_id: Optional[str] = None
    ) -> EmotionalAdjustment:
        """
        Suggest tone adjustments for the response based on detected user emotion.
        
        Args:
            detected_tone: The detected emotional tone from user
            user_id: Optional user ID for personalization
            
        Returns:
            EmotionalAdjustment with suggestions for response tone
        """
        tone = detected_tone.primary_tone
        confidence = detected_tone.confidence
        
        # Get emotional patterns for context
        patterns = self.get_patterns(user_id) if user_id else []
        
        # Determine response strategy based on detected tone
        if tone in [EmotionalTone.JOYFUL, EmotionalTone.EXCITED]:
            # Match their enthusiasm
            return EmotionalAdjustment(
                target_tone=tone,
                intensity=min(1.0, confidence * 1.2),
                suggestions=[
                    "Match their enthusiasm and energy",
                    "Use exclamation points appropriately",
                    "Share in their excitement",
                    "Be warm and encouraging"
                ],
                celebration=self._generate_celebration(tone, patterns)
            )
        
        elif tone == EmotionalTone.GRATEFUL:
            # Acknowledge and be humble
            return EmotionalAdjustment(
                target_tone=EmotionalTone.CONTENT,
                intensity=0.7,
                suggestions=[
                    "Acknowledge their gratitude warmly",
                    "Be humble and genuine",
                    "Express that you're happy to help",
                    "Maintain a warm, friendly tone"
                ]
            )
        
        elif tone in [EmotionalTone.SAD, EmotionalTone.ANXIOUS]:
            # Provide comfort and support
            return EmotionalAdjustment(
                target_tone=EmotionalTone.CONCERNED,
                intensity=0.8,
                suggestions=[
                    "Be gentle and compassionate",
                    "Acknowledge their feelings",
                    "Offer support without being dismissive",
                    "Use a calm, reassuring tone"
                ],
                support=self._generate_support(tone, patterns)
            )
        
        elif tone in [EmotionalTone.FRUSTRATED, EmotionalTone.ANGRY]:
            # Be understanding and helpful
            return EmotionalAdjustment(
                target_tone=EmotionalTone.CONTENT,
                intensity=0.6,
                suggestions=[
                    "Stay calm and professional",
                    "Acknowledge their frustration",
                    "Focus on solutions",
                    "Be patient and understanding"
                ]
            )
        
        elif tone == EmotionalTone.CONFUSED:
            # Be clear and helpful
            return EmotionalAdjustment(
                target_tone=EmotionalTone.CONTENT,
                intensity=0.7,
                suggestions=[
                    "Be extra clear and patient",
                    "Break down complex ideas",
                    "Use examples and analogies",
                    "Check for understanding"
                ]
            )
        
        elif tone == EmotionalTone.HOPEFUL:
            # Be encouraging and supportive
            return EmotionalAdjustment(
                target_tone=EmotionalTone.HOPEFUL,
                intensity=0.8,
                suggestions=[
                    "Be encouraging and positive",
                    "Support their optimism",
                    "Provide constructive guidance",
                    "Maintain realistic positivity"
                ]
            )
        
        else:  # NEUTRAL, CONTENT, CONCERNED
            # Match their tone with slight warmth
            return EmotionalAdjustment(
                target_tone=EmotionalTone.CONTENT,
                intensity=0.6,
                suggestions=[
                    "Maintain a friendly, helpful tone",
                    "Be clear and informative",
                    "Stay engaged and attentive"
                ]
            )
    
    def track_pattern(
        self,
        user_id: str,
        tone: EmotionalTone,
        confidence: float,
        context: Optional[str] = None
    ) -> None:
        """
        Track an emotional pattern for a user.
        
        Args:
            user_id: User identifier
            tone: Detected emotional tone
            confidence: Confidence in the detection
            context: Optional context about the interaction
        """
        pattern = EmotionalPattern(
            user_id=user_id,
            timestamp=datetime.now(),
            tone=tone,
            confidence=confidence,
            context=context
        )
        
        if user_id not in self.patterns:
            self.patterns[user_id] = []
        
        self.patterns[user_id].append(pattern)
        
        # Clean up old patterns
        self._cleanup_old_patterns(user_id)
    
    def get_patterns(
        self,
        user_id: str,
        days: Optional[int] = None
    ) -> List[EmotionalPattern]:
        """
        Get emotional patterns for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to look back (default: pattern_window_days)
            
        Returns:
            List of emotional patterns within the time window
        """
        if user_id not in self.patterns:
            return []
        
        days = days or self.pattern_window_days
        cutoff = datetime.now() - timedelta(days=days)
        
        return [
            pattern for pattern in self.patterns[user_id]
            if pattern.timestamp >= cutoff
        ]
    
    def analyze_emotional_trend(
        self,
        user_id: str,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze emotional trends for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis including:
            - dominant_tone: Most common emotional tone
            - tone_distribution: Distribution of tones
            - trend: "improving", "declining", or "stable"
            - recent_shift: Whether there's been a recent emotional shift
        """
        patterns = self.get_patterns(user_id, days)
        
        if not patterns:
            return {
                "dominant_tone": None,
                "tone_distribution": {},
                "trend": "unknown",
                "recent_shift": False
            }
        
        # Calculate tone distribution
        tone_counts: Dict[EmotionalTone, int] = {}
        for pattern in patterns:
            tone_counts[pattern.tone] = tone_counts.get(pattern.tone, 0) + 1
        
        tone_distribution = {
            tone.value: count / len(patterns)
            for tone, count in tone_counts.items()
        }
        
        # Find dominant tone
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])[0]
        
        # Analyze trend (simple heuristic based on positive vs negative tones)
        positive_tones = {
            EmotionalTone.JOYFUL, EmotionalTone.EXCITED,
            EmotionalTone.CONTENT, EmotionalTone.GRATEFUL,
            EmotionalTone.HOPEFUL
        }
        
        negative_tones = {
            EmotionalTone.SAD, EmotionalTone.ANXIOUS,
            EmotionalTone.FRUSTRATED, EmotionalTone.ANGRY
        }
        
        # Split patterns into first and second half
        mid_point = len(patterns) // 2
        first_half = patterns[:mid_point]
        second_half = patterns[mid_point:]
        
        def positive_ratio(pattern_list):
            if not pattern_list:
                return 0.5
            positive = sum(1 for p in pattern_list if p.tone in positive_tones)
            return positive / len(pattern_list)
        
        first_ratio = positive_ratio(first_half)
        second_ratio = positive_ratio(second_half)
        
        if second_ratio > first_ratio + 0.2:
            trend = "improving"
        elif second_ratio < first_ratio - 0.2:
            trend = "declining"
        else:
            trend = "stable"
        
        # Check for recent shift (last 3 patterns different from overall)
        recent_shift = False
        if len(patterns) >= 5:
            recent_patterns = patterns[-3:]
            recent_tone = max(
                set(p.tone for p in recent_patterns),
                key=lambda t: sum(1 for p in recent_patterns if p.tone == t)
            )
            recent_shift = recent_tone != dominant_tone
        
        return {
            "dominant_tone": dominant_tone.value,
            "tone_distribution": tone_distribution,
            "trend": trend,
            "recent_shift": recent_shift
        }
    
    def _track_pattern(self, user_id: str, detection: EmotionalDetection) -> None:
        """Internal method to track a pattern from detection."""
        self.track_pattern(
            user_id=user_id,
            tone=detection.primary_tone,
            confidence=detection.confidence,
            context=f"Indicators: {', '.join(detection.indicators[:3])}"
        )
    
    def _cleanup_old_patterns(self, user_id: str) -> None:
        """Remove patterns older than the tracking window."""
        if user_id not in self.patterns:
            return
        
        cutoff = datetime.now() - timedelta(days=self.pattern_window_days)
        self.patterns[user_id] = [
            pattern for pattern in self.patterns[user_id]
            if pattern.timestamp >= cutoff
        ]
    
    def _generate_celebration(
        self,
        tone: EmotionalTone,
        patterns: List[EmotionalPattern]
    ) -> Optional[str]:
        """Generate a celebration message for positive moments."""
        # Check if this is a shift from negative to positive
        if len(patterns) >= 3:
            recent_tones = [p.tone for p in patterns[-3:]]
            negative_tones = {
                EmotionalTone.SAD, EmotionalTone.ANXIOUS,
                EmotionalTone.FRUSTRATED, EmotionalTone.ANGRY
            }
            
            if any(t in negative_tones for t in recent_tones[:-1]):
                return "I'm so glad to hear things are looking up! 🎉"
        
        # General celebration for very positive tones
        if tone == EmotionalTone.JOYFUL:
            return "That's wonderful! I'm really happy for you! 😊"
        elif tone == EmotionalTone.EXCITED:
            return "Your excitement is contagious! This is great! 🚀"
        
        return None
    
    def _generate_support(
        self,
        tone: EmotionalTone,
        patterns: List[EmotionalPattern]
    ) -> Optional[str]:
        """Generate a support message for difficult times."""
        # Check if this is an ongoing difficult period
        if len(patterns) >= 3:
            recent_tones = [p.tone for p in patterns[-3:]]
            negative_tones = {
                EmotionalTone.SAD, EmotionalTone.ANXIOUS,
                EmotionalTone.FRUSTRATED, EmotionalTone.ANGRY
            }
            
            if all(t in negative_tones for t in recent_tones):
                return "I notice you've been going through a tough time. I'm here for you."
        
        # General support messages
        if tone == EmotionalTone.SAD:
            return "I'm here to listen and support you through this."
        elif tone == EmotionalTone.ANXIOUS:
            return "It's okay to feel anxious. Let's work through this together."
        
        return None
