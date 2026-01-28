# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Centralized Intelligence Constants for Morgan AI Assistant.

Single source of truth for emotion patterns, intensity modifiers,
and other intelligence-related constants.

SEMANTIC-FIRST ARCHITECTURE NOTE:
=================================
As of the semantic refactor, these patterns are used for VALIDATION only,
not as the primary detection method. The primary detection is now LLM-based
semantic analysis that understands meaning, context, and hidden emotions.

These patterns serve to:
1. VALIDATE semantic analysis results
2. BOOST CONFIDENCE when patterns agree with semantic results
3. Provide FALLBACK detection if semantic analysis fails
4. Catch EXPLICIT emotional language that helps verify semantic understanding

The patterns do NOT drive primary detection - they support it.

Usage:
    from morgan.intelligence.constants import (
        EMOTION_PATTERNS,  # For validation, not primary detection
        INTENSITY_MODIFIERS,
        NEGATION_PATTERNS,
        FORMALITY_INDICATORS,
        EMOTION_VALENCE,
    )
"""

# Import EmotionType from core.models (now safe due to lazy loading in core/__init__.py)
from .core.models import EmotionType


# =============================================================================
# Emotion Validation Patterns
# =============================================================================
# NOTE: These are used for VALIDATION of semantic analysis, not primary detection.
# The semantic-first approach uses LLM analysis as the primary method.

EMOTION_PATTERNS = {
    EmotionType.JOY: [
        r"\b(happy|joy|excited|thrilled|delighted|pleased|glad|cheerful|elated|wonderful)\b",
        r"\b(awesome|amazing|fantastic|great|excellent|perfect|brilliant|outstanding)\b",
        r"\b(love|adore|enjoy|celebrate|rejoice)\b",
        r"[!]{2,}",  # Multiple exclamation marks
        r":\)|:D|😊|😄|😃|🎉|❤️|🥳|✨",  # Joy emoticons and emojis
    ],
    EmotionType.SADNESS: [
        r"\b(sad|depressed|down|upset|disappointed|heartbroken|miserable|devastated)\b",
        r"\b(cry|crying|tears|weep|sob|mourn)\b",
        r"\b(lonely|alone|isolated|empty|hopeless)\b",
        r"\b(loss|grief|sorrow|despair)\b",
        r":\(|😢|😭|💔|😞|😔",  # Sad emoticons
    ],
    EmotionType.ANGER: [
        r"\b(angry|mad|furious|irritated|annoyed|frustrated|pissed|enraged)\b",
        r"\b(hate|despise|can\'t stand|loathe|detest)\b",
        r"\b(stupid|idiotic|ridiculous|absurd|outrageous)\b",
        r"\b(damn|hell|shit|fuck)\b",  # Strong language indicators
        r"[!]{3,}",  # Many exclamation marks (anger indicator)
        r"😠|😡|🤬|👿|💢",  # Angry emojis
    ],
    EmotionType.FEAR: [
        r"\b(scared|afraid|terrified|worried|anxious|nervous|panic|frightened)\b",
        r"\b(fear|phobia|dread|terror|horror)\b",
        r"\b(what if|concerned about|worried that|afraid that)\b",
        r"\b(stress|stressed|overwhelmed|helpless)\b",
        r"😰|😨|😱|😟|😧",  # Fear emojis
    ],
    EmotionType.SURPRISE: [
        r"\b(surprised|shocked|amazed|astonished|wow|whoa|incredible|unbelievable)\b",
        r"\b(unexpected|sudden|didn\'t expect|never thought)\b",
        r"\b(blown away|mind blown|can\'t believe)\b",
        r"😲|😮|🤯|😯|🙀",  # Surprise emojis
    ],
    EmotionType.DISGUST: [
        r"\b(disgusting|gross|revolting|sick|nauseating|repulsive|vile)\b",
        r"\b(ugh|eww|yuck|blech|nasty)\b",
        r"\b(horrible|awful|terrible|dreadful)\b",
        r"🤢|🤮|😷|🤧|😖",  # Disgust emojis
    ],
}


# =============================================================================
# Intensity Modifiers
# =============================================================================

INTENSITY_MODIFIERS = {
    # High intensity (amplifiers)
    "extremely": 1.5,
    "incredibly": 1.5,
    "absolutely": 1.4,
    "very": 1.3,
    "really": 1.2,
    "so": 1.2,
    "truly": 1.2,
    "quite": 1.1,
    "pretty": 1.1,
    # Medium intensity
    "fairly": 1.0,
    "moderately": 1.0,
    # Low intensity (diminishers)
    "somewhat": 0.8,
    "a little": 0.7,
    "a bit": 0.7,
    "slightly": 0.6,
    "barely": 0.4,
    "not very": 0.4,
    "hardly": 0.3,
}


# =============================================================================
# Negation Patterns
# =============================================================================

NEGATION_PATTERNS = [
    r"\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b",
    r"\b(don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t)\b",
    r"\b(isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t)\b",
    r"\b(without|lack|lacking|absent)\b",
]


# =============================================================================
# Emotion Valence (positive/negative polarity)
# =============================================================================

EMOTION_VALENCE = {
    EmotionType.JOY: 1.0,       # Highly positive
    EmotionType.SURPRISE: 0.3,  # Slightly positive (can be either)
    EmotionType.NEUTRAL: 0.0,   # Neutral
    EmotionType.FEAR: -0.6,     # Negative
    EmotionType.SADNESS: -0.8,  # Strongly negative
    EmotionType.DISGUST: -0.7,  # Negative
    EmotionType.ANGER: -0.9,    # Strongly negative
}


# =============================================================================
# Formality Indicators
# =============================================================================

FORMALITY_INDICATORS = {
    "formal": [
        r"\b(please|kindly|would you|could you|I would like|I appreciate)\b",
        r"\b(therefore|however|furthermore|nevertheless|consequently)\b",
        r"\b(regarding|concerning|with respect to|in reference to)\b",
        r"\b(sincerely|respectfully|best regards|kind regards)\b",
        r"\b(dear|sir|madam|mr\.|ms\.|dr\.)\b",
    ],
    "informal": [
        r"\b(hey|hi|yo|sup|wassup|hiya)\b",
        r"\b(gonna|wanna|gotta|kinda|sorta|yeah|yep|nope)\b",
        r"\b(lol|lmao|omg|btw|idk|tbh|rn|ngl)\b",
        r"\b(dude|bro|man|buddy|mate)\b",
        r"[!]{2,}|\.{3,}",  # Multiple punctuation
    ],
}


# =============================================================================
# Emotion Colors (for UI/visualization)
# =============================================================================

EMOTION_COLORS = {
    EmotionType.JOY: "#FFD700",      # Gold
    EmotionType.SADNESS: "#4169E1",  # Royal Blue
    EmotionType.ANGER: "#DC143C",    # Crimson
    EmotionType.FEAR: "#800080",     # Purple
    EmotionType.SURPRISE: "#FFA500", # Orange
    EmotionType.DISGUST: "#228B22",  # Forest Green
    EmotionType.NEUTRAL: "#808080",  # Gray
}


# =============================================================================
# Emotion Transitions (common transitions between emotions)
# =============================================================================

EMOTION_TRANSITIONS = {
    EmotionType.JOY: [EmotionType.NEUTRAL, EmotionType.SURPRISE],
    EmotionType.SADNESS: [EmotionType.NEUTRAL, EmotionType.ANGER, EmotionType.FEAR],
    EmotionType.ANGER: [EmotionType.SADNESS, EmotionType.DISGUST, EmotionType.NEUTRAL],
    EmotionType.FEAR: [EmotionType.SADNESS, EmotionType.NEUTRAL, EmotionType.ANGER],
    EmotionType.SURPRISE: [EmotionType.JOY, EmotionType.FEAR, EmotionType.NEUTRAL],
    EmotionType.DISGUST: [EmotionType.ANGER, EmotionType.SADNESS, EmotionType.NEUTRAL],
    EmotionType.NEUTRAL: [EmotionType.JOY, EmotionType.SADNESS, EmotionType.ANGER],
}


# =============================================================================
# Default Emotion Thresholds
# =============================================================================

EMOTION_THRESHOLDS = {
    "detection_confidence": 0.6,    # Minimum confidence for emotion detection
    "dominant_threshold": 0.4,      # Threshold for dominant emotion
    "secondary_threshold": 0.2,     # Threshold for secondary emotions
    "intensity_high": 0.8,          # High intensity threshold
    "intensity_low": 0.3,           # Low intensity threshold
}


# =============================================================================
# Semantic-First Architecture Aliases
# =============================================================================
# These aliases clarify the purpose of patterns in the semantic-first architecture.
# Use these names when the validation purpose is important to emphasize.

# Alias for EMOTION_PATTERNS - emphasizes validation role
VALIDATION_PATTERNS = EMOTION_PATTERNS

# Alias for FORMALITY_INDICATORS - used for style validation
STYLE_VALIDATION_PATTERNS = FORMALITY_INDICATORS
