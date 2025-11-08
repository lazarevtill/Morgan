"""
Emotion detection utilities.

Helper functions for common emotion detection tasks.
"""

from __future__ import annotations

from typing import List

from morgan.emotions.types import Emotion, EmotionResult, EmotionType


def format_emotion_summary(result: EmotionResult) -> str:
    """
    Format emotion result as human-readable summary.

    Args:
        result: Emotion detection result

    Returns:
        Formatted summary string
    """
    if not result.primary_emotions:
        return "No significant emotions detected"

    summary_parts = []

    # Dominant emotion
    if result.dominant_emotion:
        dom = result.dominant_emotion
        summary_parts.append(
            f"Dominant: {dom.emotion_type.value} "
            f"({dom.intensity.level}, {dom.confidence:.0%} confidence)"
        )

    # Other emotions
    other_emotions = [
        e
        for e in result.primary_emotions
        if e != result.dominant_emotion and e.is_significant
    ]

    if other_emotions:
        others = ", ".join(
            f"{e.emotion_type.value} ({e.intensity.level})"
            for e in other_emotions[:3]
        )
        summary_parts.append(f"Also: {others}")

    # Valence and arousal
    valence_desc = (
        "positive" if result.valence > 0.3
        else "negative" if result.valence < -0.3
        else "neutral"
    )
    arousal_desc = (
        "high energy" if result.arousal > 0.6
        else "low energy" if result.arousal < 0.4
        else "moderate energy"
    )

    summary_parts.append(f"Tone: {valence_desc}, {arousal_desc}")

    # Warnings
    if result.warnings:
        summary_parts.append(f"Warnings: {len(result.warnings)}")

    return " | ".join(summary_parts)


def get_console_color(emotion_type: EmotionType) -> str:
    """
    Get ANSI color code for emotion type.

    Args:
        emotion_type: Emotion type

    Returns:
        ANSI color code string
    """
    color_map = {
        EmotionType.JOY: "\033[92m",  # Green
        EmotionType.SADNESS: "\033[94m",  # Blue
        EmotionType.ANGER: "\033[91m",  # Red
        EmotionType.FEAR: "\033[95m",  # Magenta
        EmotionType.SURPRISE: "\033[93m",  # Yellow
        EmotionType.DISGUST: "\033[90m",  # Gray
        EmotionType.TRUST: "\033[96m",  # Cyan
        EmotionType.ANTICIPATION: "\033[97m",  # White
    }

    return color_map.get(emotion_type, "\033[0m")


def format_colored_output(result: EmotionResult) -> str:
    """
    Format emotion result with ANSI colors for terminal display.

    Args:
        result: Emotion detection result

    Returns:
        Colored string for terminal output
    """
    if not result.dominant_emotion:
        return "No emotions detected"

    color = get_console_color(result.dominant_emotion.emotion_type)
    reset = "\033[0m"

    summary = result.emotional_summary

    return f"{color}{summary}{reset}"


def filter_emotions_by_type(
    emotions: List[Emotion],
    emotion_types: List[EmotionType],
) -> List[Emotion]:
    """
    Filter emotions by type.

    Args:
        emotions: List of emotions
        emotion_types: Types to filter for

    Returns:
        Filtered list of emotions
    """
    return [e for e in emotions if e.emotion_type in emotion_types]


def get_strongest_emotion(emotions: List[Emotion]) -> Emotion | None:
    """
    Get strongest emotion by intensity.

    Args:
        emotions: List of emotions

    Returns:
        Strongest emotion or None
    """
    if not emotions:
        return None

    return max(emotions, key=lambda e: float(e.intensity))


def emotions_to_dict(emotions: List[Emotion]) -> dict[str, float]:
    """
    Convert emotions list to dictionary.

    Args:
        emotions: List of emotions

    Returns:
        Dictionary mapping emotion type names to intensities
    """
    return {e.emotion_type.value: float(e.intensity) for e in emotions}


def is_crisis_state(result: EmotionResult) -> bool:
    """
    Check if result indicates a crisis state.

    Args:
        result: Emotion detection result

    Returns:
        True if crisis detected
    """
    return result.is_crisis


def get_emotion_emoji(emotion_type: EmotionType) -> str:
    """
    Get emoji representation of emotion.

    Args:
        emotion_type: Emotion type

    Returns:
        Emoji string
    """
    emoji_map = {
        EmotionType.JOY: "😊",
        EmotionType.SADNESS: "😢",
        EmotionType.ANGER: "😠",
        EmotionType.FEAR: "😨",
        EmotionType.SURPRISE: "😲",
        EmotionType.DISGUST: "🤢",
        EmotionType.TRUST: "🤝",
        EmotionType.ANTICIPATION: "🤔",
    }

    return emoji_map.get(emotion_type, "😐")
