"""
Communication Module

Handles various aspects of communication analysis including
nonverbal cues (emojis, formatting) and conversational patterns.
"""

from morgan.communication.nonverbal import (
    NonverbalCommunicationAnalyzer,
    EmojiAnalysis,
    analyze_emojis,
    extract_emojis,
    analyzer,
)

__all__ = [
    "NonverbalCommunicationAnalyzer",
    "EmojiAnalysis",
    "analyze_emojis",
    "extract_emojis",
    "analyzer",
]
