"""
Nonverbal Communication Module

Handles emoji detection and analysis in text communications using secure,
non-overlapping Unicode ranges to avoid regex security issues.

This implementation addresses CodeQL security warnings about overly permissive
regex ranges by using the emoji library instead of manual Unicode ranges.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False


@dataclass
class EmojiAnalysis:
    """Results of emoji analysis in text"""

    emojis: List[str] = field(default_factory=list)
    count: int = 0
    unique_count: int = 0
    density: float = 0.0  # Emojis per character
    categories: Dict[str, int] = field(default_factory=dict)
    sentiment: Optional[str] = None  # positive, negative, neutral


class NonverbalCommunicationAnalyzer:
    """
    Analyzes nonverbal cues in text communication (emojis, punctuation, formatting).

    Uses the emoji library for secure emoji detection to avoid CodeQL warnings
    about overlapping Unicode ranges in regex patterns.
    """

    # Sentiment mappings for common emojis
    POSITIVE_EMOJIS = {
        '😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '🙃',
        '😉', '😊', '😇', '🥰', '😍', '🤩', '😘', '😗', '😚', '😙',
        '🥲', '☺️', '😋', '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭',
        '🫢', '🫣', '🤫', '🤔', '👍', '👏', '🙌', '🤝', '🎉', '🎊',
        '🌟', '✨', '💖', '💝', '💗', '💓', '💞', '💕', '❤️', '🧡',
        '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍'
    }

    NEGATIVE_EMOJIS = {
        '😢', '😭', '😤', '😠', '😡', '🤬', '😞', '😔', '😟', '😕',
        '🙁', '☹️', '😣', '😖', '😫', '😩', '🥺', '😱', '😨', '😰',
        '😥', '😓', '🤯', '😳', '🥵', '🥶', '😶‍🌫️', '😐', '😑', '😬',
        '🙄', '😒', '🤨', '💔', '👎', '😷', '🤢', '🤮'
    }

    NEUTRAL_EMOJIS = {
        '😐', '😶', '🫥', '😏', '😪', '😴', '😮', '😯', '😲', '🥱',
        '🤐', '🫡', '🤓', '🧐', '😎', '🥸', '🤠'
    }

    def __init__(self):
        """Initialize the nonverbal communication analyzer"""
        self.emoji_lib_available = HAS_EMOJI_LIB

    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text using secure method.

        Uses the emoji library if available, otherwise falls back to
        a conservative Unicode range check.

        Args:
            text: Input text to analyze

        Returns:
            List of emoji strings found in text
        """
        if not text:
            return []

        if self.emoji_lib_available:
            # Use emoji library for secure, comprehensive emoji detection
            return [char['emoji'] for char in emoji.emoji_list(text)]
        else:
            # Fallback: Conservative emoji detection using specific ranges
            # Only detects basic emojis to avoid CodeQL security warnings
            return self._extract_emojis_conservative(text)

    def _extract_emojis_conservative(self, text: str) -> List[str]:
        """
        Conservative emoji extraction using specific, non-overlapping Unicode ranges.

        This method uses a whitelist approach instead of broad Unicode ranges
        to avoid CodeQL security warnings about overlapping ranges.

        Args:
            text: Input text to analyze

        Returns:
            List of emoji characters found
        """
        emojis = []

        # Use character-by-character check against known emoji sets
        # This is slower but avoids regex security issues
        for char in text:
            if (char in self.POSITIVE_EMOJIS or
                char in self.NEGATIVE_EMOJIS or
                char in self.NEUTRAL_EMOJIS):
                emojis.append(char)

        return emojis

    def analyze(self, text: str) -> EmojiAnalysis:
        """
        Perform comprehensive emoji analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            EmojiAnalysis object with detailed results
        """
        if not text:
            return EmojiAnalysis()

        # Extract emojis securely
        emojis = self.extract_emojis(text)

        if not emojis:
            return EmojiAnalysis()

        # Calculate metrics
        count = len(emojis)
        unique_emojis = set(emojis)
        unique_count = len(unique_emojis)
        density = count / len(text) if text else 0.0

        # Analyze sentiment
        sentiment = self._analyze_sentiment(unique_emojis)

        # Categorize emojis
        categories = self._categorize_emojis(emojis)

        return EmojiAnalysis(
            emojis=emojis,
            count=count,
            unique_count=unique_count,
            density=density,
            categories=categories,
            sentiment=sentiment
        )

    def _analyze_sentiment(self, emojis: Set[str]) -> str:
        """
        Determine overall sentiment from emoji set.

        Args:
            emojis: Set of unique emojis

        Returns:
            Sentiment: "positive", "negative", or "neutral"
        """
        if not emojis:
            return "neutral"

        positive_count = len(emojis & self.POSITIVE_EMOJIS)
        negative_count = len(emojis & self.NEGATIVE_EMOJIS)
        neutral_count = len(emojis & self.NEUTRAL_EMOJIS)

        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        else:
            return "neutral"

    def _categorize_emojis(self, emojis: List[str]) -> Dict[str, int]:
        """
        Categorize emojis by type.

        Args:
            emojis: List of emoji characters

        Returns:
            Dictionary mapping categories to counts
        """
        categories = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "unknown": 0
        }

        for emoji_char in emojis:
            if emoji_char in self.POSITIVE_EMOJIS:
                categories["positive"] += 1
            elif emoji_char in self.NEGATIVE_EMOJIS:
                categories["negative"] += 1
            elif emoji_char in self.NEUTRAL_EMOJIS:
                categories["neutral"] += 1
            else:
                categories["unknown"] += 1

        return categories

    def has_emojis(self, text: str) -> bool:
        """
        Quick check if text contains any emojis.

        Args:
            text: Input text

        Returns:
            True if text contains emojis
        """
        return bool(self.extract_emojis(text))

    def emoji_density(self, text: str) -> float:
        """
        Calculate emoji density (emojis per character).

        Args:
            text: Input text

        Returns:
            Ratio of emojis to total characters
        """
        if not text:
            return 0.0

        emoji_count = len(self.extract_emojis(text))
        return emoji_count / len(text)


# Global analyzer instance
analyzer = NonverbalCommunicationAnalyzer()


def extract_emojis(text: str) -> List[str]:
    """
    Convenience function to extract emojis from text.

    Args:
        text: Input text

    Returns:
        List of emoji strings
    """
    return analyzer.extract_emojis(text)


def analyze_emojis(text: str) -> EmojiAnalysis:
    """
    Convenience function to analyze emojis in text.

    Args:
        text: Input text

    Returns:
        EmojiAnalysis object
    """
    return analyzer.analyze(text)
