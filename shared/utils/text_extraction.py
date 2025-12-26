"""
Unified text extraction utilities.

Consolidates duplicate extraction implementations from:
- memory_processor.py (entity detection, concept extraction)
- multi_stage_search.py (keyword extraction)
"""
import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class Entity:
    """Extracted entity."""

    text: str
    entity_type: str
    confidence: float = 1.0
    start_pos: int = -1
    end_pos: int = -1


# Common patterns for entity extraction
ENTITY_PATTERNS: Dict[str, str] = {
    "person": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
    "technology": r"\b(?:Python|JavaScript|TypeScript|Docker|Kubernetes|AWS|Azure|GCP|React|Vue|Angular|Node\.js|SQL|NoSQL|API|REST|GraphQL|Redis|MongoDB|PostgreSQL)\b",
    "organization": r"\b(?:Google|Microsoft|Amazon|Apple|Facebook|Meta|OpenAI|Anthropic|Netflix|Tesla)\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "url": r"https?://[^\s<>\"{}|\\^`\[\]]+",
    "date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
    "time": r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
}


# Stop words for keyword extraction
STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "although", "though",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "about", "also", "any", "back", "even", "first", "get", "go",
    "going", "got", "know", "last", "like", "look", "make", "much",
    "new", "now", "out", "over", "say", "see", "take", "think",
    "time", "up", "use", "want", "way", "well", "work", "year",
}


# Topic categories and their keywords
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "technology": ["tech", "computer", "software", "code", "programming", "api", "developer", "system", "data", "algorithm"],
    "health": ["health", "fitness", "exercise", "medical", "wellness", "doctor", "hospital", "medicine", "disease"],
    "finance": ["money", "investment", "budget", "financial", "stock", "bank", "credit", "payment", "trading"],
    "education": ["learn", "study", "course", "training", "education", "school", "university", "student", "teacher"],
    "travel": ["travel", "trip", "vacation", "destination", "flight", "hotel", "tourism", "journey"],
    "entertainment": ["movie", "music", "game", "entertainment", "show", "concert", "book", "art"],
    "food": ["food", "recipe", "cooking", "restaurant", "meal", "cuisine", "ingredient", "dish"],
    "sports": ["sport", "team", "game", "player", "match", "championship", "athlete", "training"],
    "science": ["science", "research", "experiment", "theory", "discovery", "physics", "chemistry", "biology"],
    "business": ["business", "company", "startup", "market", "customer", "product", "service", "strategy"],
}


def extract_entities(text: str, entity_types: List[str] = None) -> List[Entity]:
    """
    Extract named entities from text.

    Args:
        text: Input text
        entity_types: List of entity types to extract, or None for all

    Returns:
        List of extracted entities
    """
    entities = []
    patterns_to_use = ENTITY_PATTERNS

    if entity_types:
        patterns_to_use = {k: v for k, v in ENTITY_PATTERNS.items() if k in entity_types}

    for entity_type, pattern in patterns_to_use.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(),
                entity_type=entity_type,
                start_pos=match.start(),
                end_pos=match.end()
            ))

    return entities


def extract_keywords(
    text: str,
    max_keywords: int = 20,
    min_word_length: int = 3,
    include_bigrams: bool = False
) -> List[str]:
    """
    Extract meaningful keywords from text.

    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        min_word_length: Minimum word length
        include_bigrams: Whether to include two-word phrases

    Returns:
        List of keywords sorted by frequency
    """
    # Tokenize
    words = re.findall(r'\b[a-zA-Z]{' + str(min_word_length) + r',}\b', text.lower())

    # Remove stop words
    keywords = [w for w in words if w not in STOP_WORDS]

    # Count frequency
    freq: Dict[str, int] = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1

    # Optionally add bigrams
    if include_bigrams and len(keywords) > 1:
        for i in range(len(keywords) - 1):
            bigram = f"{keywords[i]} {keywords[i + 1]}"
            freq[bigram] = freq.get(bigram, 0) + 1

    # Sort by frequency
    sorted_keywords = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)

    return sorted_keywords[:max_keywords]


def extract_topics(text: str, min_matches: int = 1) -> List[str]:
    """
    Extract topic categories from text.

    Args:
        text: Input text
        min_matches: Minimum keyword matches to include topic

    Returns:
        List of detected topic categories
    """
    text_lower = text.lower()
    detected = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= min_matches:
            detected.append(topic)

    return detected


def extract_concepts(text: str) -> Dict[str, List[str]]:
    """
    Extract concepts organized by category.

    Args:
        text: Input text

    Returns:
        Dictionary of concepts by category
    """
    concepts = {
        "entities": [],
        "keywords": [],
        "topics": [],
    }

    # Extract entities
    entities = extract_entities(text)
    concepts["entities"] = list(set(e.text for e in entities))

    # Extract keywords
    concepts["keywords"] = extract_keywords(text, max_keywords=10)

    # Extract topics
    concepts["topics"] = extract_topics(text)

    return concepts


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", '', text)
    # Strip
    text = text.strip()

    return text


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard index.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Normalize and tokenize
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    # Remove stop words
    words1 = words1 - STOP_WORDS
    words2 = words2 - STOP_WORDS

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0
