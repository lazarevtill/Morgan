"""
Specialized Vocabulary Learning for Morgan RAG.

Learns and tracks domain-specific vocabulary, terminology, and jargon
to improve understanding and communication within specialized fields.
"""

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import get_settings
from ..emotional.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TermType(Enum):
    """Types of vocabulary terms."""

    TECHNICAL = "technical"
    JARGON = "jargon"
    ACRONYM = "acronym"
    CONCEPT = "concept"
    TOOL = "tool"
    METHODOLOGY = "methodology"


class LearningSource(Enum):
    """Sources of vocabulary learning."""

    USER_USAGE = "user_usage"
    CONTEXT_INFERENCE = "context_inference"
    EXPLICIT_DEFINITION = "explicit_definition"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class VocabularyTerm:
    """Represents a learned vocabulary term."""

    term: str
    domain: str
    term_type: TermType
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)

    # Usage tracking
    usage_count: int = 0
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    contexts: List[str] = field(default_factory=list)

    # Learning metadata
    confidence_score: float = 0.0  # 0.0 to 1.0
    learning_source: LearningSource = LearningSource.USER_USAGE
    user_defined: bool = False

    def update_usage(self, context: str):
        """Update usage statistics for the term."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

        # Keep only recent contexts (last 10)
        self.contexts.append(context)
        if len(self.contexts) > 10:
            self.contexts = self.contexts[-10:]

        # Increase confidence with usage
        self.confidence_score = min(self.confidence_score + 0.1, 1.0)


@dataclass
class DomainVocabulary:
    """Vocabulary collection for a specific domain."""

    domain_name: str
    user_id: str
    terms: Dict[str, VocabularyTerm] = field(default_factory=dict)

    # Domain statistics
    total_terms: int = 0
    active_terms: int = 0  # Terms used in last 30 days
    learning_velocity: float = 0.0  # New terms per week

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_term(self, term: VocabularyTerm):
        """Add a new term to the vocabulary."""
        self.terms[term.term.lower()] = term
        self.total_terms = len(self.terms)
        self.last_updated = datetime.utcnow()

    def get_term(self, term: str) -> Optional[VocabularyTerm]:
        """Get a term from the vocabulary."""
        return self.terms.get(term.lower())

    def update_term_usage(self, term: str, context: str):
        """Update usage for an existing term."""
        vocab_term = self.get_term(term)
        if vocab_term:
            vocab_term.update_usage(context)
            self.last_updated = datetime.utcnow()

    def get_active_terms(self, days: int = 30) -> List[VocabularyTerm]:
        """Get terms used within the specified number of days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return [term for term in self.terms.values() if term.last_used >= cutoff_date]

    def get_learning_velocity(self, weeks: int = 4) -> float:
        """Calculate learning velocity (new terms per week)."""
        cutoff_date = datetime.utcnow() - timedelta(weeks=weeks)
        new_terms = [
            term for term in self.terms.values() if term.first_seen >= cutoff_date
        ]
        return len(new_terms) / weeks

    def get_term_suggestions(
        self, partial_term: str, limit: int = 5
    ) -> List[VocabularyTerm]:
        """Get term suggestions based on partial input."""
        partial_lower = partial_term.lower()
        suggestions = []

        for term in self.terms.values():
            if partial_lower in term.term.lower():
                suggestions.append(term)

        # Sort by usage count and confidence
        suggestions.sort(
            key=lambda t: (t.usage_count, t.confidence_score), reverse=True
        )

        return suggestions[:limit]


class VocabularyLearner:
    """
    Learns and manages domain-specific vocabulary.

    Automatically identifies new terms, learns their meanings from context,
    and tracks usage patterns to improve domain understanding.

    Requirements addressed: 24.2, 24.3, 24.4
    """

    # Patterns for identifying different term types
    TERM_PATTERNS = {
        TermType.ACRONYM: [
            r"\b[A-Z]{2,}\b",  # All caps 2+ letters
            r"\b[A-Z][A-Z0-9]+\b",  # Caps with numbers
        ],
        TermType.TECHNICAL: [
            r"\b\w+\(\)\b",  # Function calls
            r"\b[a-z]+[A-Z]\w*\b",  # camelCase
            r"\b\w+\.\w+\b",  # dot notation
            r"\b\w+_\w+\b",  # snake_case
        ],
        TermType.CONCEPT: [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Title Case phrases
            r"\b\w+ (pattern|principle|method|approach)\b",
        ],
    }

    # Context patterns for definition extraction
    DEFINITION_PATTERNS = [
        r"(\w+) is (?:a |an |the )?(.+?)(?:\.|,|;)",
        r"(\w+) means (.+?)(?:\.|,|;)",
        r"(\w+)[:\-] (.+?)(?:\.|,|;)",
        r"(?:define|definition of) (\w+)[:\-]? (.+?)(?:\.|,|;)",
        r"(\w+) (?:refers to|stands for) (.+?)(?:\.|,|;)",
    ]

    def __init__(self):
        """Initialize the vocabulary learner."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_dir) / "vocabulary"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.vocabulary_cache: Dict[str, Dict[str, DomainVocabulary]] = {}

        logger.info(f"Vocabulary learner initialized at {self.storage_path}")

    def learn_from_interaction(
        self, user_id: str, domain: str, interaction_data: InteractionData
    ) -> List[VocabularyTerm]:
        """
        Learn vocabulary from a user interaction.

        Requirement 24.4: Expand domain-specific vocabulary understanding

        Args:
            user_id: User identifier
            domain: Domain name
            interaction_data: Interaction data to analyze

        Returns:
            List[VocabularyTerm]: Newly learned terms
        """
        logger.info(f"Learning vocabulary for user {user_id} in domain {domain}")

        try:
            # Extract message text
            message_text = ""
            if hasattr(interaction_data.conversation_context, "message_text"):
                message_text = interaction_data.conversation_context.message_text

            if not message_text:
                return []

            # Get or create domain vocabulary
            vocabulary = self.get_domain_vocabulary(user_id, domain)

            # Extract new terms from the message
            new_terms = self._extract_terms(message_text, domain)

            # Learn definitions from context
            definitions = self._extract_definitions(message_text)

            # Process each new term
            learned_terms = []
            for term_text, term_type in new_terms:
                # Check if term already exists
                existing_term = vocabulary.get_term(term_text)

                if existing_term:
                    # Update usage for existing term
                    vocabulary.update_term_usage(term_text, message_text[:200])
                else:
                    # Create new vocabulary term
                    new_term = VocabularyTerm(
                        term=term_text,
                        domain=domain,
                        term_type=term_type,
                        definition=definitions.get(term_text.lower()),
                        learning_source=LearningSource.USER_USAGE,
                        confidence_score=0.3,
                    )

                    # Update usage
                    new_term.update_usage(message_text[:200])

                    # Add to vocabulary
                    vocabulary.add_term(new_term)
                    learned_terms.append(new_term)

                    logger.debug(f"Learned new term: {term_text} ({term_type.value})")

            # Save updated vocabulary
            self.save_domain_vocabulary(vocabulary)

            logger.info(
                f"Learned {len(learned_terms)} new terms for {user_id} in {domain}"
            )
            return learned_terms

        except Exception as e:
            logger.error(f"Error learning vocabulary for user {user_id}: {e}")
            return []

    def get_domain_vocabulary(self, user_id: str, domain: str) -> DomainVocabulary:
        """
        Get domain vocabulary for a user.

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            DomainVocabulary: Domain vocabulary
        """
        # Check cache first
        if (
            user_id in self.vocabulary_cache
            and domain in self.vocabulary_cache[user_id]
        ):
            return self.vocabulary_cache[user_id][domain]

        # Load from storage
        vocab_path = self.storage_path / f"{user_id}_{domain}_vocabulary.json"

        if vocab_path.exists():
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to dataclass
                vocabulary = DomainVocabulary(
                    domain_name=data["domain_name"],
                    user_id=data["user_id"],
                    total_terms=data.get("total_terms", 0),
                    active_terms=data.get("active_terms", 0),
                    learning_velocity=data.get("learning_velocity", 0.0),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                )

                # Load terms
                for term_data in data.get("terms", []):
                    term = VocabularyTerm(
                        term=term_data["term"],
                        domain=term_data["domain"],
                        term_type=TermType(term_data["term_type"]),
                        definition=term_data.get("definition"),
                        synonyms=term_data.get("synonyms", []),
                        related_terms=term_data.get("related_terms", []),
                        usage_count=term_data.get("usage_count", 0),
                        first_seen=datetime.fromisoformat(term_data["first_seen"]),
                        last_used=datetime.fromisoformat(term_data["last_used"]),
                        contexts=term_data.get("contexts", []),
                        confidence_score=term_data.get("confidence_score", 0.0),
                        learning_source=LearningSource(
                            term_data.get("learning_source", "user_usage")
                        ),
                        user_defined=term_data.get("user_defined", False),
                    )
                    vocabulary.add_term(term)

                # Cache the vocabulary
                if user_id not in self.vocabulary_cache:
                    self.vocabulary_cache[user_id] = {}
                self.vocabulary_cache[user_id][domain] = vocabulary

                return vocabulary

            except Exception as e:
                logger.error(f"Error loading vocabulary for {user_id}/{domain}: {e}")

        # Create new vocabulary if none exists
        vocabulary = DomainVocabulary(domain_name=domain, user_id=user_id)

        # Cache the vocabulary
        if user_id not in self.vocabulary_cache:
            self.vocabulary_cache[user_id] = {}
        self.vocabulary_cache[user_id][domain] = vocabulary

        return vocabulary

    def add_user_definition(
        self, user_id: str, domain: str, term: str, definition: str
    ) -> VocabularyTerm:
        """
        Add a user-provided definition for a term.

        Args:
            user_id: User identifier
            domain: Domain name
            term: Term to define
            definition: User-provided definition

        Returns:
            VocabularyTerm: Updated or created term
        """
        vocabulary = self.get_domain_vocabulary(user_id, domain)

        existing_term = vocabulary.get_term(term)
        if existing_term:
            # Update existing term
            existing_term.definition = definition
            existing_term.user_defined = True
            existing_term.confidence_score = 1.0
            existing_term.learning_source = LearningSource.EXPLICIT_DEFINITION
        else:
            # Create new term
            new_term = VocabularyTerm(
                term=term,
                domain=domain,
                term_type=TermType.CONCEPT,  # Default for user-defined
                definition=definition,
                user_defined=True,
                confidence_score=1.0,
                learning_source=LearningSource.EXPLICIT_DEFINITION,
            )
            vocabulary.add_term(new_term)
            existing_term = new_term

        # Save updated vocabulary
        self.save_domain_vocabulary(vocabulary)

        logger.info(f"Added user definition for '{term}' in {domain}")
        return existing_term

    def get_term_suggestions(
        self, user_id: str, domain: str, partial_term: str, limit: int = 5
    ) -> List[VocabularyTerm]:
        """
        Get term suggestions for autocomplete.

        Args:
            user_id: User identifier
            domain: Domain name
            partial_term: Partial term input
            limit: Maximum number of suggestions

        Returns:
            List[VocabularyTerm]: Term suggestions
        """
        vocabulary = self.get_domain_vocabulary(user_id, domain)
        return vocabulary.get_term_suggestions(partial_term, limit)

    def expand_query_vocabulary(
        self, user_id: str, domain: str, query: str
    ) -> List[str]:
        """
        Expand a query with domain-specific synonyms and related terms.

        Requirement 24.3: Customize search with domain vocabulary

        Args:
            user_id: User identifier
            domain: Domain name
            query: Original query

        Returns:
            List[str]: Expanded query terms
        """
        vocabulary = self.get_domain_vocabulary(user_id, domain)

        expanded_terms = [query]  # Start with original query
        query_words = query.lower().split()

        for word in query_words:
            term = vocabulary.get_term(word)
            if term:
                # Add synonyms
                expanded_terms.extend(term.synonyms)

                # Add related terms with high confidence
                for related in term.related_terms:
                    related_term = vocabulary.get_term(related)
                    if related_term and related_term.confidence_score > 0.7:
                        expanded_terms.append(related)

        # Remove duplicates and return
        return list(set(expanded_terms))

    def get_vocabulary_stats(self, user_id: str, domain: str) -> Dict[str, Any]:
        """
        Get vocabulary statistics for a domain.

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            Dict[str, Any]: Vocabulary statistics
        """
        vocabulary = self.get_domain_vocabulary(user_id, domain)

        # Calculate statistics
        active_terms = vocabulary.get_active_terms()
        learning_velocity = vocabulary.get_learning_velocity()

        # Term type distribution
        type_distribution = Counter(
            term.term_type.value for term in vocabulary.terms.values()
        )

        # Confidence distribution
        high_confidence = len(
            [t for t in vocabulary.terms.values() if t.confidence_score > 0.7]
        )
        medium_confidence = len(
            [t for t in vocabulary.terms.values() if 0.3 < t.confidence_score <= 0.7]
        )
        low_confidence = len(
            [t for t in vocabulary.terms.values() if t.confidence_score <= 0.3]
        )

        return {
            "total_terms": vocabulary.total_terms,
            "active_terms": len(active_terms),
            "learning_velocity": learning_velocity,
            "type_distribution": dict(type_distribution),
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },
            "user_defined_terms": len(
                [t for t in vocabulary.terms.values() if t.user_defined]
            ),
            "last_updated": vocabulary.last_updated.isoformat(),
        }

    def save_domain_vocabulary(self, vocabulary: DomainVocabulary):
        """
        Save domain vocabulary to storage.

        Args:
            vocabulary: Domain vocabulary to save
        """
        # Update cache
        if vocabulary.user_id not in self.vocabulary_cache:
            self.vocabulary_cache[vocabulary.user_id] = {}
        self.vocabulary_cache[vocabulary.user_id][vocabulary.domain_name] = vocabulary

        # Save to storage
        vocab_path = (
            self.storage_path
            / f"{vocabulary.user_id}_{vocabulary.domain_name}_vocabulary.json"
        )

        try:
            # Convert terms to serializable format
            terms_data = []
            for term in vocabulary.terms.values():
                terms_data.append(
                    {
                        "term": term.term,
                        "domain": term.domain,
                        "term_type": term.term_type.value,
                        "definition": term.definition,
                        "synonyms": term.synonyms,
                        "related_terms": term.related_terms,
                        "usage_count": term.usage_count,
                        "first_seen": term.first_seen.isoformat(),
                        "last_used": term.last_used.isoformat(),
                        "contexts": term.contexts,
                        "confidence_score": term.confidence_score,
                        "learning_source": term.learning_source.value,
                        "user_defined": term.user_defined,
                    }
                )

            data = {
                "domain_name": vocabulary.domain_name,
                "user_id": vocabulary.user_id,
                "total_terms": vocabulary.total_terms,
                "active_terms": vocabulary.active_terms,
                "learning_velocity": vocabulary.learning_velocity,
                "created_at": vocabulary.created_at.isoformat(),
                "last_updated": vocabulary.last_updated.isoformat(),
                "terms": terms_data,
            }

            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                f"Error saving vocabulary for {vocabulary.user_id}/{vocabulary.domain_name}: {e}"
            )

    def _extract_terms(self, text: str, domain: str) -> List[Tuple[str, TermType]]:
        """Extract potential vocabulary terms from text."""
        import re

        extracted_terms = []

        # Extract terms by type
        for term_type, patterns in self.TERM_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) > 1:  # Filter out single characters
                        extracted_terms.append((match, term_type))

        # Extract domain-specific terms based on capitalization and context
        # This is a simplified approach - in practice, you'd use NLP libraries
        words = text.split()
        for i, word in enumerate(words):
            # Look for capitalized words that might be technical terms
            if word[0].isupper() and len(word) > 2:
                # Check if it's in a technical context
                context_words = words[max(0, i - 2) : i + 3]
                context = " ".join(context_words).lower()

                if any(
                    tech_word in context
                    for tech_word in ["using", "with", "implement", "configure"]
                ):
                    extracted_terms.append((word, TermType.TOOL))

        # Remove duplicates
        seen = set()
        unique_terms = []
        for term, term_type in extracted_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append((term, term_type))

        return unique_terms

    def _extract_definitions(self, text: str) -> Dict[str, str]:
        """Extract term definitions from text using pattern matching."""
        import re

        definitions = {}

        for pattern in self.DEFINITION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()

                # Clean up the definition
                definition = re.sub(r"\s+", " ", definition)  # Normalize whitespace

                if len(definition) > 10:  # Only keep substantial definitions
                    definitions[term.lower()] = definition

        return definitions


# Global vocabulary learner instance
_vocabulary_learner: Optional[VocabularyLearner] = None


def get_vocabulary_learner() -> VocabularyLearner:
    """
    Get the global vocabulary learner instance.

    Returns:
        VocabularyLearner: Global vocabulary learner instance
    """
    global _vocabulary_learner
    if _vocabulary_learner is None:
        _vocabulary_learner = VocabularyLearner()
    return _vocabulary_learner
