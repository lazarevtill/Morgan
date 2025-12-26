"""
Domain Knowledge Tracking for Morgan RAG.

Tracks user expertise across different domains, identifies knowledge gaps,
and maintains domain-specific interaction history for personalized assistance.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import get_settings
from ..emotional.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DomainCategory(Enum):
    """Categories of knowledge domains."""

    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    HOBBY = "hobby"


class ExpertiseLevel(Enum):
    """Levels of expertise in a domain."""

    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DomainInteraction:
    """Represents an interaction within a specific domain."""

    interaction_id: str
    domain_name: str
    timestamp: datetime
    topics_discussed: List[str]
    questions_asked: List[str]
    concepts_mentioned: List[str]
    complexity_level: float  # 0.0 to 1.0
    user_confidence: Optional[float] = None  # 0.0 to 1.0
    learning_indicators: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize interaction ID if not provided."""
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())


@dataclass
class DomainProfile:
    """Complete profile for a user's expertise in a specific domain."""

    user_id: str
    domain_name: str
    category: DomainCategory
    expertise_level: ExpertiseLevel
    confidence_score: float  # 0.0 to 1.0

    # Knowledge tracking
    known_concepts: Set[str] = field(default_factory=set)
    learning_goals: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)

    # Interaction history
    total_interactions: int = 0
    recent_interactions: List[DomainInteraction] = field(default_factory=list)

    # Learning progress
    concepts_learned: Dict[str, datetime] = field(default_factory=dict)
    learning_velocity: float = 0.0  # concepts per day

    # Metadata
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_interaction(self, interaction: DomainInteraction):
        """Add a new interaction to the domain profile."""
        self.recent_interactions.append(interaction)
        self.total_interactions += 1

        # Keep only recent interactions (last 50)
        if len(self.recent_interactions) > 50:
            self.recent_interactions = self.recent_interactions[-50:]

        # Update timestamps
        if not self.first_interaction:
            self.first_interaction = interaction.timestamp
        self.last_interaction = interaction.timestamp
        self.last_updated = datetime.utcnow()

        # Update known concepts
        self.known_concepts.update(interaction.concepts_mentioned)

        # Track newly learned concepts
        for concept in interaction.concepts_mentioned:
            if concept not in self.concepts_learned:
                self.concepts_learned[concept] = interaction.timestamp

    def get_learning_velocity(self, days: int = 30) -> float:
        """Calculate learning velocity over specified period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_concepts = [
            concept
            for concept, learned_date in self.concepts_learned.items()
            if learned_date >= cutoff_date
        ]
        return len(recent_concepts) / days

    def get_expertise_progression(self) -> List[Dict[str, Any]]:
        """Get progression of expertise over time."""
        progression = []

        # Group interactions by week
        weekly_interactions = {}
        for interaction in self.recent_interactions:
            week_key = interaction.timestamp.strftime("%Y-W%U")
            if week_key not in weekly_interactions:
                weekly_interactions[week_key] = []
            weekly_interactions[week_key].append(interaction)

        # Calculate weekly metrics
        for week, interactions in sorted(weekly_interactions.items()):
            avg_complexity = sum(i.complexity_level for i in interactions) / len(
                interactions
            )
            avg_confidence = (
                sum(
                    i.user_confidence
                    for i in interactions
                    if i.user_confidence is not None
                )
                / len([i for i in interactions if i.user_confidence is not None])
                if any(i.user_confidence is not None for i in interactions)
                else 0.0
            )

            progression.append(
                {
                    "week": week,
                    "interaction_count": len(interactions),
                    "avg_complexity": avg_complexity,
                    "avg_confidence": avg_confidence,
                    "concepts_discussed": len(
                        set(
                            concept
                            for interaction in interactions
                            for concept in interaction.concepts_mentioned
                        )
                    ),
                }
            )

        return progression


class DomainKnowledgeTracker:
    """
    Tracks user knowledge and expertise across different domains.

    Analyzes interactions to identify domains, assess expertise levels,
    track learning progress, and identify knowledge gaps.

    Requirements addressed: 24.2, 24.3, 24.4
    """

    # Domain identification patterns
    DOMAIN_PATTERNS = {
        "programming": [
            r"\b(code|coding|programming|software|development)\b",
            r"\b(python|javascript|java|c\+\+|html|css|sql)\b",
            r"\b(function|class|variable|algorithm|debug)\b",
        ],
        "data_science": [
            r"\b(data|dataset|analysis|statistics|machine learning)\b",
            r"\b(pandas|numpy|sklearn|tensorflow|pytorch)\b",
            r"\b(model|prediction|classification|regression)\b",
        ],
        "web_development": [
            r"\b(website|web|frontend|backend|api|server)\b",
            r"\b(react|vue|angular|node|express|django)\b",
            r"\b(html|css|javascript|responsive|database)\b",
        ],
        "business": [
            r"\b(business|strategy|marketing|sales|revenue)\b",
            r"\b(customer|client|market|competition|growth)\b",
            r"\b(profit|budget|investment|roi|kpi)\b",
        ],
        "design": [
            r"\b(design|ui|ux|interface|user experience)\b",
            r"\b(color|typography|layout|wireframe|prototype)\b",
            r"\b(figma|sketch|photoshop|illustrator)\b",
        ],
        "science": [
            r"\b(research|experiment|hypothesis|theory|study)\b",
            r"\b(biology|chemistry|physics|mathematics)\b",
            r"\b(analysis|methodology|results|conclusion)\b",
        ],
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        "basic": [
            r"\b(what is|how to|basic|simple|introduction)\b",
            r"\b(beginner|start|first time|new to)\b",
        ],
        "intermediate": [
            r"\b(implement|configure|optimize|integrate)\b",
            r"\b(best practice|pattern|architecture)\b",
        ],
        "advanced": [
            r"\b(performance|scalability|security|enterprise)\b",
            r"\b(advanced|complex|sophisticated|cutting-edge)\b",
        ],
    }

    def __init__(self):
        """Initialize the domain knowledge tracker."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_dir) / "domains"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.domain_cache: Dict[str, Dict[str, DomainProfile]] = {}

        logger.info(f"Domain knowledge tracker initialized at {self.storage_path}")

    def track_interaction(
        self, user_id: str, interaction_data: InteractionData
    ) -> List[DomainProfile]:
        """
        Track a user interaction and update domain profiles.

        Requirement 24.2: Track domain knowledge and expertise

        Args:
            user_id: User identifier
            interaction_data: Interaction data to analyze

        Returns:
            List[DomainProfile]: Updated domain profiles
        """
        logger.info(f"Tracking domain interaction for user {user_id}")

        try:
            # Extract message text for analysis
            message_text = ""
            if hasattr(interaction_data.conversation_context, "message_text"):
                message_text = interaction_data.conversation_context.message_text

            if not message_text:
                logger.warning("No message text found in interaction data")
                return []

            # Identify domains from the interaction
            identified_domains = self._identify_domains(message_text)

            if not identified_domains:
                logger.debug("No domains identified in interaction")
                return []

            updated_profiles = []

            for domain_name, confidence in identified_domains.items():
                # Get or create domain profile
                profile = self.get_domain_profile(user_id, domain_name)

                # Create domain interaction
                domain_interaction = self._create_domain_interaction(
                    domain_name, message_text, interaction_data
                )

                # Update profile with interaction
                profile.add_interaction(domain_interaction)

                # Update expertise level based on interaction
                self._update_expertise_level(profile, domain_interaction)

                # Save updated profile
                self.save_domain_profile(profile)
                updated_profiles.append(profile)

                logger.debug(
                    f"Updated domain '{domain_name}' for user {user_id} "
                    f"(expertise: {profile.expertise_level.value})"
                )

            return updated_profiles

        except Exception as e:
            logger.error(f"Error tracking domain interaction for user {user_id}: {e}")
            return []

    def get_domain_profile(self, user_id: str, domain_name: str) -> DomainProfile:
        """
        Get domain profile for a user.

        Args:
            user_id: User identifier
            domain_name: Domain name

        Returns:
            DomainProfile: User's domain profile
        """
        # Check cache first
        if user_id in self.domain_cache and domain_name in self.domain_cache[user_id]:
            return self.domain_cache[user_id][domain_name]

        # Load from storage
        profile_path = self.storage_path / f"{user_id}_{domain_name}_domain.json"

        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to dataclass
                profile = DomainProfile(
                    user_id=data["user_id"],
                    domain_name=data["domain_name"],
                    category=DomainCategory(data["category"]),
                    expertise_level=ExpertiseLevel(data["expertise_level"]),
                    confidence_score=data["confidence_score"],
                    known_concepts=set(data.get("known_concepts", [])),
                    learning_goals=data.get("learning_goals", []),
                    knowledge_gaps=data.get("knowledge_gaps", []),
                    total_interactions=data.get("total_interactions", 0),
                    concepts_learned={
                        concept: datetime.fromisoformat(date_str)
                        for concept, date_str in data.get(
                            "concepts_learned", {}
                        ).items()
                    },
                    learning_velocity=data.get("learning_velocity", 0.0),
                    first_interaction=(
                        datetime.fromisoformat(data["first_interaction"])
                        if data.get("first_interaction")
                        else None
                    ),
                    last_interaction=(
                        datetime.fromisoformat(data["last_interaction"])
                        if data.get("last_interaction")
                        else None
                    ),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                )

                # Cache the profile
                if user_id not in self.domain_cache:
                    self.domain_cache[user_id] = {}
                self.domain_cache[user_id][domain_name] = profile

                return profile

            except Exception as e:
                logger.error(
                    f"Error loading domain profile for {user_id}/{domain_name}: {e}"
                )

        # Create new profile if none exists
        category = self._categorize_domain(domain_name)
        profile = DomainProfile(
            user_id=user_id,
            domain_name=domain_name,
            category=category,
            expertise_level=ExpertiseLevel.NOVICE,
            confidence_score=0.0,
        )

        # Cache the profile
        if user_id not in self.domain_cache:
            self.domain_cache[user_id] = {}
        self.domain_cache[user_id][domain_name] = profile

        return profile

    def get_user_domains(self, user_id: str) -> List[DomainProfile]:
        """
        Get all domain profiles for a user.

        Args:
            user_id: User identifier

        Returns:
            List[DomainProfile]: User's domain profiles
        """
        profiles = []

        # Load all domain files for the user
        for profile_path in self.storage_path.glob(f"{user_id}_*_domain.json"):
            domain_name = profile_path.stem.replace(f"{user_id}_", "").replace(
                "_domain", ""
            )
            profile = self.get_domain_profile(user_id, domain_name)
            profiles.append(profile)

        return profiles

    def identify_knowledge_gaps(
        self, user_id: str, domain_name: str, target_concepts: List[str]
    ) -> List[str]:
        """
        Identify knowledge gaps in a domain.

        Requirement 24.3: Identify areas for learning improvement

        Args:
            user_id: User identifier
            domain_name: Domain name
            target_concepts: Concepts that should be known at current level

        Returns:
            List[str]: Identified knowledge gaps
        """
        profile = self.get_domain_profile(user_id, domain_name)

        # Find concepts not yet learned
        gaps = [
            concept
            for concept in target_concepts
            if concept not in profile.known_concepts
        ]

        # Update profile with identified gaps
        profile.knowledge_gaps = gaps
        self.save_domain_profile(profile)

        logger.info(
            f"Identified {len(gaps)} knowledge gaps for {user_id} in {domain_name}"
        )
        return gaps

    def suggest_learning_path(
        self, user_id: str, domain_name: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest a learning path based on current expertise and gaps.

        Requirement 24.4: Provide personalized learning recommendations

        Args:
            user_id: User identifier
            domain_name: Domain name

        Returns:
            List[Dict[str, Any]]: Suggested learning steps
        """
        profile = self.get_domain_profile(user_id, domain_name)

        learning_path = []

        # Base recommendations on expertise level
        if profile.expertise_level == ExpertiseLevel.NOVICE:
            learning_path.extend(
                [
                    {
                        "step": "fundamentals",
                        "title": f"Learn {domain_name} Fundamentals",
                        "description": "Start with basic concepts and terminology",
                        "priority": "high",
                        "estimated_time": "2-4 weeks",
                    },
                    {
                        "step": "practice",
                        "title": "Hands-on Practice",
                        "description": "Apply concepts through simple exercises",
                        "priority": "high",
                        "estimated_time": "1-2 weeks",
                    },
                ]
            )

        elif profile.expertise_level == ExpertiseLevel.BEGINNER:
            learning_path.extend(
                [
                    {
                        "step": "intermediate_concepts",
                        "title": "Intermediate Concepts",
                        "description": "Build on fundamentals with more complex topics",
                        "priority": "high",
                        "estimated_time": "3-6 weeks",
                    },
                    {
                        "step": "projects",
                        "title": "Small Projects",
                        "description": "Create projects to solidify understanding",
                        "priority": "medium",
                        "estimated_time": "2-4 weeks",
                    },
                ]
            )

        # Add gap-specific recommendations
        for gap in profile.knowledge_gaps[:3]:  # Top 3 gaps
            learning_path.append(
                {
                    "step": "fill_gap",
                    "title": f"Learn {gap}",
                    "description": f"Address knowledge gap in {gap}",
                    "priority": "medium",
                    "estimated_time": "1-2 weeks",
                    "gap_concept": gap,
                }
            )

        logger.info(
            f"Generated {len(learning_path)} learning steps for {user_id} in {domain_name}"
        )
        return learning_path

    def save_domain_profile(self, profile: DomainProfile):
        """
        Save domain profile to storage.

        Args:
            profile: Domain profile to save
        """
        # Update cache
        if profile.user_id not in self.domain_cache:
            self.domain_cache[profile.user_id] = {}
        self.domain_cache[profile.user_id][profile.domain_name] = profile

        # Save to storage
        profile_path = (
            self.storage_path / f"{profile.user_id}_{profile.domain_name}_domain.json"
        )

        try:
            data = {
                "user_id": profile.user_id,
                "domain_name": profile.domain_name,
                "category": profile.category.value,
                "expertise_level": profile.expertise_level.value,
                "confidence_score": profile.confidence_score,
                "known_concepts": list(profile.known_concepts),
                "learning_goals": profile.learning_goals,
                "knowledge_gaps": profile.knowledge_gaps,
                "total_interactions": profile.total_interactions,
                "concepts_learned": {
                    concept: date.isoformat()
                    for concept, date in profile.concepts_learned.items()
                },
                "learning_velocity": profile.learning_velocity,
                "first_interaction": (
                    profile.first_interaction.isoformat()
                    if profile.first_interaction
                    else None
                ),
                "last_interaction": (
                    profile.last_interaction.isoformat()
                    if profile.last_interaction
                    else None
                ),
                "last_updated": profile.last_updated.isoformat(),
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                f"Error saving domain profile for {profile.user_id}/{profile.domain_name}: {e}"
            )

    def _identify_domains(self, text: str) -> Dict[str, float]:
        """Identify domains mentioned in text with confidence scores."""
        import re

        identified_domains = {}
        text_lower = text.lower()

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            matches = 0
            total_patterns = len(patterns)

            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1

            if matches > 0:
                confidence = matches / total_patterns
                identified_domains[domain] = confidence

        return identified_domains

    def _create_domain_interaction(
        self, domain_name: str, message_text: str, interaction_data: InteractionData
    ) -> DomainInteraction:
        """Create a domain interaction from message text and interaction data."""
        import re

        # Extract topics (simple keyword extraction)
        topics = self._extract_topics(message_text)

        # Extract questions
        questions = re.findall(r"[^.!?]*\?", message_text)

        # Extract concepts (domain-specific terms)
        concepts = self._extract_concepts(domain_name, message_text)

        # Assess complexity level
        complexity = self._assess_complexity(message_text)

        # Extract learning indicators
        learning_indicators = self._extract_learning_indicators(message_text)

        return DomainInteraction(
            interaction_id=str(uuid.uuid4()),
            domain_name=domain_name,
            timestamp=datetime.utcnow(),
            topics_discussed=topics,
            questions_asked=questions,
            concepts_mentioned=concepts,
            complexity_level=complexity,
            learning_indicators=learning_indicators,
        )

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword extraction."""
        # This is a simplified implementation
        # In practice, you might use NLP libraries for better topic extraction
        import re

        # Extract noun phrases and technical terms
        topics = []

        # Simple pattern for technical terms (capitalized words, acronyms)
        technical_terms = re.findall(r"\b[A-Z][A-Za-z]*\b|\b[A-Z]{2,}\b", text)
        topics.extend(technical_terms)

        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', text)
        topics.extend(quoted_terms)

        return list(set(topics))  # Remove duplicates

    def _extract_concepts(self, domain_name: str, text: str) -> List[str]:
        """Extract domain-specific concepts from text."""
        # This would ideally use domain-specific concept dictionaries
        # For now, use simple pattern matching
        import re

        concepts = []
        text_lower = text.lower()

        # Domain-specific concept patterns
        concept_patterns = {
            "programming": [
                r"\b(function|class|method|variable|array|object|loop|condition)\b",
                r"\b(algorithm|data structure|recursion|iteration|inheritance)\b",
            ],
            "data_science": [
                r"\b(dataset|feature|model|training|prediction|accuracy)\b",
                r"\b(regression|classification|clustering|neural network)\b",
            ],
            "web_development": [
                r"\b(component|route|endpoint|middleware|database|query)\b",
                r"\b(frontend|backend|api|authentication|deployment)\b",
            ],
        }

        if domain_name in concept_patterns:
            for pattern in concept_patterns[domain_name]:
                matches = re.findall(pattern, text_lower)
                concepts.extend(matches)

        return list(set(concepts))

    def _assess_complexity(self, text: str) -> float:
        """Assess the complexity level of the text content."""
        import re

        text_lower = text.lower()
        complexity_score = 0.0

        # Check for complexity indicators
        for level, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if level == "basic":
                        complexity_score = max(complexity_score, 0.2)
                    elif level == "intermediate":
                        complexity_score = max(complexity_score, 0.6)
                    elif level == "advanced":
                        complexity_score = max(complexity_score, 0.9)

        # Default to intermediate if no indicators found
        if complexity_score == 0.0:
            complexity_score = 0.5

        return complexity_score

    def _extract_learning_indicators(self, text: str) -> List[str]:
        """Extract indicators of learning intent or confusion."""
        import re

        indicators = []
        text_lower = text.lower()

        learning_patterns = {
            "confusion": [r"\b(confused|don\'t understand|unclear|lost)\b"],
            "curiosity": [r"\b(how|why|what|when|where|curious|interested)\b"],
            "practice": [r"\b(practice|try|attempt|exercise|example)\b"],
            "mastery": [r"\b(master|expert|advanced|proficient)\b"],
        }

        for indicator, patterns in learning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    indicators.append(indicator)
                    break  # Only add each indicator once

        return indicators

    def _update_expertise_level(
        self, profile: DomainProfile, interaction: DomainInteraction
    ):
        """Update expertise level based on interaction complexity and history."""
        # Simple heuristic for expertise level progression
        avg_complexity = sum(
            i.complexity_level for i in profile.recent_interactions[-10:]
        ) / min(len(profile.recent_interactions), 10)

        concept_count = len(profile.known_concepts)
        interaction_count = profile.total_interactions

        # Calculate new expertise level
        if avg_complexity < 0.3 and concept_count < 10:
            new_level = ExpertiseLevel.NOVICE
        elif avg_complexity < 0.5 and concept_count < 25:
            new_level = ExpertiseLevel.BEGINNER
        elif avg_complexity < 0.7 and concept_count < 50:
            new_level = ExpertiseLevel.INTERMEDIATE
        elif avg_complexity < 0.9 and concept_count < 100:
            new_level = ExpertiseLevel.ADVANCED
        else:
            new_level = ExpertiseLevel.EXPERT

        # Update if level has changed
        if new_level != profile.expertise_level:
            logger.info(
                f"Expertise level updated for {profile.user_id} in {profile.domain_name}: "
                f"{profile.expertise_level.value} -> {new_level.value}"
            )
            profile.expertise_level = new_level

        # Update confidence score based on consistency
        if interaction_count > 5:
            recent_complexity = [
                i.complexity_level for i in profile.recent_interactions[-5:]
            ]
            consistency = 1.0 - (max(recent_complexity) - min(recent_complexity))
            profile.confidence_score = min(consistency * avg_complexity, 1.0)

    def _categorize_domain(self, domain_name: str) -> DomainCategory:
        """Categorize a domain into a general category."""
        domain_lower = domain_name.lower()

        if any(
            tech in domain_lower
            for tech in ["programming", "software", "code", "web", "data"]
        ):
            return DomainCategory.TECHNOLOGY
        elif any(biz in domain_lower for biz in ["business", "marketing", "sales"]):
            return DomainCategory.BUSINESS
        elif any(
            sci in domain_lower
            for sci in ["science", "research", "biology", "chemistry"]
        ):
            return DomainCategory.SCIENCE
        elif any(
            creative in domain_lower for creative in ["design", "art", "creative"]
        ):
            return DomainCategory.CREATIVE
        else:
            return DomainCategory.PERSONAL


# Global domain tracker instance
_domain_tracker: Optional[DomainKnowledgeTracker] = None


def get_domain_tracker() -> DomainKnowledgeTracker:
    """
    Get the global domain knowledge tracker instance.

    Returns:
        DomainKnowledgeTracker: Global domain tracker instance
    """
    global _domain_tracker
    if _domain_tracker is None:
        _domain_tracker = DomainKnowledgeTracker()
    return _domain_tracker
