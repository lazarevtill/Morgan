"""
Knowledge Depth Assessment for Morgan RAG.

Assesses user's knowledge depth in specific domains, tracks learning
progression, identifies skill levels, and provides depth-appropriate
content and assistance.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_settings
from ..emotional.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeLevel(Enum):
    """Knowledge depth levels."""

    SURFACE = "surface"  # Basic awareness, terminology
    FUNCTIONAL = "functional"  # Can use with guidance
    WORKING = "working"  # Independent application
    DEEP = "deep"  # Understanding principles
    EXPERT = "expert"  # Can teach and innovate


class AssessmentMethod(Enum):
    """Methods for assessing knowledge depth."""

    QUESTION_COMPLEXITY = "question_complexity"
    EXPLANATION_DEPTH = "explanation_depth"
    PROBLEM_SOLVING = "problem_solving"
    TERMINOLOGY_USAGE = "terminology_usage"
    CONCEPT_CONNECTIONS = "concept_connections"


@dataclass
class KnowledgeAssessment:
    """Assessment of knowledge in a specific area."""

    assessment_id: str
    user_id: str
    domain: str
    topic: str

    # Assessment results
    knowledge_level: KnowledgeLevel
    confidence_score: float  # 0.0 to 1.0
    depth_score: float  # 0.0 to 1.0 (normalized depth measure)

    # Assessment details
    assessment_method: AssessmentMethod
    evidence: List[str] = field(default_factory=list)
    indicators: Dict[str, float] = field(default_factory=dict)

    # Progression tracking
    previous_level: Optional[KnowledgeLevel] = None
    progression_rate: float = 0.0  # Change per week

    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    interaction_count: int = 0

    def update_progression(self, new_level: KnowledgeLevel):
        """Update progression tracking."""
        if self.knowledge_level != new_level:
            self.previous_level = self.knowledge_level
            self.knowledge_level = new_level

            # Calculate progression rate (simplified)
            level_values = {
                KnowledgeLevel.SURFACE: 1,
                KnowledgeLevel.FUNCTIONAL: 2,
                KnowledgeLevel.WORKING: 3,
                KnowledgeLevel.DEEP: 4,
                KnowledgeLevel.EXPERT: 5,
            }

            if self.previous_level:
                old_value = level_values[self.previous_level]
                new_value = level_values[new_level]
                self.progression_rate = new_value - old_value


@dataclass
class DomainDepthProfile:
    """Complete knowledge depth profile for a domain."""

    user_id: str
    domain_name: str

    # Topic assessments
    topic_assessments: Dict[str, KnowledgeAssessment] = field(default_factory=dict)

    # Overall domain metrics
    overall_level: KnowledgeLevel = KnowledgeLevel.SURFACE
    domain_confidence: float = 0.0
    breadth_score: float = 0.0  # How many topics covered
    depth_consistency: float = 0.0  # Consistency across topics

    # Learning patterns
    learning_velocity: float = 0.0  # Topics per week
    preferred_depth: KnowledgeLevel = KnowledgeLevel.WORKING
    learning_style_indicators: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    total_assessments: int = 0

    def add_assessment(self, assessment: KnowledgeAssessment):
        """Add a knowledge assessment."""
        self.topic_assessments[assessment.topic] = assessment
        self.total_assessments += 1
        self.last_updated = datetime.utcnow()

        # Update overall metrics
        self._update_overall_metrics()

    def get_assessment(self, topic: str) -> Optional[KnowledgeAssessment]:
        """Get assessment for a specific topic."""
        return self.topic_assessments.get(topic)

    def get_topics_by_level(self, level: KnowledgeLevel) -> List[str]:
        """Get topics at a specific knowledge level."""
        return [
            topic
            for topic, assessment in self.topic_assessments.items()
            if assessment.knowledge_level == level
        ]

    def get_learning_recommendations(self) -> List[Dict[str, Any]]:
        """Get learning recommendations based on depth profile."""
        recommendations = []

        # Find topics that could be deepened
        for topic, assessment in self.topic_assessments.items():
            if assessment.knowledge_level in [
                KnowledgeLevel.SURFACE,
                KnowledgeLevel.FUNCTIONAL,
            ]:
                recommendations.append(
                    {
                        "type": "deepen_knowledge",
                        "topic": topic,
                        "current_level": assessment.knowledge_level.value,
                        "target_level": KnowledgeLevel.WORKING.value,
                        "priority": (
                            "high" if assessment.confidence_score > 0.6 else "medium"
                        ),
                    }
                )

        # Suggest breadth expansion if depth is good
        deep_topics = self.get_topics_by_level(KnowledgeLevel.DEEP)
        if len(deep_topics) >= 2 and self.breadth_score < 0.7:
            recommendations.append(
                {
                    "type": "expand_breadth",
                    "suggestion": "Explore related topics to broaden understanding",
                    "priority": "medium",
                }
            )

        # Suggest specialization if breadth is good
        if self.breadth_score > 0.8 and len(deep_topics) < 1:
            working_topics = self.get_topics_by_level(KnowledgeLevel.WORKING)
            if working_topics:
                recommendations.append(
                    {
                        "type": "specialize",
                        "topic": working_topics[0],  # Suggest first working-level topic
                        "suggestion": "Develop deep expertise in this area",
                        "priority": "high",
                    }
                )

        return recommendations

    def _update_overall_metrics(self):
        """Update overall domain metrics."""
        if not self.topic_assessments:
            return

        assessments = list(self.topic_assessments.values())

        # Calculate overall level (weighted average)
        level_values = {
            KnowledgeLevel.SURFACE: 1,
            KnowledgeLevel.FUNCTIONAL: 2,
            KnowledgeLevel.WORKING: 3,
            KnowledgeLevel.DEEP: 4,
            KnowledgeLevel.EXPERT: 5,
        }

        weighted_sum = sum(
            level_values[assessment.knowledge_level] * assessment.confidence_score
            for assessment in assessments
        )
        total_weight = sum(assessment.confidence_score for assessment in assessments)

        if total_weight > 0:
            avg_level_value = weighted_sum / total_weight

            # Convert back to level
            if avg_level_value < 1.5:
                self.overall_level = KnowledgeLevel.SURFACE
            elif avg_level_value < 2.5:
                self.overall_level = KnowledgeLevel.FUNCTIONAL
            elif avg_level_value < 3.5:
                self.overall_level = KnowledgeLevel.WORKING
            elif avg_level_value < 4.5:
                self.overall_level = KnowledgeLevel.DEEP
            else:
                self.overall_level = KnowledgeLevel.EXPERT

        # Calculate domain confidence (average of topic confidences)
        self.domain_confidence = sum(
            assessment.confidence_score for assessment in assessments
        ) / len(assessments)

        # Calculate breadth score (number of topics with working+ level)
        working_plus_topics = len(
            [
                a
                for a in assessments
                if a.knowledge_level
                in [KnowledgeLevel.WORKING, KnowledgeLevel.DEEP, KnowledgeLevel.EXPERT]
            ]
        )
        self.breadth_score = min(
            working_plus_topics / 10.0, 1.0
        )  # Normalize to 10 topics

        # Calculate depth consistency (standard deviation of levels)
        if len(assessments) > 1:
            level_values_list = [level_values[a.knowledge_level] for a in assessments]
            mean_level = sum(level_values_list) / len(level_values_list)
            variance = sum((x - mean_level) ** 2 for x in level_values_list) / len(
                level_values_list
            )
            std_dev = math.sqrt(variance)
            self.depth_consistency = max(0.0, 1.0 - (std_dev / 2.0))  # Normalize


class KnowledgeDepthAssessor:
    """
    Assesses and tracks knowledge depth across domains.

    Analyzes user interactions to determine knowledge levels,
    tracks learning progression, and provides depth-appropriate
    content recommendations.

    Requirements addressed: 24.2, 24.3, 24.4
    """

    # Complexity indicators for different levels
    COMPLEXITY_INDICATORS = {
        KnowledgeLevel.SURFACE: [
            r"\b(what is|define|basic|simple|introduction)\b",
            r"\b(beginner|new to|first time|don't know)\b",
        ],
        KnowledgeLevel.FUNCTIONAL: [
            r"\b(how to|tutorial|guide|step by step)\b",
            r"\b(use|apply|implement|configure)\b",
        ],
        KnowledgeLevel.WORKING: [
            r"\b(optimize|customize|integrate|troubleshoot)\b",
            r"\b(best practice|pattern|approach|strategy)\b",
        ],
        KnowledgeLevel.DEEP: [
            r"\b(architecture|design|principle|theory)\b",
            r"\b(why|underlying|fundamental|concept)\b",
        ],
        KnowledgeLevel.EXPERT: [
            r"\b(research|innovation|cutting-edge|advanced)\b",
            r"\b(teach|mentor|lead|architect)\b",
        ],
    }

    # Terminology sophistication levels
    TERMINOLOGY_LEVELS = {
        "basic": 0.2,
        "intermediate": 0.5,
        "advanced": 0.8,
        "expert": 1.0,
    }

    def __init__(self):
        """Initialize the knowledge depth assessor."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_dir) / "depth"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.depth_cache: Dict[str, Dict[str, DomainDepthProfile]] = {}

        logger.info("Knowledge depth assessor initialized at %s", self.storage_path)

    def assess_knowledge_depth(
        self, user_id: str, domain: str, interaction_data: InteractionData
    ) -> List[KnowledgeAssessment]:
        """
        Assess knowledge depth from interaction data.

        Requirement 24.2: Assess user knowledge depth in domains

        Args:
            user_id: User identifier
            domain: Domain name
            interaction_data: Interaction data to analyze

        Returns:
            List[KnowledgeAssessment]: Knowledge assessments
        """
        logger.info(
            "Assessing knowledge depth for user %s in domain %s", user_id, domain
        )

        try:
            # Get or create depth profile
            profile = self.get_depth_profile(user_id, domain)

            # Extract message text
            message_text = ""
            if hasattr(interaction_data.conversation_context, "message_text"):
                message_text = interaction_data.conversation_context.message_text

            if not message_text:
                return []

            # Extract topics from the message
            topics = self._extract_topics(message_text, domain)

            assessments = []
            for topic in topics:
                # Assess knowledge level for this topic
                assessment = self._assess_topic_knowledge(
                    user_id, domain, topic, message_text, interaction_data
                )

                if assessment:
                    # Update or create assessment in profile
                    existing_assessment = profile.get_assessment(topic)
                    if existing_assessment:
                        # Update existing assessment
                        self._update_assessment(existing_assessment, assessment)
                        assessments.append(existing_assessment)
                    else:
                        # Add new assessment
                        profile.add_assessment(assessment)
                        assessments.append(assessment)

            # Save updated profile
            self.save_depth_profile(profile)

            logger.info("Completed %d knowledge assessments", len(assessments))
            return assessments

        except Exception as e:
            logger.error("Error assessing knowledge depth for user %s: %s", user_id, e)
            return []

    def get_depth_profile(self, user_id: str, domain: str) -> DomainDepthProfile:
        """
        Get depth profile for a user in a domain.

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            DomainDepthProfile: Depth profile
        """
        # Check cache first
        if user_id in self.depth_cache and domain in self.depth_cache[user_id]:
            return self.depth_cache[user_id][domain]

        # Load from storage
        profile_path = self.storage_path / f"{user_id}_{domain}_depth.json"

        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to dataclass
                profile = DomainDepthProfile(
                    user_id=data["user_id"],
                    domain_name=data["domain_name"],
                    overall_level=KnowledgeLevel(data.get("overall_level", "surface")),
                    domain_confidence=data.get("domain_confidence", 0.0),
                    breadth_score=data.get("breadth_score", 0.0),
                    depth_consistency=data.get("depth_consistency", 0.0),
                    learning_velocity=data.get("learning_velocity", 0.0),
                    preferred_depth=KnowledgeLevel(
                        data.get("preferred_depth", "working")
                    ),
                    learning_style_indicators=data.get("learning_style_indicators", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    total_assessments=data.get("total_assessments", 0),
                )

                # Load assessments
                for assessment_data in data.get("topic_assessments", []):
                    assessment = KnowledgeAssessment(
                        assessment_id=assessment_data["assessment_id"],
                        user_id=assessment_data["user_id"],
                        domain=assessment_data["domain"],
                        topic=assessment_data["topic"],
                        knowledge_level=KnowledgeLevel(
                            assessment_data["knowledge_level"]
                        ),
                        confidence_score=assessment_data.get("confidence_score", 0.0),
                        depth_score=assessment_data.get("depth_score", 0.0),
                        assessment_method=AssessmentMethod(
                            assessment_data["assessment_method"]
                        ),
                        evidence=assessment_data.get("evidence", []),
                        indicators=assessment_data.get("indicators", {}),
                        previous_level=(
                            KnowledgeLevel(assessment_data["previous_level"])
                            if assessment_data.get("previous_level")
                            else None
                        ),
                        progression_rate=assessment_data.get("progression_rate", 0.0),
                        assessed_at=datetime.fromisoformat(
                            assessment_data["assessed_at"]
                        ),
                        interaction_count=assessment_data.get("interaction_count", 0),
                    )
                    profile.add_assessment(assessment)

                # Cache the profile
                if user_id not in self.depth_cache:
                    self.depth_cache[user_id] = {}
                self.depth_cache[user_id][domain] = profile

                return profile

            except Exception as e:
                logger.error(
                    "Error loading depth profile for %s/%s: %s", user_id, domain, e
                )

        # Create new profile if none exists
        profile = DomainDepthProfile(user_id=user_id, domain_name=domain)

        # Cache the profile
        if user_id not in self.depth_cache:
            self.depth_cache[user_id] = {}
        self.depth_cache[user_id][domain] = profile

        return profile

    def get_appropriate_content_level(
        self, user_id: str, domain: str, topic: str
    ) -> KnowledgeLevel:
        """
        Get appropriate content level for a user and topic.

        Requirement 24.4: Provide depth-appropriate content

        Args:
            user_id: User identifier
            domain: Domain name
            topic: Topic name

        Returns:
            KnowledgeLevel: Appropriate content level
        """
        profile = self.get_depth_profile(user_id, domain)
        assessment = profile.get_assessment(topic)

        if assessment:
            # Provide content slightly above current level to encourage growth
            current_level = assessment.knowledge_level

            if current_level == KnowledgeLevel.SURFACE:
                return KnowledgeLevel.FUNCTIONAL
            elif current_level == KnowledgeLevel.FUNCTIONAL:
                return KnowledgeLevel.WORKING
            elif current_level == KnowledgeLevel.WORKING:
                return (
                    KnowledgeLevel.DEEP
                    if assessment.confidence_score > 0.7
                    else KnowledgeLevel.WORKING
                )
            else:
                return current_level  # Already at high level
        else:
            # No assessment available, start with functional level
            return KnowledgeLevel.FUNCTIONAL

    def identify_learning_opportunities(
        self, user_id: str, domain: str
    ) -> List[Dict[str, Any]]:
        """
        Identify learning opportunities based on depth assessment.

        Requirement 24.3: Identify learning opportunities

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            List[Dict[str, Any]]: Learning opportunities
        """
        profile = self.get_depth_profile(user_id, domain)
        opportunities = []

        # Get recommendations from profile
        recommendations = profile.get_learning_recommendations()
        opportunities.extend(recommendations)

        # Find topics with stagnant progress
        for topic, assessment in profile.topic_assessments.items():
            if assessment.progression_rate == 0.0 and assessment.interaction_count > 5:
                opportunities.append(
                    {
                        "type": "stagnant_progress",
                        "topic": topic,
                        "suggestion": "Try different learning approaches or seek help",
                        "priority": "medium",
                    }
                )

        # Suggest review for topics with declining confidence
        for topic, assessment in profile.topic_assessments.items():
            if assessment.confidence_score < 0.5 and assessment.interaction_count > 3:
                opportunities.append(
                    {
                        "type": "review_needed",
                        "topic": topic,
                        "suggestion": "Review fundamentals to strengthen understanding",
                        "priority": "high",
                    }
                )

        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        opportunities.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 1), reverse=True
        )

        logger.info(
            "Identified %d learning opportunities for %s", len(opportunities), user_id
        )
        return opportunities

    def save_depth_profile(self, profile: DomainDepthProfile):
        """
        Save depth profile to storage.

        Args:
            profile: Depth profile to save
        """
        # Update cache
        if profile.user_id not in self.depth_cache:
            self.depth_cache[profile.user_id] = {}
        self.depth_cache[profile.user_id][profile.domain_name] = profile

        # Save to storage
        profile_path = (
            self.storage_path / f"{profile.user_id}_{profile.domain_name}_depth.json"
        )

        try:
            # Convert assessments to serializable format
            assessments_data = []
            for assessment in profile.topic_assessments.values():
                assessments_data.append(
                    {
                        "assessment_id": assessment.assessment_id,
                        "user_id": assessment.user_id,
                        "domain": assessment.domain,
                        "topic": assessment.topic,
                        "knowledge_level": assessment.knowledge_level.value,
                        "confidence_score": assessment.confidence_score,
                        "depth_score": assessment.depth_score,
                        "assessment_method": assessment.assessment_method.value,
                        "evidence": assessment.evidence,
                        "indicators": assessment.indicators,
                        "previous_level": (
                            assessment.previous_level.value
                            if assessment.previous_level
                            else None
                        ),
                        "progression_rate": assessment.progression_rate,
                        "assessed_at": assessment.assessed_at.isoformat(),
                        "interaction_count": assessment.interaction_count,
                    }
                )

            data = {
                "user_id": profile.user_id,
                "domain_name": profile.domain_name,
                "overall_level": profile.overall_level.value,
                "domain_confidence": profile.domain_confidence,
                "breadth_score": profile.breadth_score,
                "depth_consistency": profile.depth_consistency,
                "learning_velocity": profile.learning_velocity,
                "preferred_depth": profile.preferred_depth.value,
                "learning_style_indicators": profile.learning_style_indicators,
                "created_at": profile.created_at.isoformat(),
                "last_updated": profile.last_updated.isoformat(),
                "total_assessments": profile.total_assessments,
                "topic_assessments": assessments_data,
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                "Error saving depth profile for %s/%s: %s",
                profile.user_id,
                profile.domain_name,
                e,
            )

    def _extract_topics(self, text: str, domain: str) -> List[str]:
        """Extract topics from text."""
        # Simple topic extraction - in practice, you'd use NLP libraries
        import re

        topics = []

        # Extract capitalized terms that might be topics
        capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        topics.extend(capitalized_terms)

        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', text)
        topics.extend(quoted_terms)

        # Extract technical terms (simplified)
        technical_terms = re.findall(r"\b\w+\(\)\b|\b\w+\.\w+\b", text)
        topics.extend(technical_terms)

        # Remove duplicates and filter
        unique_topics = []
        seen = set()
        for topic in topics:
            topic_clean = topic.strip().lower()
            if (
                len(topic_clean) > 2
                and topic_clean not in seen
                and not topic_clean.isdigit()
            ):
                seen.add(topic_clean)
                unique_topics.append(topic.strip())

        return unique_topics[:5]  # Limit to 5 topics per interaction

    def _assess_topic_knowledge(
        self,
        user_id: str,
        domain: str,
        topic: str,
        message_text: str,
        interaction_data: InteractionData,
    ) -> Optional[KnowledgeAssessment]:
        """Assess knowledge level for a specific topic."""
        import re
        import uuid

        # Analyze complexity indicators
        complexity_scores = {}
        for level, patterns in self.COMPLEXITY_INDICATORS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_text, re.IGNORECASE))
                score += matches
            complexity_scores[level] = score

        # Determine most likely level
        if not any(complexity_scores.values()):
            # No clear indicators, default to functional
            assessed_level = KnowledgeLevel.FUNCTIONAL
            confidence = 0.3
        else:
            # Find level with highest score
            max_level = max(
                complexity_scores.keys(), key=lambda k: complexity_scores[k]
            )
            assessed_level = max_level

            # Calculate confidence based on score distribution
            total_score = sum(complexity_scores.values())
            max_score = complexity_scores[max_level]
            confidence = min(max_score / max(total_score, 1), 1.0)

        # Assess terminology usage
        terminology_score = self._assess_terminology_usage(message_text, domain)

        # Combine assessments
        depth_score = (confidence + terminology_score) / 2.0

        # Adjust level based on depth score
        if depth_score < 0.3 and assessed_level != KnowledgeLevel.SURFACE:
            assessed_level = KnowledgeLevel.SURFACE
        elif depth_score > 0.8 and assessed_level == KnowledgeLevel.SURFACE:
            assessed_level = KnowledgeLevel.FUNCTIONAL

        return KnowledgeAssessment(
            assessment_id=str(uuid.uuid4()),
            user_id=user_id,
            domain=domain,
            topic=topic,
            knowledge_level=assessed_level,
            confidence_score=confidence,
            depth_score=depth_score,
            assessment_method=AssessmentMethod.QUESTION_COMPLEXITY,
            evidence=[f"Complexity indicators: {complexity_scores}"],
            indicators={
                "complexity_score": confidence,
                "terminology_score": terminology_score,
                "combined_score": depth_score,
            },
            interaction_count=1,
        )

    def _assess_terminology_usage(self, text: str, domain: str) -> float:
        """Assess sophistication of terminology usage."""
        # This is a simplified implementation
        # In practice, you'd have domain-specific terminology dictionaries

        import re

        # Count technical terms
        technical_patterns = [
            r"\b\w+\(\)\b",  # Function calls
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\w+\.\w+\b",  # Dot notation
            r"\b\w+_\w+\b",  # Snake case
        ]

        technical_count = 0
        total_words = len(text.split())

        for pattern in technical_patterns:
            technical_count += len(re.findall(pattern, text))

        if total_words == 0:
            return 0.0

        # Calculate terminology density
        terminology_density = technical_count / total_words

        # Normalize to 0-1 scale
        return min(terminology_density * 5, 1.0)  # Scale factor of 5

    def _update_assessment(
        self, existing: KnowledgeAssessment, new: KnowledgeAssessment
    ):
        """Update existing assessment with new information."""
        # Update progression if level changed
        if existing.knowledge_level != new.knowledge_level:
            existing.update_progression(new.knowledge_level)

        # Update confidence (weighted average)
        weight = 0.3  # Weight for new assessment
        existing.confidence_score = (
            1 - weight
        ) * existing.confidence_score + weight * new.confidence_score

        # Update depth score
        existing.depth_score = (
            1 - weight
        ) * existing.depth_score + weight * new.depth_score

        # Add new evidence
        existing.evidence.extend(new.evidence)
        if len(existing.evidence) > 10:  # Keep only recent evidence
            existing.evidence = existing.evidence[-10:]

        # Update indicators
        for key, value in new.indicators.items():
            if key in existing.indicators:
                existing.indicators[key] = (1 - weight) * existing.indicators[
                    key
                ] + weight * value
            else:
                existing.indicators[key] = value

        # Update metadata
        existing.interaction_count += 1
        existing.assessed_at = datetime.utcnow()


# Global depth assessor instance
_depth_assessor: Optional[KnowledgeDepthAssessor] = None


def get_depth_assessor() -> KnowledgeDepthAssessor:
    """
    Get the global knowledge depth assessor instance.

    Returns:
        KnowledgeDepthAssessor: Global depth assessor instance
    """
    global _depth_assessor
    if _depth_assessor is None:
        _depth_assessor = KnowledgeDepthAssessor()
    return _depth_assessor
