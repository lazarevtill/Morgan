"""
Adaptive Teaching Strategies for Morgan RAG.

Provides personalized teaching approaches based on user's knowledge level,
learning style, and progress patterns. Adapts explanations, examples,
and guidance to optimize learning outcomes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_settings
from ..emotional.models import InteractionData
from ..utils.logger import get_logger
from .depth import KnowledgeLevel, get_depth_assessor

logger = get_logger(__name__)


class TeachingStrategy(Enum):
    """Teaching strategy types."""

    SCAFFOLDING = "scaffolding"  # Gradual support reduction
    DISCOVERY = "discovery"  # Self-guided exploration
    DIRECT_INSTRUCTION = "direct"  # Clear, structured explanation
    SOCRATIC = "socratic"  # Question-based learning
    ANALOGICAL = "analogical"  # Learning through analogies
    EXPERIENTIAL = "experiential"  # Hands-on practice
    COLLABORATIVE = "collaborative"  # Learning through discussion


class LearningStyle(Enum):
    """Learning style preferences."""

    VISUAL = "visual"  # Diagrams, charts, visual aids
    AUDITORY = "auditory"  # Verbal explanations, discussions
    KINESTHETIC = "kinesthetic"  # Hands-on, practical activities
    READING = "reading"  # Text-based learning
    MULTIMODAL = "multimodal"  # Combination of styles


class ExplanationLevel(Enum):
    """Levels of explanation detail."""

    BRIEF = "brief"  # Concise, key points only
    STANDARD = "standard"  # Balanced detail
    DETAILED = "detailed"  # Comprehensive explanation
    EXHAUSTIVE = "exhaustive"  # Complete with examples


@dataclass
class TeachingApproach:
    """Specific teaching approach for a topic."""

    approach_id: str
    user_id: str
    domain: str
    topic: str

    # Strategy configuration
    primary_strategy: TeachingStrategy
    secondary_strategies: List[TeachingStrategy] = field(default_factory=list)
    explanation_level: ExplanationLevel = ExplanationLevel.STANDARD

    # Content adaptation
    use_analogies: bool = True
    include_examples: bool = True
    provide_practice: bool = True
    check_understanding: bool = True

    # Learning style adaptation
    preferred_style: LearningStyle = LearningStyle.MULTIMODAL
    visual_aids: bool = False
    step_by_step: bool = True
    interactive_elements: bool = True

    # Effectiveness tracking
    success_rate: float = 0.0  # 0.0 to 1.0
    user_satisfaction: float = 0.0  # 0.0 to 1.0
    learning_speed: float = 0.0  # Concepts per session

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0

    def update_effectiveness(self, success: bool, satisfaction: float):
        """Update effectiveness metrics."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

        # Update success rate (weighted average)
        weight = 0.2
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            new_success = 1.0 if success else 0.0
            self.success_rate = (1 - weight) * self.success_rate + weight * new_success

        # Update satisfaction (weighted average)
        if self.usage_count == 1:
            self.user_satisfaction = satisfaction
        else:
            self.user_satisfaction = (
                1 - weight
            ) * self.user_satisfaction + weight * satisfaction


@dataclass
class LearningProfile:
    """Complete learning profile for a user."""

    user_id: str

    # Learning preferences
    preferred_strategies: List[TeachingStrategy] = field(default_factory=list)
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    explanation_preference: ExplanationLevel = ExplanationLevel.STANDARD

    # Learning patterns
    optimal_session_length: int = 30  # minutes
    preferred_pace: str = "moderate"  # slow, moderate, fast
    needs_frequent_breaks: bool = False
    learns_better_with_examples: bool = True

    # Domain-specific approaches
    domain_approaches: Dict[str, Dict[str, TeachingApproach]] = field(
        default_factory=dict
    )

    # Learning analytics
    overall_success_rate: float = 0.0
    average_satisfaction: float = 0.0
    learning_velocity: float = 0.0  # concepts per week

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_approach(self, approach: TeachingApproach):
        """Add a teaching approach for a domain/topic."""
        if approach.domain not in self.domain_approaches:
            self.domain_approaches[approach.domain] = {}

        self.domain_approaches[approach.domain][approach.topic] = approach
        self.last_updated = datetime.utcnow()

    def get_approach(self, domain: str, topic: str) -> Optional[TeachingApproach]:
        """Get teaching approach for a domain/topic."""
        return self.domain_approaches.get(domain, {}).get(topic)

    def update_learning_analytics(self):
        """Update overall learning analytics."""
        all_approaches = []
        for domain_approaches in self.domain_approaches.values():
            all_approaches.extend(domain_approaches.values())

        if all_approaches:
            # Calculate overall success rate
            self.overall_success_rate = sum(
                approach.success_rate for approach in all_approaches
            ) / len(all_approaches)

            # Calculate average satisfaction
            self.average_satisfaction = sum(
                approach.user_satisfaction for approach in all_approaches
            ) / len(all_approaches)

            # Calculate learning velocity
            total_speed = sum(approach.learning_speed for approach in all_approaches)
            self.learning_velocity = total_speed / max(len(all_approaches), 1)


class AdaptiveTeachingEngine:
    """
    Provides adaptive teaching strategies based on user profiles.

    Analyzes user learning patterns, knowledge levels, and preferences
    to select optimal teaching approaches and adapt content delivery.

    Requirements addressed: 24.2, 24.3, 24.4
    """

    # Strategy selection rules based on knowledge level
    STRATEGY_BY_LEVEL = {
        KnowledgeLevel.SURFACE: [
            TeachingStrategy.DIRECT_INSTRUCTION,
            TeachingStrategy.ANALOGICAL,
            TeachingStrategy.SCAFFOLDING,
        ],
        KnowledgeLevel.FUNCTIONAL: [
            TeachingStrategy.SCAFFOLDING,
            TeachingStrategy.EXPERIENTIAL,
            TeachingStrategy.DIRECT_INSTRUCTION,
        ],
        KnowledgeLevel.WORKING: [
            TeachingStrategy.DISCOVERY,
            TeachingStrategy.EXPERIENTIAL,
            TeachingStrategy.SOCRATIC,
        ],
        KnowledgeLevel.DEEP: [
            TeachingStrategy.SOCRATIC,
            TeachingStrategy.COLLABORATIVE,
            TeachingStrategy.DISCOVERY,
        ],
        KnowledgeLevel.EXPERT: [
            TeachingStrategy.COLLABORATIVE,
            TeachingStrategy.SOCRATIC,
            TeachingStrategy.DISCOVERY,
        ],
    }

    # Content adaptation patterns
    CONTENT_PATTERNS = {
        TeachingStrategy.SCAFFOLDING: {
            "structure": "step_by_step",
            "support": "high_initially",
            "examples": "many_simple",
            "practice": "guided",
        },
        TeachingStrategy.DISCOVERY: {
            "structure": "open_ended",
            "support": "minimal",
            "examples": "few_complex",
            "practice": "self_directed",
        },
        TeachingStrategy.DIRECT_INSTRUCTION: {
            "structure": "linear",
            "support": "consistent",
            "examples": "clear_relevant",
            "practice": "structured",
        },
        TeachingStrategy.SOCRATIC: {
            "structure": "question_based",
            "support": "questioning",
            "examples": "thought_provoking",
            "practice": "reflective",
        },
        TeachingStrategy.ANALOGICAL: {
            "structure": "comparison_based",
            "support": "metaphorical",
            "examples": "analogous",
            "practice": "pattern_recognition",
        },
    }

    def __init__(self):
        """Initialize the adaptive teaching engine."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_dir) / "teaching"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.profile_cache: Dict[str, LearningProfile] = {}

        # Get depth assessor for knowledge level information
        self.depth_assessor = get_depth_assessor()

        logger.info("Adaptive teaching engine initialized at %s", self.storage_path)

    def select_teaching_strategy(
        self, user_id: str, domain: str, topic: str, interaction_data: InteractionData
    ) -> TeachingApproach:
        """
        Select optimal teaching strategy for a user and topic.

        Requirement 24.4: Provide adaptive teaching strategies

        Args:
            user_id: User identifier
            domain: Domain name
            topic: Topic name
            interaction_data: Current interaction context

        Returns:
            TeachingApproach: Selected teaching approach
        """
        logger.info("Selecting teaching strategy for user %s, topic %s", user_id, topic)

        try:
            # Get user's learning profile
            profile = self.get_learning_profile(user_id)

            # Get existing approach if available
            existing_approach = profile.get_approach(domain, topic)
            if existing_approach and existing_approach.success_rate > 0.7:
                # Use existing successful approach
                logger.debug("Using existing successful approach")
                return existing_approach

            # Get user's knowledge level for this topic
            depth_profile = self.depth_assessor.get_depth_profile(user_id, domain)
            topic_assessment = depth_profile.get_assessment(topic)

            knowledge_level = (
                topic_assessment.knowledge_level
                if topic_assessment
                else KnowledgeLevel.FUNCTIONAL
            )

            # Select primary strategy based on knowledge level
            candidate_strategies = self.STRATEGY_BY_LEVEL.get(
                knowledge_level, [TeachingStrategy.DIRECT_INSTRUCTION]
            )

            # Consider user preferences
            if profile.preferred_strategies:
                # Prioritize user's preferred strategies
                preferred_candidates = [
                    strategy
                    for strategy in candidate_strategies
                    if strategy in profile.preferred_strategies
                ]
                if preferred_candidates:
                    candidate_strategies = preferred_candidates

            # Select primary strategy (first candidate for now)
            primary_strategy = candidate_strategies[0]

            # Select secondary strategies
            secondary_strategies = candidate_strategies[1:3]  # Up to 2 secondary

            # Determine explanation level
            explanation_level = self._determine_explanation_level(
                knowledge_level, profile, interaction_data
            )

            # Create teaching approach
            approach = TeachingApproach(
                approach_id=f"{user_id}_{domain}_{topic}_{datetime.utcnow().isoformat()}",
                user_id=user_id,
                domain=domain,
                topic=topic,
                primary_strategy=primary_strategy,
                secondary_strategies=secondary_strategies,
                explanation_level=explanation_level,
                preferred_style=profile.learning_style,
                use_analogies=profile.learns_better_with_examples,
                include_examples=profile.learns_better_with_examples,
                step_by_step=(
                    knowledge_level
                    in [KnowledgeLevel.SURFACE, KnowledgeLevel.FUNCTIONAL]
                ),
            )

            # Add to profile
            profile.add_approach(approach)

            # Save updated profile
            self.save_learning_profile(profile)

            logger.info("Selected %s strategy for %s", primary_strategy.value, topic)
            return approach

        except Exception as e:
            logger.error(
                "Error selecting teaching strategy for user %s: %s", user_id, e
            )

            # Return default approach
            return TeachingApproach(
                approach_id=f"default_{user_id}_{topic}",
                user_id=user_id,
                domain=domain,
                topic=topic,
                primary_strategy=TeachingStrategy.DIRECT_INSTRUCTION,
            )

    def adapt_content(
        self, content: str, approach: TeachingApproach, user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt content based on teaching approach.

        Requirement 24.3: Adapt content delivery to learning preferences

        Args:
            content: Original content
            approach: Teaching approach to use
            user_context: Additional user context

        Returns:
            Dict[str, Any]: Adapted content with teaching elements
        """
        logger.debug(
            "Adapting content using %s strategy", approach.primary_strategy.value
        )

        adapted_content = {
            "original_content": content,
            "strategy": approach.primary_strategy.value,
            "explanation_level": approach.explanation_level.value,
        }

        # Get content pattern for the strategy
        pattern = self.CONTENT_PATTERNS.get(
            approach.primary_strategy,
            self.CONTENT_PATTERNS[TeachingStrategy.DIRECT_INSTRUCTION],
        )

        # Apply strategy-specific adaptations
        if approach.primary_strategy == TeachingStrategy.SCAFFOLDING:
            adapted_content.update(self._apply_scaffolding(content, approach))

        elif approach.primary_strategy == TeachingStrategy.DISCOVERY:
            adapted_content.update(self._apply_discovery(content, approach))

        elif approach.primary_strategy == TeachingStrategy.SOCRATIC:
            adapted_content.update(self._apply_socratic(content, approach))

        elif approach.primary_strategy == TeachingStrategy.ANALOGICAL:
            adapted_content.update(self._apply_analogical(content, approach))

        else:  # DIRECT_INSTRUCTION or others
            adapted_content.update(self._apply_direct_instruction(content, approach))

        # Apply learning style adaptations
        if approach.preferred_style == LearningStyle.VISUAL:
            adapted_content["visual_elements"] = self._suggest_visual_elements(content)

        elif approach.preferred_style == LearningStyle.KINESTHETIC:
            adapted_content["hands_on_activities"] = self._suggest_activities(content)

        # Add examples if requested
        if approach.include_examples:
            adapted_content["examples"] = self._generate_examples(
                content, approach.domain
            )

        # Add practice opportunities if requested
        if approach.provide_practice:
            adapted_content["practice_exercises"] = self._generate_practice(
                content, approach
            )

        # Add understanding checks if requested
        if approach.check_understanding:
            adapted_content["understanding_checks"] = self._generate_checks(content)

        return adapted_content

    def get_learning_profile(self, user_id: str) -> LearningProfile:
        """
        Get learning profile for a user.

        Args:
            user_id: User identifier

        Returns:
            LearningProfile: User's learning profile
        """
        # Check cache first
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]

        # Load from storage
        profile_path = self.storage_path / f"{user_id}_learning_profile.json"

        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to dataclass
                profile = LearningProfile(
                    user_id=data["user_id"],
                    preferred_strategies=[
                        TeachingStrategy(s)
                        for s in data.get("preferred_strategies", [])
                    ],
                    learning_style=LearningStyle(
                        data.get("learning_style", "multimodal")
                    ),
                    explanation_preference=ExplanationLevel(
                        data.get("explanation_preference", "standard")
                    ),
                    optimal_session_length=data.get("optimal_session_length", 30),
                    preferred_pace=data.get("preferred_pace", "moderate"),
                    needs_frequent_breaks=data.get("needs_frequent_breaks", False),
                    learns_better_with_examples=data.get(
                        "learns_better_with_examples", True
                    ),
                    overall_success_rate=data.get("overall_success_rate", 0.0),
                    average_satisfaction=data.get("average_satisfaction", 0.0),
                    learning_velocity=data.get("learning_velocity", 0.0),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                )

                # Load domain approaches
                for domain, topics in data.get("domain_approaches", {}).items():
                    for topic, approach_data in topics.items():
                        approach = TeachingApproach(
                            approach_id=approach_data["approach_id"],
                            user_id=approach_data["user_id"],
                            domain=approach_data["domain"],
                            topic=approach_data["topic"],
                            primary_strategy=TeachingStrategy(
                                approach_data["primary_strategy"]
                            ),
                            secondary_strategies=[
                                TeachingStrategy(s)
                                for s in approach_data.get("secondary_strategies", [])
                            ],
                            explanation_level=ExplanationLevel(
                                approach_data.get("explanation_level", "standard")
                            ),
                            use_analogies=approach_data.get("use_analogies", True),
                            include_examples=approach_data.get(
                                "include_examples", True
                            ),
                            provide_practice=approach_data.get(
                                "provide_practice", True
                            ),
                            check_understanding=approach_data.get(
                                "check_understanding", True
                            ),
                            preferred_style=LearningStyle(
                                approach_data.get("preferred_style", "multimodal")
                            ),
                            visual_aids=approach_data.get("visual_aids", False),
                            step_by_step=approach_data.get("step_by_step", True),
                            interactive_elements=approach_data.get(
                                "interactive_elements", True
                            ),
                            success_rate=approach_data.get("success_rate", 0.0),
                            user_satisfaction=approach_data.get(
                                "user_satisfaction", 0.0
                            ),
                            learning_speed=approach_data.get("learning_speed", 0.0),
                            created_at=datetime.fromisoformat(
                                approach_data["created_at"]
                            ),
                            last_used=datetime.fromisoformat(
                                approach_data["last_used"]
                            ),
                            usage_count=approach_data.get("usage_count", 0),
                        )
                        profile.add_approach(approach)

                # Cache the profile
                self.profile_cache[user_id] = profile
                return profile

            except Exception as e:
                logger.error("Error loading learning profile for %s: %s", user_id, e)

        # Create new profile if none exists
        profile = LearningProfile(user_id=user_id)

        # Cache the profile
        self.profile_cache[user_id] = profile
        return profile

    def update_approach_effectiveness(
        self, user_id: str, domain: str, topic: str, success: bool, satisfaction: float
    ):
        """
        Update teaching approach effectiveness based on feedback.

        Args:
            user_id: User identifier
            domain: Domain name
            topic: Topic name
            success: Whether the teaching was successful
            satisfaction: User satisfaction score (0.0 to 1.0)
        """
        profile = self.get_learning_profile(user_id)
        approach = profile.get_approach(domain, topic)

        if approach:
            approach.update_effectiveness(success, satisfaction)
            profile.update_learning_analytics()
            self.save_learning_profile(profile)

            logger.info(
                "Updated effectiveness for %s/%s: success=%s, satisfaction=%.2f",
                domain,
                topic,
                success,
                satisfaction,
            )

    def save_learning_profile(self, profile: LearningProfile):
        """
        Save learning profile to storage.

        Args:
            profile: Learning profile to save
        """
        # Update cache
        self.profile_cache[profile.user_id] = profile

        # Save to storage
        profile_path = self.storage_path / f"{profile.user_id}_learning_profile.json"

        try:
            # Convert domain approaches to serializable format
            domain_approaches_data = {}
            for domain, topics in profile.domain_approaches.items():
                domain_approaches_data[domain] = {}
                for topic, approach in topics.items():
                    domain_approaches_data[domain][topic] = {
                        "approach_id": approach.approach_id,
                        "user_id": approach.user_id,
                        "domain": approach.domain,
                        "topic": approach.topic,
                        "primary_strategy": approach.primary_strategy.value,
                        "secondary_strategies": [
                            s.value for s in approach.secondary_strategies
                        ],
                        "explanation_level": approach.explanation_level.value,
                        "use_analogies": approach.use_analogies,
                        "include_examples": approach.include_examples,
                        "provide_practice": approach.provide_practice,
                        "check_understanding": approach.check_understanding,
                        "preferred_style": approach.preferred_style.value,
                        "visual_aids": approach.visual_aids,
                        "step_by_step": approach.step_by_step,
                        "interactive_elements": approach.interactive_elements,
                        "success_rate": approach.success_rate,
                        "user_satisfaction": approach.user_satisfaction,
                        "learning_speed": approach.learning_speed,
                        "created_at": approach.created_at.isoformat(),
                        "last_used": approach.last_used.isoformat(),
                        "usage_count": approach.usage_count,
                    }

            data = {
                "user_id": profile.user_id,
                "preferred_strategies": [s.value for s in profile.preferred_strategies],
                "learning_style": profile.learning_style.value,
                "explanation_preference": profile.explanation_preference.value,
                "optimal_session_length": profile.optimal_session_length,
                "preferred_pace": profile.preferred_pace,
                "needs_frequent_breaks": profile.needs_frequent_breaks,
                "learns_better_with_examples": profile.learns_better_with_examples,
                "domain_approaches": domain_approaches_data,
                "overall_success_rate": profile.overall_success_rate,
                "average_satisfaction": profile.average_satisfaction,
                "learning_velocity": profile.learning_velocity,
                "created_at": profile.created_at.isoformat(),
                "last_updated": profile.last_updated.isoformat(),
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error("Error saving learning profile for %s: %s", profile.user_id, e)

    def _determine_explanation_level(
        self,
        knowledge_level: KnowledgeLevel,
        profile: LearningProfile,
        interaction_data: InteractionData,
    ) -> ExplanationLevel:
        """Determine appropriate explanation level."""
        # Start with user preference
        base_level = profile.explanation_preference

        # Adjust based on knowledge level
        if knowledge_level == KnowledgeLevel.SURFACE:
            if base_level == ExplanationLevel.BRIEF:
                return ExplanationLevel.STANDARD
            elif base_level == ExplanationLevel.STANDARD:
                return ExplanationLevel.DETAILED

        elif knowledge_level in [KnowledgeLevel.DEEP, KnowledgeLevel.EXPERT]:
            if base_level == ExplanationLevel.DETAILED:
                return ExplanationLevel.STANDARD
            elif base_level == ExplanationLevel.EXHAUSTIVE:
                return ExplanationLevel.DETAILED

        return base_level

    def _apply_scaffolding(
        self, content: str, approach: TeachingApproach
    ) -> Dict[str, Any]:
        """Apply scaffolding teaching strategy."""
        return {
            "structure": "step_by_step",
            "guidance": "Start with simple concepts and gradually increase complexity",
            "support_level": "high_initially",
            "progression": "guided_practice_to_independent",
            "checkpoints": "frequent_understanding_checks",
        }

    def _apply_discovery(
        self, content: str, approach: TeachingApproach
    ) -> Dict[str, Any]:
        """Apply discovery learning strategy."""
        return {
            "structure": "open_ended_exploration",
            "guidance": "Provide minimal initial guidance, let user explore",
            "questions": "thought_provoking_questions",
            "resources": "multiple_paths_to_explore",
            "reflection": "encourage_self_reflection",
        }

    def _apply_socratic(
        self, content: str, approach: TeachingApproach
    ) -> Dict[str, Any]:
        """Apply Socratic method strategy."""
        return {
            "structure": "question_based_dialogue",
            "questions": self._generate_socratic_questions(content),
            "guidance": "guide_through_questioning",
            "reflection": "encourage_critical_thinking",
            "discovery": "help_user_reach_conclusions",
        }

    def _apply_analogical(
        self, content: str, approach: TeachingApproach
    ) -> Dict[str, Any]:
        """Apply analogical learning strategy."""
        return {
            "structure": "comparison_based",
            "analogies": self._generate_analogies(content, approach.domain),
            "mapping": "connect_familiar_to_unfamiliar",
            "examples": "parallel_examples",
            "transfer": "help_apply_patterns",
        }

    def _apply_direct_instruction(
        self, content: str, approach: TeachingApproach
    ) -> Dict[str, Any]:
        """Apply direct instruction strategy."""
        return {
            "structure": "clear_linear_progression",
            "explanation": "explicit_clear_explanations",
            "examples": "concrete_relevant_examples",
            "practice": "guided_then_independent",
            "feedback": "immediate_corrective_feedback",
        }

    def _generate_socratic_questions(self, content: str) -> List[str]:
        """Generate Socratic questions for content."""
        # Simplified question generation
        questions = [
            "What do you think this means?",
            "How does this relate to what you already know?",
            "What would happen if...?",
            "Why do you think this is important?",
            "Can you think of an example?",
        ]
        return questions[:3]  # Return first 3 questions

    def _generate_analogies(self, content: str, domain: str) -> List[str]:
        """Generate analogies for content."""
        # Simplified analogy generation
        analogies = [
            "Think of this like...",
            "This is similar to how...",
            "You can compare this to...",
        ]
        return analogies[:2]  # Return first 2 analogies

    def _suggest_visual_elements(self, content: str) -> List[str]:
        """Suggest visual elements for content."""
        return [
            "Create a diagram showing the relationships",
            "Use flowcharts to show the process",
            "Add visual examples or screenshots",
        ]

    def _suggest_activities(self, content: str) -> List[str]:
        """Suggest hands-on activities."""
        return [
            "Try implementing this yourself",
            "Practice with a real example",
            "Experiment with different approaches",
        ]

    def _generate_examples(self, content: str, domain: str) -> List[str]:
        """Generate examples for content."""
        return [
            "Here's a practical example...",
            "Consider this scenario...",
            "For instance...",
        ]

    def _generate_practice(self, content: str, approach: TeachingApproach) -> List[str]:
        """Generate practice exercises."""
        return [
            "Try applying this concept to...",
            "Practice by working through...",
            "Test your understanding with...",
        ]

    def _generate_checks(self, content: str) -> List[str]:
        """Generate understanding checks."""
        return [
            "Can you explain this in your own words?",
            "What questions do you have so far?",
            "How confident do you feel about this concept?",
        ]


# Global teaching engine instance
_teaching_engine: Optional[AdaptiveTeachingEngine] = None


def get_teaching_engine() -> AdaptiveTeachingEngine:
    """
    Get the global adaptive teaching engine instance.

    Returns:
        AdaptiveTeachingEngine: Global teaching engine instance
    """
    global _teaching_engine
    if _teaching_engine is None:
        _teaching_engine = AdaptiveTeachingEngine()
    return _teaching_engine
