"""
Domain Context Understanding for Morgan RAG.

Builds and maintains contextual understanding within specific domains,
tracking relationships between concepts, identifying context patterns,
and providing domain-aware context for better assistance.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..intelligence.core.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ContextType(Enum):
    """Types of domain context."""

    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    FACTUAL = "factual"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"


class ContextRelevance(Enum):
    """Relevance levels for context information."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContextElement:
    """Individual element of domain context."""

    element_id: str
    content: str
    context_type: ContextType
    relevance: ContextRelevance
    domain: str

    # Relationships
    related_elements: List[str] = field(default_factory=list)
    prerequisite_elements: List[str] = field(default_factory=list)
    dependent_elements: List[str] = field(default_factory=list)

    # Metadata
    confidence_score: float = 0.0  # 0.0 to 1.0
    usage_frequency: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Context patterns
    common_patterns: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)

    def update_access(self):
        """Update access statistics."""
        self.usage_frequency += 1
        self.last_accessed = datetime.now(timezone.utc)

        # Increase confidence with usage
        self.confidence_score = min(self.confidence_score + 0.05, 1.0)


@dataclass
class DomainContext:
    """Complete contextual understanding for a domain."""

    domain_name: str
    user_id: str

    # Context elements
    elements: Dict[str, ContextElement] = field(default_factory=dict)

    # Context relationships
    concept_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    workflow_patterns: List[List[str]] = field(default_factory=list)
    common_sequences: List[List[str]] = field(default_factory=list)

    # Context statistics
    total_elements: int = 0
    active_elements: int = 0
    context_depth: float = 0.0  # Average relationship depth

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_element(self, element: ContextElement):
        """Add a context element."""
        self.elements[element.element_id] = element
        self.total_elements = len(self.elements)
        self.last_updated = datetime.now(timezone.utc)

    def get_element(self, element_id: str) -> Optional[ContextElement]:
        """Get a context element by ID."""
        return self.elements.get(element_id)

    def get_related_elements(
        self, element_id: str, max_depth: int = 2
    ) -> List[ContextElement]:
        """Get related elements up to specified depth."""
        if element_id not in self.elements:
            return []

        related = []
        visited = set()
        queue = [(element_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            current_element = self.elements.get(current_id)

            if current_element and depth > 0:  # Don't include the starting element
                related.append(current_element)

            if current_element and depth < max_depth:
                # Add related elements to queue
                for related_id in current_element.related_elements:
                    if related_id not in visited:
                        queue.append((related_id, depth + 1))

        return related

    def get_context_path(self, from_element: str, to_element: str) -> List[str]:
        """Find context path between two elements."""
        if from_element not in self.elements or to_element not in self.elements:
            return []

        # Simple BFS to find path
        queue = [(from_element, [from_element])]
        visited = set()

        while queue:
            current_id, path = queue.pop(0)

            if current_id == to_element:
                return path

            if current_id in visited:
                continue

            visited.add(current_id)
            current_element = self.elements.get(current_id)

            if current_element:
                for related_id in current_element.related_elements:
                    if related_id not in visited:
                        queue.append((related_id, path + [related_id]))

        return []  # No path found


class DomainContextEngine:
    """
    Builds and maintains contextual understanding within domains.

    Analyzes interactions to identify context patterns, builds concept
    hierarchies, and provides domain-aware contextual assistance.

    Requirements addressed: 24.2, 24.3, 24.4
    """

    # Context extraction patterns
    CONTEXT_PATTERNS = {
        "conceptual": [
            r"(\w+) is (?:a |an |the )?(.+?)(?:\.|,|;)",
            r"(\w+) refers to (.+?)(?:\.|,|;)",
            r"the concept of (\w+) (.+?)(?:\.|,|;)",
        ],
        "procedural": [
            r"(?:first|then|next|finally),?\s+(.+?)(?:\.|,|;)",
            r"step \d+[:\-]\s*(.+?)(?:\.|,|;)",
            r"to (\w+),?\s+(.+?)(?:\.|,|;)",
        ],
        "relational": [
            r"(\w+) (?:depends on|requires|needs) (.+?)(?:\.|,|;)",
            r"(\w+) is (?:related to|connected to|part of) (.+?)(?:\.|,|;)",
            r"(\w+) (?:leads to|results in|causes) (.+?)(?:\.|,|;)",
        ],
    }

    # Workflow indicators
    WORKFLOW_INDICATORS = [
        "first",
        "then",
        "next",
        "after",
        "before",
        "finally",
        "step",
        "phase",
        "stage",
        "process",
        "workflow",
    ]

    def __init__(self):
        """Initialize the domain context engine."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.morgan_data_dir) / "context"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.context_cache: Dict[str, Dict[str, DomainContext]] = {}

        logger.info("Domain context engine initialized at %s", self.storage_path)

    def build_context(
        self, user_id: str, domain: str, interaction_data: InteractionData
    ) -> DomainContext:
        """
        Build domain context from interaction data.

        Requirement 24.3: Build contextual understanding within domains

        Args:
            user_id: User identifier
            domain: Domain name
            interaction_data: Interaction data to analyze

        Returns:
            DomainContext: Updated domain context
        """
        logger.info("Building context for user %s in domain %s", user_id, domain)

        try:
            # Get or create domain context
            context = self.get_domain_context(user_id, domain)

            # Extract message text
            message_text = ""
            if hasattr(interaction_data.conversation_context, "message_text"):
                message_text = interaction_data.conversation_context.message_text

            if not message_text:
                return context

            # Extract context elements from the message
            new_elements = self._extract_context_elements(message_text, domain, user_id)

            # Add new elements to context
            for element in new_elements:
                context.add_element(element)

            # Identify relationships between elements
            self._identify_relationships(context, new_elements)

            # Update workflow patterns
            self._update_workflow_patterns(context, message_text)

            # Save updated context
            self.save_domain_context(context)

            logger.debug("Added %d new context elements", len(new_elements))
            return context

        except Exception as e:
            logger.error("Error building context for user %s: %s", user_id, e)
            return self.get_domain_context(user_id, domain)

    def get_domain_context(self, user_id: str, domain: str) -> DomainContext:
        """
        Get domain context for a user.

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            DomainContext: Domain context
        """
        # Check cache first
        cache_key = f"{user_id}_{domain}"
        if user_id in self.context_cache and domain in self.context_cache[user_id]:
            return self.context_cache[user_id][domain]

        # Load from storage
        context_path = self.storage_path / f"{cache_key}_context.json"

        if context_path.exists():
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to dataclass
                context = DomainContext(
                    domain_name=data["domain_name"],
                    user_id=data["user_id"],
                    total_elements=data.get("total_elements", 0),
                    active_elements=data.get("active_elements", 0),
                    context_depth=data.get("context_depth", 0.0),
                    concept_hierarchy=data.get("concept_hierarchy", {}),
                    workflow_patterns=data.get("workflow_patterns", []),
                    common_sequences=data.get("common_sequences", []),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                )

                # Load context elements
                for element_data in data.get("elements", []):
                    element = ContextElement(
                        element_id=element_data["element_id"],
                        content=element_data["content"],
                        context_type=ContextType(element_data["context_type"]),
                        relevance=ContextRelevance(element_data["relevance"]),
                        domain=element_data["domain"],
                        related_elements=element_data.get("related_elements", []),
                        prerequisite_elements=element_data.get(
                            "prerequisite_elements", []
                        ),
                        dependent_elements=element_data.get("dependent_elements", []),
                        confidence_score=element_data.get("confidence_score", 0.0),
                        usage_frequency=element_data.get("usage_frequency", 0),
                        last_accessed=datetime.fromisoformat(
                            element_data["last_accessed"]
                        ),
                        created_at=datetime.fromisoformat(element_data["created_at"]),
                        common_patterns=element_data.get("common_patterns", []),
                        trigger_keywords=element_data.get("trigger_keywords", []),
                    )
                    context.add_element(element)

                # Cache the context
                if user_id not in self.context_cache:
                    self.context_cache[user_id] = {}
                self.context_cache[user_id][domain] = context

                return context

            except Exception as e:
                logger.error("Error loading context for %s/%s: %s", user_id, domain, e)

        # Create new context if none exists
        context = DomainContext(domain_name=domain, user_id=user_id)

        # Cache the context
        if user_id not in self.context_cache:
            self.context_cache[user_id] = {}
        self.context_cache[user_id][domain] = context

        return context

    def get_contextual_suggestions(
        self, user_id: str, domain: str, current_topic: str
    ) -> List[Dict[str, Any]]:
        """
        Get contextual suggestions based on current topic.

        Requirement 24.4: Provide context-aware assistance

        Args:
            user_id: User identifier
            domain: Domain name
            current_topic: Current topic or concept

        Returns:
            List[Dict[str, Any]]: Contextual suggestions
        """
        context = self.get_domain_context(user_id, domain)
        suggestions = []

        # Find elements related to current topic
        topic_elements = []
        for element in context.elements.values():
            if current_topic.lower() in element.content.lower() or any(
                keyword.lower() in current_topic.lower()
                for keyword in element.trigger_keywords
            ):
                topic_elements.append(element)

        # Generate suggestions based on related elements
        for element in topic_elements:
            # Get related elements
            related = context.get_related_elements(element.element_id, max_depth=2)

            for related_element in related:
                suggestions.append(
                    {
                        "type": "related_concept",
                        "title": f"Related: {related_element.content[:50]}...",
                        "content": related_element.content,
                        "relevance": related_element.relevance.value,
                        "confidence": related_element.confidence_score,
                    }
                )

            # Suggest prerequisites if available
            for prereq_id in element.prerequisite_elements:
                prereq = context.get_element(prereq_id)
                if prereq:
                    suggestions.append(
                        {
                            "type": "prerequisite",
                            "title": f"Prerequisite: {prereq.content[:50]}...",
                            "content": prereq.content,
                            "relevance": "high",
                            "confidence": prereq.confidence_score,
                        }
                    )

            # Suggest next steps if available
            for dependent_id in element.dependent_elements:
                dependent = context.get_element(dependent_id)
                if dependent:
                    suggestions.append(
                        {
                            "type": "next_step",
                            "title": f"Next: {dependent.content[:50]}...",
                            "content": dependent.content,
                            "relevance": "medium",
                            "confidence": dependent.confidence_score,
                        }
                    )

        # Sort by relevance and confidence
        suggestions.sort(
            key=lambda s: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(
                    s.get("relevance", "low"), 1
                ),
                s.get("confidence", 0.0),
            ),
            reverse=True,
        )

        # Remove duplicates and limit results
        seen_content = set()
        unique_suggestions = []
        for suggestion in suggestions:
            content_key = suggestion["content"][:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_suggestions.append(suggestion)

                if len(unique_suggestions) >= 10:  # Limit to 10 suggestions
                    break

        logger.info(
            "Generated %d contextual suggestions for %s",
            len(unique_suggestions),
            current_topic,
        )
        return unique_suggestions

    def identify_context_gaps(self, user_id: str, domain: str) -> List[Dict[str, Any]]:
        """
        Identify gaps in contextual understanding.

        Args:
            user_id: User identifier
            domain: Domain name

        Returns:
            List[Dict[str, Any]]: Identified context gaps
        """
        context = self.get_domain_context(user_id, domain)
        gaps = []

        # Find elements with missing prerequisites
        for element in context.elements.values():
            missing_prereqs = []
            for prereq_id in element.prerequisite_elements:
                if prereq_id not in context.elements:
                    missing_prereqs.append(prereq_id)

            if missing_prereqs:
                gaps.append(
                    {
                        "type": "missing_prerequisite",
                        "element": element.content,
                        "missing": missing_prereqs,
                        "impact": (
                            "high"
                            if element.relevance == ContextRelevance.CRITICAL
                            else "medium"
                        ),
                    }
                )

        # Find isolated elements (no relationships)
        for element in context.elements.values():
            if (
                not element.related_elements
                and not element.prerequisite_elements
                and not element.dependent_elements
            ):
                gaps.append(
                    {
                        "type": "isolated_element",
                        "element": element.content,
                        "suggestion": "Connect to related concepts",
                        "impact": "medium",
                    }
                )

        # Find incomplete workflows
        incomplete_workflows = []
        for workflow in context.workflow_patterns:
            if len(workflow) < 3:  # Workflows should have at least 3 steps
                incomplete_workflows.append(workflow)

        if incomplete_workflows:
            gaps.append(
                {
                    "type": "incomplete_workflow",
                    "workflows": incomplete_workflows,
                    "suggestion": "Complete workflow sequences",
                    "impact": "medium",
                }
            )

        logger.info(
            "Identified %d context gaps for %s in %s", len(gaps), user_id, domain
        )
        return gaps

    def save_domain_context(self, context: DomainContext):
        """
        Save domain context to storage.

        Args:
            context: Domain context to save
        """
        # Update cache
        if context.user_id not in self.context_cache:
            self.context_cache[context.user_id] = {}
        self.context_cache[context.user_id][context.domain_name] = context

        # Save to storage
        cache_key = f"{context.user_id}_{context.domain_name}"
        context_path = self.storage_path / f"{cache_key}_context.json"

        try:
            # Convert elements to serializable format
            elements_data = []
            for element in context.elements.values():
                elements_data.append(
                    {
                        "element_id": element.element_id,
                        "content": element.content,
                        "context_type": element.context_type.value,
                        "relevance": element.relevance.value,
                        "domain": element.domain,
                        "related_elements": element.related_elements,
                        "prerequisite_elements": element.prerequisite_elements,
                        "dependent_elements": element.dependent_elements,
                        "confidence_score": element.confidence_score,
                        "usage_frequency": element.usage_frequency,
                        "last_accessed": element.last_accessed.isoformat(),
                        "created_at": element.created_at.isoformat(),
                        "common_patterns": element.common_patterns,
                        "trigger_keywords": element.trigger_keywords,
                    }
                )

            data = {
                "domain_name": context.domain_name,
                "user_id": context.user_id,
                "total_elements": context.total_elements,
                "active_elements": context.active_elements,
                "context_depth": context.context_depth,
                "concept_hierarchy": context.concept_hierarchy,
                "workflow_patterns": context.workflow_patterns,
                "common_sequences": context.common_sequences,
                "created_at": context.created_at.isoformat(),
                "last_updated": context.last_updated.isoformat(),
                "elements": elements_data,
            }

            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                "Error saving context for %s/%s: %s",
                context.user_id,
                context.domain_name,
                e,
            )

    def _extract_context_elements(
        self, text: str, domain: str, user_id: str
    ) -> List[ContextElement]:
        """Extract context elements from text."""
        import re
        import uuid

        elements = []

        # Extract different types of context
        for context_type, patterns in self.CONTEXT_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        concept = match.group(1).strip()
                        description = match.group(2).strip()

                        if len(concept) > 2 and len(description) > 5:
                            element = ContextElement(
                                element_id=str(uuid.uuid4()),
                                content=f"{concept}: {description}",
                                context_type=ContextType(context_type),
                                relevance=ContextRelevance.MEDIUM,
                                domain=domain,
                                confidence_score=0.5,
                                trigger_keywords=[concept.lower()],
                            )
                            elements.append(element)

        # Extract procedural elements from numbered lists or steps
        step_pattern = r"(?:step\s+)?(\d+)[.\-:]\s*(.+?)(?:\n|$)"
        step_matches = re.finditer(step_pattern, text, re.IGNORECASE | re.MULTILINE)

        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2).strip()

            if len(step_content) > 5:
                element = ContextElement(
                    element_id=str(uuid.uuid4()),
                    content=f"Step {step_num}: {step_content}",
                    context_type=ContextType.PROCEDURAL,
                    relevance=ContextRelevance.HIGH,
                    domain=domain,
                    confidence_score=0.7,
                    trigger_keywords=[f"step {step_num}", "procedure", "process"],
                )
                elements.append(element)

        return elements

    def _identify_relationships(
        self, context: DomainContext, new_elements: List[ContextElement]
    ):
        """Identify relationships between context elements."""
        # Simple relationship identification based on content similarity
        for new_element in new_elements:
            for existing_id, existing_element in context.elements.items():
                if existing_id == new_element.element_id:
                    continue

                # Check for content overlap
                new_words = set(new_element.content.lower().split())
                existing_words = set(existing_element.content.lower().split())

                overlap = len(new_words.intersection(existing_words))
                total_words = len(new_words.union(existing_words))

                if total_words > 0:
                    similarity = overlap / total_words

                    # Create relationship if similarity is high enough
                    if similarity > 0.3:
                        if existing_id not in new_element.related_elements:
                            new_element.related_elements.append(existing_id)
                        if (
                            new_element.element_id
                            not in existing_element.related_elements
                        ):
                            existing_element.related_elements.append(
                                new_element.element_id
                            )

                # Check for prerequisite relationships
                if any(
                    word in new_element.content.lower()
                    for word in ["requires", "needs", "depends on", "after"]
                ):
                    # This might be a prerequisite relationship
                    if existing_id not in new_element.prerequisite_elements:
                        new_element.prerequisite_elements.append(existing_id)
                    if (
                        new_element.element_id
                        not in existing_element.dependent_elements
                    ):
                        existing_element.dependent_elements.append(
                            new_element.element_id
                        )

    def _update_workflow_patterns(self, context: DomainContext, text: str):
        """Update workflow patterns from text."""
        import re

        # Look for sequential indicators
        workflow_indicators = []
        for indicator in self.WORKFLOW_INDICATORS:
            if indicator in text.lower():
                workflow_indicators.append(indicator)

        # If we found workflow indicators, try to extract the sequence
        if len(workflow_indicators) >= 2:
            # Simple extraction of sentences containing workflow indicators
            sentences = re.split(r"[.!?]+", text)
            workflow_steps = []

            for sentence in sentences:
                sentence = sentence.strip()
                if any(
                    indicator in sentence.lower()
                    for indicator in self.WORKFLOW_INDICATORS
                ):
                    # Clean up the sentence
                    clean_sentence = re.sub(
                        r"^\W+", "", sentence
                    )  # Remove leading punctuation
                    if len(clean_sentence) > 10:
                        workflow_steps.append(clean_sentence)

            if len(workflow_steps) >= 2:
                context.workflow_patterns.append(workflow_steps)

                # Keep only recent workflow patterns (last 20)
                if len(context.workflow_patterns) > 20:
                    context.workflow_patterns = context.workflow_patterns[-20:]


# Global context engine instance
_context_engine: Optional[DomainContextEngine] = None


def get_context_engine() -> DomainContextEngine:
    """
    Get the global domain context engine instance.

    Returns:
        DomainContextEngine: Global context engine instance
    """
    global _context_engine
    if _context_engine is None:
        _context_engine = DomainContextEngine()
    return _context_engine
