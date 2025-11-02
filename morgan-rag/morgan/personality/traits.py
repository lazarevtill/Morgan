"""
Personality Trait Modeling for Morgan RAG.

Models user personality traits based on the Big Five personality model
and adapts assistant behavior to match user personality preferences.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger
from ..emotional.models import (
    InteractionData, ConversationContext, EmotionalState
)

logger = get_logger(__name__)


class PersonalityTrait(Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class TraitLevel(Enum):
    """Levels of personality traits."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TraitScore:
    """Score for a specific personality trait."""
    trait: PersonalityTrait
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def level(self) -> TraitLevel:
        """Get trait level based on score."""
        if self.score < 0.2:
            return TraitLevel.VERY_LOW
        elif self.score < 0.4:
            return TraitLevel.LOW
        elif self.score < 0.6:
            return TraitLevel.MODERATE
        elif self.score < 0.8:
            return TraitLevel.HIGH
        else:
            return TraitLevel.VERY_HIGH


@dataclass
class PersonalityProfile:
    """Complete personality profile for a user."""
    user_id: str
    trait_scores: Dict[PersonalityTrait, TraitScore] = field(
        default_factory=dict
    )
    overall_confidence: float = 0.0
    profile_stability: float = 0.0  # How stable the profile is over time
    last_analysis: datetime = field(default_factory=datetime.utcnow)
    analysis_count: int = 0
    
    def get_trait_score(self, trait: PersonalityTrait) -> Optional[TraitScore]:
        """Get score for a specific trait."""
        return self.trait_scores.get(trait)
    
    def get_trait_level(self, trait: PersonalityTrait) -> TraitLevel:
        """Get level for a specific trait."""
        trait_score = self.trait_scores.get(trait)
        if trait_score:
            return trait_score.level
        return TraitLevel.MODERATE  # Default to moderate if unknown
    
    def update_trait(self, trait_score: TraitScore):
        """Update a trait score."""
        self.trait_scores[trait_score.trait] = trait_score
        self.last_analysis = datetime.utcnow()
        self.analysis_count += 1
        self._update_overall_confidence()
    
    def _update_overall_confidence(self):
        """Update overall confidence based on trait confidences."""
        if not self.trait_scores:
            self.overall_confidence = 0.0
            return
        
        total_confidence = sum(
            score.confidence for score in self.trait_scores.values()
        )
        self.overall_confidence = total_confidence / len(self.trait_scores)


class PersonalityTraitModeler:
    """
    Models user personality traits based on interaction patterns.
    
    Uses linguistic analysis, behavioral patterns, and emotional responses
    to infer Big Five personality traits.
    """
    
    # Trait indicators based on linguistic and behavioral patterns
    TRAIT_INDICATORS = {
        PersonalityTrait.OPENNESS: {
            'high': [
                r'\b(creative|innovative|imaginative|artistic)\b',
                r'\b(new|novel|different|unique|original)\b',
                r'\b(explore|discover|experiment|try)\b',
                r'\b(abstract|theoretical|philosophical)\b'
            ],
            'low': [
                r'\b(traditional|conventional|practical|realistic)\b',
                r'\b(proven|established|standard|normal)\b',
                r'\b(concrete|specific|factual|literal)\b'
            ]
        },
        PersonalityTrait.CONSCIENTIOUSNESS: {
            'high': [
                r'\b(organized|planned|structured|systematic)\b',
                r'\b(careful|thorough|detailed|precise)\b',
                r'\b(responsible|reliable|dependable)\b',
                r'\b(goal|objective|target|deadline)\b'
            ],
            'low': [
                r'\b(spontaneous|flexible|adaptable|casual)\b',
                r'\b(quick|rough|approximate|general)\b',
                r'\b(relaxed|easygoing|informal)\b'
            ]
        },
        PersonalityTrait.EXTRAVERSION: {
            'high': [
                r'\b(social|people|group|team|community)\b',
                r'\b(excited|enthusiastic|energetic|active)\b',
                r'\b(talk|discuss|share|communicate)\b',
                r'\b(outgoing|friendly|sociable)\b'
            ],
            'low': [
                r'\b(quiet|private|alone|individual|personal)\b',
                r'\b(calm|peaceful|reserved|thoughtful)\b',
                r'\b(reflect|consider|think|contemplate)\b'
            ]
        },
        PersonalityTrait.AGREEABLENESS: {
            'high': [
                r'\b(help|support|assist|cooperate)\b',
                r'\b(kind|nice|friendly|pleasant|polite)\b',
                r'\b(understand|empathize|care|concern)\b',
                r'\b(agree|harmony|peace|compromise)\b'
            ],
            'low': [
                r'\b(compete|challenge|argue|debate)\b',
                r'\b(critical|skeptical|questioning)\b',
                r'\b(direct|blunt|straightforward)\b'
            ]
        },
        PersonalityTrait.NEUROTICISM: {
            'high': [
                r'\b(worry|anxious|stress|nervous|concern)\b',
                r'\b(problem|issue|difficulty|trouble)\b',
                r'\b(upset|frustrated|annoyed|irritated)\b',
                r'\b(uncertain|unsure|doubt|confused)\b'
            ],
            'low': [
                r'\b(calm|relaxed|confident|stable)\b',
                r'\b(positive|optimistic|hopeful)\b',
                r'\b(handle|manage|cope|deal)\b'
            ]
        }
    }
    
    def __init__(self):
        """Initialize personality trait modeler."""
        logger.info("Personality trait modeler initialized")
    
    def analyze_personality(
        self,
        user_id: str,
        interactions: List[InteractionData],
        existing_profile: Optional[PersonalityProfile] = None
    ) -> PersonalityProfile:
        """
        Analyze user personality from interactions.
        
        Args:
            user_id: User identifier
            interactions: List of user interactions
            existing_profile: Existing personality profile to update
            
        Returns:
            PersonalityProfile: Updated personality profile
        """
        logger.info(
            "Analyzing personality for user %s from %d interactions",
            user_id, len(interactions)
        )
        
        if existing_profile:
            profile = existing_profile
        else:
            profile = PersonalityProfile(user_id=user_id)
        
        # Collect text data from interactions
        text_data = self._extract_text_data(interactions)
        
        if not text_data:
            logger.warning("No text data available for personality analysis")
            return profile
        
        # Analyze each personality trait
        for trait in PersonalityTrait:
            trait_score = self._analyze_trait(
                trait, text_data, interactions, user_id
            )
            if trait_score:
                profile.update_trait(trait_score)
        
        # Update profile stability
        profile.profile_stability = self._calculate_stability(profile)
        
        logger.info(
            "Personality analysis complete for user %s. "
            "Overall confidence: %.2f",
            user_id, profile.overall_confidence
        )
        
        return profile
    
    def _extract_text_data(
        self, interactions: List[InteractionData]
    ) -> List[str]:
        """Extract text data from interactions."""
        text_data = []
        
        for interaction in interactions:
            # Extract message text
            if hasattr(interaction.conversation_context, 'message_text'):
                text = interaction.conversation_context.message_text
                if text and isinstance(text, str):
                    text_data.append(text.lower())
            
            # Extract any additional text fields
            if hasattr(interaction, 'user_message'):
                if interaction.user_message:
                    text_data.append(interaction.user_message.lower())
        
        return text_data
    
    def _analyze_trait(
        self,
        trait: PersonalityTrait,
        text_data: List[str],
        interactions: List[InteractionData],
        user_id: str
    ) -> Optional[TraitScore]:
        """Analyze a specific personality trait."""
        if trait not in self.TRAIT_INDICATORS:
            return None
        
        indicators = self.TRAIT_INDICATORS[trait]
        
        # Calculate linguistic indicators
        high_score = self._calculate_pattern_score(
            text_data, indicators.get('high', [])
        )
        low_score = self._calculate_pattern_score(
            text_data, indicators.get('low', [])
        )
        
        # Calculate behavioral indicators
        behavioral_score = self._analyze_behavioral_indicators(
            trait, interactions
        )
        
        # Combine scores (weighted average)
        linguistic_weight = 0.6
        behavioral_weight = 0.4
        
        if high_score > low_score:
            base_score = high_score
            trait_direction = 1.0
        else:
            base_score = low_score
            trait_direction = 0.0
        
        # Combine linguistic and behavioral evidence
        combined_score = (
            linguistic_weight * base_score +
            behavioral_weight * behavioral_score
        )
        
        # Adjust score based on direction
        if trait_direction == 0.0:  # Low trait indicators
            final_score = max(0.0, 0.5 - combined_score)
        else:  # High trait indicators
            final_score = min(1.0, 0.5 + combined_score)
        
        # Calculate confidence based on amount of evidence
        confidence = min(
            (len(text_data) / 10.0) * combined_score,
            1.0
        )
        
        # Generate evidence list
        evidence = self._generate_evidence(
            trait, high_score, low_score, behavioral_score
        )
        
        return TraitScore(
            trait=trait,
            score=final_score,
            confidence=confidence,
            evidence=evidence
        )
    
    def _calculate_pattern_score(
        self, text_data: List[str], patterns: List[str]
    ) -> float:
        """Calculate pattern match score for text data."""
        import re
        
        if not patterns or not text_data:
            return 0.0
        
        total_matches = 0
        total_words = 0
        
        for text in text_data:
            words = text.split()
            total_words += len(words)
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                total_matches += len(matches)
        
        if total_words == 0:
            return 0.0
        
        # Normalize per 100 words and cap at 1.0
        return min(total_matches / (total_words / 100), 1.0)
    
    def _analyze_behavioral_indicators(
        self,
        trait: PersonalityTrait,
        interactions: List[InteractionData]
    ) -> float:
        """Analyze behavioral indicators for a trait."""
        if not interactions:
            return 0.0
        
        behavioral_score = 0.0
        
        if trait == PersonalityTrait.EXTRAVERSION:
            # Analyze interaction frequency and length
            avg_session_length = self._calculate_avg_session_length(
                interactions
            )
            if avg_session_length > 600:  # More than 10 minutes
                behavioral_score += 0.3
            
            # Analyze feedback frequency (extraverts give more feedback)
            feedback_ratio = self._calculate_feedback_ratio(interactions)
            behavioral_score += feedback_ratio * 0.2
        
        elif trait == PersonalityTrait.CONSCIENTIOUSNESS:
            # Analyze consistency in interaction patterns
            consistency_score = self._calculate_interaction_consistency(
                interactions
            )
            behavioral_score += consistency_score * 0.4
        
        elif trait == PersonalityTrait.NEUROTICISM:
            # Analyze emotional volatility
            emotional_volatility = self._calculate_emotional_volatility(
                interactions
            )
            behavioral_score += emotional_volatility * 0.3
        
        elif trait == PersonalityTrait.OPENNESS:
            # Analyze topic diversity
            topic_diversity = self._calculate_topic_diversity(interactions)
            behavioral_score += topic_diversity * 0.3
        
        elif trait == PersonalityTrait.AGREEABLENESS:
            # Analyze positive sentiment in interactions
            positive_sentiment = self._calculate_positive_sentiment(
                interactions
            )
            behavioral_score += positive_sentiment * 0.3
        
        return min(behavioral_score, 1.0)
    
    def _calculate_avg_session_length(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate average session length in seconds."""
        durations = []
        
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'session_duration'):
                duration = interaction.conversation_context.session_duration
                if duration:
                    durations.append(duration.total_seconds())
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def _calculate_feedback_ratio(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate ratio of interactions with feedback."""
        feedback_count = 0
        
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'user_feedback'):
                if interaction.conversation_context.user_feedback is not None:
                    feedback_count += 1
        
        return feedback_count / len(interactions) if interactions else 0.0
    
    def _calculate_interaction_consistency(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate consistency in interaction patterns."""
        if len(interactions) < 3:
            return 0.0
        
        # Analyze time patterns (simplified)
        timestamps = []
        for interaction in interactions:
            if hasattr(interaction, 'timestamp'):
                timestamps.append(interaction.timestamp)
        
        if len(timestamps) < 3:
            return 0.0
        
        # Calculate variance in interaction intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        mean_interval = sum(intervals) / len(intervals)
        variance = sum(
            (interval - mean_interval) ** 2 for interval in intervals
        ) / len(intervals)
        
        # Lower variance = higher consistency
        consistency = max(0.0, 1.0 - (variance / (mean_interval ** 2)))
        return consistency
    
    def _calculate_emotional_volatility(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate emotional volatility from interactions."""
        emotional_states = []
        
        for interaction in interactions:
            if interaction.emotional_state:
                emotional_states.append(interaction.emotional_state.intensity)
        
        if len(emotional_states) < 2:
            return 0.0
        
        # Calculate variance in emotional intensity
        mean_intensity = sum(emotional_states) / len(emotional_states)
        variance = sum(
            (intensity - mean_intensity) ** 2 
            for intensity in emotional_states
        ) / len(emotional_states)
        
        # Higher variance = higher volatility
        return min(variance * 2, 1.0)
    
    def _calculate_topic_diversity(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate diversity of topics discussed."""
        topics = set()
        
        for interaction in interactions:
            if hasattr(interaction, 'topics_discussed'):
                topics.update(interaction.topics_discussed or [])
        
        # Normalize by interaction count
        diversity = len(topics) / len(interactions) if interactions else 0.0
        return min(diversity, 1.0)
    
    def _calculate_positive_sentiment(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate overall positive sentiment."""
        positive_emotions = ['joy', 'surprise']
        positive_count = 0
        total_count = 0
        
        for interaction in interactions:
            if interaction.emotional_state:
                total_count += 1
                if interaction.emotional_state.primary_emotion.value in positive_emotions:
                    positive_count += 1
        
        return positive_count / total_count if total_count > 0 else 0.0
    
    def _generate_evidence(
        self,
        trait: PersonalityTrait,
        high_score: float,
        low_score: float,
        behavioral_score: float
    ) -> List[str]:
        """Generate evidence list for trait analysis."""
        evidence = []
        
        if high_score > 0.1:
            evidence.append(
                f"High {trait.value} linguistic indicators (score: {high_score:.2f})"
            )
        
        if low_score > 0.1:
            evidence.append(
                f"Low {trait.value} linguistic indicators (score: {low_score:.2f})"
            )
        
        if behavioral_score > 0.1:
            evidence.append(
                f"Behavioral patterns suggest {trait.value} "
                f"(score: {behavioral_score:.2f})"
            )
        
        return evidence
    
    def _calculate_stability(self, profile: PersonalityProfile) -> float:
        """Calculate profile stability over time."""
        # This would compare current profile with historical profiles
        # For now, return a default value based on analysis count
        if profile.analysis_count < 3:
            return 0.3  # Low stability with few analyses
        elif profile.analysis_count < 10:
            return 0.6  # Medium stability
        else:
            return 0.8  # High stability with many analyses
    
    def get_trait_description(
        self, trait: PersonalityTrait, level: TraitLevel
    ) -> str:
        """Get human-readable description of a trait level."""
        descriptions = {
            PersonalityTrait.OPENNESS: {
                TraitLevel.VERY_LOW: "Prefers routine and conventional approaches",
                TraitLevel.LOW: "Somewhat traditional and practical",
                TraitLevel.MODERATE: "Balanced between tradition and innovation",
                TraitLevel.HIGH: "Open to new experiences and ideas",
                TraitLevel.VERY_HIGH: "Highly creative and imaginative"
            },
            PersonalityTrait.CONSCIENTIOUSNESS: {
                TraitLevel.VERY_LOW: "Very flexible and spontaneous",
                TraitLevel.LOW: "Somewhat disorganized but adaptable",
                TraitLevel.MODERATE: "Moderately organized and reliable",
                TraitLevel.HIGH: "Well-organized and dependable",
                TraitLevel.VERY_HIGH: "Extremely disciplined and systematic"
            },
            PersonalityTrait.EXTRAVERSION: {
                TraitLevel.VERY_LOW: "Very introverted and reserved",
                TraitLevel.LOW: "Somewhat quiet and reflective",
                TraitLevel.MODERATE: "Balanced between social and solitary",
                TraitLevel.HIGH: "Outgoing and sociable",
                TraitLevel.VERY_HIGH: "Highly energetic and social"
            },
            PersonalityTrait.AGREEABLENESS: {
                TraitLevel.VERY_LOW: "Very competitive and skeptical",
                TraitLevel.LOW: "Somewhat direct and challenging",
                TraitLevel.MODERATE: "Balanced between cooperation and assertion",
                TraitLevel.HIGH: "Cooperative and trusting",
                TraitLevel.VERY_HIGH: "Extremely helpful and empathetic"
            },
            PersonalityTrait.NEUROTICISM: {
                TraitLevel.VERY_LOW: "Very emotionally stable and calm",
                TraitLevel.LOW: "Generally stable with good coping skills",
                TraitLevel.MODERATE: "Moderate emotional reactivity",
                TraitLevel.HIGH: "Somewhat anxious and sensitive",
                TraitLevel.VERY_HIGH: "Highly sensitive and emotionally reactive"
            }
        }
        
        return descriptions.get(trait, {}).get(
            level, f"{trait.value} level: {level.value}"
        )