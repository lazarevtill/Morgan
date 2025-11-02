"""
Emotional pattern recognition module.

Provides focused emotional pattern detection, trend analysis, and behavioral
pattern recognition for enhanced emotional intelligence and user understanding.
"""

import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import statistics
import math

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotions.memory import get_emotional_memory_storage, EmotionalMemory
from morgan.emotional.models import EmotionalState, EmotionType, CompanionProfile

logger = get_logger(__name__)


class EmotionalPattern:
    """
    Represents a detected emotional pattern.
    
    Features:
    - Pattern type classification
    - Confidence scoring
    - Temporal characteristics
    - Behavioral implications
    """
    
    def __init__(
        self,
        pattern_id: str,
        pattern_type: str,
        description: str,
        confidence: float,
        frequency: float,
        temporal_characteristics: Dict[str, Any],
        emotional_characteristics: Dict[str, Any],
        behavioral_implications: List[str],
        supporting_evidence: List[str]
    ):
        """Initialize emotional pattern."""
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.description = description
        self.confidence = confidence
        self.frequency = frequency
        self.temporal_characteristics = temporal_characteristics
        self.emotional_characteristics = emotional_characteristics
        self.behavioral_implications = behavioral_implications
        self.supporting_evidence = supporting_evidence
        self.detected_at = datetime.utcnow()
        self.last_observed = datetime.utcnow()
        self.observation_count = 1
    
    def update_observation(self):
        """Update pattern observation tracking."""
        self.last_observed = datetime.utcnow()
        self.observation_count += 1
    
    def calculate_pattern_strength(self) -> float:
        """
        Calculate overall pattern strength.
        
        Returns:
            Pattern strength score (0.0 to 1.0)
        """
        # Combine confidence, frequency, and observation count
        observation_factor = min(1.0, self.observation_count / 10.0)
        return (self.confidence * 0.5) + (self.frequency * 0.3) + (observation_factor * 0.2)
    
    def is_active(self, max_age_days: int = 30) -> bool:
        """
        Check if pattern is still active.
        
        Args:
            max_age_days: Maximum days since last observation
            
        Returns:
            True if pattern is still active
        """
        age = datetime.utcnow() - self.last_observed
        return age.days <= max_age_days


class EmotionalPatternRecognizer:
    """
    Recognizes and analyzes emotional patterns in user behavior.
    
    Features:
    - Multi-type pattern detection
    - Temporal pattern analysis
    - Behavioral pattern recognition
    - Adaptive pattern learning
    """
    
    # Pattern type definitions
    PATTERN_TYPES = {
        'cyclical': 'Recurring emotional cycles',
        'trigger_response': 'Consistent emotional responses to triggers',
        'escalation': 'Emotional escalation patterns',
        'recovery': 'Emotional recovery patterns',
        'stability': 'Emotional stability patterns',
        'volatility': 'Emotional volatility patterns',
        'seasonal': 'Time-based emotional patterns',
        'contextual': 'Context-dependent emotional patterns'
    }
    
    def __init__(self):
        """Initialize emotional pattern recognizer."""
        self.settings = get_settings()
        self.memory_storage = get_emotional_memory_storage()
        
        # Pattern storage
        self._detected_patterns = {}  # user_id -> List[EmotionalPattern]
        self._pattern_cache = {}
        
        logger.info("Emotional Pattern Recognizer initialized")
    
    def detect_patterns(
        self,
        user_id: str,
        analysis_days: int = 30,
        min_pattern_confidence: float = 0.6
    ) -> List[EmotionalPattern]:
        """
        Detect emotional patterns for a user.
        
        Args:
            user_id: User identifier
            analysis_days: Days of history to analyze
            min_pattern_confidence: Minimum confidence threshold
            
        Returns:
            List of detected emotional patterns
        """
        # Retrieve emotional memories for analysis
        memories = self.memory_storage.retrieve_memories(
            user_id=user_id,
            max_age_days=analysis_days,
            min_importance=0.2,
            limit=200
        )
        
        if len(memories) < 10:
            logger.debug(f"Insufficient data for pattern detection: {len(memories)} memories")
            return []
        
        detected_patterns = []
        
        # Detect different types of patterns
        cyclical_patterns = self._detect_cyclical_patterns(user_id, memories)
        trigger_patterns = self._detect_trigger_response_patterns(user_id, memories)
        escalation_patterns = self._detect_escalation_patterns(user_id, memories)
        recovery_patterns = self._detect_recovery_patterns(user_id, memories)
        stability_patterns = self._detect_stability_patterns(user_id, memories)
        seasonal_patterns = self._detect_seasonal_patterns(user_id, memories)
        contextual_patterns = self._detect_contextual_patterns(user_id, memories)
        
        # Combine all patterns
        all_patterns = (
            cyclical_patterns + trigger_patterns + escalation_patterns +
            recovery_patterns + stability_patterns + seasonal_patterns +
            contextual_patterns
        )
        
        # Filter by confidence threshold
        detected_patterns = [
            pattern for pattern in all_patterns
            if pattern.confidence >= min_pattern_confidence
        ]
        
        # Store patterns for user
        self._detected_patterns[user_id] = detected_patterns
        
        logger.info(f"Detected {len(detected_patterns)} emotional patterns for user {user_id}")
        return detected_patterns
    
    def get_user_patterns(
        self,
        user_id: str,
        pattern_types: Optional[List[str]] = None,
        active_only: bool = True
    ) -> List[EmotionalPattern]:
        """
        Get detected patterns for a user.
        
        Args:
            user_id: User identifier
            pattern_types: Optional filter by pattern types
            active_only: Only return active patterns
            
        Returns:
            List of user's emotional patterns
        """
        user_patterns = self._detected_patterns.get(user_id, [])
        
        # Filter by pattern types
        if pattern_types:
            user_patterns = [
                p for p in user_patterns
                if p.pattern_type in pattern_types
            ]
        
        # Filter by active status
        if active_only:
            user_patterns = [p for p in user_patterns if p.is_active()]
        
        return user_patterns
    
    def analyze_pattern_implications(
        self,
        patterns: List[EmotionalPattern]
    ) -> Dict[str, Any]:
        """
        Analyze implications of detected patterns.
        
        Args:
            patterns: List of emotional patterns
            
        Returns:
            Pattern implications analysis
        """
        if not patterns:
            return {'implications': 'no_patterns_detected'}
        
        # Categorize patterns by strength
        strong_patterns = [p for p in patterns if p.calculate_pattern_strength() > 0.8]
        moderate_patterns = [p for p in patterns if 0.6 <= p.calculate_pattern_strength() <= 0.8]
        weak_patterns = [p for p in patterns if p.calculate_pattern_strength() < 0.6]
        
        # Analyze pattern types
        pattern_type_counts = Counter(p.pattern_type for p in patterns)
        
        # Extract behavioral implications
        all_implications = []
        for pattern in patterns:
            all_implications.extend(pattern.behavioral_implications)
        
        implication_counts = Counter(all_implications)
        
        # Determine overall emotional profile
        emotional_profile = self._determine_emotional_profile(patterns)
        
        # Generate recommendations
        recommendations = self._generate_pattern_recommendations(patterns)
        
        return {
            'total_patterns': len(patterns),
            'pattern_strength_distribution': {
                'strong': len(strong_patterns),
                'moderate': len(moderate_patterns),
                'weak': len(weak_patterns)
            },
            'pattern_type_distribution': dict(pattern_type_counts),
            'top_implications': dict(implication_counts.most_common(5)),
            'emotional_profile': emotional_profile,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def predict_emotional_response(
        self,
        user_id: str,
        context_factors: Dict[str, Any],
        confidence_threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Predict likely emotional response based on patterns.
        
        Args:
            user_id: User identifier
            context_factors: Current context factors
            confidence_threshold: Minimum prediction confidence
            
        Returns:
            Emotional response prediction or None
        """
        user_patterns = self.get_user_patterns(user_id, active_only=True)
        
        if not user_patterns:
            return None
        
        # Find patterns that match current context
        matching_patterns = []
        for pattern in user_patterns:
            match_score = self._calculate_context_match(pattern, context_factors)
            if match_score > 0.5:
                matching_patterns.append((pattern, match_score))
        
        if not matching_patterns:
            return None
        
        # Sort by match score and pattern strength
        matching_patterns.sort(
            key=lambda x: x[1] * x[0].calculate_pattern_strength(),
            reverse=True
        )
        
        # Use top matching pattern for prediction
        best_pattern, match_score = matching_patterns[0]
        prediction_confidence = match_score * best_pattern.calculate_pattern_strength()
        
        if prediction_confidence < confidence_threshold:
            return None
        
        # Generate prediction
        predicted_emotions = self._extract_pattern_emotions(best_pattern)
        
        return {
            'predicted_emotions': predicted_emotions,
            'confidence': prediction_confidence,
            'based_on_pattern': {
                'type': best_pattern.pattern_type,
                'description': best_pattern.description
            },
            'context_match_score': match_score,
            'behavioral_implications': best_pattern.behavioral_implications
        }
    
    def _detect_cyclical_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect cyclical emotional patterns."""
        patterns = []
        
        # Group memories by time periods
        daily_emotions = defaultdict(list)
        weekly_emotions = defaultdict(list)
        
        for memory in memories:
            day_of_week = memory.created_at.weekday()
            hour_of_day = memory.created_at.hour
            
            daily_emotions[hour_of_day].append(memory.emotional_state)
            weekly_emotions[day_of_week].append(memory.emotional_state)
        
        # Detect daily cycles
        daily_pattern = self._analyze_temporal_cycle(daily_emotions, 'daily')
        if daily_pattern:
            patterns.append(daily_pattern)
        
        # Detect weekly cycles
        weekly_pattern = self._analyze_temporal_cycle(weekly_emotions, 'weekly')
        if weekly_pattern:
            patterns.append(weekly_pattern)
        
        return patterns
    
    def _detect_trigger_response_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect trigger-response emotional patterns."""
        patterns = []
        
        # Group memories by emotional indicators (triggers)
        trigger_responses = defaultdict(list)
        
        for memory in memories:
            for indicator in memory.emotional_state.emotional_indicators:
                trigger_responses[indicator].append(memory.emotional_state.primary_emotion)
        
        # Find consistent trigger-response patterns
        for trigger, responses in trigger_responses.items():
            if len(responses) >= 3:  # Minimum occurrences
                response_counts = Counter(responses)
                most_common_response, count = response_counts.most_common(1)[0]
                
                consistency = count / len(responses)
                if consistency >= 0.7:  # High consistency threshold
                    pattern = EmotionalPattern(
                        pattern_id=f"{user_id}_trigger_{hash(trigger)}",
                        pattern_type='trigger_response',
                        description=f"Consistently responds with {most_common_response.value} to '{trigger}'",
                        confidence=consistency,
                        frequency=len(responses) / len(memories),
                        temporal_characteristics={},
                        emotional_characteristics={
                            'trigger': trigger,
                            'response_emotion': most_common_response.value,
                            'consistency': consistency
                        },
                        behavioral_implications=[
                            f"predictable_response_to_{trigger.replace(' ', '_')}",
                            f"emotional_sensitivity_to_{most_common_response.value}"
                        ],
                        supporting_evidence=[f"Observed {count} times out of {len(responses)} occurrences"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_escalation_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect emotional escalation patterns."""
        patterns = []
        
        # Sort memories by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        # Look for escalation sequences
        escalation_sequences = []
        current_sequence = []
        
        for i, memory in enumerate(sorted_memories):
            if i == 0:
                current_sequence = [memory]
                continue
            
            # Check if this continues an escalation
            prev_memory = sorted_memories[i-1]
            time_diff = memory.created_at - prev_memory.created_at
            
            # Must be within reasonable time window
            if time_diff <= timedelta(hours=6):
                prev_intensity = prev_memory.emotional_state.intensity
                curr_intensity = memory.emotional_state.intensity
                
                # Check for intensity increase
                if curr_intensity > prev_intensity + 0.2:
                    current_sequence.append(memory)
                else:
                    # End of sequence
                    if len(current_sequence) >= 3:
                        escalation_sequences.append(current_sequence)
                    current_sequence = [memory]
            else:
                # Time gap too large
                if len(current_sequence) >= 3:
                    escalation_sequences.append(current_sequence)
                current_sequence = [memory]
        
        # Analyze escalation sequences
        if escalation_sequences:
            avg_sequence_length = sum(len(seq) for seq in escalation_sequences) / len(escalation_sequences)
            escalation_frequency = len(escalation_sequences) / len(memories)
            
            if escalation_frequency > 0.1:  # At least 10% of interactions show escalation
                pattern = EmotionalPattern(
                    pattern_id=f"{user_id}_escalation",
                    pattern_type='escalation',
                    description=f"Shows emotional escalation in {len(escalation_sequences)} sequences",
                    confidence=min(1.0, escalation_frequency * 2),
                    frequency=escalation_frequency,
                    temporal_characteristics={
                        'average_sequence_length': avg_sequence_length,
                        'escalation_count': len(escalation_sequences)
                    },
                    emotional_characteristics={
                        'escalation_tendency': escalation_frequency,
                        'typical_sequence_length': avg_sequence_length
                    },
                    behavioral_implications=[
                        'emotional_escalation_tendency',
                        'requires_de_escalation_support',
                        'benefits_from_early_intervention'
                    ],
                    supporting_evidence=[f"Detected {len(escalation_sequences)} escalation sequences"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_recovery_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect emotional recovery patterns."""
        patterns = []
        
        # Find negative emotion to positive emotion transitions
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        recovery_sequences = []
        
        negative_emotions = {EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR}
        positive_emotions = {EmotionType.JOY, EmotionType.SURPRISE}
        
        for i in range(len(sorted_memories) - 1):
            current_memory = sorted_memories[i]
            next_memory = sorted_memories[i + 1]
            
            # Check for negative to positive transition
            if (current_memory.emotional_state.primary_emotion in negative_emotions and
                next_memory.emotional_state.primary_emotion in positive_emotions):
                
                time_diff = next_memory.created_at - current_memory.created_at
                recovery_sequences.append({
                    'from_emotion': current_memory.emotional_state.primary_emotion,
                    'to_emotion': next_memory.emotional_state.primary_emotion,
                    'recovery_time': time_diff,
                    'from_intensity': current_memory.emotional_state.intensity,
                    'to_intensity': next_memory.emotional_state.intensity
                })
        
        if recovery_sequences:
            avg_recovery_time = sum(
                seq['recovery_time'].total_seconds() for seq in recovery_sequences
            ) / len(recovery_sequences)
            
            recovery_frequency = len(recovery_sequences) / len(memories)
            
            pattern = EmotionalPattern(
                pattern_id=f"{user_id}_recovery",
                pattern_type='recovery',
                description=f"Shows emotional recovery in {len(recovery_sequences)} instances",
                confidence=min(1.0, recovery_frequency * 3),
                frequency=recovery_frequency,
                temporal_characteristics={
                    'average_recovery_time_seconds': avg_recovery_time,
                    'recovery_count': len(recovery_sequences)
                },
                emotional_characteristics={
                    'recovery_tendency': recovery_frequency,
                    'typical_recovery_time': avg_recovery_time
                },
                behavioral_implications=[
                    'good_emotional_recovery',
                    'resilient_emotional_processing',
                    'benefits_from_positive_reinforcement'
                ],
                supporting_evidence=[f"Observed {len(recovery_sequences)} recovery instances"]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_stability_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect emotional stability patterns."""
        patterns = []
        
        # Calculate emotional stability metrics
        intensities = [m.emotional_state.intensity for m in memories]
        emotions = [m.emotional_state.primary_emotion for m in memories]
        
        if len(intensities) < 5:
            return patterns
        
        # Intensity stability
        intensity_variance = statistics.variance(intensities)
        intensity_stability = max(0.0, 1.0 - (intensity_variance * 2))
        
        # Emotion consistency
        emotion_counts = Counter(emotions)
        most_common_emotion, count = emotion_counts.most_common(1)[0]
        emotion_consistency = count / len(emotions)
        
        # Overall stability
        overall_stability = (intensity_stability + emotion_consistency) / 2
        
        if overall_stability > 0.7:
            pattern = EmotionalPattern(
                pattern_id=f"{user_id}_stability",
                pattern_type='stability',
                description=f"Shows high emotional stability (score: {overall_stability:.2f})",
                confidence=overall_stability,
                frequency=1.0,  # Stability is a continuous characteristic
                temporal_characteristics={
                    'analysis_period_days': (memories[-1].created_at - memories[0].created_at).days
                },
                emotional_characteristics={
                    'intensity_stability': intensity_stability,
                    'emotion_consistency': emotion_consistency,
                    'dominant_emotion': most_common_emotion.value,
                    'overall_stability': overall_stability
                },
                behavioral_implications=[
                    'emotionally_stable',
                    'predictable_emotional_responses',
                    'low_maintenance_emotional_support'
                ],
                supporting_evidence=[f"Stability analysis over {len(memories)} interactions"]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_seasonal_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect seasonal/temporal emotional patterns."""
        patterns = []
        
        # Group by time periods
        time_groups = {
            'morning': defaultdict(list),    # 6-12
            'afternoon': defaultdict(list),  # 12-18
            'evening': defaultdict(list),    # 18-24
            'night': defaultdict(list)       # 0-6
        }
        
        for memory in memories:
            hour = memory.created_at.hour
            emotion = memory.emotional_state.primary_emotion
            
            if 6 <= hour < 12:
                time_groups['morning'][emotion].append(memory)
            elif 12 <= hour < 18:
                time_groups['afternoon'][emotion].append(memory)
            elif 18 <= hour < 24:
                time_groups['evening'][emotion].append(memory)
            else:
                time_groups['night'][emotion].append(memory)
        
        # Analyze each time period
        for period, emotion_groups in time_groups.items():
            if not emotion_groups:
                continue
            
            total_memories = sum(len(memories) for memories in emotion_groups.values())
            if total_memories < 3:
                continue
            
            # Find dominant emotion for this period
            dominant_emotion = max(emotion_groups.keys(), key=lambda e: len(emotion_groups[e]))
            dominance_ratio = len(emotion_groups[dominant_emotion]) / total_memories
            
            if dominance_ratio > 0.6:  # Strong temporal association
                pattern = EmotionalPattern(
                    pattern_id=f"{user_id}_seasonal_{period}",
                    pattern_type='seasonal',
                    description=f"Tends to feel {dominant_emotion.value} during {period}",
                    confidence=dominance_ratio,
                    frequency=total_memories / len(memories),
                    temporal_characteristics={
                        'time_period': period,
                        'dominant_emotion': dominant_emotion.value,
                        'dominance_ratio': dominance_ratio
                    },
                    emotional_characteristics={
                        'temporal_emotion_association': {
                            period: dominant_emotion.value
                        }
                    },
                    behavioral_implications=[
                        f'emotional_pattern_{period}',
                        f'time_sensitive_{dominant_emotion.value}'
                    ],
                    supporting_evidence=[f"{len(emotion_groups[dominant_emotion])} occurrences during {period}"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_contextual_patterns(
        self,
        user_id: str,
        memories: List[EmotionalMemory]
    ) -> List[EmotionalPattern]:
        """Detect context-dependent emotional patterns."""
        patterns = []
        
        # Group by conversation context characteristics
        context_groups = defaultdict(list)
        
        for memory in memories:
            # Group by message length (proxy for conversation complexity)
            msg_length = len(memory.conversation_context.message_text)
            if msg_length < 50:
                context_groups['short_messages'].append(memory)
            elif msg_length < 200:
                context_groups['medium_messages'].append(memory)
            else:
                context_groups['long_messages'].append(memory)
            
            # Group by presence of previous context
            if memory.conversation_context.previous_messages:
                context_groups['continued_conversation'].append(memory)
            else:
                context_groups['new_conversation'].append(memory)
        
        # Analyze each context group
        for context_type, context_memories in context_groups.items():
            if len(context_memories) < 5:
                continue
            
            # Find dominant emotion for this context
            emotions = [m.emotional_state.primary_emotion for m in context_memories]
            emotion_counts = Counter(emotions)
            dominant_emotion, count = emotion_counts.most_common(1)[0]
            
            dominance_ratio = count / len(context_memories)
            
            if dominance_ratio > 0.6:
                pattern = EmotionalPattern(
                    pattern_id=f"{user_id}_contextual_{context_type}",
                    pattern_type='contextual',
                    description=f"Tends to feel {dominant_emotion.value} in {context_type.replace('_', ' ')}",
                    confidence=dominance_ratio,
                    frequency=len(context_memories) / len(memories),
                    temporal_characteristics={},
                    emotional_characteristics={
                        'context_type': context_type,
                        'dominant_emotion': dominant_emotion.value,
                        'dominance_ratio': dominance_ratio
                    },
                    behavioral_implications=[
                        f'context_sensitive_{context_type}',
                        f'predictable_in_{context_type}'
                    ],
                    supporting_evidence=[f"{count} occurrences in {context_type}"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_temporal_cycle(
        self,
        temporal_emotions: Dict[int, List[EmotionalState]],
        cycle_type: str
    ) -> Optional[EmotionalPattern]:
        """Analyze temporal emotional cycles."""
        if len(temporal_emotions) < 3:
            return None
        
        # Find periods with consistent emotions
        consistent_periods = {}
        for period, emotions in temporal_emotions.items():
            if len(emotions) >= 3:
                emotion_counts = Counter(e.primary_emotion for e in emotions)
                dominant_emotion, count = emotion_counts.most_common(1)[0]
                consistency = count / len(emotions)
                
                if consistency > 0.6:
                    consistent_periods[period] = {
                        'emotion': dominant_emotion,
                        'consistency': consistency,
                        'count': len(emotions)
                    }
        
        if len(consistent_periods) >= 2:
            total_observations = sum(data['count'] for data in consistent_periods.values())
            avg_consistency = sum(data['consistency'] for data in consistent_periods.values()) / len(consistent_periods)
            
            pattern = EmotionalPattern(
                pattern_id=f"cyclical_{cycle_type}_{hash(str(consistent_periods))}",
                pattern_type='cyclical',
                description=f"Shows {cycle_type} emotional cycles across {len(consistent_periods)} periods",
                confidence=avg_consistency,
                frequency=len(consistent_periods) / (24 if cycle_type == 'daily' else 7),
                temporal_characteristics={
                    'cycle_type': cycle_type,
                    'consistent_periods': len(consistent_periods),
                    'period_details': consistent_periods
                },
                emotional_characteristics={
                    'cyclical_emotions': {
                        str(period): data['emotion'].value
                        for period, data in consistent_periods.items()
                    }
                },
                behavioral_implications=[
                    f'{cycle_type}_emotional_cycles',
                    'predictable_temporal_patterns',
                    'time_aware_interaction_optimization'
                ],
                supporting_evidence=[f"Consistent patterns across {len(consistent_periods)} time periods"]
            )
            return pattern
        
        return None
    
    def _determine_emotional_profile(
        self,
        patterns: List[EmotionalPattern]
    ) -> Dict[str, Any]:
        """Determine overall emotional profile from patterns."""
        if not patterns:
            return {'profile': 'insufficient_data'}
        
        # Analyze pattern characteristics
        stability_patterns = [p for p in patterns if p.pattern_type == 'stability']
        volatility_patterns = [p for p in patterns if p.pattern_type == 'volatility']
        recovery_patterns = [p for p in patterns if p.pattern_type == 'recovery']
        escalation_patterns = [p for p in patterns if p.pattern_type == 'escalation']
        
        profile = {}
        
        # Emotional stability assessment
        if stability_patterns:
            profile['stability'] = 'high'
        elif volatility_patterns:
            profile['stability'] = 'low'
        else:
            profile['stability'] = 'moderate'
        
        # Recovery capability
        if recovery_patterns:
            profile['recovery_ability'] = 'good'
        else:
            profile['recovery_ability'] = 'unknown'
        
        # Escalation tendency
        if escalation_patterns:
            profile['escalation_tendency'] = 'present'
        else:
            profile['escalation_tendency'] = 'low'
        
        # Pattern complexity
        profile['pattern_complexity'] = len(set(p.pattern_type for p in patterns))
        
        # Overall predictability
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        if avg_confidence > 0.8:
            profile['predictability'] = 'high'
        elif avg_confidence > 0.6:
            profile['predictability'] = 'moderate'
        else:
            profile['predictability'] = 'low'
        
        return profile
    
    def _generate_pattern_recommendations(
        self,
        patterns: List[EmotionalPattern]
    ) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []
        
        # Pattern-specific recommendations
        for pattern in patterns:
            if pattern.pattern_type == 'escalation':
                recommendations.append("Implement early intervention strategies")
                recommendations.append("Monitor for escalation triggers")
            elif pattern.pattern_type == 'recovery':
                recommendations.append("Leverage natural recovery patterns")
                recommendations.append("Provide positive reinforcement during recovery")
            elif pattern.pattern_type == 'stability':
                recommendations.append("Maintain consistent interaction style")
            elif pattern.pattern_type == 'cyclical':
                recommendations.append("Adapt responses based on temporal patterns")
            elif pattern.pattern_type == 'trigger_response':
                recommendations.append("Be mindful of identified emotional triggers")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def _calculate_context_match(
        self,
        pattern: EmotionalPattern,
        context_factors: Dict[str, Any]
    ) -> float:
        """Calculate how well a pattern matches current context."""
        match_score = 0.0
        total_factors = 0
        
        # Check temporal factors
        if 'time_of_day' in context_factors and pattern.pattern_type == 'seasonal':
            current_hour = context_factors['time_of_day']
            pattern_periods = pattern.temporal_characteristics.get('period_details', {})
            
            # Map hour to period
            if 6 <= current_hour < 12:
                period = 'morning'
            elif 12 <= current_hour < 18:
                period = 'afternoon'
            elif 18 <= current_hour < 24:
                period = 'evening'
            else:
                period = 'night'
            
            if period in str(pattern_periods):
                match_score += 1.0
            total_factors += 1
        
        # Check contextual factors
        if pattern.pattern_type == 'contextual':
            context_type = pattern.emotional_characteristics.get('context_type', '')
            
            # Match message length context
            if 'message_length' in context_factors:
                msg_length = context_factors['message_length']
                if ('short' in context_type and msg_length < 50) or \
                   ('medium' in context_type and 50 <= msg_length < 200) or \
                   ('long' in context_type and msg_length >= 200):
                    match_score += 1.0
                total_factors += 1
            
            # Match conversation continuity
            if 'has_previous_context' in context_factors:
                has_context = context_factors['has_previous_context']
                if ('continued' in context_type and has_context) or \
                   ('new' in context_type and not has_context):
                    match_score += 1.0
                total_factors += 1
        
        return match_score / max(1, total_factors)
    
    def _extract_pattern_emotions(
        self,
        pattern: EmotionalPattern
    ) -> List[Dict[str, Any]]:
        """Extract predicted emotions from a pattern."""
        emotions = []
        
        if 'dominant_emotion' in pattern.emotional_characteristics:
            emotions.append({
                'emotion': pattern.emotional_characteristics['dominant_emotion'],
                'confidence': pattern.confidence
            })
        
        if 'response_emotion' in pattern.emotional_characteristics:
            emotions.append({
                'emotion': pattern.emotional_characteristics['response_emotion'],
                'confidence': pattern.confidence
            })
        
        return emotions if emotions else [{'emotion': 'neutral', 'confidence': 0.5}]


# Singleton instance
_pattern_recognizer_instance = None
_pattern_recognizer_lock = threading.Lock()


def get_emotional_pattern_recognizer() -> EmotionalPatternRecognizer:
    """
    Get singleton emotional pattern recognizer instance.
    
    Returns:
        Shared EmotionalPatternRecognizer instance
    """
    global _pattern_recognizer_instance
    
    if _pattern_recognizer_instance is None:
        with _pattern_recognizer_lock:
            if _pattern_recognizer_instance is None:
                _pattern_recognizer_instance = EmotionalPatternRecognizer()
    
    return _pattern_recognizer_instance