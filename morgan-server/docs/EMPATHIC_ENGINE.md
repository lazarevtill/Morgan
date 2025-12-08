# Empathic Engine - Emotional Intelligence

## Overview

The Emotional Intelligence module is the first component of Morgan's Empathic Engine. It provides emotional awareness and responsiveness to make Morgan feel more human and emotionally intelligent.

## Features

### 1. Emotional Tone Detection

Detects emotional tones from user messages across 12 categories:
- **Positive**: Joyful, Excited, Content, Grateful, Hopeful
- **Negative**: Sad, Anxious, Frustrated, Angry
- **Neutral**: Neutral, Concerned, Confused

**Key capabilities:**
- Pattern matching with emotional indicators (words, phrases, emojis)
- Intensity modifiers (very, really, extremely, etc.)
- Confidence scoring based on indicator strength and message length
- Secondary tone detection for mixed emotions

### 2. Response Tone Adjustment

Provides intelligent suggestions for adjusting response tone based on detected user emotion:
- **Match enthusiasm** for joyful/excited users
- **Provide comfort and support** for sad/anxious users
- **Stay calm and solution-focused** for frustrated/angry users
- **Be clear and patient** for confused users
- **Acknowledge gratitude** warmly and humbly

**Includes:**
- Target tone recommendations
- Intensity levels (0.0 to 1.0)
- Specific suggestions for response style
- Celebration messages for positive moments
- Support messages for difficult times

### 3. Emotional Pattern Tracking

Tracks emotional patterns over time to understand user's emotional journey:
- Stores patterns with timestamps and confidence scores
- Configurable tracking window (default: 30 days)
- Automatic cleanup of old patterns

### 4. Emotional Trend Analysis

Analyzes emotional trends to provide insights:
- **Dominant tone**: Most common emotional state
- **Tone distribution**: Percentage breakdown of all tones
- **Trend analysis**: Improving, declining, or stable
- **Recent shift detection**: Identifies sudden emotional changes

## Usage Examples

### Basic Tone Detection

```python
from morgan_server.empathic import EmotionalIntelligence

ei = EmotionalIntelligence()

# Detect tone from a message
detection = ei.detect_tone("I'm so happy! This is wonderful!")

print(f"Primary tone: {detection.primary_tone.value}")
print(f"Confidence: {detection.confidence:.2f}")
print(f"Indicators: {detection.indicators}")
```

### Response Adjustment

```python
# Get adjustment recommendations
adjustment = ei.adjust_response_tone(detection)

print(f"Target tone: {adjustment.target_tone.value}")
print(f"Intensity: {adjustment.intensity:.2f}")
print(f"Suggestions: {adjustment.suggestions}")

if adjustment.celebration:
    print(f"Celebration: {adjustment.celebration}")
```

### Pattern Tracking

```python
# Track patterns with user ID
user_id = "user123"
detection = ei.detect_tone("I'm feeling great today!", user_id=user_id)

# Get patterns for analysis
patterns = ei.get_patterns(user_id, days=7)
print(f"Patterns in last 7 days: {len(patterns)}")

# Analyze emotional trend
analysis = ei.analyze_emotional_trend(user_id)
print(f"Dominant tone: {analysis['dominant_tone']}")
print(f"Trend: {analysis['trend']}")
print(f"Recent shift: {analysis['recent_shift']}")
```

## Implementation Details

### Emotional Indicators

The system uses a comprehensive dictionary of emotional indicators:
- **Words and phrases**: "happy", "sad", "worried", "excited", etc.
- **Emojis**: 😊, 😢, 🎉, etc.
- **Punctuation patterns**: "!!!", "??", etc.

### Scoring Algorithm

1. **Base scoring**: Each indicator found adds to the tone's score
2. **Intensity modifiers**: Multipliers applied for words like "very", "extremely"
3. **Confidence calculation**: Based on score strength relative to message length
4. **Secondary tones**: Identified when score is >30% of primary tone

### Pattern Analysis

Trends are determined by comparing positive vs. negative tone ratios between the first and second half of the tracking window:
- **Improving**: Second half has >20% more positive tones
- **Declining**: Second half has >20% fewer positive tones
- **Stable**: Change is within ±20%

## Testing

Comprehensive test suite with 33 unit tests covering:
- ✅ Tone detection for all 12 emotional categories
- ✅ Intensity modifier handling
- ✅ Mixed emotion detection
- ✅ Emoji support
- ✅ Response adjustment for all tones
- ✅ Celebration and support message generation
- ✅ Pattern tracking and retrieval
- ✅ Trend analysis (improving, declining, stable)
- ✅ Recent shift detection
- ✅ Full workflow integration tests

All tests passing: **33/33** ✅

## Integration with Other Systems

The Emotional Intelligence module integrates seamlessly with:
- **Personality System** - Provides emotional context for personality-driven responses
- **Roleplay System** - Enables emotionally intelligent roleplay behavior
- **Relationship Management** - Tracks emotional patterns for relationship depth calculation

## Next Steps

The Emotional Intelligence, Personality System, and Roleplay System modules are complete. Next components to implement:
1. **Relationship Management** - Trust building and milestone tracking
2. **Communication Style Adaptation** - Learning user preferences

## Requirements Validated

This implementation validates **Requirement 1.1** from the design document:
- ✅ Emotional tone detection from user messages
- ✅ Emotional tone adjustment for responses
- ✅ Emotional pattern tracking over time
- ✅ Support for celebrating positive moments and providing support


---

# Roleplay System

## Overview

The Roleplay System is the third component of Morgan's Empathic Engine. It provides context-aware, emotionally intelligent roleplay behavior by integrating the Emotional Intelligence and Personality System modules.

## Features

### 1. Roleplay Configuration

Comprehensive configuration for character behavior:
- **Character identity**: Name, description, background story
- **Roleplay tone**: Professional, Friendly, Casual, Playful, Supportive, Mentor, Companion
- **Response style**: Concise, Detailed, Conversational, Technical, Empathetic
- **Personality traits**: Integration with Personality System
- **Expertise areas**: List of knowledge domains
- **Communication preferences**: Customizable preferences
- **Feature toggles**: Enable/disable emotional intelligence and relationship awareness

### 2. Context-Aware Response Generation

Generates enhanced responses based on rich context:
- **User identification**: Track individual users
- **Conversation history**: Maintain context across interactions
- **Relationship depth**: Adapt behavior based on relationship (0.0 to 1.0)
- **Emotional state**: Integrate detected emotional tone
- **User preferences**: Respect individual communication preferences
- **Session metadata**: Additional contextual information

**Response includes:**
- Enhanced response text
- Applied tone and style
- Emotional adjustments
- Relationship notes
- Personality notes
- Context usage summary

### 3. Emotional Intelligence Integration

Seamlessly integrates with the Emotional Intelligence module:
- **Emotion detection**: Detect emotional tone from user messages
- **Response adjustment**: Adjust tone based on detected emotion
- **Celebration messages**: Celebrate positive emotional shifts
- **Support messages**: Provide support during difficult times
- **Emotional trend analysis**: Track emotional patterns over time

### 4. Relationship-Aware Behavior

Adapts behavior based on relationship depth:
- **New relationships** (0.0-0.5): Professional, respectful, establishing rapport
- **Established relationships** (0.5-0.8): Familiar, friendly, more personal
- **Close relationships** (0.8-1.0): Warm, supportive, very personal

### 5. System Prompt Generation

Generates comprehensive system prompts for LLMs:
- Personality traits and conversational style
- Roleplay tone and response style
- Expertise areas
- Emotional intelligence instructions
- Relationship context
- User preferences

## Usage Examples

### Basic Roleplay System

```python
from morgan_server.empathic import RoleplaySystem, RoleplayConfig

# Create with default configuration
system = RoleplaySystem()

# Generate a response
response = system.generate_response("Hello! How can I help you?")

print(f"Response: {response.response_text}")
print(f"Tone: {response.tone_applied.value}")
print(f"Style: {response.style_applied.value}")
```

### Custom Configuration

```python
from morgan_server.empathic import (
    RoleplaySystem,
    RoleplayConfig,
    RoleplayTone,
    ResponseStyle,
    PersonalityTrait
)

# Create custom configuration
config = RoleplayConfig(
    character_name="TechMentor",
    character_description="A friendly technical mentor",
    tone=RoleplayTone.MENTOR,
    response_style=ResponseStyle.TECHNICAL,
    personality_traits={
        PersonalityTrait.WARMTH: 0.8,
        PersonalityTrait.FORMALITY: 0.4,
        PersonalityTrait.EMPATHY: 0.9
    },
    background_story="Expert software engineer with 15 years of experience",
    expertise_areas=["Python", "System Design", "Testing"]
)

system = RoleplaySystem(config)
```

### Context-Aware Response

```python
from morgan_server.empathic import RoleplayContext, EmotionalDetection, EmotionalTone

# Detect emotion from user message
message = "I'm struggling with this bug and feeling frustrated"
emotional_state = system.detect_and_integrate_emotion(message, "user123")

# Create context
context = RoleplayContext(
    user_id="user123",
    relationship_depth=0.7,
    emotional_state=emotional_state,
    user_preferences={
        "response_length": "detailed",
        "technical_level": "high"
    }
)

# Generate response with full context
response = system.generate_response(
    "Let me help you debug that issue.",
    context
)

print(f"Emotional adjustment: {response.emotional_adjustment}")
print(f"Relationship notes: {response.relationship_notes}")
print(f"Personality notes: {response.personality_notes}")
```

### System Prompt Generation

```python
# Generate system prompt for LLM
context = RoleplayContext(
    user_id="user123",
    relationship_depth=0.8,
    user_preferences={"response_length": "concise"}
)

prompt = system.get_system_prompt(context)
print(prompt)
# Output includes personality, tone, style, expertise, and relationship context
```

### Emotional Trend Analysis

```python
# Track emotional patterns
system.detect_and_integrate_emotion("I'm happy today!", "user123")
system.detect_and_integrate_emotion("Feeling great!", "user123")

# Get emotional trend
trend = system.get_emotional_trend("user123")
print(f"Dominant tone: {trend['dominant_tone']}")
print(f"Trend: {trend['trend']}")
```

## Implementation Details

### Tone Adaptation

Tone adapts based on relationship depth:
- **Professional** → **Friendly** (at depth > 0.7)
- **Friendly** → **Companion** (at depth > 0.7)

### Style Adaptation

Style adapts based on user preferences:
- `response_length: "concise"` → **Concise** style
- `response_length: "detailed"` → **Detailed** style
- `technical_level: "high"` → **Technical** style

### Integration Architecture

```
RoleplaySystem
├── EmotionalIntelligence (emotion detection & adjustment)
├── PersonalitySystem (personality traits & consistency)
└── Configuration (roleplay settings)
```

## Testing

Comprehensive test suite with 37 unit tests covering:
- ✅ Roleplay configuration loading and updates
- ✅ Context-aware response generation
- ✅ Emotional intelligence integration
- ✅ Relationship-aware behavior
- ✅ System prompt generation
- ✅ Tone and style adaptation
- ✅ Context summary generation
- ✅ Integration with emotional intelligence
- ✅ Integration with personality system
- ✅ Full workflow integration

All tests passing: **37/37** ✅

## Requirements Validated

This implementation validates **Requirement 1.1** from the design document:
- ✅ Base roleplay configuration (personality, tone, style)
- ✅ Context-aware response logic
- ✅ Emotional intelligence integration
- ✅ Relationship-aware behavior

## API Reference

### RoleplayConfig

Configuration dataclass for roleplay behavior:
- `character_name: str` - Character name (default: "Morgan")
- `character_description: Optional[str]` - Character description
- `tone: RoleplayTone` - Roleplay tone (default: FRIENDLY)
- `response_style: ResponseStyle` - Response style (default: CONVERSATIONAL)
- `personality_traits: Dict[PersonalityTrait, float]` - Personality trait values
- `background_story: Optional[str]` - Character background
- `expertise_areas: List[str]` - Areas of expertise
- `communication_preferences: Dict[str, Any]` - Communication preferences
- `emotional_intelligence_enabled: bool` - Enable EI (default: True)
- `relationship_awareness_enabled: bool` - Enable relationship awareness (default: True)

### RoleplayContext

Context dataclass for response generation:
- `user_id: Optional[str]` - User identifier
- `conversation_history: List[Dict[str, str]]` - Conversation history
- `relationship_depth: float` - Relationship depth (0.0 to 1.0)
- `emotional_state: Optional[EmotionalDetection]` - Detected emotional state
- `user_preferences: Dict[str, Any]` - User preferences
- `session_metadata: Dict[str, Any]` - Session metadata

### RoleplayResponse

Response dataclass with metadata:
- `response_text: str` - Enhanced response text
- `tone_applied: RoleplayTone` - Applied tone
- `style_applied: ResponseStyle` - Applied style
- `emotional_adjustment: Optional[str]` - Emotional adjustment made
- `relationship_notes: List[str]` - Relationship-related notes
- `personality_notes: List[str]` - Personality-related notes
- `context_used: Dict[str, Any]` - Context information used

### RoleplaySystem Methods

- `generate_response(base_response: str, context: Optional[RoleplayContext]) -> RoleplayResponse`
  - Generate an enhanced roleplay response
  
- `get_system_prompt(context: Optional[RoleplayContext]) -> str`
  - Generate system prompt for LLM
  
- `update_config(**kwargs) -> None`
  - Update roleplay configuration
  
- `detect_and_integrate_emotion(message: str, user_id: Optional[str]) -> EmotionalDetection`
  - Detect emotion and integrate with EI system
  
- `get_emotional_trend(user_id: str, days: Optional[int]) -> Dict[str, Any]`
  - Get emotional trend analysis
  
- `get_context_summary(context: RoleplayContext) -> Dict[str, Any]`
  - Get context summary for debugging/logging
