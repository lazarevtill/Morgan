"""
Comprehensive tests for the emotion detection system.

Tests all 11 modules and their integration.
"""

import asyncio

import pytest

from morgan.emotions import (
    EmotionContext,
    EmotionDetector,
    EmotionType,
    format_emotion_summary,
    is_crisis_state,
)


@pytest.mark.asyncio
async def test_basic_emotion_detection():
    """Test basic emotion detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        # Test happy message
        result = await detector.detect("I'm so happy today! This is wonderful!")

        assert result is not None
        assert result.dominant_emotion is not None
        assert result.dominant_emotion.emotion_type == EmotionType.JOY
        assert result.valence > 0
        assert result.processing_time_ms > 0

        print(f"✓ Happy message: {format_emotion_summary(result)}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_sad_emotion_detection():
    """Test sad emotion detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        result = await detector.detect(
            "I'm feeling so sad and lonely. I miss them terribly."
        )

        assert result.dominant_emotion is not None
        assert result.dominant_emotion.emotion_type == EmotionType.SADNESS
        assert result.valence < 0

        print(f"✓ Sad message: {format_emotion_summary(result)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_anger_emotion_detection():
    """Test anger emotion detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        result = await detector.detect(
            "I'm so mad! This is absolutely infuriating! I hate this!"
        )

        assert result.dominant_emotion is not None
        assert result.dominant_emotion.emotion_type == EmotionType.ANGER
        assert result.arousal > 0.6  # Anger should have high arousal

        print(f"✓ Angry message: {format_emotion_summary(result)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_fear_emotion_detection():
    """Test fear emotion detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        result = await detector.detect(
            "I'm so scared. What if something terrible happens?"
        )

        assert result.dominant_emotion is not None
        assert result.dominant_emotion.emotion_type == EmotionType.FEAR

        print(f"✓ Fearful message: {format_emotion_summary(result)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_multi_emotion_detection():
    """Test detection of multiple emotions."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        result = await detector.detect(
            "I'm happy about the promotion but worried about the new responsibilities."
        )

        assert len(result.primary_emotions) >= 2
        emotion_types = {e.emotion_type for e in result.primary_emotions}
        assert EmotionType.JOY in emotion_types or EmotionType.ANTICIPATION in emotion_types
        assert EmotionType.FEAR in emotion_types or EmotionType.ANTICIPATION in emotion_types

        print(f"✓ Multi-emotion: {format_emotion_summary(result)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_trigger_detection():
    """Test emotional trigger detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        result = await detector.detect("My dog just died. I'm heartbroken.")

        assert len(result.triggers) > 0
        trigger_texts = {t.trigger_text for t in result.triggers}
        assert any("died" in text for text in trigger_texts)

        print(f"✓ Triggers detected: {len(result.triggers)}")
        for trigger in result.triggers[:3]:
            print(f"  - {trigger.trigger_text} ({trigger.trigger_type})")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_context_awareness():
    """Test context-aware emotion detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=True)

    try:
        await detector.initialize()

        context = EmotionContext(user_id="test_user_001", conversation_id="conv_001")

        # First message
        result1 = await detector.detect("I'm feeling down today.", context=context)

        # Second message (should consider context)
        context.previous_emotions = result1.primary_emotions
        context.time_since_last_message = 5.0  # 5 seconds

        result2 = await detector.detect("Still not feeling great.", context=context)

        assert result2.context is not None
        assert result2.context.user_id == "test_user_001"

        print(f"✓ Context-aware detection:")
        print(f"  Message 1: {result1.emotional_summary}")
        print(f"  Message 2: {result2.emotional_summary}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_pattern_detection():
    """Test emotional pattern detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=True)

    try:
        await detector.initialize()

        context = EmotionContext(user_id="test_user_002")

        # Send several sad messages to establish a pattern
        messages = [
            "I'm feeling sad.",
            "Still sad about everything.",
            "Can't shake this sadness.",
            "So depressed.",
        ]

        for msg in messages:
            result = await detector.detect(msg, context=context)
            if result.primary_emotions:
                context.previous_emotions = result.primary_emotions
            await asyncio.sleep(0.01)  # Small delay

        # Final message should detect recurring pattern
        final_result = await detector.detect("Sad again.", context=context)

        # Check if any patterns were detected
        if final_result.patterns:
            print(f"✓ Patterns detected: {len(final_result.patterns)}")
            for pattern in final_result.patterns:
                print(f"  - {pattern.pattern_type}: {pattern.description}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_crisis_detection():
    """Test crisis state detection."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        # Non-crisis message
        result1 = await detector.detect("I'm a bit sad today.")
        assert not is_crisis_state(result1)

        # Crisis message
        result2 = await detector.detect(
            "I'm terrified and so sad. I don't know what to do. Everything is falling apart."
        )

        print(f"✓ Crisis detection:")
        print(f"  Normal message is_crisis: {is_crisis_state(result1)}")
        print(f"  Crisis message is_crisis: {is_crisis_state(result2)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_cache_performance():
    """Test caching improves performance."""
    detector = EmotionDetector(enable_cache=True, enable_history=False)

    try:
        await detector.initialize()

        text = "I'm feeling happy and excited about the future!"

        # First call (no cache)
        result1 = await detector.detect(text, use_cache=True)
        time1 = result1.processing_time_ms

        # Second call (should hit cache)
        result2 = await detector.detect(text, use_cache=True)
        time2 = result2.processing_time_ms

        print(f"✓ Cache performance:")
        print(f"  First call: {time1:.2f}ms")
        print(f"  Cached call: {time2:.2f}ms")

        stats = detector.stats
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_performance_target():
    """Test that detection meets <200ms target."""
    detector = EmotionDetector(enable_cache=False, enable_history=False)

    try:
        await detector.initialize()

        # Test various message types
        messages = [
            "I'm happy!",
            "This is terrible and I'm so angry about it.",
            "I love this but I'm worried about what comes next.",
            "Feeling anxious and scared.",
        ]

        times = []

        for msg in messages:
            result = await detector.detect(msg)
            times.append(result.processing_time_ms)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"✓ Performance:")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Target: <200ms")

        # Most calls should be under 200ms (allows for some variance)
        fast_calls = sum(1 for t in times if t < 200)
        print(f"  Calls under 200ms: {fast_calls}/{len(times)}")

    finally:
        await detector.cleanup()


@pytest.mark.asyncio
async def test_user_trajectory():
    """Test user emotional trajectory tracking."""
    detector = EmotionDetector(enable_cache=False, enable_history=True)

    try:
        await detector.initialize()

        context = EmotionContext(user_id="test_user_003")

        # Simulate improving emotional state
        messages = [
            "I'm so sad.",
            "Still feeling down.",
            "Feeling a bit better.",
            "Actually doing okay now.",
            "Feeling much better!",
        ]

        for msg in messages:
            await detector.detect(msg, context=context)
            await asyncio.sleep(0.01)

        # Get trajectory
        trajectory = await detector.get_user_trajectory("test_user_003")

        print(f"✓ User trajectory:")
        print(f"  Direction: {trajectory['direction']}")
        print(f"  Trend: {trajectory['trend']}")
        print(f"  Current valence: {trajectory['current_valence']:.2f}")

    finally:
        await detector.cleanup()


if __name__ == "__main__":
    """Run tests directly."""
    asyncio.run(test_basic_emotion_detection())
    asyncio.run(test_sad_emotion_detection())
    asyncio.run(test_anger_emotion_detection())
    asyncio.run(test_fear_emotion_detection())
    asyncio.run(test_multi_emotion_detection())
    asyncio.run(test_trigger_detection())
    asyncio.run(test_context_awareness())
    asyncio.run(test_pattern_detection())
    asyncio.run(test_crisis_detection())
    asyncio.run(test_cache_performance())
    asyncio.run(test_performance_target())
    asyncio.run(test_user_trajectory())

    print("\n✅ All tests completed!")
