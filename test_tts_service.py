"""
TTS Service Integration Test
Tests the CSM-streaming implementation in the TTS service
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.tts.service import TTSService
from shared.config.base import ServiceConfig
from shared.models.base import TTSRequest


async def test_tts_service():
    """Test TTS service with CSM model"""
    print("=" * 80)
    print("TTS Service CSM Integration Test")
    print("=" * 80)

    # Check for HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n⚠️  WARNING: HF_TOKEN environment variable not set")
        print("The CSM model may require authentication to download.")
        print("Set it with: export HF_TOKEN=your_token_here")
        print("\nProceeding anyway (public models may work)...\n")

    try:
        # Initialize TTS service
        print("\n1. Initializing TTS Service...")
        config = ServiceConfig("tts", config_dir=str(project_root / "config"))
        tts_service = TTSService(config)

        # Start service (loads model)
        print("\n2. Starting TTS Service (loading CSM model)...")
        print("   This may take a few minutes on first run to download model...")
        await tts_service.start()
        print("   ✓ TTS Service started successfully")

        # Check model details
        print("\n3. Model Information:")
        print(f"   Model Type: {tts_service.model_type}")
        print(f"   Device: {tts_service.device}")
        print(f"   Sample Rate: {tts_service.csm_sample_rate}Hz")
        print(f"   Model Class: {type(tts_service.csm_model).__name__}")

        # Test 1: Health check
        print("\n4. Running Health Check...")
        health = await tts_service.health_check()
        print(f"   Status: {health['status']}")
        if health['status'] == 'healthy':
            print("   ✓ Health check passed")
        else:
            print(f"   ✗ Health check failed: {health.get('error', 'Unknown error')}")
            return

        # Test 2: List voices
        print("\n5. Listing Available Voices...")
        voices = await tts_service.list_voices()
        print(f"   Available voices: {voices['voices']}")
        print(f"   Current voice: {voices['current_voice']}")

        # Test 3: Generate simple speech
        print("\n6. Testing Speech Generation...")
        test_text = "Hello, this is a test of the CSM streaming text to speech system."
        request = TTSRequest(
            text=test_text,
            voice="default"
        )

        print(f"   Input text: '{test_text}'")
        response = await tts_service.generate_speech(request)

        print(f"   ✓ Audio generated successfully")
        print(f"   Audio format: {response.format}")
        print(f"   Sample rate: {response.sample_rate}Hz")
        print(f"   Duration: {response.duration:.2f}s")
        print(f"   Audio size: {len(response.audio_data)} bytes")

        # Save test audio
        output_file = project_root / "test_output.wav"
        with open(output_file, 'wb') as f:
            f.write(response.audio_data)
        print(f"   Audio saved to: {output_file}")

        # Test 4: Streaming generation
        print("\n7. Testing Streaming Speech Generation...")
        stream_request = TTSRequest(
            text="This is a streaming test.",
            voice="default"
        )

        chunk_count = 0
        total_bytes = 0
        async for chunk in tts_service.generate_speech_stream(stream_request):
            chunk_count += 1
            total_bytes += len(chunk)

        print(f"   ✓ Streaming completed")
        print(f"   Chunks received: {chunk_count}")
        print(f"   Total bytes: {total_bytes}")

        # Test 5: Long text
        print("\n8. Testing Long Text Generation...")
        long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet. "
            "Text to speech synthesis has come a long way in recent years. "
            "Modern neural models can produce very natural sounding speech. "
            "The CSM model is designed for real-time streaming applications."
        )
        long_request = TTSRequest(text=long_text, voice="default")
        long_response = await tts_service.generate_speech(long_request)

        print(f"   ✓ Long text generated successfully")
        print(f"   Duration: {long_response.duration:.2f}s")
        print(f"   Audio size: {len(long_response.audio_data)} bytes")

        # Test 6: Special characters handling
        print("\n9. Testing Special Characters...")
        special_text = "Hello! How are you? I'm fine. Testing 1-2-3... Done."
        special_request = TTSRequest(text=special_text, voice="default")
        special_response = await tts_service.generate_speech(special_request)
        print(f"   ✓ Special characters handled successfully")

        # Stop service
        print("\n10. Stopping TTS Service...")
        await tts_service.stop()
        print("   ✓ Service stopped successfully")

        # Summary
        print("\n" + "=" * 80)
        print("✓ All TTS Service Tests Passed!")
        print("=" * 80)
        print("\nKey Results:")
        print(f"  • CSM model loaded successfully")
        print(f"  • Sample rate: {tts_service.csm_sample_rate}Hz")
        print(f"  • Speech generation: Working")
        print(f"  • Streaming: Working")
        print(f"  • Special characters: Handled")
        print(f"  • Test audio: {output_file}")
        print("\nThe TTS service is fully operational with CSM-streaming integration!")

    except ImportError as ie:
        print(f"\n✗ Import Error: {ie}")
        print("\nThis usually means transformers>=4.52.1 is not installed.")
        print("Please ensure you have the correct transformers version:")
        print("  pip install 'transformers>=4.52.1'")
        return 1

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_tts_service())
    sys.exit(exit_code)
