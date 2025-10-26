#!/usr/bin/env python3
"""
Test script to verify end-to-end audio flow in Morgan
"""
import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"
TTS_URL = "http://localhost:8002"

def test_health():
    """Test system health"""
    print("=" * 60)
    print("Testing system health...")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ System status: {data['status']}")
        print(f"✓ Services: LLM={data['services']['llm']}, TTS={data['services']['tts']}, STT={data['services']['stt']}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False

def test_tts_direct():
    """Test TTS service directly"""
    print("\n" + "=" * 60)
    print("Testing TTS service directly...")
    print("=" * 60)

    payload = {
        "text": "Hello, this is a test",
        "voice": "af_heart"
    }

    response = requests.post(f"{TTS_URL}/generate", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"✓ TTS response received")
        print(f"  - audio_data length: {len(data['audio_data'])} hex chars")
        print(f"  - audio_data bytes: {len(data['audio_data']) // 2} bytes")
        print(f"  - format: {data['format']}")
        print(f"  - sample_rate: {data['sample_rate']}")
        print(f"  - duration: {data['duration']:.2f}s")

        # Validate hex string
        hex_str = data['audio_data']
        try:
            # Try to convert first 100 chars to validate it's hex
            test_bytes = bytes.fromhex(hex_str[:100])
            print(f"✓ Hex string is valid")
            print(f"  - First 8 bytes (hex): {test_bytes[:8].hex()}")
            print(f"  - First 4 bytes (ascii): {test_bytes[:4].decode('ascii', errors='ignore')}")

            # Check if it's a WAV file
            if test_bytes[:4] == b'RIFF':
                print(f"✓ Audio is valid WAV format")
                return True, data
            else:
                print(f"✗ Audio header is not RIFF: {test_bytes[:4]}")
                return False, None

        except ValueError as e:
            print(f"✗ Invalid hex string: {e}")
            return False, None
    else:
        print(f"✗ TTS request failed: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        return False, None

def test_text_to_audio():
    """Test full text-to-audio flow"""
    print("\n" + "=" * 60)
    print("Testing full text-to-audio flow...")
    print("=" * 60)

    payload = {
        "text": "What is the weather today?",
        "user_id": "test_user"
    }

    response = requests.post(f"{BASE_URL}/api/text", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Text processing successful")
        print(f"  - Response text: {data['text'][:100]}...")

        if data.get('audio'):
            print(f"  - Audio included: YES")
            print(f"    - audio length: {len(data['audio'])} hex chars")
            print(f"    - audio bytes: {len(data['audio']) // 2} bytes")

            # Validate hex
            try:
                test_bytes = bytes.fromhex(data['audio'][:100])
                print(f"  - Audio hex valid: YES")
                print(f"  - Audio header: {test_bytes[:4]}")
                return True, data
            except:
                print(f"  - Audio hex valid: NO")
                return False, None
        else:
            print(f"  - Audio included: NO")
            return False, None
    else:
        print(f"✗ Text processing failed: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        return False, None

def test_audio_metadata():
    """Test audio processing with metadata"""
    print("\n" + "=" * 60)
    print("Testing audio endpoint metadata...")
    print("=" * 60)

    # Create a simple audio file to test
    # We'll use the TTS service to generate one
    tts_response = requests.post(f"{TTS_URL}/generate", json={"text": "Test", "voice": "af_heart"})

    if tts_response.status_code != 200:
        print(f"✗ Could not generate test audio")
        return False

    # Get hex audio
    tts_data = tts_response.json()
    hex_audio = tts_data['audio_data']
    audio_bytes = bytes.fromhex(hex_audio)

    print(f"Generated test audio: {len(audio_bytes)} bytes")

    # Now test audio endpoint (we'll just check if it accepts the format)
    files = {'file': ('test.wav', audio_bytes, 'audio/wav')}
    data = {'user_id': 'test_user', 'device_type': 'test', 'language': 'auto'}

    response = requests.post(f"{BASE_URL}/api/audio", files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Audio endpoint successful")
        print(f"  - Transcription: {result.get('transcription', 'N/A')}")
        print(f"  - AI Response: {result.get('ai_response', 'N/A')[:100]}...")

        if result.get('audio'):
            print(f"  - Response audio: YES ({len(result['audio']) // 2} bytes)")
            return True
        else:
            print(f"  - Response audio: NO")
            print(f"  - Metadata: {json.dumps(result.get('metadata', {}), indent=2)}")
            return False
    else:
        print(f"✗ Audio endpoint failed: {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MORGAN AUDIO FLOW TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Health
    results.append(("System Health", test_health()))
    time.sleep(0.5)

    # Test 2: Direct TTS
    success, data = test_tts_direct()
    results.append(("Direct TTS", success))
    time.sleep(0.5)

    # Test 3: Text-to-Audio
    success, data = test_text_to_audio()
    results.append(("Text-to-Audio", success))
    time.sleep(0.5)

    # Test 4: Audio endpoint metadata
    success = test_audio_metadata()
    results.append(("Audio Endpoint", success))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")

    print("=" * 60)

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
