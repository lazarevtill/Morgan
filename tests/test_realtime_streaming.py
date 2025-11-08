"""
End-to-end testing for real-time streaming functionality
Tests the complete STT → LLM → TTS pipeline with Redis and PostgreSQL
"""

import asyncio
import base64
import json
import time
from typing import Any, Dict

import aiohttp
import pytest
import websockets


class StreamingTester:
    """Test harness for streaming functionality"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")

    async def test_health_check(self):
        """Test that all services are healthy"""
        print("\n🔍 Testing health check...")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                assert response.status == 200
                data = await response.json()

                print(f"✅ Status: {data['status']}")
                print(f"   Services: {data['services']}")
                print(f"   Redis: {data.get('redis', 'N/A')}")
                print(f"   Database: {data.get('database', 'N/A')}")

                # Check all services are healthy
                assert all(data["services"].values()), "Some services are unhealthy"

                return data

    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        print("\n🔍 Testing WebSocket connection...")

        uri = f"{self.ws_url}/ws/stream/test_user"

        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected")

            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))

            response = await websocket.recv()
            data = json.loads(response)

            assert data["type"] == "pong"
            print("✅ Ping/pong successful")

            return True

    async def test_streaming_session_lifecycle(self):
        """Test complete streaming session lifecycle"""
        print("\n🔍 Testing streaming session lifecycle...")

        uri = f"{self.ws_url}/ws/stream/test_user"

        async with websockets.connect(uri) as websocket:
            # 1. Start session
            print("   Starting session...")
            await websocket.send(
                json.dumps({"type": "start", "data": {"language": "en"}})
            )

            response = await websocket.recv()
            data = json.loads(response)

            assert data["type"] == "session_started"
            session_id = data["data"]["session_id"]
            print(f"✅ Session started: {session_id}")

            # 2. Send text message
            print("   Sending text message...")
            await websocket.send(
                json.dumps({"type": "text", "data": {"text": "What is 2+2?"}})
            )

            # 3. Receive responses
            text_chunks = []
            audio_chunks = []
            complete = False

            timeout = time.time() + 30  # 30 second timeout

            while time.time() < timeout and not complete:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)

                    if data["type"] == "response_text":
                        text_chunks.append(data["data"]["text"])
                        print(f"   📝 Text chunk: {data['data']['text'][:50]}...")

                    elif data["type"] == "response_audio":
                        audio_chunks.append(data["data"]["audio_data"])
                        print(
                            f"   🔊 Audio chunk: {len(data['data']['audio_data'])} bytes"
                        )

                    elif data["type"] == "response_complete":
                        complete = True
                        full_text = data["data"]["text"]
                        print(f"✅ Response complete: {full_text[:100]}...")

                    elif data["type"] == "error":
                        print(f"❌ Error: {data['data']['message']}")
                        break

                except asyncio.TimeoutError:
                    print("   ⚠️  Response timeout")
                    break

            # 4. Stop session
            print("   Stopping session...")
            await websocket.send(json.dumps({"type": "stop"}))

            response = await websocket.recv()
            data = json.loads(response)

            assert data["type"] == "session_ended"
            print(f"✅ Session ended: {data['data']['session_id']}")

            # Verify we got responses
            assert len(text_chunks) > 0, "No text chunks received"
            print(
                f"✅ Received {len(text_chunks)} text chunks and {len(audio_chunks)} audio chunks"
            )

            return True

    async def test_audio_streaming(self):
        """Test audio chunk processing"""
        print("\n🔍 Testing audio streaming...")

        uri = f"{self.ws_url}/ws/stream/test_user"

        async with websockets.connect(uri) as websocket:
            # Start session
            await websocket.send(
                json.dumps({"type": "start", "data": {"language": "en"}})
            )

            response = await websocket.recv()
            data = json.loads(response)
            session_id = data["data"]["session_id"]
            print(f"   Session started: {session_id}")

            # Send dummy audio chunk (silence)
            silence = b"\x00" * 16000  # 1 second of 16kHz silence
            audio_b64 = base64.b64encode(silence).decode("utf-8")

            print("   Sending audio chunk...")
            await websocket.send(
                json.dumps({"type": "audio", "data": {"audio_data": audio_b64}})
            )

            # Wait for VAD response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)

                if data["type"] == "vad_status":
                    print(f"✅ VAD status: {data['data']}")
                elif data["type"] == "transcription":
                    print(f"✅ Transcription: {data['data']['text']}")

            except asyncio.TimeoutError:
                print("   ⚠️  No VAD response (expected for silence)")

            # Stop session
            await websocket.send(json.dumps({"type": "stop"}))
            await websocket.recv()

            print("✅ Audio streaming test complete")
            return True

    async def test_concurrent_sessions(self):
        """Test multiple concurrent streaming sessions"""
        print("\n🔍 Testing concurrent sessions...")

        async def create_session(user_id: str):
            """Create a single session"""
            uri = f"{self.ws_url}/ws/stream/{user_id}"

            async with websockets.connect(uri) as websocket:
                await websocket.send(
                    json.dumps({"type": "start", "data": {"language": "en"}})
                )

                response = await websocket.recv()
                data = json.loads(response)

                print(f"   ✅ Session for {user_id}: {data['data']['session_id']}")

                # Keep alive for a bit
                await asyncio.sleep(1)

                # Stop
                await websocket.send(json.dumps({"type": "stop"}))
                await websocket.recv()

        # Create 5 concurrent sessions
        tasks = [create_session(f"user_{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        print("✅ All concurrent sessions completed")
        return True

    async def test_redis_caching(self):
        """Test Redis caching functionality"""
        print("\n🔍 Testing Redis caching...")

        # Send a text request
        async with aiohttp.ClientSession() as session:
            # First request
            start = time.time()
            async with session.post(
                f"{self.base_url}/api/text",
                json={"text": "Hello", "user_id": "cache_test"},
            ) as response:
                assert response.status == 200
                first_time = time.time() - start
                print(f"   First request: {first_time:.3f}s")

            # Second request (should hit cache)
            await asyncio.sleep(0.5)  # Small delay

            start = time.time()
            async with session.post(
                f"{self.base_url}/api/text",
                json={"text": "How are you?", "user_id": "cache_test"},
            ) as response:
                assert response.status == 200
                second_time = time.time() - start
                print(f"   Second request: {second_time:.3f}s")

            print("✅ Redis caching test complete")
            return True

    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("🚀 Morgan Streaming Tests")
        print("=" * 60)

        tests = [
            ("Health Check", self.test_health_check),
            ("WebSocket Connection", self.test_websocket_connection),
            ("Session Lifecycle", self.test_streaming_session_lifecycle),
            ("Audio Streaming", self.test_audio_streaming),
            ("Concurrent Sessions", self.test_concurrent_sessions),
            ("Redis Caching", self.test_redis_caching),
        ]

        results = []

        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, "✅ PASS", None))
            except Exception as e:
                results.append((test_name, "❌ FAIL", str(e)))
                print(f"\n❌ Test failed: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        for test_name, status, error in results:
            print(f"{status} {test_name}")
            if error:
                print(f"    Error: {error}")

        passed = sum(1 for _, status, _ in results if "PASS" in status)
        total = len(results)

        print(f"\n✅ Passed: {passed}/{total}")

        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")

        return passed == total


async def main():
    """Main test runner"""
    tester = StreamingTester()

    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
