"""
Example: Using the refactored Morgan Assistant core system.

Demonstrates:
- Full async/await architecture
- Emotion detection integration
- Memory management
- Context handling
- RAG search
- Response generation
- Learning system updates
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from morgan.core import (
    MorganAssistant,
    Message,
    MessageRole,
)
from morgan.services.embedding_service import EmbeddingService, EmbeddingConfig
from morgan.vector_db.client import QdrantClient, QdrantConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main example demonstrating the assistant."""
    logger.info("Starting Morgan Assistant example")

    # Configuration
    storage_path = Path.home() / ".morgan" / "examples"
    storage_path.mkdir(parents=True, exist_ok=True)

    # Initialize embedding service (optional, for RAG)
    embedding_config = EmbeddingConfig(
        base_url="http://localhost:11434",
        model="qwen3-embedding:latest",
    )
    embedding_service = EmbeddingService(config=embedding_config)

    # Initialize vector database (optional, for RAG)
    # Note: Requires Qdrant running
    try:
        qdrant_config = QdrantConfig(
            host="localhost",
            port=6333,
        )
        vector_db = QdrantClient(config=qdrant_config)
        enable_rag = True
        logger.info("Vector DB connected, RAG enabled")
    except Exception as e:
        logger.warning(f"Vector DB not available: {e}, RAG disabled")
        vector_db = None
        enable_rag = False

    # Initialize Morgan Assistant
    assistant = MorganAssistant(
        storage_path=storage_path,
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:latest",
        vector_db=vector_db,
        embedding_service=embedding_service,
        enable_emotion_detection=True,
        enable_learning=True,
        enable_rag=enable_rag,
    )

    # Initialize assistant
    await assistant.initialize()

    try:
        # Test user
        user_id = "example_user_001"
        session_id = "session_001"

        # Example 1: Simple message processing
        logger.info("\n=== Example 1: Simple Message ===")
        response1 = await assistant.process_message(
            user_id=user_id,
            message="Hello! I'm having a great day!",
            session_id=session_id,
        )

        logger.info(f"Response: {response1.content}")
        logger.info(f"Emotion detected: {response1.emotion}")
        logger.info(f"Generation time: {response1.generation_time_ms:.2f}ms")

        # Example 2: Message with context
        logger.info("\n=== Example 2: Message with Context ===")
        response2 = await assistant.process_message(
            user_id=user_id,
            message="Can you tell me more about what we just discussed?",
            session_id=session_id,
        )

        logger.info(f"Response: {response2.content}")
        logger.info(f"Sources used: {len(response2.sources)}")

        # Example 3: Emotional message
        logger.info("\n=== Example 3: Emotional Message ===")
        response3 = await assistant.process_message(
            user_id=user_id,
            message="I'm feeling a bit worried about my upcoming presentation.",
            session_id=session_id,
        )

        logger.info(f"Response: {response3.content}")
        if response3.emotion and response3.emotion.primary_emotion:
            logger.info(
                f"Detected emotion: {response3.emotion.primary_emotion.emotion_type.value} "
                f"(intensity: {response3.emotion.primary_emotion.intensity:.2f})"
            )

        # Example 4: Streaming response
        logger.info("\n=== Example 4: Streaming Response ===")
        logger.info("Streaming response: ", end="", flush=True)

        full_response = ""
        async for chunk in assistant.stream_response(
            user_id=user_id,
            message="Tell me a short story about a robot learning to dance.",
            session_id=session_id,
        ):
            print(chunk, end="", flush=True)
            full_response += chunk

        print()  # New line after streaming
        logger.info(f"Streamed {len(full_response)} characters")

        # Example 5: Check assistant stats
        logger.info("\n=== Example 5: Assistant Statistics ===")
        stats = assistant.get_stats()

        logger.info(f"Total requests: {stats['metrics']['total_requests']}")
        logger.info(f"Successful: {stats['metrics']['successful_requests']}")
        logger.info(f"Failed: {stats['metrics']['failed_requests']}")
        logger.info(f"Degraded: {stats['metrics']['degraded_requests']}")

        logger.info("\nMemory stats:")
        logger.info(f"  Short-term sessions: {stats['memory']['short_term_sessions']}")
        logger.info(f"  Short-term messages: {stats['memory']['short_term_messages']}")
        logger.info(f"  User profiles cached: {stats['memory']['user_profiles_cached']}")

        logger.info("\nContext stats:")
        logger.info(f"  Contexts built: {stats['context']['metrics']['contexts_built']}")
        logger.info(f"  Contexts pruned: {stats['context']['metrics']['contexts_pruned']}")

        logger.info("\nResponse generation stats:")
        logger.info(f"  Total generations: {stats['response_generator']['metrics']['total_generations']}")
        logger.info(f"  Successful: {stats['response_generator']['metrics']['successful_generations']}")

        # Example 6: New session
        logger.info("\n=== Example 6: New Session ===")
        session_id_2 = "session_002"

        response6 = await assistant.process_message(
            user_id=user_id,
            message="Hi! This is a new conversation.",
            session_id=session_id_2,
        )

        logger.info(f"Response: {response6.content}")
        logger.info(f"New session created: {session_id_2}")

        # Example 7: Performance test
        logger.info("\n=== Example 7: Performance Test ===")
        start_time = datetime.now()

        perf_response = await assistant.process_message(
            user_id=user_id,
            message="What is machine learning?",
            session_id=session_id,
        )

        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(f"Total end-to-end time: {total_time_ms:.2f}ms")
        logger.info(f"Target is < 2000ms: {'PASS' if total_time_ms < 2000 else 'FAIL'}")

        if perf_response.metadata.get("metrics"):
            metrics = perf_response.metadata["metrics"]
            logger.info("\nDetailed breakdown:")
            timing = metrics.get("timing", {})
            for key, value in timing.items():
                logger.info(f"  {key}: {value:.2f}ms")

    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        await assistant.cleanup()
        await embedding_service.cleanup()

        logger.info("Example completed successfully!")


async def example_direct_components():
    """
    Example: Using components directly.

    Shows how to use individual components for more fine-grained control.
    """
    logger.info("\n=== Direct Component Usage ===")

    from morgan.core.memory import MemorySystem
    from morgan.core.context import ContextManager
    from morgan.core.response_generator import ResponseGenerator

    # Initialize components
    memory = MemorySystem()
    await memory.initialize()

    context_mgr = ContextManager()
    response_gen = ResponseGenerator()

    try:
        # Create some messages
        messages = [
            Message(
                role=MessageRole.USER,
                content="Hello!",
                timestamp=datetime.now(),
                message_id="msg_001",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Hi! How can I help you?",
                timestamp=datetime.now(),
                message_id="msg_002",
            ),
            Message(
                role=MessageRole.USER,
                content="Tell me about Python.",
                timestamp=datetime.now(),
                message_id="msg_003",
            ),
        ]

        # Store messages
        session_id = "direct_session_001"
        for msg in messages:
            await memory.store_message(session_id, msg)

        logger.info(f"Stored {len(messages)} messages")

        # Retrieve context
        retrieved = await memory.retrieve_context(session_id)
        logger.info(f"Retrieved {len(retrieved)} messages")

        # Build conversation context
        context = await context_mgr.build_context(
            messages=retrieved,
            user_id="test_user",
            session_id=session_id,
        )

        logger.info(f"Built context with {context.message_count} messages")
        logger.info(f"Total tokens: {context.total_tokens}")

        # Generate response
        response = await response_gen.generate(
            context=context,
            user_message="Tell me about Python.",
        )

        logger.info(f"Generated response: {response.content[:100]}...")

        # Get stats
        mem_stats = memory.get_stats()
        ctx_stats = context_mgr.get_stats()
        gen_stats = response_gen.get_stats()

        logger.info(f"\nMemory: {mem_stats['metrics']}")
        logger.info(f"Context: {ctx_stats['metrics']}")
        logger.info(f"Generator: {gen_stats['metrics']}")

    finally:
        await memory.cleanup()
        await response_gen.cleanup()


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Optionally run direct component example
    # asyncio.run(example_direct_components())
