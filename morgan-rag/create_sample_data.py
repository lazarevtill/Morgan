#!/usr/bin/env python3
"""
Create Sample Data Script

Create sample documents and memories to test the enhanced search integration.
"""

import sys
import os
import uuid
from datetime import datetime, timedelta

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.vector_db.client import VectorDBClient
from morgan.embeddings.service import get_embedding_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_documents():
    """Create sample documents for testing."""
    sample_docs = [
        {
            "content": "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers. To deploy Docker containers, you first need to create a Dockerfile that defines your application environment, then build an image using 'docker build', and finally run the container with 'docker run'. For production deployments, consider using Docker Compose or Kubernetes for orchestration.",
            "source": "docker-guide.md",
            "category": "documentation",
            "metadata": {
                "title": "Docker Deployment Guide",
                "author": "DevOps Team",
                "tags": ["docker", "containers", "deployment"],
            },
        },
        {
            "content": "Python API best practices include using proper HTTP status codes, implementing authentication and authorization, validating input data, handling errors gracefully, and documenting your API with tools like OpenAPI/Swagger. Use frameworks like FastAPI or Flask for rapid development. Always implement rate limiting and logging for production APIs.",
            "source": "python-api-guide.md",
            "category": "documentation",
            "metadata": {
                "title": "Python API Best Practices",
                "author": "Backend Team",
                "tags": ["python", "api", "best-practices"],
            },
        },
        {
            "content": "Database connection troubleshooting steps: 1) Check network connectivity, 2) Verify credentials and permissions, 3) Ensure the database service is running, 4) Check firewall settings, 5) Review connection string format, 6) Test with a simple client tool, 7) Check database logs for errors. Common issues include timeout settings, SSL configuration, and connection pool limits.",
            "source": "database-troubleshooting.md",
            "category": "troubleshooting",
            "metadata": {
                "title": "Database Connection Troubleshooting",
                "author": "DBA Team",
                "tags": ["database", "troubleshooting", "connectivity"],
            },
        },
        {
            "content": "Machine learning tutorial: Start with understanding the problem type (classification, regression, clustering). Prepare your data by cleaning, normalizing, and splitting into train/test sets. Choose appropriate algorithms (linear regression, decision trees, neural networks). Train your model, evaluate performance using metrics like accuracy or RMSE, and iterate to improve results. Use libraries like scikit-learn, TensorFlow, or PyTorch.",
            "source": "ml-tutorial.md",
            "category": "tutorial",
            "metadata": {
                "title": "Machine Learning Tutorial",
                "author": "ML Team",
                "tags": ["machine-learning", "tutorial", "data-science"],
            },
        },
    ]

    return sample_docs


def create_sample_memories():
    """Create sample conversation memories for testing."""
    sample_memories = [
        {
            "content": "Q: How do I fix Docker permission errors on Linux?\nA: You can fix Docker permission errors by adding your user to the docker group with 'sudo usermod -aG docker $USER', then log out and back in. Alternatively, you can run Docker commands with sudo, but adding to the group is the recommended approach for development.",
            "conversation_id": "conv_001",
            "turn_id": "turn_001",
            "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "feedback_rating": 5,
            "user_id": "user123",
            "emotional_context": {
                "primary_emotion": "frustration",
                "intensity": 0.7,
                "confidence": 0.8,
            },
        },
        {
            "content": "Q: What's the difference between REST and GraphQL APIs?\nA: REST uses multiple endpoints with standard HTTP methods, while GraphQL uses a single endpoint with a query language. GraphQL allows clients to request exactly the data they need, reducing over-fetching. REST is simpler and more widely adopted, while GraphQL offers more flexibility for complex data requirements.",
            "conversation_id": "conv_002",
            "turn_id": "turn_001",
            "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "feedback_rating": 4,
            "user_id": "user123",
            "emotional_context": {
                "primary_emotion": "curiosity",
                "intensity": 0.6,
                "confidence": 0.9,
            },
        },
        {
            "content": "Q: My database keeps timing out, what should I check?\nA: Database timeouts can be caused by several factors: slow queries, insufficient connection pool size, network latency, or database overload. Start by checking your query performance with EXPLAIN, monitor connection pool usage, and review database server resources (CPU, memory, disk I/O). Also check your connection timeout settings.",
            "conversation_id": "conv_003",
            "turn_id": "turn_001",
            "timestamp": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
            "feedback_rating": 5,
            "user_id": "user456",
            "emotional_context": {
                "primary_emotion": "anxiety",
                "intensity": 0.8,
                "confidence": 0.7,
            },
        },
    ]

    return sample_memories


def create_sample_data():
    """Create sample data for testing enhanced search integration."""
    print("📝 Creating sample data for enhanced search testing...")

    try:
        # Initialize services
        vector_db = VectorDBClient()
        embedding_service = get_embedding_service()

        # Create sample documents
        print("📚 Creating sample documents...")
        sample_docs = create_sample_documents()

        doc_points = []
        for i, doc in enumerate(sample_docs):
            try:
                # Generate embedding for the document
                embedding = embedding_service.encode(
                    text=doc["content"], instruction="document"
                )

                # Create point for vector database
                point = {
                    "id": f"doc_{i+1}",
                    "vector": {"fine": embedding},  # Using fine embedding for now
                    "payload": {
                        "content": doc["content"],
                        "source": doc["source"],
                        "category": doc["category"],
                        "metadata": doc["metadata"],
                        "indexed_at": datetime.utcnow().isoformat(),
                    },
                }
                doc_points.append(point)

            except Exception as e:
                print(f"⚠️ Failed to create embedding for document {i+1}: {e}")
                continue

        # Insert documents into knowledge collection
        if doc_points:
            success = vector_db.upsert_points("morgan_knowledge", doc_points)
            if success:
                print(f"✅ Created {len(doc_points)} sample documents")
            else:
                print("❌ Failed to insert sample documents")

        # Create sample memories
        print("💭 Creating sample memories...")
        sample_memories = create_sample_memories()

        memory_points = []
        for i, memory in enumerate(sample_memories):
            try:
                # Generate embedding for the memory content
                embedding = embedding_service.encode(
                    text=memory["content"], instruction="document"
                )

                # Create point for memory collection
                point = {
                    "id": f"memory_{i+1}",
                    "vector": embedding,
                    "payload": {
                        "question": memory["content"]
                        .split("A:")[0]
                        .replace("Q:", "")
                        .strip(),
                        "answer": (
                            memory["content"].split("A:")[1].strip()
                            if "A:" in memory["content"]
                            else ""
                        ),
                        "conversation_id": memory["conversation_id"],
                        "turn_id": memory["turn_id"],
                        "timestamp": memory["timestamp"],
                        "feedback_rating": memory["feedback_rating"],
                        "user_id": memory["user_id"],
                        "emotional_context": memory["emotional_context"],
                    },
                }
                memory_points.append(point)

            except Exception as e:
                print(f"⚠️ Failed to create embedding for memory {i+1}: {e}")
                continue

        # Insert memories into turns collection
        if memory_points:
            success = vector_db.upsert_points("morgan_turns", memory_points)
            if success:
                print(f"✅ Created {len(memory_points)} sample memories")
            else:
                print("❌ Failed to insert sample memories")

        # Create sample companion profile
        print("👤 Creating sample companion profile...")
        try:
            profile_embedding = embedding_service.encode(
                text="Technical user interested in backend development, Docker, Python APIs, and database management",
                instruction="document",
            )

            profile_success = vector_db.upsert_companion_profile(
                user_id="user123",
                profile_embedding=profile_embedding,
                profile_data={
                    "user_id": "user123",
                    "preferred_name": "Alex",
                    "communication_style": "technical",
                    "interests": ["python", "docker", "api", "database", "backend"],
                    "interaction_count": 15,
                    "relationship_duration_days": 30,
                    "last_interaction": datetime.utcnow().isoformat(),
                },
            )

            if profile_success:
                print("✅ Created sample companion profile")
            else:
                print("❌ Failed to create companion profile")

        except Exception as e:
            print(f"⚠️ Failed to create companion profile: {e}")

        print("\n📊 Final Collection Statistics:")
        stats = vector_db.get_collection_stats()

        for collection_name, info in stats.items():
            if info.points_count > 0:
                print(f"  {collection_name}: {info.points_count} points")

        print("\n🎉 Sample data creation complete!")
        return True

    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        logger.error(f"Sample data creation failed: {e}")
        return False


if __name__ == "__main__":
    success = create_sample_data()
    sys.exit(0 if success else 1)
