"""
Integration test for hierarchical embedding service.

Tests the service with realistic content to demonstrate the three-scale approach.
"""

import pytest
from unittest.mock import Mock, patch
from morgan.vectorization.hierarchical_embeddings import (
    get_hierarchical_embedding_service,
)


class TestHierarchicalEmbeddingIntegration:
    """Integration tests for hierarchical embedding service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = get_hierarchical_embedding_service()

        # Mock embedding service to avoid actual API calls
        self.mock_embedding_service = Mock()
        self.service.embedding_service = self.mock_embedding_service

        # Mock clustering engine to return embeddings unchanged (avoid contrastive bias transformation)
        self.mock_clustering_engine = Mock()
        self.mock_clustering_engine.apply_contrastive_bias = Mock(side_effect=lambda emb, cat, scale: emb)
        self.service.clustering_engine = self.mock_clustering_engine

        # Mock different embeddings for each scale to verify they're different
        self.mock_embedding_service.encode_batch.return_value = [
            [0.1, 0.2, 0.3, 0.4],  # coarse embedding
            [0.5, 0.6, 0.7, 0.8],  # medium embedding
            [0.9, 1.0, 1.1, 1.2],  # fine embedding
        ]

    def test_realistic_code_document(self):
        """Test hierarchical embeddings for a realistic code document."""
        code_content = '''
def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate user credentials against the database.
    
    Args:
        username: User's login name
        password: User's password
        
    Returns:
        True if authentication successful, False otherwise
    """
    if not username or not password:
        return False
    
    user = get_user_by_username(username)
    if not user:
        return False
    
    return verify_password(password, user.password_hash)

class UserManager:
    """Manages user operations and authentication."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_user(self, username: str, email: str, password: str):
        """Create a new user account."""
        password_hash = hash_password(password)
        return self.db.insert_user(username, email, password_hash)
'''

        metadata = {
            "title": "User Authentication Module",
            "source": "auth/user_auth.py",
            "filename": "user_auth.py",
        }

        result = self.service.create_hierarchical_embeddings(
            code_content, metadata, use_cache=False
        )

        # Verify hierarchical structure
        assert result.coarse == [0.1, 0.2, 0.3, 0.4]
        assert result.medium == [0.5, 0.6, 0.7, 0.8]
        assert result.fine == [0.9, 1.0, 1.1, 1.2]

        # Verify coarse text focuses on high-level info
        coarse_text = result.get_text("coarse")
        assert "Category: code" in coarse_text
        assert "Title: User Authentication Module" in coarse_text
        assert "Type: Python code" in coarse_text
        assert "Source: auth/user_auth.py" in coarse_text

        # Verify medium text includes code elements and concepts
        medium_text = result.get_text("medium")
        assert "authenticate_user" in medium_text
        assert "UserManager" in medium_text
        assert "create_user" in medium_text

        # Verify fine text includes full content
        fine_text = result.get_text("fine")
        assert "def authenticate_user" in fine_text
        assert "class UserManager" in fine_text
        assert "Authenticate user credentials" in fine_text

        # Verify metadata
        assert result.metadata["category"] == "code"
        assert result.metadata["content_length"] == len(code_content)

    def test_realistic_documentation(self):
        """Test hierarchical embeddings for documentation."""
        doc_content = """
# Docker Deployment Guide

This guide explains how to deploy applications using Docker containers.

## Prerequisites

Before starting, ensure you have:
- Docker installed on your system
- Basic understanding of containerization
- Access to a Docker registry

## Building Images

### Creating a Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### Building the Image

Run the following command to build your image:

```bash
docker build -t myapp:latest .
```

## Deployment

### Local Deployment

For local testing:

```bash
docker run -p 3000:3000 myapp:latest
```

### Production Deployment

For production, use Docker Compose or Kubernetes for orchestration.
"""

        metadata = {
            "title": "Docker Deployment Guide",
            "source": "docs/deployment/docker.md",
            "filename": "docker.md",
        }

        result = self.service.create_hierarchical_embeddings(
            doc_content, metadata, use_cache=False
        )

        # Verify coarse text focuses on categories and topics
        coarse_text = result.get_text("coarse")
        assert "Category: documentation" in coarse_text
        assert "Title: Docker Deployment Guide" in coarse_text
        assert "Docker containers" in coarse_text or "Docker" in coarse_text

        # Verify medium text includes section headers
        medium_text = result.get_text("medium")
        assert "Prerequisites" in medium_text or "Building Images" in medium_text
        assert "Deployment" in medium_text or "Docker" in medium_text

        # Verify fine text includes full content
        fine_text = result.get_text("fine")
        assert "# Docker Deployment Guide" in fine_text
        assert "FROM node:16-alpine" in fine_text
        assert "docker build -t myapp:latest" in fine_text

    def test_batch_processing_mixed_content(self):
        """Test batch processing with mixed content types."""
        contents = [
            "def login(user, pwd): return authenticate(user, pwd)",
            "# API Reference\n\nThis endpoint handles user authentication.",
            "ERROR: Connection timeout. Check network settings.",
        ]

        metadatas = [
            {"source": "auth.py", "title": "Login Function"},
            {"source": "api.md", "title": "API Documentation"},
            {"source": "error.log", "title": "Error Log"},
        ]

        # Mock 9 embeddings (3 items × 3 scales each)
        batch_embeddings = [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],  # Item 1
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],  # Item 2
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],  # Item 3
        ]
        self.mock_embedding_service.encode_batch.return_value = batch_embeddings

        results = self.service.create_batch_hierarchical_embeddings(
            contents, metadatas, show_progress=False, use_cache=False
        )

        # Verify correct categorization
        assert results[0].metadata["category"] == "code"
        assert results[1].metadata["category"] == "api"  # Should detect API content
        assert (
            results[2].metadata["category"] == "troubleshooting"
        )  # Should detect error content

        # Verify different embeddings for each item
        assert results[0].coarse == [0.1, 0.1]
        assert results[1].coarse == [0.4, 0.4]
        assert results[2].coarse == [0.7, 0.7]

        # Verify coarse texts are category-appropriate
        assert "Category: code" in results[0].get_text("coarse")
        assert "Category: api" in results[1].get_text("coarse")
        assert "Category: troubleshooting" in results[2].get_text("coarse")

    def test_embedding_service_integration(self):
        """Test that the service correctly calls the embedding service."""
        content = "Sample content for testing"
        metadata = {"source": "test.txt"}

        self.service.create_hierarchical_embeddings(content, metadata, use_cache=False)

        # Verify embedding service was called
        self.mock_embedding_service.encode_batch.assert_called_once()

        # Verify call parameters
        call_args = self.mock_embedding_service.encode_batch.call_args
        texts = call_args[0][0]  # First positional argument (list of texts)
        kwargs = call_args[1]  # Keyword arguments

        # Should have 3 texts (coarse, medium, fine)
        assert len(texts) == 3

        # Should use document instruction
        assert kwargs["instruction"] == "document"

        # Should disable progress for single item
        assert kwargs["show_progress"] == False

        # Should respect cache setting
        assert kwargs["use_cache"] == False
