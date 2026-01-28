"""
Tests for hierarchical embedding service.

Tests core functionality of three-scale embedding generation:
- Text construction for different scales
- Category detection
- Batch processing
- Integration with embedding service
"""

import pytest
from unittest.mock import Mock, patch
from morgan.vectorization.hierarchical_embeddings import (
    HierarchicalEmbeddingService,
    HierarchicalEmbedding,
    get_hierarchical_embedding_service,
)


class TestHierarchicalEmbeddingService:
    """Test hierarchical embedding service functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = HierarchicalEmbeddingService()

        # Mock embedding service to avoid actual API calls
        self.mock_embedding_service = Mock()
        self.service.embedding_service = self.mock_embedding_service

        # Mock embeddings (different for each scale)
        self.mock_embeddings = [
            [0.1, 0.2, 0.3],  # coarse
            [0.4, 0.5, 0.6],  # medium
            [0.7, 0.8, 0.9],  # fine
        ]
        self.mock_embedding_service.encode_batch.return_value = self.mock_embeddings

        # Mock clustering engine to returned unmodified embeddings
        self.mock_clustering_engine = Mock()
        self.mock_clustering_engine.apply_contrastive_bias.side_effect = (
            lambda emb, *args: emb
        )
        self.service.clustering_engine = self.mock_clustering_engine

    def test_create_hierarchical_embeddings(self):
        """Test creating hierarchical embeddings for single content."""
        content = "def login(username, password):\n    return authenticate(username, password)"
        metadata = {"title": "Login Function", "source": "auth.py"}

        result = self.service.create_hierarchical_embeddings(
            content, metadata, category="code"
        )

        # Verify structure
        assert isinstance(result, HierarchicalEmbedding)
        assert result.coarse == [0.1, 0.2, 0.3]
        assert result.medium == [0.4, 0.5, 0.6]
        assert result.fine == [0.7, 0.8, 0.9]

        # Verify texts were created
        assert "coarse" in result.texts
        assert "medium" in result.texts
        assert "fine" in result.texts

        # Verify metadata
        assert result.metadata["category"] == "code"
        assert result.metadata["title"] == "Login Function"

        # Verify embedding service was called correctly
        self.mock_embedding_service.encode_batch.assert_called_once()
        call_args = self.mock_embedding_service.encode_batch.call_args
        assert len(call_args[0][0]) == 3  # Three texts (coarse, medium, fine)
        assert call_args[1]["instruction"] == "document"

    def test_build_coarse_text(self):
        """Test coarse text construction."""
        content = "This is a Docker deployment guide for containerized applications."
        metadata = {"title": "Docker Guide", "source": "docs/docker.md"}

        coarse_text = self.service.build_coarse_text(content, "documentation", metadata)

        # Should contain category, title, and content preview
        assert "Category: documentation" in coarse_text
        assert "Title: Docker Guide" in coarse_text
        assert "Source: docs/docker.md" in coarse_text
        assert "Docker deployment guide" in coarse_text
        assert "Type: Markdown document" in coarse_text

    def test_build_medium_text(self):
        """Test medium text construction."""
        content = """# Docker Guide
        
        ## Installation
        Install Docker on your system.
        
        ## Configuration  
        Configure Docker settings.
        
        def setup_docker():
            return configure_settings()
        """
        metadata = {"title": "Docker Guide", "source": "docker.md"}

        medium_text = self.service.build_medium_text(content, "documentation", metadata)

        # Should contain title, category, sections, and content sample
        assert "Title: Docker Guide" in medium_text
        assert "Category: documentation" in medium_text
        assert "Sections:" in medium_text
        assert "Installation" in medium_text or "Configuration" in medium_text

    def test_build_fine_text(self):
        """Test fine text construction."""
        content = "def authenticate(user, pass): return check_credentials(user, pass)"
        metadata = {"title": "Auth Module", "source": "auth.py"}

        fine_text = self.service.build_fine_text(content, "code", metadata)

        # Should contain document context and full content
        assert "Document: Auth Module" in fine_text
        assert "def authenticate(user, pass)" in fine_text

    def test_category_detection(self):
        """Test automatic category detection."""
        # Test code detection
        code_content = "def main():\n    print('hello')"
        code_metadata = {"source": "main.py"}
        assert self.service._detect_category(code_content, code_metadata) == "code"

        # Test documentation detection
        doc_content = "# User Guide\nThis is a tutorial for new users."
        doc_metadata = {"source": "guide.md"}
        assert (
            self.service._detect_category(doc_content, doc_metadata) == "documentation"
        )

        # Test API detection
        api_content = (
            "This endpoint handles authentication requests and returns tokens."
        )
        api_metadata = {"source": "api.md"}
        assert self.service._detect_category(api_content, api_metadata) == "api"

    def test_batch_processing(self):
        """Test batch hierarchical embedding creation."""
        contents = [
            "def login(): pass",
            "# User Guide\nWelcome to our app",
            "API endpoint for user authentication",
        ]
        metadatas = [
            {"source": "auth.py"},
            {"source": "guide.md"},
            {"source": "api.md"},
        ]

        # Mock batch embeddings (9 embeddings for 3 items × 3 scales)
        batch_embeddings = [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],  # Item 1: coarse, medium, fine
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],  # Item 2: coarse, medium, fine
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],  # Item 3: coarse, medium, fine
        ]
        self.mock_embedding_service.encode_batch.return_value = batch_embeddings

        results = self.service.create_batch_hierarchical_embeddings(
            contents, metadatas, show_progress=False
        )

        # Verify results
        assert len(results) == 3

        # Check first item (code)
        assert results[0].coarse == [0.1, 0.1]
        assert results[0].medium == [0.2, 0.2]
        assert results[0].fine == [0.3, 0.3]
        assert results[0].metadata["category"] == "code"

        # Check second item (documentation)
        assert results[1].coarse == [0.4, 0.4]
        assert results[1].metadata["category"] == "documentation"

        # Check third item (api)
        assert results[2].coarse == [0.7, 0.7]
        assert results[2].metadata["category"] == "api"

    def test_hierarchical_embedding_methods(self):
        """Test HierarchicalEmbedding helper methods."""
        embedding = HierarchicalEmbedding(
            coarse=[0.1, 0.2],
            medium=[0.3, 0.4],
            fine=[0.5, 0.6],
            texts={
                "coarse": "coarse text",
                "medium": "medium text",
                "fine": "fine text",
            },
            metadata={"test": True},
        )

        # Test get_embedding
        assert embedding.get_embedding("coarse") == [0.1, 0.2]
        assert embedding.get_embedding("medium") == [0.3, 0.4]
        assert embedding.get_embedding("fine") == [0.5, 0.6]

        # Test invalid scale
        with pytest.raises(ValueError):
            embedding.get_embedding("invalid")

        # Test get_text
        assert embedding.get_text("coarse") == "coarse text"
        assert embedding.get_text("medium") == "medium text"
        assert embedding.get_text("fine") == "fine text"

    def test_code_element_extraction(self):
        """Test extraction of code elements."""
        code_content = """
        def login(username, password):
            return authenticate(username, password)
        
        class UserManager:
            def create_user(self, data):
                return User(data)
        
        function validateInput(input) {
            return input.length > 0;
        }
        """

        elements = self.service._extract_code_elements(code_content)

        # Should extract function and class names
        assert "login" in elements
        assert "UserManager" in elements
        assert "create_user" in elements
        assert "validateInput" in elements

    def test_header_extraction(self):
        """Test extraction of headers from markdown."""
        markdown_content = """
        # Main Title
        
        ## Section 1
        Some content here.
        
        ### Subsection 1.1
        More content.
        
        ## Section 2
        Final content.
        """

        headers = self.service._extract_headers(markdown_content)

        # Should extract markdown headers
        assert "Main Title" in headers
        assert "Section 1" in headers
        assert "Subsection 1.1" in headers
        assert "Section 2" in headers

    def test_singleton_service(self):
        """Test singleton pattern for service access."""
        service1 = get_hierarchical_embedding_service()
        service2 = get_hierarchical_embedding_service()

        # Should return same instance
        assert service1 is service2
        assert isinstance(service1, HierarchicalEmbeddingService)
