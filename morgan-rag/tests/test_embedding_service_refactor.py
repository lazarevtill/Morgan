import pytest
from morgan.services.embeddings import get_embedding_service


def test_embedding_service_initialization():
    """Test that the embedding service relies on the new package."""
    service = get_embedding_service()
    assert service is not None
    # We might expect it to fallback to local if no API key, which is fine
    # This test just checks it constructs without crashing due to import errors
    assert hasattr(service, "encode")
    assert hasattr(service, "encode_batch")


def test_embedding_functionality():
    """Test basic encoding functionality."""
    service = get_embedding_service()
    if not service.is_available():
        pytest.skip(
            "Embedding service not available (no provider configured or local model missing)"
        )

    embedding = service.encode("test")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
