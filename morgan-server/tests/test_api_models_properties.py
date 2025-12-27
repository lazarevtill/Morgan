"""
Property-based tests for API models.

This module tests API model consistency, validation, and error response structure
using property-based testing with Hypothesis.
"""

from datetime import datetime

from hypothesis import given, strategies as st, settings
from pydantic import ValidationError
import pytest

from morgan_server.api.models import (
    ChatRequest,
    ChatResponse,
    MemorySearchRequest,
    LearnRequest,
    HealthResponse,
    PreferenceUpdate,
    ErrorResponse,
    ComponentStatus,
)


# ============================================================================
# Property 15: API Consistency
# ============================================================================


class TestAPIConsistency:
    """
    Property-based tests for API consistency.

    **Feature: client-server-separation, Property 15: API consistency**

    For any API endpoint, responses should follow consistent formats (JSON structure,
    HTTP status codes, error format) as defined in the OpenAPI specification.

    **Validates: Requirements 7.2**
    """

    @given(
        message=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip() != ""),
        user_id=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        conversation_id=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=100)
    def test_property_chat_request_consistency(self, message, user_id, conversation_id):
        """
        Property: ChatRequest models are consistently validated.

        For any valid message, user_id, and conversation_id, the ChatRequest
        should be created successfully and maintain consistent field structure.
        """
        request = ChatRequest(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # Verify consistent structure
        assert hasattr(request, "message")
        assert hasattr(request, "user_id")
        assert hasattr(request, "conversation_id")
        assert hasattr(request, "metadata")

        # Verify message is stripped
        assert request.message == message.strip()

        # Verify optional fields
        assert request.user_id == user_id
        assert request.conversation_id == conversation_id
        assert isinstance(request.metadata, dict)

    @given(
        answer=st.text(min_size=1, max_size=5000),
        conversation_id=st.text(min_size=1, max_size=100),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_property_chat_response_consistency(
        self, answer, conversation_id, confidence
    ):
        """
        Property: ChatResponse models have consistent structure.

        For any valid answer, conversation_id, and confidence, the ChatResponse
        should maintain consistent field structure and types.
        """
        response = ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            confidence=confidence,
        )

        # Verify required fields
        assert response.answer == answer
        assert response.conversation_id == conversation_id
        assert response.confidence == confidence

        # Verify consistent structure
        assert hasattr(response, "emotional_tone")
        assert hasattr(response, "empathy_level")
        assert hasattr(response, "personalization_elements")
        assert hasattr(response, "milestone_celebration")
        assert hasattr(response, "sources")
        assert hasattr(response, "metadata")

        # Verify default values
        assert isinstance(response.personalization_elements, list)
        assert isinstance(response.sources, list)
        assert isinstance(response.metadata, dict)

    @given(
        query=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip() != ""),
        user_id=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_property_memory_search_request_consistency(self, query, user_id, limit):
        """
        Property: MemorySearchRequest models are consistently validated.

        For any valid query, user_id, and limit, the MemorySearchRequest
        should be created successfully with consistent structure.
        """
        request = MemorySearchRequest(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        # Verify consistent structure
        assert hasattr(request, "query")
        assert hasattr(request, "user_id")
        assert hasattr(request, "limit")

        # Verify query is stripped
        assert request.query == query.strip()
        assert request.user_id == user_id
        assert request.limit == limit

    @given(
        doc_type=st.sampled_from(["auto", "pdf", "markdown", "text", "html", "docx"]),
        content=st.text(min_size=1, max_size=5000),
    )
    @settings(max_examples=100)
    def test_property_learn_request_consistency(self, doc_type, content):
        """
        Property: LearnRequest models are consistently validated.

        For any valid doc_type and content, the LearnRequest should be
        created successfully with consistent structure.
        """
        request = LearnRequest(
            content=content,
            doc_type=doc_type,
        )

        # Verify consistent structure
        assert hasattr(request, "source")
        assert hasattr(request, "url")
        assert hasattr(request, "content")
        assert hasattr(request, "doc_type")
        assert hasattr(request, "metadata")

        # Verify values
        assert request.content == content
        assert request.doc_type == doc_type
        assert isinstance(request.metadata, dict)

    @given(
        status=st.sampled_from(["healthy", "degraded", "unhealthy"]),
        version=st.text(min_size=1, max_size=50),
        uptime_seconds=st.floats(min_value=0.0, max_value=1e9),
    )
    @settings(max_examples=100)
    def test_property_health_response_consistency(
        self, status, version, uptime_seconds
    ):
        """
        Property: HealthResponse models have consistent structure.

        For any valid status, version, and uptime, the HealthResponse
        should maintain consistent field structure and types.
        """
        response = HealthResponse(
            status=status,
            version=version,
            uptime_seconds=uptime_seconds,
        )

        # Verify consistent structure
        assert hasattr(response, "status")
        assert hasattr(response, "timestamp")
        assert hasattr(response, "version")
        assert hasattr(response, "uptime_seconds")

        # Verify values
        assert response.status == status
        assert response.version == version
        assert response.uptime_seconds == uptime_seconds
        assert isinstance(response.timestamp, datetime)

    @given(
        communication_style=st.one_of(
            st.none(),
            st.sampled_from(
                ["casual", "professional", "friendly", "technical", "playful"]
            ),
        ),
        response_length=st.one_of(
            st.none(), st.sampled_from(["brief", "moderate", "detailed"])
        ),
    )
    @settings(max_examples=100)
    def test_property_preference_update_consistency(
        self, communication_style, response_length
    ):
        """
        Property: PreferenceUpdate models are consistently validated.

        For any valid communication_style and response_length, the PreferenceUpdate
        should be created successfully with consistent structure.
        """
        request = PreferenceUpdate(
            communication_style=communication_style,
            response_length=response_length,
        )

        # Verify consistent structure
        assert hasattr(request, "communication_style")
        assert hasattr(request, "response_length")
        assert hasattr(request, "topics_of_interest")
        assert hasattr(request, "preferred_name")

        # Verify values
        assert request.communication_style == communication_style
        assert request.response_length == response_length


# ============================================================================
# Property 16: Error Response Structure
# ============================================================================


class TestErrorResponseStructure:
    """
    Property-based tests for error response structure.

    **Feature: client-server-separation, Property 16: Error response structure**

    For any API request that fails (validation error, server error, not found),
    the response should be a structured JSON object containing an error code,
    message, and optional details.

    **Validates: Requirements 7.3**
    """

    @given(
        error_code=st.from_regex(r"[A-Z]+_[A-Z_]+", fullmatch=True),
        message=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_property_error_response_structure(self, error_code, message):
        """
        Property: ErrorResponse models have consistent structure.

        For any valid error_code and message, the ErrorResponse should
        maintain consistent field structure with required fields.
        """
        response = ErrorResponse(
            error=error_code,
            message=message,
        )

        # Verify required fields exist
        assert hasattr(response, "error")
        assert hasattr(response, "message")
        assert hasattr(response, "details")
        assert hasattr(response, "timestamp")
        assert hasattr(response, "request_id")

        # Verify values
        assert response.error == error_code
        assert response.message == message
        assert isinstance(response.timestamp, datetime)

        # Verify error code format (uppercase with underscores)
        assert response.error.isupper()
        assert " " not in response.error

    @given(
        error_code=st.from_regex(r"[A-Z]+_[A-Z_]+", fullmatch=True),
        message=st.text(min_size=1, max_size=500),
        details=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(max_size=200),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            max_size=10,
        ),
        request_id=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=100)
    def test_property_error_response_with_details(
        self, error_code, message, details, request_id
    ):
        """
        Property: ErrorResponse with details maintains structure.

        For any valid error_code, message, details, and request_id,
        the ErrorResponse should maintain consistent structure.
        """
        response = ErrorResponse(
            error=error_code,
            message=message,
            details=details,
            request_id=request_id,
        )

        # Verify all fields
        assert response.error == error_code
        assert response.message == message
        assert response.details == details
        assert response.request_id == request_id
        assert isinstance(response.timestamp, datetime)

    @given(
        invalid_error_code=st.text(min_size=1, max_size=100).filter(
            lambda x: not x.isupper() or " " in x
        ),
        message=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_property_error_response_rejects_invalid_codes(
        self, invalid_error_code, message
    ):
        """
        Property: ErrorResponse rejects invalid error codes.

        For any error code that is not uppercase with underscores,
        the ErrorResponse should raise a validation error.
        """
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(
                error=invalid_error_code,
                message=message,
            )

        # Verify validation error contains information about error code
        assert "error" in str(exc_info.value).lower()

    @given(
        component_status=st.sampled_from(["up", "down", "degraded"]),
        name=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_property_component_status_consistency(self, component_status, name):
        """
        Property: ComponentStatus models have consistent structure.

        For any valid component status and name, the ComponentStatus
        should maintain consistent field structure.
        """
        status = ComponentStatus(
            name=name,
            status=component_status,
        )

        # Verify consistent structure
        assert hasattr(status, "name")
        assert hasattr(status, "status")
        assert hasattr(status, "latency_ms")
        assert hasattr(status, "error")
        assert hasattr(status, "details")

        # Verify values
        assert status.name == name
        assert status.status == component_status
        assert isinstance(status.details, dict)

    @given(
        whitespace_message=st.from_regex(r"[\s]+", fullmatch=True),
    )
    @settings(max_examples=100)
    def test_property_chat_request_rejects_whitespace(self, whitespace_message):
        """
        Property: ChatRequest rejects whitespace-only messages.

        For any message that is only whitespace, the ChatRequest
        should raise a validation error.
        """
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=whitespace_message)

        # Verify validation error mentions message
        assert "message" in str(exc_info.value).lower()

    @given(
        whitespace_query=st.from_regex(r"[\s]+", fullmatch=True),
    )
    @settings(max_examples=100)
    def test_property_memory_search_rejects_whitespace(self, whitespace_query):
        """
        Property: MemorySearchRequest rejects whitespace-only queries.

        For any query that is only whitespace, the MemorySearchRequest
        should raise a validation error.
        """
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query=whitespace_query)

        # Verify validation error mentions query
        assert "query" in str(exc_info.value).lower()

    @given(
        invalid_doc_type=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in ["auto", "pdf", "markdown", "text", "html", "docx"]
        ),
    )
    @settings(max_examples=100)
    def test_property_learn_request_rejects_invalid_doc_type(self, invalid_doc_type):
        """
        Property: LearnRequest rejects invalid document types.

        For any doc_type that is not in the valid list, the LearnRequest
        should raise a validation error.
        """
        with pytest.raises(ValidationError) as exc_info:
            LearnRequest(
                content="test content",
                doc_type=invalid_doc_type,
            )

        # Verify validation error mentions doc_type
        assert "doc_type" in str(exc_info.value).lower()
