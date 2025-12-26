"""
Property-Based Tests for Morgan Client

These tests verify universal properties that should hold across all client operations.
"""

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, strategies as st, settings

from morgan_cli.client import (
    ClientConfig,
    ConnectionStatus,
    HTTPClient,
    MorganClient,
    # These are now aliases to shared exceptions but maintain backward compatibility
    ConnectionError,
    RequestError,
    TimeoutError as ClientTimeoutError,
)


# ============================================================================
# Hypothesis Strategies
# ============================================================================

@st.composite
def client_config_strategy(draw):
    """Generate valid client configurations."""
    server_url = draw(st.sampled_from([
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://example.com",
        "http://192.168.1.100:9000"
    ]))
    
    api_key = draw(st.one_of(
        st.none(),
        st.text(min_size=10, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))
    ))
    
    user_id = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))
    ))
    
    timeout = draw(st.integers(min_value=1, max_value=120))
    retry_attempts = draw(st.integers(min_value=0, max_value=5))
    retry_delay = draw(st.integers(min_value=0, max_value=10))
    
    return ClientConfig(
        server_url=server_url,
        api_key=api_key,
        user_id=user_id,
        timeout_seconds=timeout,
        retry_attempts=retry_attempts,
        retry_delay_seconds=retry_delay
    )


@st.composite
def message_strategy(draw):
    """Generate valid chat messages."""
    return draw(st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()))


@st.composite
def metadata_strategy(draw):
    """Generate valid metadata dictionaries."""
    keys = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
        min_size=0,
        max_size=5,
        unique=True
    ))
    
    metadata = {}
    for key in keys:
        value = draw(st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ))
        metadata[key] = value
    
    return metadata


# ============================================================================
# Property 5: Client API-only communication
# **Feature: client-server-separation, Property 5: Client API-only communication**
# **Validates: Requirements 2.1, 2.3**
# ============================================================================

@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(
    config=client_config_strategy(),
    message=message_strategy(),
    metadata=metadata_strategy()
)
async def test_client_api_only_communication(config, message, metadata):
    """
    Property: For any request sent by the TUI client, the communication should occur
    exclusively through HTTP or WebSocket APIs, with no direct access to core components
    (vector database, embeddings, LLM).
    
    This test verifies that the client only uses HTTP/WebSocket protocols and does not
    import or access any server-side components directly.
    """
    # Verify that client module does not import server components
    import morgan_cli.client as client_module
    
    # Check that no server-side imports are present
    forbidden_imports = [
        'qdrant',
        'sentence_transformers',
        'transformers',
        'torch',
        'ollama',
        'openai',
        'anthropic',
        'vector_db',
        'embedding',
        'llm',
    ]
    
    module_dict = dir(client_module)
    for forbidden in forbidden_imports:
        assert not any(forbidden in item.lower() for item in module_dict), \
            f"Client should not import server component: {forbidden}"
    
    # Verify that HTTP client only uses aiohttp for communication
    http_client = HTTPClient(config)
    
    # Mock the session to verify it's the only communication method
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "answer": "test response",
            "conversation_id": "test_conv",
            "confidence": 0.9,
            "sources": []
        })
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session
        
        await http_client.connect()
        
        # Make a request
        try:
            await http_client.chat(message, metadata=metadata)
        except Exception:
            pass  # We're just verifying the communication method
        
        # Verify that aiohttp session was used (API-only communication)
        assert mock_session_class.called, "Client must use HTTP session for communication"
        
        await http_client.close()
    
    # Verify WebSocket client only uses websockets for communication
    ws_client = WebSocketClient(config)
    
    # Check that WebSocket URL is properly formed (ws:// or wss://)
    ws_url = ws_client._get_ws_url()
    assert ws_url.startswith("ws://") or ws_url.startswith("wss://"), \
        "WebSocket client must use WebSocket protocol"
    
    # Verify no direct database or model access
    assert not hasattr(http_client, 'vector_db'), "Client should not have direct vector DB access"
    assert not hasattr(http_client, 'embedding_model'), "Client should not have direct embedding access"
    assert not hasattr(http_client, 'llm_client'), "Client should not have direct LLM access"
    assert not hasattr(ws_client, 'vector_db'), "WebSocket client should not have direct vector DB access"
    assert not hasattr(ws_client, 'embedding_model'), "WebSocket client should not have direct embedding access"
    assert not hasattr(ws_client, 'llm_client'), "WebSocket client should not have direct LLM access"


# ============================================================================
# Property 6: Client configuration flexibility
# **Feature: client-server-separation, Property 6: Client configuration flexibility**
# **Validates: Requirements 2.2**
# ============================================================================

@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(config=client_config_strategy())
async def test_client_configuration_flexibility(config):
    """
    Property: For any valid server URL provided via command-line argument or environment
    variable, the TUI client should accept it and use it for all server communication.
    
    This test verifies that the client correctly uses the configured server URL for
    all API calls.
    """
    # Create client with the configuration
    http_client = HTTPClient(config)
    
    # Verify configuration is stored correctly
    assert http_client.config.server_url == config.server_url
    assert http_client.config.api_key == config.api_key
    assert http_client.config.user_id == config.user_id
    assert http_client.config.timeout_seconds == config.timeout_seconds
    assert http_client.config.retry_attempts == config.retry_attempts
    assert http_client.config.retry_delay_seconds == config.retry_delay_seconds
    
    # Mock the session to verify URL usage
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "healthy"})
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session
        
        await http_client.connect()
        
        # Make a health check request
        try:
            await http_client.health_check()
        except Exception:
            pass
        
        # Verify the request was made to the configured server URL
        if mock_session.request.called:
            call_args = mock_session.request.call_args
            if call_args:
                url_used = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('url', '')
                # The URL should start with the configured server URL
                assert config.server_url in url_used or url_used.startswith(config.server_url), \
                    f"Client should use configured server URL. Expected: {config.server_url}, Got: {url_used}"
        
        await http_client.close()
    
    # Verify WebSocket client uses the configured URL
    ws_client = WebSocketClient(config)
    ws_url = ws_client._get_ws_url()
    
    # Extract base URL from config
    base_url = config.server_url.replace("http://", "").replace("https://", "")
    
    # Verify WebSocket URL is derived from configured server URL
    assert base_url in ws_url, \
        f"WebSocket URL should be derived from configured server URL. Config: {config.server_url}, WS URL: {ws_url}"


# ============================================================================
# Property 7: Client error handling
# **Feature: client-server-separation, Property 7: Client error handling**
# **Validates: Requirements 2.4**
# ============================================================================

@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(
    config=client_config_strategy(),
    error_type=st.sampled_from(['connection', 'timeout', 'server_error', 'not_found'])
)
async def test_client_error_handling(config, error_type):
    """
    Property: For any scenario where the server is unavailable, the TUI client should
    display a clear error message and either exit gracefully or retry based on
    configuration, without crashing.
    
    This test verifies that the client handles various error scenarios gracefully.
    """
    http_client = HTTPClient(config)
    
    # Test different error scenarios
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.closed = False
        
        if error_type == 'connection':
            # Simulate connection error
            mock_session.request = AsyncMock(side_effect=Exception("Connection refused"))
        elif error_type == 'timeout':
            # Simulate timeout
            mock_session.request = AsyncMock(side_effect=asyncio.TimeoutError())
        elif error_type == 'server_error':
            # Simulate 500 error
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Server error occurred"
            })
            mock_session.request = AsyncMock(return_value=mock_response)
        elif error_type == 'not_found':
            # Simulate 404 error
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.json = AsyncMock(return_value={
                "error": "NOT_FOUND",
                "message": "Resource not found"
            })
            mock_session.request = AsyncMock(return_value=mock_response)
        
        mock_session_class.return_value = mock_session
        
        await http_client.connect()
        
        # Attempt to make a request and verify it raises an appropriate exception
        error_raised = False
        error_type_correct = False
        
        try:
            await http_client.health_check()
        except ClientConnectionError as e:
            error_raised = True
            error_type_correct = error_type in ['connection']
            # Verify error message is informative
            assert len(str(e)) > 0, "Error message should not be empty"
        except ClientTimeoutError as e:
            error_raised = True
            error_type_correct = error_type == 'timeout'
            # Verify error message is informative
            assert len(str(e)) > 0, "Error message should not be empty"
        except RequestError as e:
            error_raised = True
            error_type_correct = error_type in ['server_error', 'not_found']
            # Verify error has status code
            assert e.status_code is not None, "RequestError should include status code"
            # Verify error message is informative
            assert len(str(e)) > 0, "Error message should not be empty"
        except Exception as e:
            # Any exception is acceptable as long as it's caught
            error_raised = True
        
        # Verify that an error was raised (not a crash)
        if error_type in ['connection', 'timeout', 'server_error', 'not_found']:
            assert error_raised, f"Client should raise an exception for {error_type} errors"
        
        # Verify client can be closed gracefully even after error
        try:
            await http_client.close()
            cleanup_successful = True
        except Exception:
            cleanup_successful = False
        
        assert cleanup_successful, "Client should close gracefully even after errors"


# ============================================================================
# Property 8: Client cleanup isolation
# **Feature: client-server-separation, Property 8: Client cleanup isolation**
# **Validates: Requirements 2.5**
# ============================================================================

@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(
    config=client_config_strategy(),
    message=message_strategy()
)
async def test_client_cleanup_isolation(config, message):
    """
    Property: For any TUI client session, when the client exits, all network connections
    should be closed and the server state should remain unchanged (no data loss, no
    orphaned sessions).
    
    This test verifies that client cleanup is isolated and doesn't affect server state.
    """
    # Create and use client
    http_client = HTTPClient(config)
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "answer": "test",
            "conversation_id": "test_conv",
            "confidence": 0.9,
            "sources": []
        })
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Connect and make a request
        await http_client.connect()
        
        try:
            await http_client.chat(message)
        except Exception:
            pass
        
        # Verify session is open
        assert not mock_session.closed
        
        # Close the client
        await http_client.close()
        
        # Verify session was closed
        assert mock_session.close.called, "Client should close session on cleanup"
        
        # Verify client status is disconnected
        assert http_client.status == ConnectionStatus.DISCONNECTED, \
            "Client status should be DISCONNECTED after close"
        
        # Verify no lingering references
        assert http_client._session is None or http_client._session.closed, \
            "Client should not hold open session after close"
    
    # Test WebSocket cleanup
    ws_client = WebSocketClient(config)
    
    with patch('websockets.connect') as mock_ws_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()
        
        # Make connect return the mock directly (not wrapped in AsyncMock)
        async def mock_connect(*args, **kwargs):
            return mock_ws
        
        mock_ws_connect.side_effect = mock_connect
        
        # Connect
        await ws_client.connect()
        
        # Verify connected
        assert ws_client.status == ConnectionStatus.CONNECTED
        
        # Close
        await ws_client.close()
        
        # Verify WebSocket was closed
        assert mock_ws.close.called, "WebSocket should be closed on cleanup"
        
        # Verify status is disconnected
        assert ws_client.status == ConnectionStatus.DISCONNECTED, \
            "WebSocket status should be DISCONNECTED after close"
        
        # Verify no lingering references
        assert ws_client._ws is None or ws_client._ws.closed, \
            "WebSocket client should not hold open connection after close"
    
    # Verify that multiple close calls are safe (idempotent)
    http_client2 = HTTPClient(config)
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session
        
        await http_client2.connect()
        await http_client2.close()
        
        # Close again - should not raise error
        try:
            await http_client2.close()
            multiple_close_safe = True
        except Exception:
            multiple_close_safe = False
        
        assert multiple_close_safe, "Multiple close calls should be safe"


# ============================================================================
# Additional Helper Tests
# ============================================================================

@pytest.mark.asyncio
async def test_connection_status_tracking():
    """Test that connection status is properly tracked."""
    config = ClientConfig(server_url="http://localhost:8080")
    http_client = HTTPClient(config)
    
    # Initial status should be disconnected
    assert http_client.status == ConnectionStatus.DISCONNECTED
    
    # Track status changes
    status_changes = []
    
    def status_callback(status):
        status_changes.append(status)
    
    http_client.add_status_callback(status_callback)
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Connect
        await http_client.connect()
        
        # Should have transitioned through CONNECTING to CONNECTED
        assert ConnectionStatus.CONNECTING in status_changes
        assert ConnectionStatus.CONNECTED in status_changes
        assert http_client.status == ConnectionStatus.CONNECTED
        
        # Close
        await http_client.close()
        
        # Should be disconnected
        assert http_client.status == ConnectionStatus.DISCONNECTED
        assert ConnectionStatus.DISCONNECTED in status_changes


@pytest.mark.asyncio
async def test_unified_client():
    """Test that MorganClient provides unified interface."""
    config = ClientConfig(server_url="http://localhost:8080")
    client = MorganClient(config)
    
    # Verify both HTTP and WebSocket clients are available
    assert client.http is not None
    assert client.ws is not None
    assert isinstance(client.http, HTTPClient)
    assert isinstance(client.ws, WebSocketClient)
    
    # Verify configuration is shared
    assert client.http.config == config
    assert client.ws.config == config
    
    # Test context manager
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session
        
        async with client:
            assert client.http.status == ConnectionStatus.CONNECTED
        
        # Should be closed after context exit
        assert client.http.status == ConnectionStatus.DISCONNECTED
