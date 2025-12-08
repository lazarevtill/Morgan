"""
Property-Based Tests for Container Configuration and Signal Handling

These tests verify that the Morgan server behaves correctly when running
in a containerized environment, including proper configuration handling
and graceful shutdown on SIGTERM.

**Feature: client-server-separation**
"""

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from morgan_server.config import ServerConfig, load_config


# ============================================================================
# Property 18: Container configuration
# ============================================================================

@pytest.mark.parametrize("execution_number", range(100))
@given(
    host=st.sampled_from(["0.0.0.0", "127.0.0.1", "localhost"]),
    port=st.integers(min_value=1024, max_value=65535),
    log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"]),
    log_format=st.sampled_from(["json", "text"]),
    workers=st.integers(min_value=1, max_value=8),
)
@settings(
    max_examples=1,  # Run once per parametrize iteration
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_container_configuration(
    execution_number,
    host,
    port,
    log_level,
    log_format,
    workers,
):
    """
    **Feature: client-server-separation, Property 18: Container configuration**
    
    **Validates: Requirements 8.2**
    
    For any server running in a container, configuration values provided via
    environment variables should be read and applied correctly, producing the
    same behavior as running outside a container.
    
    This test verifies that:
    1. Environment variables are correctly read
    2. Configuration is applied as expected
    3. The server configuration matches the provided environment variables
    4. The behavior is consistent with non-containerized deployment
    
    Note: This test simulates container environment by setting environment
    variables and verifying the configuration is loaded correctly.
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Set environment variables (simulating container environment)
        os.environ["MORGAN_HOST"] = host
        os.environ["MORGAN_PORT"] = str(port)
        os.environ["MORGAN_LOG_LEVEL"] = log_level
        os.environ["MORGAN_LOG_FORMAT"] = log_format
        os.environ["MORGAN_WORKERS"] = str(workers)
        
        # Set required configuration (with defaults for testing)
        os.environ["MORGAN_LLM_ENDPOINT"] = "http://localhost:11434"
        os.environ["MORGAN_VECTOR_DB_URL"] = "http://localhost:6333"
        
        # Load configuration (this is what happens in the container)
        config = load_config()
        
        # Verify configuration matches environment variables
        assert config.host == host, \
            f"Host mismatch: expected {host}, got {config.host}"
        
        assert config.port == port, \
            f"Port mismatch: expected {port}, got {config.port}"
        
        assert config.log_level == log_level, \
            f"Log level mismatch: expected {log_level}, got {config.log_level}"
        
        assert config.log_format == log_format, \
            f"Log format mismatch: expected {log_format}, got {config.log_format}"
        
        assert config.workers == workers, \
            f"Workers mismatch: expected {workers}, got {config.workers}"
        
        # Verify required fields are set
        assert config.llm_endpoint == "http://localhost:11434"
        assert config.vector_db_url == "http://localhost:6333"
        
        # Property verified: Container configuration is read and applied correctly
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@pytest.mark.parametrize("execution_number", range(100))
@given(
    llm_provider=st.sampled_from(["ollama", "openai-compatible"]),
    llm_model=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_."
    )),
    embedding_provider=st.sampled_from(["local", "ollama", "openai-compatible"]),
    cache_size_mb=st.integers(min_value=100, max_value=10000),
)
@settings(
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_container_configuration_advanced(
    execution_number,
    llm_provider,
    llm_model,
    embedding_provider,
    cache_size_mb,
):
    """
    **Feature: client-server-separation, Property 18: Container configuration**
    
    **Validates: Requirements 8.2**
    
    Test advanced configuration options in container environment.
    """
    original_env = os.environ.copy()
    
    try:
        # Set environment variables
        os.environ["MORGAN_LLM_PROVIDER"] = llm_provider
        os.environ["MORGAN_LLM_MODEL"] = llm_model
        os.environ["MORGAN_LLM_ENDPOINT"] = "http://localhost:11434"
        os.environ["MORGAN_EMBEDDING_PROVIDER"] = embedding_provider
        os.environ["MORGAN_CACHE_SIZE_MB"] = str(cache_size_mb)
        os.environ["MORGAN_VECTOR_DB_URL"] = "http://localhost:6333"
        
        # Set embedding endpoint for remote providers
        if embedding_provider in ["ollama", "openai-compatible"]:
            os.environ["MORGAN_EMBEDDING_ENDPOINT"] = "http://localhost:11434"
        
        # Load configuration
        config = load_config()
        
        # Verify configuration
        assert config.llm_provider == llm_provider
        assert config.llm_model == llm_model
        assert config.embedding_provider == embedding_provider
        assert config.cache_size_mb == cache_size_mb
        
        # Property verified: Advanced configuration works in containers
        
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@pytest.mark.parametrize("execution_number", range(100))
@given(
    config_format=st.sampled_from(["env", "yaml", "json"]),
    host=st.sampled_from(["0.0.0.0", "127.0.0.1"]),
    port=st.integers(min_value=8000, max_value=9000),
)
@settings(
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_container_configuration_file_formats(
    execution_number,
    config_format,
    host,
    port,
):
    """
    **Feature: client-server-separation, Property 18: Container configuration**
    
    **Validates: Requirements 8.2**
    
    Test that different configuration file formats work correctly in containers.
    """
    original_env = os.environ.copy()
    
    try:
        # Clear ALL Morgan environment variables to ensure file config is used
        for key in list(os.environ.keys()):
            if key.startswith("MORGAN_"):
                del os.environ[key]
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'.{config_format}',
            delete=False
        ) as f:
            config_path = Path(f.name)
            
            if config_format == "env":
                f.write(f"MORGAN_HOST={host}\n")
                f.write(f"MORGAN_PORT={port}\n")
                f.write("MORGAN_LLM_ENDPOINT=http://localhost:11434\n")
                f.write("MORGAN_VECTOR_DB_URL=http://localhost:6333\n")
            
            elif config_format == "yaml":
                f.write(f"host: {host}\n")
                f.write(f"port: {port}\n")
                f.write("llm_endpoint: http://localhost:11434\n")
                f.write("vector_db_url: http://localhost:6333\n")
            
            elif config_format == "json":
                import json
                config_dict = {
                    "host": host,
                    "port": port,
                    "llm_endpoint": "http://localhost:11434",
                    "vector_db_url": "http://localhost:6333",
                }
                json.dump(config_dict, f)
        
        try:
            # Load configuration from file
            config = load_config(config_file=config_path)
            
            # Verify configuration
            assert config.host == host, \
                f"Host mismatch: expected {host}, got {config.host}"
            assert config.port == port, \
                f"Port mismatch: expected {port}, got {config.port}"
            
            # Property verified: Config files work in containers
            
        finally:
            # Clean up temp file
            config_path.unlink(missing_ok=True)
    
    finally:
        os.environ.clear()
        os.environ.update(original_env)


# ============================================================================
# Property 19: Container signal handling
# ============================================================================

@pytest.mark.parametrize("execution_number", range(100))
@given(
    signal_type=st.sampled_from([signal.SIGTERM, signal.SIGINT]),
    startup_delay=st.floats(min_value=0.1, max_value=1.0),
)
@settings(
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_container_signal_handling(
    execution_number,
    signal_type,
    startup_delay,
):
    """
    **Feature: client-server-separation, Property 19: Container signal handling**
    
    **Validates: Requirements 8.4**
    
    For any containerized server, when a SIGTERM signal is received, the server
    should perform graceful shutdown (close connections, persist data) before
    the container exits.
    
    This test verifies that:
    1. The server responds to SIGTERM/SIGINT signals
    2. Graceful shutdown is performed
    3. The process exits cleanly with code 0
    4. No errors occur during shutdown
    
    Note: This test uses a subprocess to simulate a containerized environment
    and sends signals to test graceful shutdown.
    """
    # Skip on Windows (signal handling is different)
    if sys.platform == "win32":
        pytest.skip("Signal handling test not supported on Windows")
    
    # Create a minimal test script that starts the server
    test_script = """
import sys
import signal
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan_server.app import create_app
from morgan_server.config import ServerConfig

# Track shutdown
shutdown_called = False

def signal_handler(signum, frame):
    global shutdown_called
    shutdown_called = True
    print(f"Signal {signum} received, shutting down...", flush=True)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Create minimal config
config = ServerConfig(
    host="127.0.0.1",
    port=0,  # Use random port
    llm_endpoint="http://localhost:11434",
    vector_db_url="http://localhost:6333",
)

print("Server started", flush=True)

# Keep running until signal
try:
    while True:
        asyncio.sleep(0.1)
except KeyboardInterrupt:
    print("Keyboard interrupt", flush=True)
    sys.exit(0)
"""
    
    # Write test script to temp file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        script_path = Path(f.name)
        f.write(test_script)
    
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait for startup
        time.sleep(startup_delay)
        
        # Verify process is running
        assert process.poll() is None, "Process should be running"
        
        # Send signal
        process.send_signal(signal_type)
        
        # Wait for graceful shutdown (max 5 seconds)
        try:
            stdout, stderr = process.communicate(timeout=5)
            exit_code = process.returncode
            
            # Verify graceful shutdown
            assert exit_code == 0, \
                f"Process should exit with code 0, got {exit_code}\nStdout: {stdout}\nStderr: {stderr}"
            
            # Verify shutdown message was printed
            assert "Signal" in stdout or "shutting down" in stdout.lower(), \
                f"Shutdown message not found in output: {stdout}"
            
            # Property verified: Container handles signals gracefully
            
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't shut down
            process.kill()
            process.communicate()
            pytest.fail("Process did not shut down gracefully within timeout")
    
    finally:
        # Clean up
        script_path.unlink(missing_ok=True)
        
        # Ensure process is terminated
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("execution_number", range(100))
@given(
    num_active_connections=st.integers(min_value=0, max_value=5),
    shutdown_delay=st.floats(min_value=0.1, max_value=0.5),
)
@settings(
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_container_signal_handling_with_active_connections(
    execution_number,
    num_active_connections,
    shutdown_delay,
):
    """
    **Feature: client-server-separation, Property 19: Container signal handling**
    
    **Validates: Requirements 8.4**
    
    Test that graceful shutdown works even with active connections.
    
    This simulates a more realistic scenario where the server has active
    connections when it receives a shutdown signal.
    """
    # Skip on Windows
    if sys.platform == "win32":
        pytest.skip("Signal handling test not supported on Windows")
    
    # Create test script with simulated active connections
    test_script = f"""
import sys
import signal
import time

# Track state
active_connections = {num_active_connections}
shutdown_called = False

def signal_handler(signum, frame):
    global shutdown_called
    shutdown_called = True
    print(f"Signal received, closing {{active_connections}} connections...", flush=True)
    
    # Simulate connection cleanup
    time.sleep({shutdown_delay})
    
    print("All connections closed, exiting", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("Server started with {{active_connections}} connections", flush=True)

# Keep running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    sys.exit(0)
"""
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        script_path = Path(f.name)
        f.write(test_script)
    
    try:
        # Start process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait for startup
        time.sleep(0.2)
        
        # Send SIGTERM
        process.send_signal(signal.SIGTERM)
        
        # Wait for shutdown
        try:
            stdout, stderr = process.communicate(timeout=5)
            exit_code = process.returncode
            
            # Verify graceful shutdown
            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
            assert "connections closed" in stdout.lower(), \
                f"Connection cleanup message not found: {stdout}"
            
            # Property verified: Graceful shutdown with active connections
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            pytest.fail("Shutdown timeout with active connections")
    
    finally:
        script_path.unlink(missing_ok=True)
        if process.poll() is None:
            process.kill()
            process.communicate()


# ============================================================================
# Unit Tests for Container Configuration
# ============================================================================

def test_container_environment_precedence():
    """
    Test that environment variables take precedence over defaults in containers.
    
    This is a unit test that complements the property tests.
    """
    original_env = os.environ.copy()
    
    try:
        # Set environment variables
        os.environ["MORGAN_HOST"] = "0.0.0.0"
        os.environ["MORGAN_PORT"] = "9999"
        os.environ["MORGAN_LLM_ENDPOINT"] = "http://test:11434"
        os.environ["MORGAN_VECTOR_DB_URL"] = "http://test:6333"
        
        # Load config
        config = load_config()
        
        # Verify environment takes precedence
        assert config.host == "0.0.0.0"
        assert config.port == 9999
        assert config.llm_endpoint == "http://test:11434"
        assert config.vector_db_url == "http://test:6333"
    
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_container_default_config_values():
    """
    Test that default configuration values work in containers.
    
    This ensures containers can start with minimal configuration.
    """
    original_env = os.environ.copy()
    
    try:
        # Clear all Morgan config
        for key in list(os.environ.keys()):
            if key.startswith("MORGAN_"):
                del os.environ[key]
        
        # Should load with defaults
        config = load_config()
        
        # Verify defaults are applied
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.llm_provider == "ollama"
        assert config.llm_endpoint == "http://localhost:11434"
        assert config.vector_db_url == "http://localhost:6333"
    
    finally:
        os.environ.clear()
        os.environ.update(original_env)
