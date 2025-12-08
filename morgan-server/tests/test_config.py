"""
Tests for server configuration system.

This module tests configuration loading from multiple sources,
precedence rules, validation, and error handling.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
import yaml

from morgan_server.config import (
    ServerConfig,
    ConfigurationError,
    load_config,
    load_config_from_file,
    validate_required_config,
    get_config,
)


class TestServerConfig:
    """Test ServerConfig model and validation."""
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        config = ServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.workers == 4
        assert config.llm_provider == "ollama"
        assert config.llm_model == "gemma3"
        assert config.llm_endpoint == "http://localhost:11434"
        assert config.vector_db_url == "http://localhost:6333"
        assert config.log_level == "INFO"
        assert config.log_format == "json"
    
    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            llm_model="llama3",
            log_level="DEBUG"
        )
        
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.llm_model == "llama3"
        assert config.log_level == "DEBUG"
    
    def test_invalid_port(self):
        """Test that invalid port raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            ServerConfig(port=0)
        
        with pytest.raises(Exception):
            ServerConfig(port=70000)
    
    def test_invalid_llm_provider(self):
        """Test that invalid LLM provider raises validation error."""
        with pytest.raises(Exception):
            ServerConfig(llm_provider="invalid")
    
    def test_invalid_log_level(self):
        """Test that invalid log level raises validation error."""
        with pytest.raises(Exception):
            ServerConfig(log_level="INVALID")
    
    def test_invalid_llm_endpoint(self):
        """Test that invalid LLM endpoint raises validation error."""
        with pytest.raises(Exception):
            ServerConfig(llm_endpoint="")
        
        with pytest.raises(Exception):
            ServerConfig(llm_endpoint="not-a-url")
    
    def test_invalid_vector_db_url(self):
        """Test that invalid vector DB URL raises validation error."""
        with pytest.raises(Exception):
            ServerConfig(vector_db_url="")
        
        with pytest.raises(Exception):
            ServerConfig(vector_db_url="invalid-url")
    
    def test_invalid_embedding_device(self):
        """Test that invalid embedding device raises validation error."""
        with pytest.raises(Exception):
            ServerConfig(embedding_device="invalid")


class TestLoadConfigFromFile:
    """Test loading configuration from files."""
    
    def test_load_yaml_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'host': '192.168.1.1',
                'port': 9090,
                'llm_model': 'llama3'
            }, f)
            temp_path = Path(f.name)
        
        try:
            config_data = load_config_from_file(temp_path)
            assert config_data['host'] == '192.168.1.1'
            assert config_data['port'] == 9090
            assert config_data['llm_model'] == 'llama3'
        finally:
            temp_path.unlink()
    
    def test_load_json_file(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'host': '192.168.1.2',
                'port': 8888,
                'llm_model': 'mistral'
            }, f)
            temp_path = Path(f.name)
        
        try:
            config_data = load_config_from_file(temp_path)
            assert config_data['host'] == '192.168.1.2'
            assert config_data['port'] == 8888
            assert config_data['llm_model'] == 'mistral'
        finally:
            temp_path.unlink()
    
    def test_load_env_file(self):
        """Test loading .env file returns empty dict (handled by pydantic-settings)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("MORGAN_HOST=127.0.0.1\n")
            f.write("MORGAN_PORT=7777\n")
            temp_path = Path(f.name)
        
        try:
            config_data = load_config_from_file(temp_path)
            # .env files return empty dict as they're handled by pydantic-settings
            assert config_data == {}
        finally:
            temp_path.unlink()
    
    def test_file_not_found(self):
        """Test that missing file raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_config_from_file(Path("/nonexistent/file.yaml"))
    
    def test_unsupported_format(self):
        """Test that unsupported file format raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError, match="Unsupported configuration file format"):
                load_config_from_file(temp_path)
        finally:
            temp_path.unlink()
    
    def test_invalid_yaml(self):
        """Test that invalid YAML raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
                load_config_from_file(temp_path)
        finally:
            temp_path.unlink()
    
    def test_invalid_json(self):
        """Test that invalid JSON raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json}")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse JSON"):
                load_config_from_file(temp_path)
        finally:
            temp_path.unlink()


class TestLoadConfig:
    """Test configuration loading with precedence rules."""
    
    def test_load_with_defaults(self):
        """Test loading configuration with only defaults."""
        config = load_config()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.llm_provider == "ollama"
    
    def test_load_with_overrides(self):
        """Test that explicit overrides have highest precedence."""
        config = load_config(
            host="10.0.0.1",
            port=5000,
            llm_model="custom-model"
        )
        
        assert config.host == "10.0.0.1"
        assert config.port == 5000
        assert config.llm_model == "custom-model"
    
    def test_load_with_yaml_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'host': '172.16.0.1',
                'port': 3000,
                'llm_model': 'yaml-model'
            }, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(config_file=temp_path)
            assert config.host == '172.16.0.1'
            assert config.port == 3000
            assert config.llm_model == 'yaml-model'
        finally:
            temp_path.unlink()
    
    def test_load_with_json_file(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'host': '172.16.0.2',
                'port': 4000,
                'llm_model': 'json-model'
            }, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(config_file=temp_path)
            assert config.host == '172.16.0.2'
            assert config.port == 4000
            assert config.llm_model == 'json-model'
        finally:
            temp_path.unlink()
    
    def test_precedence_override_over_file(self):
        """Test that explicit overrides take precedence over file config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'host': '172.16.0.1',
                'port': 3000,
            }, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(
                config_file=temp_path,
                port=9999  # Override
            )
            assert config.host == '172.16.0.1'  # From file
            assert config.port == 9999  # From override
        finally:
            temp_path.unlink()
    
    def test_invalid_config_raises_error(self):
        """Test that invalid configuration raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            load_config(port=70000)  # Invalid port


class TestValidateRequiredConfig:
    """Test validation of required configuration."""
    
    def test_valid_config_passes(self):
        """Test that valid configuration passes validation."""
        config = ServerConfig()
        # Should not raise
        validate_required_config(config)
    
    def test_empty_llm_endpoint_fails(self):
        """Test that empty LLM endpoint fails validation."""
        # This should be caught by pydantic validation before reaching validate_required_config
        with pytest.raises(Exception):
            config = ServerConfig(llm_endpoint="")
    
    def test_empty_vector_db_url_fails(self):
        """Test that empty vector DB URL fails validation."""
        # This should be caught by pydantic validation before reaching validate_required_config
        with pytest.raises(Exception):
            config = ServerConfig(vector_db_url="")
    
    def test_invalid_cache_dir(self):
        """Test that invalid cache directory fails validation."""
        # Use a path that cannot be created (e.g., under a file)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            config = ServerConfig(cache_dir=str(temp_file / "subdir"))
            with pytest.raises(ConfigurationError, match="Cannot create cache directory"):
                validate_required_config(config)
        finally:
            temp_file.unlink()


class TestGetConfig:
    """Test main configuration entry point."""
    
    def test_get_config_with_validation(self):
        """Test getting configuration with validation enabled."""
        config = get_config(validate=True)
        
        assert isinstance(config, ServerConfig)
        assert config.host == "0.0.0.0"
        assert config.port == 8080
    
    def test_get_config_without_validation(self):
        """Test getting configuration with validation disabled."""
        config = get_config(validate=False)
        
        assert isinstance(config, ServerConfig)
    
    def test_get_config_with_overrides(self):
        """Test getting configuration with overrides."""
        config = get_config(
            validate=False,
            host="192.168.1.100",
            port=7000
        )
        
        assert config.host == "192.168.1.100"
        assert config.port == 7000


class TestEnvironmentVariables:
    """Test configuration loading from environment variables."""
    
    def test_env_var_precedence(self):
        """Test that environment variables override defaults."""
        # Set environment variable
        os.environ["MORGAN_HOST"] = "env-host"
        os.environ["MORGAN_PORT"] = "6000"
        
        try:
            config = load_config()
            assert config.host == "env-host"
            assert config.port == 6000
        finally:
            # Clean up
            del os.environ["MORGAN_HOST"]
            del os.environ["MORGAN_PORT"]
    
    def test_env_var_with_file(self):
        """Test that environment variables override file config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'host': 'file-host',
                'port': 5000,
            }, f)
            temp_path = Path(f.name)
        
        os.environ["MORGAN_PORT"] = "7000"
        
        try:
            config = load_config(config_file=temp_path)
            assert config.host == 'file-host'  # From file
            assert config.port == 7000  # From env var (higher precedence)
        finally:
            temp_path.unlink()
            del os.environ["MORGAN_PORT"]


# Property-Based Tests using Hypothesis
from hypothesis import given, strategies as st, settings


class TestConfigurationFormatSupport:
    """
    Property-based tests for configuration format support.
    
    **Feature: client-server-separation, Property 9: Configuration format support**
    
    For any valid configuration file in YAML, JSON, or .env format containing
    the same configuration values, the server should parse it correctly and
    produce equivalent runtime configuration.
    
    **Validates: Requirements 3.2**
    """
    
    @given(
        host=st.text(min_size=1, max_size=50).filter(lambda x: '\n' not in x),
        port=st.integers(min_value=1024, max_value=65535),
        llm_model=st.text(min_size=1, max_size=50).filter(lambda x: '\n' not in x),
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        workers=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100)
    def test_property_yaml_json_equivalence(
        self, host, port, llm_model, log_level, workers
    ):
        """
        Property: YAML and JSON formats produce equivalent configuration.
        
        For any valid configuration values, loading from YAML and JSON files
        should produce the same ServerConfig object.
        """
        config_data = {
            'host': host,
            'port': port,
            'llm_model': llm_model,
            'log_level': log_level,
            'workers': workers,
        }
        
        # Create YAML file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)
        
        # Create JSON file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            json_path = Path(f.name)
        
        try:
            # Load from both formats
            config_yaml = load_config(config_file=yaml_path)
            config_json = load_config(config_file=json_path)
            
            # Verify equivalence
            assert config_yaml.host == config_json.host == host
            assert config_yaml.port == config_json.port == port
            assert config_yaml.llm_model == config_json.llm_model == llm_model
            assert config_yaml.log_level == config_json.log_level == log_level
            assert config_yaml.workers == config_json.workers == workers
            
        finally:
            yaml_path.unlink()
            json_path.unlink()
    
    @given(
        llm_provider=st.sampled_from(["ollama", "openai-compatible"]),
        cache_size_mb=st.integers(min_value=1, max_value=10000),
        request_timeout_seconds=st.integers(min_value=1, max_value=300),
    )
    @settings(max_examples=100)
    def test_property_format_preserves_types(
        self, llm_provider, cache_size_mb, request_timeout_seconds
    ):
        """
        Property: All formats correctly preserve data types.
        
        For any configuration values with different types (strings, integers),
        all supported formats should correctly parse and preserve the types.
        """
        config_data = {
            'llm_provider': llm_provider,
            'cache_size_mb': cache_size_mb,
            'request_timeout_seconds': request_timeout_seconds,
        }
        
        # Test YAML
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)
        
        # Test JSON
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            json_path = Path(f.name)
        
        try:
            config_yaml = load_config(config_file=yaml_path)
            config_json = load_config(config_file=json_path)
            
            # Verify types are preserved
            assert isinstance(config_yaml.llm_provider, str)
            assert isinstance(config_yaml.cache_size_mb, int)
            assert isinstance(config_yaml.request_timeout_seconds, int)
            
            assert isinstance(config_json.llm_provider, str)
            assert isinstance(config_json.cache_size_mb, int)
            assert isinstance(config_json.request_timeout_seconds, int)
            
            # Verify values match
            assert config_yaml.llm_provider == config_json.llm_provider
            assert config_yaml.cache_size_mb == config_json.cache_size_mb
            assert (
                config_yaml.request_timeout_seconds ==
                config_json.request_timeout_seconds
            )
            
        finally:
            yaml_path.unlink()
            json_path.unlink()
    
    @given(
        embedding_device=st.sampled_from(["cpu", "cuda", "mps"]),
        log_format=st.sampled_from(["json", "text"]),
    )
    @settings(max_examples=100)
    def test_property_format_handles_literals(
        self, embedding_device, log_format
    ):
        """
        Property: All formats correctly handle Literal types.
        
        For any configuration with Literal type fields, all formats should
        correctly parse and validate the values.
        """
        config_data = {
            'embedding_device': embedding_device,
            'log_format': log_format,
        }
        
        # Test YAML
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)
        
        # Test JSON
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            json_path = Path(f.name)
        
        try:
            config_yaml = load_config(config_file=yaml_path)
            config_json = load_config(config_file=json_path)
            
            # Verify values are correctly parsed
            assert config_yaml.embedding_device == embedding_device
            assert config_yaml.log_format == log_format
            assert config_json.embedding_device == embedding_device
            assert config_json.log_format == log_format
            
        finally:
            yaml_path.unlink()
            json_path.unlink()


class TestInvalidConfigurationRejection:
    """
    Property-based tests for invalid configuration rejection.
    
    **Feature: client-server-separation, Property 3: Invalid configuration rejection**
    
    For any invalid server configuration (missing required values, invalid formats,
    out-of-range values), the server should fail to start and provide a clear error
    message indicating which configuration is invalid.
    
    **Validates: Requirements 1.4, 3.4, 3.5**
    """
    
    @given(
        port=st.one_of(
            st.integers(max_value=0),  # Negative or zero ports
            st.integers(min_value=65536),  # Ports above valid range
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_port_rejected(self, port):
        """
        Property: Invalid port values should be rejected with clear error.
        
        For any port value outside the valid range [1, 65535], configuration
        should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(port=port)
        
        # Verify error message is clear and mentions the issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['port', 'validation', 'greater', 'less'])
    
    @given(
        llm_provider=st.text(min_size=1).filter(
            lambda x: x not in ["ollama", "openai-compatible"]
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_llm_provider_rejected(self, llm_provider):
        """
        Property: Invalid LLM provider values should be rejected.
        
        For any LLM provider that is not 'ollama' or 'openai-compatible',
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(llm_provider=llm_provider)
        
        # Verify error message mentions the invalid provider
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['llm_provider', 'literal', 'input'])
    
    @given(
        log_level=st.text(min_size=1).filter(
            lambda x: x not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_log_level_rejected(self, log_level):
        """
        Property: Invalid log level values should be rejected.
        
        For any log level that is not one of the valid levels,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(log_level=log_level)
        
        # Verify error message mentions log level
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['log_level', 'literal', 'input'])
    
    @given(
        llm_endpoint=st.one_of(
            st.just(""),  # Empty string
            st.text(min_size=1, max_size=50).filter(
                lambda x: not x.startswith("http://") and not x.startswith("https://")
            ),  # Invalid URL format
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_llm_endpoint_rejected(self, llm_endpoint):
        """
        Property: Invalid LLM endpoint URLs should be rejected.
        
        For any LLM endpoint that is empty or doesn't start with http:// or https://,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(llm_endpoint=llm_endpoint)
        
        # Verify error message mentions the endpoint issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['endpoint', 'http', 'empty', 'value'])
    
    @given(
        vector_db_url=st.one_of(
            st.just(""),  # Empty string
            st.text(min_size=1, max_size=50).filter(
                lambda x: not x.startswith("http://") and not x.startswith("https://")
            ),  # Invalid URL format
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_vector_db_url_rejected(self, vector_db_url):
        """
        Property: Invalid vector database URLs should be rejected.
        
        For any vector DB URL that is empty or doesn't start with http:// or https://,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(vector_db_url=vector_db_url)
        
        # Verify error message mentions the URL issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['url', 'http', 'empty', 'value', 'database'])
    
    @given(
        embedding_device=st.text(min_size=1).filter(
            lambda x: x not in ["cpu", "cuda", "mps"]
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_embedding_device_rejected(self, embedding_device):
        """
        Property: Invalid embedding device values should be rejected.
        
        For any embedding device that is not 'cpu', 'cuda', or 'mps',
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(embedding_device=embedding_device)
        
        # Verify error message mentions the device issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['device', 'embedding', 'cpu', 'cuda', 'mps'])
    
    @given(
        cache_size_mb=st.integers(max_value=0)  # Zero or negative cache size
    )
    @settings(max_examples=100)
    def test_property_invalid_cache_size_rejected(self, cache_size_mb):
        """
        Property: Invalid cache size values should be rejected.
        
        For any cache size that is zero or negative,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(cache_size_mb=cache_size_mb)
        
        # Verify error message mentions the size issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['cache_size_mb', 'greater', 'validation'])
    
    @given(
        workers=st.integers(max_value=0)  # Zero or negative workers
    )
    @settings(max_examples=100)
    def test_property_invalid_workers_rejected(self, workers):
        """
        Property: Invalid worker count values should be rejected.
        
        For any worker count that is zero or negative,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(workers=workers)
        
        # Verify error message mentions the workers issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['workers', 'greater', 'validation'])
    
    @given(
        timeout=st.integers(max_value=0)  # Zero or negative timeout
    )
    @settings(max_examples=100)
    def test_property_invalid_timeout_rejected(self, timeout):
        """
        Property: Invalid timeout values should be rejected.
        
        For any timeout that is zero or negative,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(request_timeout_seconds=timeout)
        
        # Verify error message mentions the timeout issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['timeout', 'greater', 'validation'])
    
    @given(
        cache_dir=st.just("")  # Empty cache directory
    )
    @settings(max_examples=100)
    def test_property_empty_cache_dir_rejected(self, cache_dir):
        """
        Property: Empty cache directory should be rejected.
        
        For any empty cache directory path,
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(cache_dir=cache_dir)
        
        # Verify error message mentions the cache directory issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['cache', 'directory', 'empty', 'value'])
    
    @given(
        embedding_provider=st.text(min_size=1).filter(
            lambda x: x not in ["local", "ollama", "openai-compatible"]
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_embedding_provider_rejected(self, embedding_provider):
        """
        Property: Invalid embedding provider values should be rejected.
        
        For any embedding provider that is not 'local', 'ollama', or 'openai-compatible',
        configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(embedding_provider=embedding_provider)
        
        # Verify error message mentions the embedding provider issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['embedding_provider', 'literal', 'input'])
    
    @given(
        embedding_endpoint=st.one_of(
            st.just(""),  # Empty string
            st.text(min_size=1, max_size=50).filter(
                lambda x: not x.startswith("http://") and not x.startswith("https://")
            ),  # Invalid URL format
        )
    )
    @settings(max_examples=100)
    def test_property_invalid_embedding_endpoint_rejected(self, embedding_endpoint):
        """
        Property: Invalid embedding endpoint URLs should be rejected for remote providers.
        
        For any embedding endpoint that is empty or doesn't start with http:// or https://
        when using a remote provider, configuration should fail with a validation error.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(
                embedding_provider="ollama",  # Remote provider
                embedding_endpoint=embedding_endpoint
            )
        
        # Verify error message mentions the endpoint issue
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['endpoint', 'http', 'empty', 'value', 'required'])
    
    @given(
        provider=st.sampled_from(["ollama", "openai-compatible"])
    )
    @settings(max_examples=100)
    def test_property_remote_provider_requires_endpoint(self, provider):
        """
        Property: Remote embedding providers require an endpoint.
        
        For any remote embedding provider (ollama, openai-compatible),
        configuration should fail if no endpoint is provided.
        """
        with pytest.raises(Exception) as exc_info:
            ServerConfig(
                embedding_provider=provider,
                embedding_endpoint=None  # Missing endpoint
            )
        
        # Verify error message mentions the endpoint requirement
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['endpoint', 'required'])
