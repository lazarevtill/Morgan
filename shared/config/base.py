"""
Base configuration management for Morgan AI Assistant services
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class BaseConfig:
    """Base configuration class with environment variable support"""

    def __init__(self, config_name: str = "config.yaml"):
        self.config_name = config_name
        self.config_dir = self._get_config_dir()
        self.config_path = self.config_dir / config_name
        self._config = {}
        self._load_config()

    def _get_config_dir(self) -> Path:
        """Get configuration directory from environment or default"""
        config_dir = os.getenv("MORGAN_CONFIG_DIR")
        if config_dir:
            return Path(config_dir)

        # Default config directories to check
        default_dirs = [
            Path.cwd() / "config",
            Path.home() / ".morgan" / "config",
            Path("/etc/morgan")
        ]

        for config_dir in default_dirs:
            if config_dir.exists():
                return config_dir

        # Create default config directory
        default_dir = Path.cwd() / "config"
        default_dir.mkdir(exist_ok=True)
        return default_dir

    def _load_config(self):
        """Load configuration from file and environment variables"""
        config = {}

        # Load from YAML file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                raise ConfigError(f"Failed to load config file {self.config_path}: {e}")

        # Override with environment variables
        config = self._override_from_env(config)

        self._config = config

    def _override_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration values with environment variables"""
        result = config.copy()

        # Convert config keys to environment variable names
        for key, value in self._flatten_config(config).items():
            env_var = f"MORGAN_{key.replace('.', '_').upper()}"
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                result = self._set_nested_value(result, key, self._convert_env_value(env_value))

        return result

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Flatten nested config dict for environment variable mapping"""
        flattened = {}

        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, full_key))
            else:
                flattened[full_key] = str(value)

        return flattened

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """Set a nested value in config dict"""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config = self._set_nested_value(self._config, key, value)

    def all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()

    def save(self):
        """Save configuration to file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigError(f"Failed to save config file {self.config_path}: {e}")

    def reload(self):
        """Reload configuration from file and environment"""
        self._load_config()


class ServiceConfig(BaseConfig):
    """Configuration for a specific service"""

    def __init__(self, service_name: str, config_name: Optional[str] = None):
        self.service_name = service_name
        config_name = config_name or f"{service_name}.yaml"
        super().__init__(config_name)

    def get_service_url(self, default_port: int = 8000) -> str:
        """Get service URL with fallback to localhost"""
        host = self.get("host", "localhost")
        port = self.get("port", default_port)
        return f"http://{host}:{port}"

    def get_database_url(self) -> str:
        """Get database connection URL"""
        db_type = self.get("database.type", "sqlite")
        if db_type == "sqlite":
            db_path = self.get("database.path", "data/morgan.db")
            return f"sqlite:///{db_path}"
        elif db_type == "postgresql":
            host = self.get("database.host", "localhost")
            port = self.get("database.port", 5432)
            database = self.get("database.name", "morgan")
            user = self.get("database.user", "morgan")
            password = self.get("database.password", "")
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ConfigError(f"Unsupported database type: {db_type}")


def get_default_config(service_name: str) -> Dict[str, Any]:
    """Get default configuration for a service"""
    defaults = {
        "llm": {
            "host": "0.0.0.0",
            "port": 8001,
            "model": "superdrew100/llama3-abliterated:latest",
            "ollama_url": "http://192.168.101.3:11434/",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 30,
            "gpu_layers": -1,
            "context_window": 4096,
            "system_prompt": "You are Morgan, a helpful AI assistant.",
            "log_level": "INFO"
        },
        "tts": {
            "host": "0.0.0.0",
            "port": 8002,
            "model": "kokoro",
            "device": "cuda",
            "language": "en-us",
            "voice": "af_heart",
            "speed": 1.0,
            "output_format": "wav",
            "sample_rate": 22050,
            "log_level": "INFO"
        },
        "stt": {
            "host": "0.0.0.0",
            "port": 8003,
            "model": "whisper-base",
            "device": "cuda",
            "language": "auto",
            "sample_rate": 16000,
            "chunk_size": 1024,
            "threshold": 0.5,
            "min_silence_duration": 0.5,
            "log_level": "INFO"
        },
        "vad": {
            "host": "0.0.0.0",
            "port": 8004,
            "model": "silero_vad",
            "threshold": 0.5,
            "min_speech_duration": 0.25,
            "max_speech_duration": 30.0,
            "window_size": 512,
            "sample_rate": 16000,
            "device": "cpu",
            "log_level": "INFO"
        },
        "core": {
            "host": "0.0.0.0",
            "port": 8000,
            "llm_service_url": "http://llm:8001",
            "tts_service_url": "http://tts:8002",
            "stt_service_url": "http://stt:8003",
            "vad_service_url": "http://vad:8004",
            "conversation_timeout": 1800,
            "max_history": 50,
            "log_level": "INFO"
        }
    }

    return defaults.get(service_name, {})
