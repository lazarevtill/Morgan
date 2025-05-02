"""
Configuration management for Morgan Core
"""
import os
from typing import Dict, Any, Optional

import yaml


class ConfigManager:
    """Manages configuration loading and validation for Morgan Core"""

    def __init__(self, config_dir: str = "/opt/morgan/config"):
        self.config_dir = config_dir
        self._core_config = None
        self._handlers_config = None
        self._devices_config = None
        self._voices_config = None

    def load_core_config(self) -> Dict[str, Any]:
        """Load the core configuration file"""
        if self._core_config:
            return self._core_config

        config_path = os.path.join(self.config_dir, "core.yaml")
        with open(config_path, 'r') as file:
            self._core_config = yaml.safe_load(file)

        # Apply default values for missing configuration
        self._set_defaults()

        # Validate the configuration
        self._validate_core_config()

        return self._core_config

    def load_handlers_config(self) -> Dict[str, Any]:
        """Load the handlers configuration file"""
        if self._handlers_config:
            return self._handlers_config

        config_path = os.path.join(self.config_dir, "handlers.yaml")
        with open(config_path, 'r') as file:
            self._handlers_config = yaml.safe_load(file)

        return self._handlers_config

    def load_devices_config(self) -> Dict[str, Any]:
        """Load the device mappings configuration file"""
        if self._devices_config:
            return self._devices_config

        config_path = os.path.join(self.config_dir, "devices.yaml")
        with open(config_path, 'r') as file:
            self._devices_config = yaml.safe_load(file)

        return self._devices_config

    def load_voices_config(self) -> Dict[str, Any]:
        """Load the voice profiles configuration file"""
        if self._voices_config:
            return self._voices_config

        config_path = os.path.join(self.config_dir, "voices.yaml")
        with open(config_path, 'r') as file:
            self._voices_config = yaml.safe_load(file)

        return self._voices_config

    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific service"""
        core_config = self.load_core_config()
        return core_config.get('services', {}).get(service_name)

    def save_config(self, config_type: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        config_path = os.path.join(self.config_dir, f"{config_type}.yaml")

        try:
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False)

            # Reset cached config
            setattr(self, f"_{config_type}_config", None)
            return True
        except Exception:
            return False

    def _set_defaults(self):
        """Set default values for missing configuration"""
        # System defaults
        system = self._core_config.setdefault('system', {})
        system.setdefault('name', 'Morgan')
        system.setdefault('log_level', 'info')
        system.setdefault('data_dir', '/app/data')

        # Services defaults
        services = self._core_config.setdefault('services', {})

        # LLM defaults
        llm = services.setdefault('llm', {})
        llm.setdefault('url', 'http://llm-service:8001')
        llm.setdefault('model', 'mistral')

        # TTS defaults
        tts = services.setdefault('tts', {})
        tts.setdefault('url', 'http://tts-service:8002')
        tts.setdefault('default_voice', 'morgan_default')

        # STT defaults
        stt = services.setdefault('stt', {})
        stt.setdefault('url', 'http://stt-service:8003')
        stt.setdefault('model', 'whisper-large-v3')

    def _validate_core_config(self):
        """Validate the core configuration"""
        # Check required fields
        if 'services' not in self._core_config:
            raise ValueError("Missing 'services' section in core configuration")

        services = self._core_config['services']
        required_services = ['llm', 'tts', 'stt']
        for service in required_services:
            if service not in services:
                raise ValueError(f"Missing '{service}' configuration in services section")

            if 'url' not in services[service]:
                raise ValueError(f"Missing 'url' in {service} service configuration")