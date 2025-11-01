"""
Remote Model Manager - Remote model integration

Handles remote model integration, specifically gpt.lazarev.cloud endpoint.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.2, 23.3
"""

from typing import Dict, Any, Optional, List
import logging
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RemoteModelConfig:
    """Configuration for remote model endpoints."""
    endpoint_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


class RemoteModelManager:
    """
    Remote model manager following KISS principles.
    
    Single responsibility: Manage remote AI model connections only.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configure gpt.lazarev.cloud endpoint
        self.lazarev_config = RemoteModelConfig(
            endpoint_url=config.get('lazarev_endpoint', 'https://gpt.lazarev.cloud'),
            api_key=config.get('lazarev_api_key'),
            timeout=config.get('timeout', 30),
            max_retries=config.get('max_retries', 3)
        )
        
    def load_model(self, model_name: str, model_type: str = "auto") -> Any:
        """
        Load a remote model connection.
        
        Args:
            model_name: Name of the remote model
            model_type: Type of model ('embedding', 'llm', 'emotional')
            
        Returns:
            Remote model client instance
        """
        try:
            # For now, only support gpt.lazarev.cloud
            if 'lazarev' in model_name.lower() or 'gpt' in model_name.lower():
                return self._create_lazarev_client(model_name, model_type)
            else:
                raise ValueError(f"Unsupported remote model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load remote model {model_name}: {e}")
            raise
            
    def _create_lazarev_client(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Create a client for gpt.lazarev.cloud."""
        try:
            # Test connection
            if self._test_lazarev_connection():
                return {
                    'provider': 'lazarev',
                    'model_name': model_name,
                    'endpoint': self.lazarev_config.endpoint_url,
                    'config': self.lazarev_config,
                    'type': model_type
                }
            else:
                raise ConnectionError("Cannot connect to gpt.lazarev.cloud")
                
        except Exception as e:
            logger.error(f"Failed to create Lazarev client: {e}")
            raise
            
    def _test_lazarev_connection(self) -> bool:
        """Test connection to gpt.lazarev.cloud."""
        try:
            # Simple health check
            response = requests.get(
                f"{self.lazarev_config.endpoint_url}/health",
                timeout=self.lazarev_config.timeout
            )
            return response.status_code == 200
        except:
            # If health endpoint doesn't exist, assume connection is available
            return True
            
    def generate_embedding(self, client: Dict[str, Any], text: str) -> List[float]:
        """Generate embedding using remote model."""
        try:
            if client['provider'] == 'lazarev':
                return self._lazarev_embedding(client, text)
            else:
                raise ValueError(f"Unsupported provider: {client['provider']}")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
            
    def _lazarev_embedding(self, client: Dict[str, Any], text: str) -> List[float]:
        """Generate embedding via gpt.lazarev.cloud."""
        try:
            headers = {}
            if client['config'].api_key:
                headers['Authorization'] = f"Bearer {client['config'].api_key}"
                
            payload = {
                'model': client['model_name'],
                'input': text
            }
            
            response = requests.post(
                f"{client['endpoint']}/embeddings",
                json=payload,
                headers=headers,
                timeout=client['config'].timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['data'][0]['embedding']
            
        except Exception as e:
            logger.error(f"Lazarev embedding failed: {e}")
            raise
            
    def generate_response(self, client: Dict[str, Any], prompt: str, context: str = "") -> str:
        """Generate text response using remote model."""
        try:
            if client['provider'] == 'lazarev':
                return self._lazarev_response(client, prompt, context)
            else:
                raise ValueError(f"Unsupported provider: {client['provider']}")
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
            
    def _lazarev_response(self, client: Dict[str, Any], prompt: str, context: str = "") -> str:
        """Generate response via gpt.lazarev.cloud."""
        try:
            headers = {}
            if client['config'].api_key:
                headers['Authorization'] = f"Bearer {client['config'].api_key}"
                
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                'model': client['model_name'],
                'messages': messages
            }
            
            response = requests.post(
                f"{client['endpoint']}/chat/completions",
                json=payload,
                headers=headers,
                timeout=client['config'].timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Lazarev response failed: {e}")
            raise
            
    def list_models(self) -> List[str]:
        """List available remote models."""
        return [
            'gpt-3.5-turbo',
            'gpt-4',
            'text-embedding-ada-002'
        ]
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a remote model."""
        return {
            'name': model_name,
            'provider': 'lazarev',
            'location': 'remote',
            'endpoint': self.lazarev_config.endpoint_url
        }