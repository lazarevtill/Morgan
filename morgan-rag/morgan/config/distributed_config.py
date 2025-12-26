"""
Distributed Configuration Loader for Morgan.

100% Self-Hosted - No API Keys Required.
Loads distributed architecture configuration from YAML files,
environment variables, or defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


# Default config file locations (searched in order)
DEFAULT_CONFIG_PATHS = [
    Path("config/distributed.local.yaml"),  # Local override
    Path("config/distributed.yaml"),  # Project config
    Path.home() / ".morgan" / "distributed.yaml",  # User config
    Path("/etc/morgan/distributed.yaml"),  # System config
]


@dataclass
class LLMConfig:
    """LLM configuration (Ollama)."""

    main_model: str = "qwen2.5:32b-instruct-q4_K_M"
    fast_model: str = "qwen2.5:7b-instruct-q5_K_M"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    """
    Embedding configuration (self-hosted via Ollama).

    Models (Qwen3-Embedding via Ollama):
    - qwen3-embedding:0.6b: 896 dims (lightweight)
    - qwen3-embedding:4b: 2048 dims (recommended for RTX 4070)
    - qwen3-embedding:8b: 4096 dims (best quality, RTX 3090)

    Fallback:
    - all-MiniLM-L6-v2: Via sentence-transformers, 384 dims
    """

    model: str = "qwen3-embedding:4b"
    dimensions: int = 2048
    local_fallback_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 100


@dataclass
class RerankingConfig:
    """
    Reranking configuration (self-hosted).

    Models (via sentence-transformers CrossEncoder):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality
    - BAAI/bge-reranker-base: Multilingual
    """

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_weight: float = 0.6
    original_weight: float = 0.4
    top_k: int = 20


@dataclass
class HostDefinition:
    """Host definition from config."""

    host_id: str
    address: str
    port: int
    role: str
    gpu_model: Optional[str] = None
    gpu_vram_gb: float = 0.0
    models: List[str] = field(default_factory=list)
    api_path: str = "/v1"
    description: str = ""


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    prefix: str = "morgan:"


@dataclass
class QdrantConfig:
    """Qdrant configuration."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    prometheus_host: str = "localhost"
    prometheus_port: int = 9090
    grafana_host: str = "localhost"
    grafana_port: int = 3000


@dataclass
class ModelCacheConfig:
    """Model cache configuration for persistent model storage."""

    base_dir: str = "~/.morgan/models"
    sentence_transformers_home: str = "~/.morgan/models/sentence-transformers"
    hf_home: str = "~/.morgan/models/huggingface"
    ollama_models: str = "~/.ollama/models"
    preload_on_startup: bool = True

    def get_expanded_paths(self) -> Dict[str, Path]:
        """Get paths with ~ expanded."""
        st_home = Path(self.sentence_transformers_home).expanduser()
        return {
            "base_dir": Path(self.base_dir).expanduser(),
            "sentence_transformers_home": st_home,
            "hf_home": Path(self.hf_home).expanduser(),
            "ollama_models": Path(self.ollama_models).expanduser(),
        }

    def ensure_directories(self):
        """Create cache directories if they don't exist."""
        for _, path in self.get_expanded_paths().items():
            path.mkdir(parents=True, exist_ok=True)

    def set_environment_variables(self):
        """
        Set environment variables for model caching and HF authentication.

        Also loads HF_TOKEN from environment for gated model downloads.
        """
        paths = self.get_expanded_paths()
        st_home = str(paths["sentence_transformers_home"])
        hf_home = str(paths["hf_home"])

        # Set cache directories
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = st_home
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = hf_home
        os.environ["HF_DATASETS_CACHE"] = str(paths["hf_home"] / "datasets")

        # Configure HF_TOKEN for gated model downloads
        # Check multiple possible env var names
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        if hf_token:
            # Set all possible HF token env vars for compatibility
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


@dataclass
class DistributedSettings:
    """Global distributed settings."""

    health_check_interval: int = 60
    default_timeout: float = 30.0
    load_balancing_strategy: str = "round_robin"
    enable_failover: bool = True
    max_retries: int = 3


@dataclass
class DistributedArchitectureConfig:
    """Complete distributed architecture configuration."""

    settings: DistributedSettings = field(default_factory=DistributedSettings)
    model_cache: ModelCacheConfig = field(default_factory=ModelCacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    hosts: List[HostDefinition] = field(default_factory=list)
    redis: RedisConfig = field(default_factory=RedisConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Source tracking
    config_source: Optional[str] = None

    def setup_model_cache(self):
        """Setup model cache directories and environment variables."""
        self.model_cache.ensure_directories()
        self.model_cache.set_environment_variables()

    def get_hosts_by_role(self, role: str) -> List[HostDefinition]:
        """Get all hosts with a specific role."""
        return [h for h in self.hosts if h.role == role]

    def get_host(self, host_id: str) -> Optional[HostDefinition]:
        """Get a host by ID."""
        for host in self.hosts:
            if host.host_id == host_id:
                return host
        return None


def _parse_host(host_data: Dict[str, Any]) -> HostDefinition:
    """Parse a host definition from config data."""
    return HostDefinition(
        host_id=host_data.get("host_id", "unknown"),
        address=host_data.get("address", "localhost"),
        port=host_data.get("port", 8080),
        role=host_data.get("role", "orchestrator"),
        gpu_model=host_data.get("gpu_model"),
        gpu_vram_gb=host_data.get("gpu_vram_gb", 0.0),
        models=host_data.get("models", []),
        api_path=host_data.get("api_path", "/v1"),
        description=host_data.get("description", ""),
    )


def _parse_config(data: Dict[str, Any]) -> DistributedArchitectureConfig:
    """Parse configuration from dictionary."""
    config = DistributedArchitectureConfig()

    # Parse settings
    if "settings" in data:
        s = data["settings"]
        lb_strategy = s.get("load_balancing_strategy", "round_robin")
        config.settings = DistributedSettings(
            health_check_interval=s.get("health_check_interval", 60),
            default_timeout=s.get("default_timeout", 30.0),
            load_balancing_strategy=lb_strategy,
            enable_failover=s.get("enable_failover", True),
            max_retries=s.get("max_retries", 3),
        )

    # Parse model cache config
    if "model_cache" in data:
        mc = data["model_cache"]
        st_default = "~/.morgan/models/sentence-transformers"
        config.model_cache = ModelCacheConfig(
            base_dir=mc.get("base_dir", "~/.morgan/models"),
            sentence_transformers_home=mc.get("sentence_transformers_home", st_default),
            hf_home=mc.get("hf_home", "~/.morgan/models/huggingface"),
            ollama_models=mc.get("ollama_models", "~/.ollama/models"),
            preload_on_startup=mc.get("preload_on_startup", True),
        )

    # Parse LLM config
    if "llm" in data:
        llm_cfg = data["llm"]
        main_default = "qwen2.5:32b-instruct-q4_K_M"
        fast_default = "qwen2.5:7b-instruct-q5_K_M"
        config.llm = LLMConfig(
            main_model=llm_cfg.get("main_model", main_default),
            fast_model=llm_cfg.get("fast_model", fast_default),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )

    # Parse embeddings config
    if "embeddings" in data:
        e = data["embeddings"]
        fallback_model = e.get("local_fallback_model", "all-MiniLM-L6-v2")
        config.embeddings = EmbeddingConfig(
            model=e.get("model", "qwen3-embedding:4b"),
            dimensions=e.get("dimensions", 2048),
            local_fallback_model=fallback_model,
            batch_size=e.get("batch_size", 100),
        )

    # Parse reranking config
    if "reranking" in data:
        r = data["reranking"]
        config.reranking = RerankingConfig(
            model=r.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            rerank_weight=r.get("rerank_weight", 0.6),
            original_weight=r.get("original_weight", 0.4),
            top_k=r.get("top_k", 20),
        )

    # Parse hosts
    if "hosts" in data:
        config.hosts = [_parse_host(h) for h in data["hosts"]]

    # Parse Redis config
    if "redis" in data:
        rd = data["redis"]
        config.redis = RedisConfig(
            host=rd.get("host", "localhost"),
            port=rd.get("port", 6379),
            db=rd.get("db", 0),
            password=rd.get("password", ""),
            prefix=rd.get("prefix", "morgan:"),
        )

    # Parse Qdrant config
    if "qdrant" in data:
        q = data["qdrant"]
        config.qdrant = QdrantConfig(
            host=q.get("host", "localhost"),
            port=q.get("port", 6333),
            grpc_port=q.get("grpc_port", 6334),
        )

    # Parse monitoring config
    if "monitoring" in data:
        m = data["monitoring"]
        config.monitoring = MonitoringConfig(
            prometheus_host=m.get("prometheus", {}).get("host", "localhost"),
            prometheus_port=m.get("prometheus", {}).get("port", 9090),
            grafana_host=m.get("grafana", {}).get("host", "localhost"),
            grafana_port=m.get("grafana", {}).get("port", 3000),
        )

    return config


def load_distributed_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
) -> DistributedArchitectureConfig:
    """
    Load distributed architecture configuration.

    Search order:
    1. Explicit config_path parameter
    2. MORGAN_DISTRIBUTED_CONFIG environment variable
    3. Default config paths (local override, project, user, system)
    4. Built-in defaults

    Args:
        config_path: Explicit path to config file
        use_env: Whether to check environment variable

    Returns:
        Loaded configuration
    """
    if not YAML_AVAILABLE:
        logger.warning(
            "PyYAML not installed, using default configuration. "
            "Install with: pip install pyyaml"
        )
        return DistributedArchitectureConfig()

    # Determine config path
    paths_to_try = []

    # 1. Explicit path
    if config_path:
        paths_to_try.append(Path(config_path))

    # 2. Environment variable
    if use_env:
        env_path = os.environ.get("MORGAN_DISTRIBUTED_CONFIG")
        if env_path:
            paths_to_try.append(Path(env_path))

    # 3. Default paths
    paths_to_try.extend(DEFAULT_CONFIG_PATHS)

    # Try to load from each path
    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                config = _parse_config(data or {})
                config.config_source = str(path)

                logger.info("Loaded distributed config from: %s", path)
                logger.info("Configured %d hosts across roles", len(config.hosts))

                return config

            except yaml.YAMLError as e:
                logger.error("Failed to parse YAML config %s: %s", path, e)
                continue
            except Exception as e:
                logger.error("Failed to load config %s: %s", path, e)
                continue

    # No config found, use defaults
    logger.warning(
        "No distributed config found, using defaults. "
        "Create config/distributed.yaml to customize."
    )
    return DistributedArchitectureConfig()


def save_distributed_config(
    config: DistributedArchitectureConfig,
    config_path: str,
) -> bool:
    """
    Save distributed configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save to

    Returns:
        True if successful
    """
    if not YAML_AVAILABLE:
        logger.error("PyYAML not installed, cannot save configuration")
        return False

    try:
        # Convert to dictionary
        sett = config.settings
        data = {
            "settings": {
                "health_check_interval": sett.health_check_interval,
                "default_timeout": sett.default_timeout,
                "load_balancing_strategy": sett.load_balancing_strategy,
                "enable_failover": sett.enable_failover,
                "max_retries": sett.max_retries,
            },
            "llm": {
                "main_model": config.llm.main_model,
                "fast_model": config.llm.fast_model,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
            },
            "embeddings": {
                "model": config.embeddings.model,
                "dimensions": config.embeddings.dimensions,
                "local_fallback_model": config.embeddings.local_fallback_model,
                "batch_size": config.embeddings.batch_size,
            },
            "reranking": {
                "model": config.reranking.model,
                "rerank_weight": config.reranking.rerank_weight,
                "original_weight": config.reranking.original_weight,
                "top_k": config.reranking.top_k,
            },
            "hosts": [
                {
                    "host_id": h.host_id,
                    "address": h.address,
                    "port": h.port,
                    "role": h.role,
                    "gpu_model": h.gpu_model,
                    "gpu_vram_gb": h.gpu_vram_gb,
                    "models": h.models,
                    "api_path": h.api_path,
                    "description": h.description,
                }
                for h in config.hosts
            ],
            "redis": {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db,
                "password": config.redis.password,
                "prefix": config.redis.prefix,
            },
            "qdrant": {
                "host": config.qdrant.host,
                "port": config.qdrant.port,
                "grpc_port": config.qdrant.grpc_port,
            },
            "monitoring": {
                "prometheus": {
                    "host": config.monitoring.prometheus_host,
                    "port": config.monitoring.prometheus_port,
                },
                "grafana": {
                    "host": config.monitoring.grafana_host,
                    "port": config.monitoring.grafana_port,
                },
            },
        }

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved distributed config to: %s", config_path)
        return True

    except Exception as e:
        logger.error("Failed to save config to %s: %s", config_path, e)
        return False


# Singleton cached config
_cached_config: Optional[DistributedArchitectureConfig] = None


def get_distributed_config(
    reload: bool = False,
    config_path: Optional[str] = None,
) -> DistributedArchitectureConfig:
    """
    Get distributed configuration (cached singleton).

    Args:
        reload: Force reload from file
        config_path: Override config path

    Returns:
        Distributed configuration
    """
    global _cached_config

    if _cached_config is None or reload:
        _cached_config = load_distributed_config(config_path=config_path)

    return _cached_config
