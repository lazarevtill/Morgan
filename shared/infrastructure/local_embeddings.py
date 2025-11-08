"""
Local embeddings infrastructure for offline or low-latency deployments

Provides:
- Async embedding generation with local models
- Model loading and management
- Batch processing
- Memory management
- GPU acceleration support
- Comprehensive error handling
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..utils.errors import ModelError, ErrorCode

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported model backends"""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    FASTEMBED = "fastembed"


@dataclass
class LocalEmbeddingConfig:
    """Local embedding configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: ModelBackend = ModelBackend.SENTENCE_TRANSFORMERS
    device: str = "cpu"  # cpu, cuda, mps
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    use_async: bool = True
    cache_folder: Optional[str] = None
    trust_remote_code: bool = False


class LocalEmbeddingModel:
    """
    Local embedding model wrapper with async support

    Supports multiple backends:
    - sentence-transformers (recommended)
    - Hugging Face transformers
    - ONNX Runtime
    - FastEmbed
    """

    def __init__(self, config: Optional[LocalEmbeddingConfig] = None):
        self.config = config or LocalEmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.embedding_dimension = 0

        # Metrics
        self.embedding_count = 0
        self.total_processing_time = 0.0

        logger.info(
            f"Local embedding model initialized: "
            f"model={self.config.model_name}, "
            f"backend={self.config.backend.value}, "
            f"device={self.config.device}"
        )

    async def load(self):
        """Load the embedding model"""
        try:
            if self.config.backend == ModelBackend.SENTENCE_TRANSFORMERS:
                await self._load_sentence_transformers()

            elif self.config.backend == ModelBackend.HUGGINGFACE:
                await self._load_huggingface()

            elif self.config.backend == ModelBackend.ONNX:
                await self._load_onnx()

            elif self.config.backend == ModelBackend.FASTEMBED:
                await self._load_fastembed()

            else:
                raise ModelError(
                    f"Unsupported backend: {self.config.backend}",
                    ErrorCode.MODEL_LOAD_ERROR
                )

            self.is_loaded = True
            logger.info(
                f"Model loaded successfully: "
                f"dimension={self.embedding_dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(
                f"Model loading failed: {e}",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def _load_sentence_transformers(self):
        """Load using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_folder
                )
            )

            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        except ImportError:
            raise ModelError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def _load_huggingface(self):
        """Load using Hugging Face transformers"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            loop = asyncio.get_event_loop()

            # Load tokenizer and model
            self.tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_folder,
                    trust_remote_code=self.config.trust_remote_code
                )
            )

            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModel.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_folder,
                    trust_remote_code=self.config.trust_remote_code
                )
            )

            # Move to device
            device = torch.device(self.config.device)
            self.model.to(device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dimension = self.model.config.hidden_size

        except ImportError:
            raise ModelError(
                "transformers not installed. "
                "Install with: pip install transformers torch",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def _load_onnx(self):
        """Load using ONNX Runtime"""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            loop = asyncio.get_event_loop()

            # Load tokenizer
            self.tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_folder
                )
            )

            # Create ONNX session
            # Note: This assumes the model is already converted to ONNX
            model_path = f"{self.config.model_name}.onnx"

            providers = ['CPUExecutionProvider']
            if self.config.device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')

            self.model = await loop.run_in_executor(
                None,
                lambda: ort.InferenceSession(model_path, providers=providers)
            )

            # Get embedding dimension from model output
            output_shape = self.model.get_outputs()[0].shape
            self.embedding_dimension = output_shape[-1]

        except ImportError:
            raise ModelError(
                "onnxruntime not installed. "
                "Install with: pip install onnxruntime",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def _load_fastembed(self):
        """Load using FastEmbed"""
        try:
            from fastembed import TextEmbedding

            loop = asyncio.get_event_loop()

            self.model = await loop.run_in_executor(
                None,
                lambda: TextEmbedding(
                    model_name=self.config.model_name,
                    cache_dir=self.config.cache_folder
                )
            )

            # Get embedding dimension
            # FastEmbed doesn't provide this directly, so we'll generate
            # a test embedding
            test_embedding = list(self.model.embed(["test"]))[0]
            self.embedding_dimension = len(test_embedding)

        except ImportError:
            raise ModelError(
                "fastembed not installed. "
                "Install with: pip install fastembed",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def unload(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if using GPU
        if self.config.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        logger.info("Model unloaded")

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            batch_size: Optional batch size override

        Returns:
            Single embedding or list of embeddings

        Raises:
            ModelError: If model not loaded or embedding fails
        """
        if not self.is_loaded:
            raise ModelError(
                "Model not loaded. Call load() first.",
                ErrorCode.MODEL_NOT_LOADED
            )

        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        try:
            batch_size = batch_size or self.config.batch_size

            # Generate embeddings based on backend
            if self.config.backend == ModelBackend.SENTENCE_TRANSFORMERS:
                embeddings = await self._embed_sentence_transformers(
                    texts, batch_size
                )

            elif self.config.backend == ModelBackend.HUGGINGFACE:
                embeddings = await self._embed_huggingface(texts, batch_size)

            elif self.config.backend == ModelBackend.ONNX:
                embeddings = await self._embed_onnx(texts, batch_size)

            elif self.config.backend == ModelBackend.FASTEMBED:
                embeddings = await self._embed_fastembed(texts, batch_size)

            else:
                raise ModelError(
                    f"Unsupported backend: {self.config.backend}",
                    ErrorCode.MODEL_INFERENCE_ERROR
                )

            # Normalize if configured
            if self.config.normalize_embeddings:
                embeddings = self._normalize(embeddings)

            # Track metrics
            self.embedding_count += len(texts)

            # Return single embedding if input was single text
            if is_single:
                return embeddings[0]

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ModelError(
                f"Embedding generation failed: {e}",
                ErrorCode.MODEL_INFERENCE_ERROR
            )

    async def _embed_sentence_transformers(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using sentence-transformers"""
        loop = asyncio.get_event_loop()

        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        )

        return embeddings.tolist()

    async def _embed_huggingface(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using Hugging Face transformers"""
        import torch

        loop = asyncio.get_event_loop()

        def _generate():
            all_embeddings = []

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]

                    # Tokenize
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    )

                    # Move to device
                    device = torch.device(self.config.device)
                    encoded = {k: v.to(device) for k, v in encoded.items()}

                    # Generate embeddings
                    outputs = self.model(**encoded)

                    # Mean pooling
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state,
                        encoded['attention_mask']
                    )

                    all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings)

        embeddings = await loop.run_in_executor(None, _generate)
        return embeddings.tolist()

    async def _embed_onnx(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using ONNX Runtime"""
        loop = asyncio.get_event_loop()

        def _generate():
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='np'
                )

                # Run inference
                outputs = self.model.run(
                    None,
                    {
                        'input_ids': encoded['input_ids'],
                        'attention_mask': encoded['attention_mask']
                    }
                )

                # Mean pooling
                embeddings = self._mean_pooling_numpy(
                    outputs[0],
                    encoded['attention_mask']
                )

                all_embeddings.append(embeddings)

            return np.vstack(all_embeddings)

        embeddings = await loop.run_in_executor(None, _generate)
        return embeddings.tolist()

    async def _embed_fastembed(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using FastEmbed"""
        loop = asyncio.get_event_loop()

        embeddings = await loop.run_in_executor(
            None,
            lambda: list(self.model.embed(texts, batch_size=batch_size))
        )

        return embeddings

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        """Mean pooling for Hugging Face models"""
        import torch

        # Expand attention mask
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Sum mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    @staticmethod
    def _mean_pooling_numpy(token_embeddings, attention_mask):
        """Mean pooling for ONNX models"""
        # Expand attention mask
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], axis=-1)
        )

        # Sum embeddings
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)

        # Sum mask
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    @staticmethod
    def _normalize(embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length"""
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / np.clip(norms, a_min=1e-9, a_max=None)
        return normalized.tolist()

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "backend": self.config.backend.value,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
            "embedding_dimension": self.embedding_dimension,
            "embedding_count": self.embedding_count
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics"""
        return {
            "embedding_count": self.embedding_count,
            "avg_processing_time": (
                self.total_processing_time / self.embedding_count
                if self.embedding_count > 0 else 0.0
            )
        }


class LocalEmbeddingPool:
    """
    Pool of local embedding models for concurrent processing

    Allows multiple model instances for parallel processing,
    useful for high-throughput scenarios.
    """

    def __init__(
        self,
        config: LocalEmbeddingConfig,
        pool_size: int = 1
    ):
        """
        Initialize embedding pool

        Args:
            config: Embedding configuration
            pool_size: Number of model instances
        """
        self.config = config
        self.pool_size = pool_size
        self.models: List[LocalEmbeddingModel] = []
        self.semaphore = asyncio.Semaphore(pool_size)

        logger.info(
            f"Local embedding pool initialized: pool_size={pool_size}"
        )

    async def start(self):
        """Initialize all models in the pool"""
        for i in range(self.pool_size):
            model = LocalEmbeddingModel(self.config)
            await model.load()
            self.models.append(model)

        logger.info(f"Embedding pool started with {self.pool_size} models")

    async def stop(self):
        """Unload all models"""
        for model in self.models:
            await model.unload()

        self.models.clear()
        logger.info("Embedding pool stopped")

    async def embed(
        self,
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using the pool

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings
        """
        async with self.semaphore:
            # Use the least loaded model
            model = min(self.models, key=lambda m: m.embedding_count)
            return await model.embed(texts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics"""
        return {
            "pool_size": self.pool_size,
            "models": [model.get_metrics() for model in self.models]
        }
