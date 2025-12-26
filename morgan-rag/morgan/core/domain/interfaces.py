from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class EmbeddingProvider(ABC):
    """Interface for embedding services."""
    @abstractmethod
    async def embed_texts(self, texts: List[str], instruction: Optional[str] = None) -> List[List[float]]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

class RerankerProvider(ABC):
    """Interface for reranking services."""
    @abstractmethod
    async def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        pass

class ScraperProvider(ABC):
    """Interface for web scraping services."""
    @abstractmethod
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        pass
