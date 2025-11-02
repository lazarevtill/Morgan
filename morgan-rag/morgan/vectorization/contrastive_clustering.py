"""
Contrastive Clustering Engine for Morgan RAG.

Implements category-specific bias vectors to improve clustering quality:
- Applies bias to group similar content together (minimize intra-category distance)
- Separates different content types (maximize inter-category distance)
- Uses scale-aware bias strength for hierarchical embeddings
- Maintains unit sphere properties after bias application

Features:
- Reproducible category-specific bias generation
- Scale-aware bias strength (coarse 1.5x, medium 1.0x, fine 0.5x)
- Normalized embedding output for consistent similarity calculations
- Integration with hierarchical embedding service
"""

import hashlib
from typing import List, Dict, Any
import numpy as np

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ContrastiveClusteringEngine:
    """
    Engine for applying contrastive clustering to embeddings.
    
    Implements category-specific bias vectors that:
    - Pull similar content together (same category)
    - Push different content apart (different categories)
    - Use scale-aware bias strength for hierarchical search
    - Maintain unit sphere properties for consistent similarity
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize contrastive clustering engine.
        
        Args:
            seed: Random seed for reproducible bias generation
        """
        self.seed = seed
        self.bias_cache: Dict[str, np.ndarray] = {}
        
        # Scale-aware bias strengths as per requirements
        self.bias_strengths = {
            "coarse": 1.5,   # Strongest for category filtering
            "medium": 1.0,   # Balanced for pattern matching
            "fine": 0.5      # Weakest for precise retrieval
        }
        
        logger.info(
            "ContrastiveClusteringEngine initialized with seed=%d, "
            "bias_strengths=%s", seed, self.bias_strengths
        )
    
    def apply_contrastive_bias(
        self,
        embedding: List[float],
        category: str,
        scale: str
    ) -> List[float]:
        """
        Apply category-specific contrastive bias to embedding.
        
        This method implements the core contrastive clustering algorithm:
        1. Generate reproducible category-specific bias vector
        2. Apply scale-aware bias strength
        3. Add bias to original embedding
        4. Normalize to maintain unit sphere properties
        
        Args:
            embedding: Original embedding vector
            category: Content category for bias generation
            scale: Embedding scale (coarse, medium, fine)
            
        Returns:
            Biased and normalized embedding
            
        Raises:
            ValueError: If scale is not supported
            
        Example:
            >>> engine = ContrastiveClusteringEngine()
            >>> embedding = [0.1, 0.2, 0.3, 0.4]
            >>> biased = engine.apply_contrastive_bias(
            ...     embedding, "code", "coarse"
            ... )
        """
        if scale not in self.bias_strengths:
            raise ValueError(
                f"Unsupported scale '{scale}'. "
                f"Must be one of: {list(self.bias_strengths.keys())}"
            )
        
        # Convert to numpy array for efficient computation
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_dim = len(embedding_array)
        
        # Generate category-specific bias vector
        bias_vector = self.generate_category_bias(category, embedding_dim)
        
        # Apply scale-aware bias strength
        bias_strength = self.bias_strengths[scale]
        scaled_bias = bias_vector * bias_strength
        
        # Add bias to original embedding
        biased_embedding = embedding_array + scaled_bias
        
        # Normalize to maintain unit sphere properties
        normalized_embedding = self.normalize_embedding(biased_embedding)
        
        logger.debug(
            "Applied contrastive bias: category=%s, scale=%s, "
            "bias_strength=%.1f, original_norm=%.3f, final_norm=%.3f",
            category, scale, bias_strength,
            np.linalg.norm(embedding_array),
            np.linalg.norm(normalized_embedding)
        )
        
        return normalized_embedding.tolist()
    
    def generate_category_bias(
        self,
        category: str,
        embedding_dim: int
    ) -> np.ndarray:
        """
        Generate reproducible category-specific bias vector.
        
        Uses deterministic random generation based on category name
        to ensure same category always gets same bias vector.
        This creates consistent clustering behavior.
        
        Args:
            category: Content category
            embedding_dim: Dimension of embedding space
            
        Returns:
            Category-specific bias vector
        """
        # Create cache key
        cache_key = f"{category}_{embedding_dim}"
        
        # Return cached bias if available
        if cache_key in self.bias_cache:
            return self.bias_cache[cache_key]
        
        # Generate deterministic seed from category name
        category_hash = hashlib.md5(category.encode()).hexdigest()
        category_seed = int(category_hash[:8], 16) + self.seed
        
        # Generate reproducible random bias vector
        rng = np.random.RandomState(category_seed)
        
        # Generate bias vector with small magnitude
        # Use normal distribution for better clustering properties
        bias_vector = rng.normal(0, 0.1, embedding_dim).astype(np.float32)
        
        # Normalize bias vector to unit length for consistent scaling
        bias_norm = np.linalg.norm(bias_vector)
        if bias_norm > 0:
            bias_vector = bias_vector / bias_norm
        
        # Scale to appropriate magnitude (small bias for subtle clustering)
        bias_vector = bias_vector * 0.1
        
        # Cache for future use
        self.bias_cache[cache_key] = bias_vector
        
        logger.debug(
            "Generated category bias: category=%s, dim=%d, "
            "seed=%d, norm=%.3f",
            category, embedding_dim, category_seed,
            np.linalg.norm(bias_vector)
        )
        
        return bias_vector
    
    def normalize_embedding(
        self,
        embedding: np.ndarray
    ) -> np.ndarray:
        """
        Normalize embedding to unit sphere.
        
        Maintains unit sphere properties required for cosine similarity
        and consistent distance calculations.
        
        Args:
            embedding: Embedding vector to normalize
            
        Returns:
            Normalized embedding with unit length
        """
        # Calculate L2 norm
        norm = np.linalg.norm(embedding)
        
        # Avoid division by zero
        if norm == 0:
            logger.warning("Zero norm embedding encountered, returning zeros")
            return embedding
        
        # Normalize to unit length
        normalized = embedding / norm
        
        return normalized
    
    def apply_batch_contrastive_bias(
        self,
        embeddings: List[List[float]],
        categories: List[str],
        scales: List[str]
    ) -> List[List[float]]:
        """
        Apply contrastive bias to batch of embeddings efficiently.
        
        Args:
            embeddings: List of embedding vectors
            categories: List of categories for each embedding
            scales: List of scales for each embedding
            
        Returns:
            List of biased and normalized embeddings
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(embeddings) == len(categories) == len(scales)):
            raise ValueError(
                "embeddings, categories, and scales must have same length"
            )
        
        if not embeddings:
            return []
        
        logger.debug(
            "Applying batch contrastive bias to %d embeddings",
            len(embeddings)
        )
        
        # Process each embedding
        biased_embeddings = []
        for embedding, category, scale in zip(embeddings, categories, scales):
            biased_embedding = self.apply_contrastive_bias(
                embedding, category, scale
            )
            biased_embeddings.append(biased_embedding)
        
        return biased_embeddings
    
    def compute_category_separation(
        self,
        embeddings: List[List[float]],
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Compute separation metrics between categories.
        
        Measures how well the contrastive clustering separates
        different categories and groups similar categories.
        
        Args:
            embeddings: List of embedding vectors
            categories: List of categories for each embedding
            
        Returns:
            Dictionary with separation metrics
        """
        if len(embeddings) != len(categories):
            raise ValueError(
                "embeddings and categories must have same length"
            )
        
        if len(embeddings) < 2:
            return {"intra_category_distance": 0.0, 
                   "inter_category_distance": 0.0,
                   "separation_ratio": 0.0}
        
        embeddings_array = np.array(embeddings)
        
        # Group embeddings by category
        category_groups = {}
        for i, category in enumerate(categories):
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(embeddings_array[i])
        
        # Compute intra-category distances (should be small)
        intra_distances = []
        for category, group_embeddings in category_groups.items():
            if len(group_embeddings) > 1:
                group_array = np.array(group_embeddings)
                # Compute pairwise distances within category
                for i in range(len(group_array)):
                    for j in range(i + 1, len(group_array)):
                        distance = np.linalg.norm(
                            group_array[i] - group_array[j]
                        )
                        intra_distances.append(distance)
        
        # Compute inter-category distances (should be large)
        inter_distances = []
        category_list = list(category_groups.keys())
        for i in range(len(category_list)):
            for j in range(i + 1, len(category_list)):
                cat1_embeddings = np.array(category_groups[category_list[i]])
                cat2_embeddings = np.array(category_groups[category_list[j]])
                
                # Compute distances between different categories
                for emb1 in cat1_embeddings:
                    for emb2 in cat2_embeddings:
                        distance = np.linalg.norm(emb1 - emb2)
                        inter_distances.append(distance)
        
        # Calculate metrics
        avg_intra = np.mean(intra_distances) if intra_distances else 0.0
        avg_inter = np.mean(inter_distances) if inter_distances else 0.0
        separation_ratio = (
            avg_inter / avg_intra if avg_intra > 0 else float('inf')
        )
        
        metrics = {
            "intra_category_distance": float(avg_intra),
            "inter_category_distance": float(avg_inter),
            "separation_ratio": float(separation_ratio),
            "num_categories": len(category_groups),
            "total_embeddings": len(embeddings)
        }
        
        logger.debug(
            "Category separation metrics: intra=%.3f, inter=%.3f, "
            "ratio=%.3f, categories=%d",
            avg_intra, avg_inter, separation_ratio, len(category_groups)
        )
        
        return metrics
    
    def clear_cache(self):
        """Clear the bias vector cache."""
        self.bias_cache.clear()
        logger.debug("Cleared bias vector cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the bias vector cache.
        
        Returns:
            Dictionary with cache statistics
        """
        cached_categories = []
        cached_dimensions = []
        
        for key in self.bias_cache.keys():
            # Split from the right to handle categories with underscores
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                category, dim_str = parts
                try:
                    dimension = int(dim_str)
                    cached_categories.append(category)
                    cached_dimensions.append(dimension)
                except ValueError:
                    # Skip malformed keys
                    continue
        
        return {
            "cache_size": len(self.bias_cache),
            "cached_categories": list(set(cached_categories)),
            "cached_dimensions": list(set(cached_dimensions))
        }


# Singleton instance for global access
_contrastive_clustering_engine = None


def get_contrastive_clustering_engine() -> ContrastiveClusteringEngine:
    """
    Get singleton contrastive clustering engine instance.
    
    Returns:
        Shared ContrastiveClusteringEngine instance
    """
    global _contrastive_clustering_engine
    
    if _contrastive_clustering_engine is None:
        _contrastive_clustering_engine = ContrastiveClusteringEngine()
    
    return _contrastive_clustering_engine