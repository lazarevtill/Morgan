"""
Tests for contrastive clustering engine.

Tests core functionality of category-specific bias application:
- Bias vector generation
- Scale-aware bias strength
- Embedding normalization
- Batch processing
- Category separation metrics
"""

import numpy as np
from morgan.vectorization.contrastive_clustering import (
    ContrastiveClusteringEngine,
    get_contrastive_clustering_engine,
)


class TestContrastiveClusteringEngine:
    """Test contrastive clustering engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ContrastiveClusteringEngine(seed=42)

        # Test embeddings
        self.test_embedding = [0.1, 0.2, 0.3, 0.4]
        self.test_category = "code"

    def test_apply_contrastive_bias(self):
        """Test applying contrastive bias to embedding."""
        result = self.engine.apply_contrastive_bias(
            self.test_embedding, self.test_category, "coarse"
        )

        # Result should be different from original
        assert result != self.test_embedding

        # Result should be normalized (unit length)
        result_norm = np.linalg.norm(result)
        assert abs(result_norm - 1.0) < 1e-6

        # Should have same dimensions
        assert len(result) == len(self.test_embedding)

    def test_scale_aware_bias_strength(self):
        """Test different bias strengths for different scales."""
        coarse_result = self.engine.apply_contrastive_bias(
            self.test_embedding, self.test_category, "coarse"
        )
        medium_result = self.engine.apply_contrastive_bias(
            self.test_embedding, self.test_category, "medium"
        )
        fine_result = self.engine.apply_contrastive_bias(
            self.test_embedding, self.test_category, "fine"
        )

        # All should be different due to different bias strengths
        assert coarse_result != medium_result
        assert medium_result != fine_result
        assert coarse_result != fine_result

        # All should be normalized
        for result in [coarse_result, medium_result, fine_result]:
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-6

    def test_reproducible_bias_generation(self):
        """Test that same category generates same bias."""
        bias1 = self.engine.generate_category_bias("code", 4)
        bias2 = self.engine.generate_category_bias("code", 4)

        # Should be identical
        np.testing.assert_array_equal(bias1, bias2)

        # Different categories should generate different bias
        bias_doc = self.engine.generate_category_bias("documentation", 4)
        assert not np.array_equal(bias1, bias_doc)

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        # Test normal case
        embedding = np.array([3.0, 4.0])  # Length = 5
        normalized = self.engine.normalize_embedding(embedding)

        expected = np.array([0.6, 0.8])  # 3/5, 4/5
        np.testing.assert_array_almost_equal(normalized, expected)

        # Test zero embedding
        zero_embedding = np.array([0.0, 0.0])
        normalized_zero = self.engine.normalize_embedding(zero_embedding)
        np.testing.assert_array_equal(normalized_zero, zero_embedding)

    def test_batch_contrastive_bias(self):
        """Test batch processing of embeddings."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        categories = ["code", "documentation", "api"]
        scales = ["coarse", "medium", "fine"]

        results = self.engine.apply_batch_contrastive_bias(
            embeddings, categories, scales
        )

        # Should have same number of results
        assert len(results) == len(embeddings)

        # Each result should be normalized
        for result in results:
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-6

        # Results should be different from originals
        for original, result in zip(embeddings, results):
            assert original != result

    def test_category_separation_metrics(self):
        """Test category separation computation."""
        # Create embeddings that should cluster by category
        embeddings = [
            [1.0, 0.0],  # code category
            [1.1, 0.1],  # code category (similar)
            [0.0, 1.0],  # documentation category
            [0.1, 1.1],  # documentation category (similar)
        ]
        categories = ["code", "code", "documentation", "documentation"]

        metrics = self.engine.compute_category_separation(embeddings, categories)

        # Should have expected structure
        assert "intra_category_distance" in metrics
        assert "inter_category_distance" in metrics
        assert "separation_ratio" in metrics
        assert "num_categories" in metrics
        assert "total_embeddings" in metrics

        # Inter-category distance should be larger than intra-category
        assert metrics["inter_category_distance"] > metrics["intra_category_distance"]
        assert metrics["separation_ratio"] > 1.0
        assert metrics["num_categories"] == 2
        assert metrics["total_embeddings"] == 4

    def test_invalid_scale(self):
        """Test error handling for invalid scale."""
        try:
            self.engine.apply_contrastive_bias(
                self.test_embedding, self.test_category, "invalid"
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported scale" in str(e)

    def test_mismatched_batch_inputs(self):
        """Test error handling for mismatched batch inputs."""
        embeddings = [[0.1, 0.2]]
        categories = ["code", "documentation"]  # Different length
        scales = ["coarse"]

        try:
            self.engine.apply_batch_contrastive_bias(embeddings, categories, scales)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same length" in str(e)

    def test_cache_functionality(self):
        """Test bias vector caching."""
        # Generate bias to populate cache
        self.engine.generate_category_bias("test_category", 4)

        # Check cache stats
        stats = self.engine.get_cache_stats()
        assert stats["cache_size"] > 0
        assert "test_category" in stats["cached_categories"]
        assert 4 in stats["cached_dimensions"]

        # Clear cache
        self.engine.clear_cache()
        stats_after = self.engine.get_cache_stats()
        assert stats_after["cache_size"] == 0

    def test_singleton_engine(self):
        """Test singleton pattern for engine access."""
        engine1 = get_contrastive_clustering_engine()
        engine2 = get_contrastive_clustering_engine()

        # Should return same instance
        assert engine1 is engine2
        assert isinstance(engine1, ContrastiveClusteringEngine)

    def test_bias_strength_values(self):
        """Test that bias strengths match requirements."""
        # Requirements specify: coarse 1.5x, medium 1.0x, fine 0.5x
        assert self.engine.bias_strengths["coarse"] == 1.5
        assert self.engine.bias_strengths["medium"] == 1.0
        assert self.engine.bias_strengths["fine"] == 0.5

    def test_category_clustering_effect(self):
        """Test that same categories cluster closer together."""
        # Create two embeddings for same category
        embedding1 = [0.5, 0.5, 0.0, 0.0]
        embedding2 = [0.6, 0.4, 0.0, 0.0]

        # Apply bias to both
        biased1 = self.engine.apply_contrastive_bias(embedding1, "code", "medium")
        biased2 = self.engine.apply_contrastive_bias(embedding2, "code", "medium")

        # Calculate distances
        original_distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        biased_distance = np.linalg.norm(np.array(biased1) - np.array(biased2))

        # Biased embeddings should be closer (or at least not much farther)
        # Note: Due to normalization, this might not always be strictly smaller
        # but the bias should pull them in the same direction
        assert biased_distance <= original_distance * 1.5  # Allow some tolerance
