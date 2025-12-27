"""
Unified Deduplication Utility for Morgan AI Assistant.

Provides a single deduplication implementation that can be used across
memory processing, search, and companion modules.

Usage:
    from morgan.utils.deduplication import ResultDeduplicator

    # Create deduplicator
    deduplicator = ResultDeduplicator(
        strategy="content_hash",
        similarity_threshold=0.95
    )

    # Deduplicate results
    unique_results = deduplicator.deduplicate(results, key_fn=lambda x: x.content)

    # With embeddings for semantic deduplication
    unique_results = deduplicator.deduplicate_semantic(
        results,
        embeddings,
        key_fn=lambda x: x.content
    )
"""

import hashlib
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from morgan.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DeduplicationStrategy(str, Enum):
    """Deduplication strategies."""

    CONTENT_HASH = "content_hash"  # Hash-based exact matching
    SEMANTIC = "semantic"  # Embedding-based similarity
    HYBRID = "hybrid"  # Combine hash and semantic


class ResultDeduplicator:
    """
    Unified deduplication utility.

    Supports multiple deduplication strategies:
    - content_hash: Fast hash-based exact matching
    - semantic: Embedding-based similarity matching
    - hybrid: Combine both strategies

    Example:
        >>> deduplicator = ResultDeduplicator()

        >>> # Simple deduplication
        >>> unique = deduplicator.deduplicate(
        ...     items=["hello", "world", "hello"],
        ...     key_fn=lambda x: x
        ... )
        >>> assert unique == ["hello", "world"]

        >>> # With custom objects
        >>> class Result:
        ...     def __init__(self, content, score):
        ...         self.content = content
        ...         self.score = score

        >>> results = [Result("doc1", 0.9), Result("doc1", 0.8), Result("doc2", 0.7)]
        >>> unique = deduplicator.deduplicate(results, key_fn=lambda r: r.content)
        >>> assert len(unique) == 2
    """

    def __init__(
        self,
        strategy: str = "content_hash",
        similarity_threshold: float = 0.95,
        normalize_text: bool = True,
    ):
        """
        Initialize deduplicator.

        Args:
            strategy: Deduplication strategy (content_hash, semantic, hybrid)
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
            normalize_text: Whether to normalize text before hashing
        """
        self.strategy = DeduplicationStrategy(strategy)
        self.similarity_threshold = similarity_threshold
        self.normalize_text = normalize_text

    def deduplicate(
        self,
        items: List[T],
        key_fn: Callable[[T], str],
        keep_first: bool = True,
    ) -> List[T]:
        """
        Deduplicate items using content hash strategy.

        Args:
            items: List of items to deduplicate
            key_fn: Function to extract text content from item
            keep_first: Keep first occurrence (True) or last (False)

        Returns:
            List of unique items
        """
        if not items:
            return []

        seen_hashes: Dict[str, int] = {}
        unique_items: List[T] = []

        for i, item in enumerate(items):
            content = key_fn(item)
            content_hash = self._hash_content(content)

            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = i
                unique_items.append(item)
            elif not keep_first:
                # Replace with later occurrence
                old_idx = seen_hashes[content_hash]
                for j, u in enumerate(unique_items):
                    if key_fn(u) == key_fn(items[old_idx]):
                        unique_items[j] = item
                        break
                seen_hashes[content_hash] = i

        logger.debug(
            "Deduplicated %d items to %d unique (%.1f%% reduction)",
            len(items),
            len(unique_items),
            (1 - len(unique_items) / len(items)) * 100 if items else 0,
        )

        return unique_items

    def deduplicate_semantic(
        self,
        items: List[T],
        embeddings: List[List[float]],
        key_fn: Optional[Callable[[T], str]] = None,
    ) -> List[T]:
        """
        Deduplicate items using semantic similarity.

        Args:
            items: List of items to deduplicate
            embeddings: Pre-computed embeddings for each item
            key_fn: Optional function to extract text (for logging)

        Returns:
            List of unique items
        """
        if not items or len(items) != len(embeddings):
            return items

        try:
            import numpy as np

            # Convert to numpy array
            emb_array = np.array(embeddings)

            # Normalize embeddings
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            emb_normalized = emb_array / norms

            # Track which items to keep
            keep_indices: List[int] = []
            kept_embeddings: List[np.ndarray] = []

            for i, emb in enumerate(emb_normalized):
                if not kept_embeddings:
                    keep_indices.append(i)
                    kept_embeddings.append(emb)
                    continue

                # Calculate similarity to all kept items
                similarities = np.dot(np.array(kept_embeddings), emb)
                max_similarity = np.max(similarities)

                # Keep if not too similar to existing items
                if max_similarity < self.similarity_threshold:
                    keep_indices.append(i)
                    kept_embeddings.append(emb)

            unique_items = [items[i] for i in keep_indices]

            logger.debug(
                "Semantic deduplication: %d items to %d unique (threshold=%.2f)",
                len(items),
                len(unique_items),
                self.similarity_threshold,
            )

            return unique_items

        except ImportError:
            logger.warning("numpy not available, falling back to hash deduplication")
            if key_fn:
                return self.deduplicate(items, key_fn)
            return items

    def deduplicate_hybrid(
        self,
        items: List[T],
        embeddings: List[List[float]],
        key_fn: Callable[[T], str],
    ) -> List[T]:
        """
        Deduplicate using both hash and semantic strategies.

        First removes exact duplicates, then applies semantic deduplication.

        Args:
            items: List of items to deduplicate
            embeddings: Pre-computed embeddings for each item
            key_fn: Function to extract text content from item

        Returns:
            List of unique items
        """
        if not items:
            return []

        # First pass: hash-based deduplication
        hash_unique = self.deduplicate(items, key_fn)

        if len(hash_unique) == len(items):
            # No exact duplicates, apply semantic deduplication
            return self.deduplicate_semantic(items, embeddings, key_fn)

        # Get embeddings for hash-unique items
        unique_indices = []
        seen_hashes = set()
        for i, item in enumerate(items):
            content_hash = self._hash_content(key_fn(item))
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_indices.append(i)

        unique_embeddings = [embeddings[i] for i in unique_indices]

        # Second pass: semantic deduplication
        return self.deduplicate_semantic(hash_unique, unique_embeddings, key_fn)

    def _hash_content(self, content: str) -> str:
        """
        Hash content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hash string
        """
        if self.normalize_text:
            # Normalize: lowercase, strip whitespace, collapse spaces
            content = " ".join(content.lower().split())

        return hashlib.sha256(content.encode()).hexdigest()

    def find_duplicates(
        self,
        items: List[T],
        key_fn: Callable[[T], str],
    ) -> Dict[str, List[int]]:
        """
        Find duplicate groups in items.

        Args:
            items: List of items to analyze
            key_fn: Function to extract text content from item

        Returns:
            Dict mapping content hash to list of indices
        """
        groups: Dict[str, List[int]] = {}

        for i, item in enumerate(items):
            content = key_fn(item)
            content_hash = self._hash_content(content)

            if content_hash not in groups:
                groups[content_hash] = []
            groups[content_hash].append(i)

        # Return only groups with duplicates
        return {k: v for k, v in groups.items() if len(v) > 1}


# Convenience function for simple deduplication
def deduplicate_strings(
    strings: List[str],
    normalize: bool = True,
) -> List[str]:
    """
    Simple string deduplication.

    Args:
        strings: List of strings to deduplicate
        normalize: Whether to normalize text before comparing

    Returns:
        List of unique strings
    """
    deduplicator = ResultDeduplicator(normalize_text=normalize)
    return deduplicator.deduplicate(strings, key_fn=lambda x: x)


def deduplicate_dicts(
    dicts: List[Dict[str, Any]],
    key: str,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Deduplicate list of dictionaries by a key.

    Args:
        dicts: List of dictionaries to deduplicate
        key: Dictionary key to use for deduplication
        normalize: Whether to normalize text before comparing

    Returns:
        List of unique dictionaries
    """
    deduplicator = ResultDeduplicator(normalize_text=normalize)
    return deduplicator.deduplicate(dicts, key_fn=lambda d: str(d.get(key, "")))
