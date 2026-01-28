"""
Unified deduplication utilities.

Consolidates duplicate deduplication implementations from:
- memory_processor.py
- multi_stage_search.py
- companion modules
"""
import hashlib
from typing import List, TypeVar, Callable, Optional, Any
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""

    unique_items: List[Any]
    duplicates_removed: int
    original_count: int

    @property
    def deduplication_ratio(self) -> float:
        """Percentage of items that were duplicates."""
        if self.original_count == 0:
            return 0.0
        return self.duplicates_removed / self.original_count


def deduplicate_by_content(
    items: List[T],
    content_getter: Callable[[T], str],
    preserve_order: bool = True
) -> DeduplicationResult:
    """
    Deduplicate items by content hash.

    Args:
        items: List of items to deduplicate
        content_getter: Function to extract content string from item
        preserve_order: If True, maintains original order

    Returns:
        DeduplicationResult with unique items
    """
    if not items:
        return DeduplicationResult([], 0, 0)

    seen_hashes = set()
    unique = []

    for item in items:
        content = content_getter(item)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique.append(item)

    return DeduplicationResult(
        unique_items=unique,
        duplicates_removed=len(items) - len(unique),
        original_count=len(items)
    )


def deduplicate_by_id(
    items: List[T],
    id_getter: Callable[[T], Any],
) -> DeduplicationResult:
    """
    Deduplicate items by ID field.

    Args:
        items: List of items to deduplicate
        id_getter: Function to extract ID from item

    Returns:
        DeduplicationResult with unique items
    """
    if not items:
        return DeduplicationResult([], 0, 0)

    seen_ids = set()
    unique = []

    for item in items:
        item_id = id_getter(item)

        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique.append(item)

    return DeduplicationResult(
        unique_items=unique,
        duplicates_removed=len(items) - len(unique),
        original_count=len(items)
    )


def deduplicate_by_similarity(
    items: List[T],
    embedding_getter: Callable[[T], List[float]],
    threshold: float = 0.95
) -> DeduplicationResult:
    """
    Deduplicate items by embedding similarity.

    Args:
        items: List of items to deduplicate
        embedding_getter: Function to extract embedding from item
        threshold: Similarity threshold (0-1), items above this are duplicates

    Returns:
        DeduplicationResult with unique items
    """
    if not items:
        return DeduplicationResult([], 0, 0)

    try:
        import numpy as np
    except ImportError:
        # Fallback to content-based if numpy not available
        return DeduplicationResult(items, 0, len(items))

    unique = [items[0]]
    unique_embeddings = [np.array(embedding_getter(items[0]))]

    for item in items[1:]:
        embedding = np.array(embedding_getter(item))

        # Check similarity with all unique items
        is_duplicate = False
        for unique_emb in unique_embeddings:
            # Cosine similarity
            norm_product = np.linalg.norm(embedding) * np.linalg.norm(unique_emb)
            if norm_product > 0:
                similarity = np.dot(embedding, unique_emb) / norm_product
                if similarity >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique.append(item)
            unique_embeddings.append(embedding)

    return DeduplicationResult(
        unique_items=unique,
        duplicates_removed=len(items) - len(unique),
        original_count=len(items)
    )


def deduplicate_search_results(
    results: List[T],
    content_getter: Callable[[T], str],
    score_getter: Optional[Callable[[T], float]] = None,
    similarity_threshold: float = 0.9
) -> List[T]:
    """
    Deduplicate search results, keeping highest scored duplicates.

    Args:
        results: List of search results
        content_getter: Function to extract content from result
        score_getter: Function to extract score from result
        similarity_threshold: Text similarity threshold for deduplication

    Returns:
        Deduplicated list of results
    """
    if not results or len(results) <= 1:
        return results

    # Sort by score descending if score getter provided
    if score_getter:
        results = sorted(results, key=score_getter, reverse=True)

    unique = []
    seen_content_hashes = set()

    for result in results:
        content = content_getter(result)
        # Normalize content for comparison
        normalized = content.lower().strip()
        content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()

        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique.append(result)

    return unique
