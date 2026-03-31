# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hybrid Memory Search.

Combines BM25-inspired keyword scoring with vector cosine similarity
for searching over daily memory logs.

BM25 parameters: k1=1.5, b=0.75
Hybrid weights default: vector_weight=0.6, keyword_weight=0.4

Ported from OpenClaw's memory-core and Claude Code's memdir patterns.
"""

import math
import re
from pathlib import Path
from typing import Dict, List, Optional


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridMemorySearch:
    """
    Search over daily memory log files using BM25 keyword scoring
    and optional vector cosine similarity.

    Args:
        memory_dir: Directory containing YYYY-MM-DD.md log files.

    Example:
        >>> search = HybridMemorySearch(Path("memory"))
        >>> results = search.keyword_search("Python refactoring", limit=5)
        >>> results = search.hybrid_search("API design", vector_weight=0.6)
    """

    # BM25 parameters
    K1 = 1.5
    B = 0.75

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = Path(memory_dir)

    def _load_documents(self) -> List[Dict]:
        """
        Load all .md log files from the memory directory.

        Returns:
            List of dicts with ``path``, ``content``, and ``tokens`` keys.
        """
        if not self.memory_dir.exists():
            return []

        docs = []
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

        for log_file in sorted(self.memory_dir.iterdir()):
            if not log_file.is_file() or not date_pattern.match(log_file.name):
                continue
            content = log_file.read_text(encoding="utf-8")
            tokens = _tokenize(content)
            docs.append(
                {
                    "path": str(log_file),
                    "content": content,
                    "tokens": tokens,
                }
            )
        return docs

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        BM25-inspired keyword search over memory logs.

        Score formula per query term per document:
            idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))

        Where:
            - tf = term frequency in document
            - dl = document length (token count)
            - avgdl = average document length
            - idf = log((N - df + 0.5) / (df + 0.5) + 1)
            - N = total number of documents
            - df = number of documents containing the term

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with ``path``, ``content``, and ``score`` keys,
            sorted by score descending.
        """
        docs = self._load_documents()
        if not docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(docs)
        avgdl = sum(len(d["tokens"]) for d in docs) / n

        # Compute document frequency for each query term
        df: Dict[str, int] = {}
        for term in query_tokens:
            df[term] = sum(1 for d in docs if term in d["tokens"])

        # Score each document
        scored = []
        for doc in docs:
            dl = len(doc["tokens"])
            score = 0.0

            for term in query_tokens:
                if df[term] == 0:
                    continue

                tf = doc["tokens"].count(term)
                if tf == 0:
                    continue

                # IDF component
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)

                # TF component with length normalization
                tf_norm = (tf * (self.K1 + 1.0)) / (
                    tf + self.K1 * (1.0 - self.B + self.B * (dl / avgdl))
                )

                score += idf * tf_norm

            if score > 0:
                scored.append(
                    {
                        "path": doc["path"],
                        "content": doc["content"],
                        "score": score,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 keyword scores with vector cosine
        similarity. Falls back to keyword-only if embeddings are unavailable.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            vector_weight: Weight for vector similarity scores (default 0.6).
            keyword_weight: Weight for keyword BM25 scores (default 0.4).

        Returns:
            List of dicts with ``path``, ``content``, and ``score`` keys,
            sorted by combined score descending.
        """
        # Get keyword results first (always available)
        keyword_results = self.keyword_search(query, limit=limit * 2)

        # Try to get vector scores
        vector_scores: Optional[Dict[str, float]] = None
        try:
            vector_scores = self._compute_vector_scores(query)
        except Exception:
            # Embedding service unavailable, fall back to keyword-only
            pass

        if vector_scores is None:
            # Pure keyword fallback
            return keyword_results[:limit]

        # Normalize keyword scores to [0, 1]
        kw_by_path: Dict[str, Dict] = {}
        max_kw_score = max(
            (r["score"] for r in keyword_results), default=0.0
        )

        for r in keyword_results:
            norm_score = r["score"] / max_kw_score if max_kw_score > 0 else 0.0
            kw_by_path[r["path"]] = {
                "content": r["content"],
                "kw_score": norm_score,
            }

        # Collect all candidate paths
        all_paths = set(kw_by_path.keys()) | set(vector_scores.keys())

        # Normalize vector scores to [0, 1]
        max_vec_score = max(vector_scores.values(), default=0.0)

        combined = []
        for path in all_paths:
            kw_data = kw_by_path.get(path)
            kw_score = kw_data["kw_score"] if kw_data else 0.0
            content = kw_data["content"] if kw_data else ""

            vec_score = vector_scores.get(path, 0.0)
            if max_vec_score > 0:
                vec_score = vec_score / max_vec_score

            # If we don't have content yet (only found via vector), load it
            if not content and Path(path).exists():
                content = Path(path).read_text(encoding="utf-8")

            final_score = (
                keyword_weight * kw_score + vector_weight * vec_score
            )

            if final_score > 0:
                combined.append(
                    {
                        "path": path,
                        "content": content,
                        "score": final_score,
                    }
                )

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    def _compute_vector_scores(self, query: str) -> Dict[str, float]:
        """
        Compute cosine similarity between query embedding and each document.

        Raises if embedding service is unavailable.

        Args:
            query: Search query string.

        Returns:
            Dict mapping file path to cosine similarity score.
        """
        from morgan.services import get_embedding_service

        embedding_svc = get_embedding_service()
        query_embedding = embedding_svc.encode(query)

        docs = self._load_documents()
        scores = {}

        for doc in docs:
            doc_embedding = embedding_svc.encode(doc["content"])
            similarity = _cosine_similarity(query_embedding, doc_embedding)
            # Shift from [-1, 1] to [0, 1] for scoring
            scores[doc["path"]] = max(0.0, similarity)

        return scores
